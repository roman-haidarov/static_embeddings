require "digest"

module StaticEmbeddings
  module Format
    MAGIC = "SEMBv1\0\0"
    VERSION = 2
    HEADER_SIZE = 320
    ALIGNMENT = 64

    TOKENIZER_BERT_WORDPIECE_V1 = 1
    DTYPE_F32 = 1
    POOLING_MEAN = 1
    NORMALIZATION_NONE = 0
    NORMALIZATION_L2 = 1
    TRUNCATE_IDS_BEFORE_POOLING = 1
    UNK_INCLUDE = 0
    UNK_DROP = 1
    EMPTY_ZERO_VECTOR = 0
    EMPTY_RAISE = 1

    SLOT_EMPTY = 0xFFFFFFFF
    HASH_SEED = 2_166_136_261
    LOAD_FACTOR = 0.70

    MAX_TOKEN_CHARS_OFFSET = 124
    CHECKSUM_OFFSET = 240
    CHECKSUM_SIZE = 32
    MAX_PROBE_OFFSET = 304

    SECTION_FIELDS = {
      vocab_strings: 128,
      vocab_hash: 144,
      embeddings: 160,
      norm_tables: 176,
      provenance: 192,
      root_trie: 208,
      continuation_trie: 224
    }.freeze

    HEADER_U32 = {
      8 => VERSION,
      12 => HEADER_SIZE,
      16 => 1,
      28 => TOKENIZER_BERT_WORDPIECE_V1,
      32 => DTYPE_F32,
      36 => POOLING_MEAN,
      48 => TRUNCATE_IDS_BEFORE_POOLING,
      108 => HASH_SEED
    }.freeze

    META_U32 = {
      20 => :dim,
      40 => :normalization_type,
      44 => :max_tokens_default,
      56 => :unk_policy,
      60 => :empty_policy,
      80 => :max_input_chars_per_word,
      84 => :pad_id,
      88 => :unk_id,
      92 => :cls_id,
      96 => :sep_id,
      100 => :mask_id
    }.freeze

    META_BOOL = {
      52 => :add_special_tokens,
      64 => :do_lower_case,
      68 => :strip_accents,
      72 => :handle_chinese_chars,
      76 => :clean_text
    }.freeze

    VERIFY_CHUNK_BYTES = 1024 * 1024

    module_function

    def hash_bytes(str, seed = HASH_SEED)
      str.each_byte.reduce(seed) { |h, b| ((h ^ b) * 16_777_619) & 0xFFFFFFFF }
    end

    def next_power_of_two(n)
      1 << (n - 1).bit_length
    end

    def build_hash_table(tokens)
      size = next_power_of_two((tokens.length / LOAD_FACTOR).ceil + 1)
      slots = Array.new(size)
      strings = binary_string
      max_probe = 0

      tokens.each_with_index do |token, id|
        bytes = token.b
        offset = strings.bytesize
        strings << bytes
        probe = insert_slot!(slots, tokens, size, bytes, hash_bytes(bytes), offset, id)
        max_probe = probe if probe > max_probe
      end

      [size, strings, pack_slots(slots), max_probe]
    end

    def write(path:, meta:, tokens:, matrix:, norm_tables:, provenance:)
      hash_size, strings, hash_blob, max_probe = build_hash_table(tokens)
      root_trie, continuation_trie = build_wordpiece_tries(tokens, meta.fetch(:subword_prefix))
      sections, body = build_body(
        vocab_strings: strings,
        vocab_hash: hash_blob,
        embeddings: matrix,
        norm_tables: norm_tables,
        provenance: provenance,
        root_trie: root_trie,
        continuation_trie: continuation_trie
      )

      header = build_header(meta, tokens.length, hash_size, max_probe, sections)
      file = header << body
      digest = Digest::SHA256.digest(file)
      file[CHECKSUM_OFFSET, CHECKSUM_SIZE] = digest

      File.binwrite(path, file)
      { bytes: file.bytesize, sha256: digest.unpack1("H*"), hash_table_size: hash_size }
    end

    def verify(path)
      raise InvalidModelError, "file too small" if File.size(path) < HEADER_SIZE

      File.open(path, "rb") do |io|
        header = io.read(HEADER_SIZE)
        raise InvalidModelError, "bad magic" unless header.byteslice(0, 8) == MAGIC.b

        stored = header.byteslice(CHECKSUM_OFFSET, CHECKSUM_SIZE)
        actual = streaming_checksum(io, header)

        { ok: stored == actual, expected: actual.unpack1("H*"), stored: stored.unpack1("H*") }
      end
    end

    def streaming_checksum(io, header)
      digest = Digest::SHA256.new
      zeroed = header.dup
      zeroed[CHECKSUM_OFFSET, CHECKSUM_SIZE] = "\0".b * CHECKSUM_SIZE
      digest << zeroed

      buffer = String.new(capacity: VERIFY_CHUNK_BYTES)
      digest << buffer while io.read(VERIFY_CHUNK_BYTES, buffer)
      digest.digest
    end

    def binary_string(capacity = nil)
      str = capacity ? String.new(capacity: capacity) : +""
      str.force_encoding(Encoding::BINARY)
    end

    def insert_slot!(slots, tokens, size, bytes, hash, offset, id)
      pos = hash & (size - 1)
      probe = 1
      loop do
        slot = slots[pos]
        unless slot
          slots[pos] = [hash, offset, bytes.bytesize, id]
          return probe
        end

        if slot[0] == hash && slot[2] == bytes.bytesize && tokens.fetch(slot[3]).b == bytes
          raise ArgumentError, "duplicate token in vocabulary: #{tokens.fetch(slot[3]).inspect}"
        end

        pos = (pos + 1) & (size - 1)
        probe += 1
      end
    end

    def pack_slots(slots)
      empty = [0, 0, 0, SLOT_EMPTY].pack("V4")
      packed = binary_string(slots.length * 16)
      slots.each { |slot| packed << (slot ? slot.pack("V4") : empty) }
      packed
    end

    def build_wordpiece_tries(tokens, prefix)
      root = TrieBuilder.new
      continuation = TrieBuilder.new
      prefix_bytes = prefix.b

      tokens.each_with_index do |token, id|
        bytes = token.b
        if !prefix_bytes.empty? && bytes.start_with?(prefix_bytes) && bytes.bytesize > prefix_bytes.bytesize
          continuation.insert(bytes.byteslice(prefix_bytes.bytesize, bytes.bytesize - prefix_bytes.bytesize), id)
        else
          root.insert(bytes, id)
        end
      end

      [root.pack, continuation.pack]
    end

    class TrieBuilder
      Node = Struct.new(:terminal, :children, keyword_init: true)

      def initialize
        @nodes = [Node.new(terminal: SLOT_EMPTY, children: {})]
      end

      def insert(bytes, id)
        return if bytes.empty?

        node_index = 0
        bytes.each_byte do |byte|
          node = @nodes[node_index]
          child = node.children[byte]
          unless child
            child = @nodes.length
            node.children[byte] = child
            @nodes << Node.new(terminal: SLOT_EMPTY, children: {})
          end
          node_index = child
        end

        node = @nodes[node_index]
        raise ArgumentError, "duplicate trie key for token id #{id}" unless node.terminal == SLOT_EMPTY

        node.terminal = id
      end

      def pack
        edges = []
        node_records = @nodes.map do |node|
          start = edges.length
          node.children.sort_by { |byte, _| byte }.each do |byte, child|
            edges << [byte, child]
          end
          [start, node.children.length, node.terminal, 0]
        end

        packed = Format.binary_string(16 + node_records.length * 16 + edges.length * 8)
        packed << [node_records.length, edges.length, 0, 0].pack("V4")
        node_records.each { |record| packed << record.pack("V4") }
        edges.each { |edge| packed << edge.pack("V2") }
        packed
      end
    end

    def build_body(payloads)
      body = binary_string
      sections = {}
      payloads.each do |name, payload|
        align_body!(body)
        sections[name] = [HEADER_SIZE + body.bytesize, payload.bytesize]
        body << payload
      end
      [sections, body]
    end

    def align_body!(body)
      padding = (ALIGNMENT - ((HEADER_SIZE + body.bytesize) % ALIGNMENT)) % ALIGNMENT
      body << "\0".b * padding if padding.positive?
    end

    def build_header(meta, vocab_size, hash_size, max_probe, sections)
      header = "\0".b * HEADER_SIZE
      header[0, 8] = MAGIC.b

      HEADER_U32.each { |offset, value| put_u32(header, offset, value) }
      META_U32.each { |offset, key| put_u32(header, offset, meta.fetch(key)) }
      META_BOOL.each { |offset, key| put_u32(header, offset, meta.fetch(key) ? 1 : 0) }

      put_u32(header, 24, vocab_size)
      put_u32(header, 104, hash_size)
      put_u32(header, MAX_TOKEN_CHARS_OFFSET, meta.fetch(:max_token_chars))
      put_u32(header, MAX_PROBE_OFFSET, max_probe)
      put_prefix(header, meta.fetch(:subword_prefix))
      SECTION_FIELDS.each { |name, field| put_section(header, field, sections.fetch(name)) }

      header
    end

    def put_prefix(header, prefix)
      bytes = prefix.b
      raise ArgumentError, "subword prefix too long" if bytes.bytesize > 8

      put_u32(header, 112, bytes.bytesize)
      header[116, 8] = bytes.ljust(8, "\0")
    end

    def put_section(header, offset, section)
      off, size = section
      put_u64(header, offset, off)
      put_u64(header, offset + 8, size)
    end

    def put_u32(buffer, offset, value)
      buffer[offset, 4] = [value].pack("V")
    end

    def put_u64(buffer, offset, value)
      buffer[offset, 8] = [value & 0xFFFFFFFF, value >> 32].pack("V2")
    end

  end
end
