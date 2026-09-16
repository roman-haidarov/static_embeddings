require "digest"
require "static_embeddings/format/constants"
require "static_embeddings/format/hash_table"
require "static_embeddings/format/trie"

module StaticEmbeddings
  module Format
    module Writer
      module_function

      def call(path:, meta:, tokens:, matrix:, norm_tables:, provenance:)
        hash_size, strings, hash_blob, max_probe = HashTable.build(tokens)
        root_trie, continuation_trie = WordPieceTrie.build(tokens, meta.fetch(:subword_prefix))
        payloads = payloads(strings, hash_blob, matrix, norm_tables, provenance, root_trie, continuation_trie)
        sections, file_size = layout(payloads)
        header = build_header(meta, tokens.length, hash_size, max_probe, sections)
        checksum = write_file(path, header, payloads, sections)

        { bytes: file_size, sha256: checksum.unpack1("H*"), hash_table_size: hash_size }
      end

      def payloads(strings, hash_blob, matrix, norm_tables, provenance, root_trie, continuation_trie)
        {
          vocab_strings: strings,
          vocab_hash: hash_blob,
          embeddings: matrix,
          norm_tables: norm_tables,
          provenance: provenance,
          root_trie: root_trie,
          continuation_trie: continuation_trie
        }
      end

      def layout(payloads)
        offset = HEADER_SIZE
        sections = payloads.each_with_object({}) do |(name, payload), out|
          offset += (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT
          out[name] = [offset, payload.bytesize]
          offset += payload.bytesize
        end
        [sections, offset]
      end

      def write_file(path, header, payloads, sections)
        digest = Digest::SHA256.new
        File.open(path, "wb") do |io|
          write_chunk(io, digest, header)
          offset = HEADER_SIZE

          payloads.each do |name, payload|
            target = sections.fetch(name).first
            padding = target - offset
            write_chunk(io, digest, "\0".b * padding) if padding.positive?
            write_payload(io, digest, payload)
            offset = target + payload.bytesize
          end
        end

        digest.digest.tap do |checksum|
          File.open(path, "r+b") do |io|
            io.seek(CHECKSUM_OFFSET, IO::SEEK_SET)
            io.write(checksum)
          end
        end
      end

      def write_payload(io, digest, payload)
        if payload.respond_to?(:each_chunk)
          payload.each_chunk { |chunk| write_chunk(io, digest, chunk) }
        else
          write_chunk(io, digest, payload)
        end
      end

      def write_chunk(io, digest, chunk)
        io.write(chunk)
        digest << chunk
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
        put_u32(header, ADDED_TOKEN_MASK_OFFSET, meta.fetch(:added_token_mask, 0))
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
        put_u64(header, offset, section.fetch(0))
        put_u64(header, offset + 8, section.fetch(1))
      end

      def put_u32(buffer, offset, value)
        integer = Integer(value)
        unless integer.between?(0, UINT32_MAX)
          raise ArgumentError, "u32 value out of range: #{integer}"
        end
        buffer[offset, 4] = [integer].pack("V")
      end

      def put_u64(buffer, offset, value)
        integer = Integer(value)
        unless integer.between?(0, UINT64_MAX)
          raise ArgumentError, "u64 value out of range: #{integer}"
        end
        buffer[offset, 8] = [integer & UINT32_MAX, integer >> 32].pack("V2")
      end
    end

    def self.write(**kwargs)
      Writer.call(**kwargs)
    end
  end
end
