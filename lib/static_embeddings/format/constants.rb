module StaticEmbeddings
  module Format
    MAGIC = "SEMBv1\0\0"
    VERSION = 3
    VERSION_WORDPIECE = VERSION
    HEADER_SIZE = 320
    ALIGNMENT = 64

    TOKENIZER_BERT_WORDPIECE_V1 = 1
    DTYPE_F32 = 1
    POOLING_MEAN = 1
    NORMALIZATION_NONE = 0
    NORMALIZATION_L2 = 1
    TRUNCATE_USABLE_IDS_BEFORE_POOLING = 2
    UNK_INCLUDE = 0
    UNK_DROP = 1
    EMPTY_ZERO_VECTOR = 0
    EMPTY_RAISE = 1

    SLOT_EMPTY = 0xFFFFFFFF
    HASH_SEED = 2_166_136_261
    LOAD_FACTOR = 0.70
    UINT32_MAX = 0xFFFFFFFF
    UINT64_MAX = 0xFFFFFFFFFFFFFFFF

    MAX_TOKEN_CHARS_OFFSET = 124
    CHECKSUM_OFFSET = 240
    CHECKSUM_SIZE = 32
    MAX_PROBE_OFFSET = 304
    ADDED_TOKEN_MASK_OFFSET = 308

    ADDED_PAD  = 1 << 0
    ADDED_UNK  = 1 << 1
    ADDED_CLS  = 1 << 2
    ADDED_SEP  = 1 << 3
    ADDED_MASK = 1 << 4
    ADDED_TOKEN_MASK_ALL = ADDED_PAD | ADDED_UNK | ADDED_CLS | ADDED_SEP | ADDED_MASK

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
      48 => TRUNCATE_USABLE_IDS_BEFORE_POOLING,
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

    module_function

    def binary_string(capacity = nil)
      string = capacity ? String.new(capacity: capacity) : +""
      string.force_encoding(Encoding::BINARY)
    end

    def verify(path)
      require "static_embeddings/format/verifier"
      Verifier.call(path)
    end

    def write(**kwargs)
      require "static_embeddings/format/writer"
      Writer.call(**kwargs)
    end

    def hash_bytes(string, seed = HASH_SEED)
      require "static_embeddings/format/hash_table"
      HashTable.hash_bytes(string, seed)
    end

    def next_power_of_two(value)
      require "static_embeddings/format/hash_table"
      HashTable.next_power_of_two(value)
    end
  end
end
