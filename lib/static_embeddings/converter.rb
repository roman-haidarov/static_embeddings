require "json"
require "digest"

module StaticEmbeddings
  class Converter
    REFERENCE_IMPL = "model2vec.StaticModel"
    REFERENCE_MAX_TOKENS = 512

    ALLOWED_NORMALIZER_KEYS = %w[type clean_text handle_chinese_chars strip_accents lowercase].freeze
    STANDARD_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"].freeze
    SOURCE_FILES = %w[tokenizer.json config.json tokenizer_config.json model.safetensors].freeze
    TOKENIZER_PROFILE = "BERT_WORDPIECE_V1"

    attr_reader :source_dir, :report

    def initialize(source_dir)
      @source_dir = source_dir
      @report = {}
    end

    def convert(output_path:, model_id: nil, max_tokens: REFERENCE_MAX_TOKENS)
      source = load_source
      profile = audit_tokenizer(source[:tokenizer], source[:tokenizer_config])
      tokens = extract_vocab(source[:tokenizer])
      matrix, dim = extract_matrix(tokens.length)
      meta = runtime_meta(source[:config], profile, tokens, max_tokens)

      result = write_model(output_path, meta, tokens, matrix,
                           provenance_for(model_id, meta, profile, source[:config], tokens.length, dim))
      @report = result.merge(vocab_size: tokens.length, dim: dim, provenance: JSON.parse(result[:provenance]))
    end

    private

    def load_source
      {
        tokenizer: load_json("tokenizer.json"),
        config: load_json("config.json", optional: true) || {},
        tokenizer_config: load_json("tokenizer_config.json", optional: true) || {}
      }
    end

    def load_json(name, optional: false)
      path = File.join(source_dir, name)
      return nil if optional && !File.file?(path)
      raise ConversionError, "missing #{name} in #{source_dir}" unless File.file?(path)

      JSON.parse(File.binread(path))
    end

    def reject!(reason)
      raise UnsupportedModelError,
            "#{reason}. This converter only accepts the #{TOKENIZER_PROFILE} profile; " \
            "supporting this model would require implementing that behaviour in the runtime first."
    end

    def audit_tokenizer(tokenizer, tokenizer_config)
      model = tokenizer["model"] or reject!("tokenizer.json has no model section")
      normalizer = tokenizer["normalizer"] or reject!("tokenizer has no normalizer")
      pre_tokenizer = tokenizer["pre_tokenizer"]

      assert_wordpiece!(model)
      assert_normalizer!(normalizer)
      assert_pre_tokenizer!(pre_tokenizer)
      audit_added_tokens(tokenizer)

      profile = profile_from(model, normalizer, tokenizer_config)
      reject!("clean_text=false is not supported by the runtime") unless profile[:clean_text]
      profile
    end

    def assert_wordpiece!(model)
      reject!("tokenizer model.type is #{model['type'].inspect}, expected WordPiece") unless model["type"] == "WordPiece"

      prefix = model.fetch("continuing_subword_prefix", "##")
      reject!("continuing_subword_prefix #{prefix.inspect} is not supported") unless prefix == "##"

      unk = model.fetch("unk_token", "[UNK]")
      reject!("unk_token #{unk.inspect} is not supported") unless unk == "[UNK]"
    end

    def assert_normalizer!(normalizer)
      reject!("normalizer type #{normalizer['type'].inspect} is not BertNormalizer") unless normalizer["type"] == "BertNormalizer"

      unknown = normalizer.keys - ALLOWED_NORMALIZER_KEYS
      reject!("normalizer has unsupported keys #{unknown.inspect}") unless unknown.empty?
    end

    def assert_pre_tokenizer!(pre_tokenizer)
      return if pre_tokenizer && pre_tokenizer["type"] == "BertPreTokenizer"

      reject!("pre_tokenizer type #{pre_tokenizer && pre_tokenizer['type'].inspect} is not BertPreTokenizer")
    end

    def profile_from(model, normalizer, tokenizer_config)
      lowercase = flag_value(normalizer, "lowercase", tokenizer_config["do_lower_case"], true)
      {
        lowercase: lowercase,
        strip_accents: strip_accents?(normalizer, lowercase),
        clean_text: flag_value(normalizer, "clean_text", nil, true),
        handle_chinese_chars: flag_value(normalizer, "handle_chinese_chars",
                                         tokenizer_config["tokenize_chinese_chars"], true),
        continuing_subword_prefix: model.fetch("continuing_subword_prefix", "##"),
        unk_token: model.fetch("unk_token", "[UNK]"),
        max_input_chars_per_word: model.fetch("max_input_chars_per_word", 100).to_i,
        tokenizer_class: tokenizer_config["tokenizer_class"]
      }
    end

    def flag_value(hash, key, fallback, default)
      return !!hash[key] if hash.key?(key) && !hash[key].nil?
      return !!fallback unless fallback.nil?

      default
    end

    def strip_accents?(normalizer, lowercase)
      return !!normalizer["strip_accents"] if normalizer.key?("strip_accents") && !normalizer["strip_accents"].nil?

      lowercase
    end

    def audit_added_tokens(tokenizer)
      added = tokenizer["added_tokens"] || []
      bad_content = added.reject { |token| standard_special?(token) }
      reject!("tokenizer declares non-standard added_tokens #{bad_content.map { |t| t['content'] }.inspect}") unless bad_content.empty?

      whitespace = added.find { |token| token["content"].to_s.match?(/\s/) }
      reject!("added token #{whitespace['content'].inspect} contains whitespace") if whitespace

      flagged = added.find { |token| token["lstrip"] || token["rstrip"] || token["single_word"] }
      reject!("added token #{flagged['content'].inspect} uses lstrip/rstrip/single_word") if flagged
    end

    def standard_special?(token)
      token["special"] && STANDARD_SPECIAL_TOKENS.include?(token["content"])
    end

    def extract_vocab(tokenizer)
      vocab = tokenizer.dig("model", "vocab") or reject!("tokenizer.json has no model.vocab")
      vocab.each_with_object(Array.new(vocab.length)) do |(token, id), tokens|
        validate_vocab_id!(token, id, vocab.length, tokens)
        tokens[id] = token
      end.tap do |tokens|
        missing = tokens.index(nil)
        raise ConversionError, "vocab has a hole at id #{missing}" if missing
      end
    end

    def validate_vocab_id!(token, id, size, tokens)
      reject!("vocab id #{id.inspect} is not an integer") unless id.is_a?(Integer)
      unless id.between?(0, size - 1)
        raise ConversionError, "vocab id #{id} for #{token.inspect} is outside 0...#{size}"
      end
      raise ConversionError, "duplicate vocab id #{id}" unless tokens[id].nil?
    end

    def extract_matrix(vocab_size)
      path = File.join(source_dir, "model.safetensors")
      raise ConversionError, "missing model.safetensors in #{source_dir}" unless File.file?(path)

      name, tensor = sole_matrix_tensor(Safetensors.read(path)[:tensors])
      rows, dim = tensor[:shape]
      if rows != vocab_size
        raise ConversionError,
              "embedding matrix #{name} has #{rows} rows but the tokenizer has " \
              "#{vocab_size} tokens — refusing to guess the mapping"
      end

      [tensor[:bytes], dim]
    end

    def sole_matrix_tensor(tensors)
      matrices = tensors.select { |_, tensor| tensor[:shape].length == 2 }
      raise ConversionError, "model.safetensors contains no 2-D tensor" if matrices.empty?
      return matrices.first if matrices.length == 1

      raise ConversionError,
            "model.safetensors contains several 2-D tensors (#{matrices.keys.inspect}); " \
            "a static embedding model must have exactly one"
    end

    def runtime_meta(config, profile, tokens, max_tokens)
      base_meta(config, max_tokens)
        .merge(tokenizer_meta(profile, tokens))
        .merge(token_ids(tokens, profile))
    end

    def base_meta(config, max_tokens)
      {
        dim: nil,
        normalization_type: normalization_from(config),
        max_tokens_default: max_tokens.to_i,
        add_special_tokens: false,
        unk_policy: Format::UNK_DROP,
        empty_policy: Format::EMPTY_ZERO_VECTOR
      }
    end

    def tokenizer_meta(profile, tokens)
      {
        do_lower_case: profile[:lowercase],
        strip_accents: profile[:strip_accents],
        handle_chinese_chars: profile[:handle_chinese_chars],
        clean_text: profile[:clean_text],
        max_input_chars_per_word: profile[:max_input_chars_per_word],
        max_token_chars: max_token_chars(tokens, profile[:continuing_subword_prefix]),
        subword_prefix: profile[:continuing_subword_prefix]
      }
    end

    def max_token_chars(tokens, prefix)
      tokens.map do |token|
        body = token.start_with?(prefix) ? token[prefix.length..] : token
        body.each_char.count
      end.max || 1
    end

    def token_ids(tokens, profile)
      {
        pad_id: token_id(tokens, "[PAD]", 0),
        unk_id: token_id(tokens, profile[:unk_token]),
        cls_id: token_id(tokens, "[CLS]", 0),
        sep_id: token_id(tokens, "[SEP]", 0),
        mask_id: token_id(tokens, "[MASK]", 0)
      }
    end

    def normalization_from(config)
      config.key?("normalize") && !config["normalize"] ? Format::NORMALIZATION_NONE : Format::NORMALIZATION_L2
    end

    def token_id(tokens, token, fallback = nil)
      id = tokens.index(token)
      return id unless id.nil?
      return fallback unless fallback.nil?

      raise ConversionError, "vocabulary has no #{token.inspect}"
    end

    def source_digests
      SOURCE_FILES.each_with_object({}) do |name, acc|
        path = File.join(source_dir, name)
        acc[name] = Digest::SHA256.hexdigest(File.binread(path)) if File.file?(path)
      end
    end

    def provenance_for(model_id, meta, profile, config, vocab_size, dim)
      ordered_json(
        "format_version" => Format::VERSION,
        "converter_version" => StaticEmbeddings::VERSION,
        "source_model_id" => model_id || File.basename(File.expand_path(source_dir)),
        "source_files_sha256" => source_digests,
        "reference_impl" => REFERENCE_IMPL,
        "reference_max_tokens" => meta[:max_tokens_default],
        "unicode_source" => UnicodeTables.source_stamp,
        "tokenizer_profile" => TOKENIZER_PROFILE,
        "tokenizer_class" => profile[:tokenizer_class],
        "vocab_size" => vocab_size,
        "dim" => dim,
        "normalize" => meta[:normalization_type] == Format::NORMALIZATION_L2,
        "config_seq_length" => config["seq_length"],
        "notes" => "Vectors are only reference-compatible if docs/MODEL_AUDIT.md records a passing oracle run for this source revision."
      )
    end

    def ordered_json(hash)
      JSON.generate(hash.sort_by { |key, _| key }.to_h)
    end

    def write_model(output_path, meta, tokens, matrix, provenance)
      dim = matrix.bytesize / (tokens.length * 4)
      result = Format.write(
        path: output_path,
        meta: meta.merge(dim: dim),
        tokens: tokens,
        matrix: matrix,
        norm_tables: UnicodeTables.packed,
        provenance: provenance
      )
      result.merge(provenance: provenance)
    end
  end
end
