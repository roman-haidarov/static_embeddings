require "static_embeddings/errors"
require "static_embeddings/format/constants"

module StaticEmbeddings
  module BertWordPiece
    TOKENIZER_PROFILE = "BERT_WORDPIECE_V1"
    ALLOWED_NORMALIZER_KEYS = %w[type clean_text handle_chinese_chars strip_accents lowercase].freeze
    STANDARD_SPECIAL_TOKENS = {
      "[PAD]" => Format::ADDED_PAD,
      "[UNK]" => Format::ADDED_UNK,
      "[CLS]" => Format::ADDED_CLS,
      "[SEP]" => Format::ADDED_SEP,
      "[MASK]" => Format::ADDED_MASK
    }.freeze

    module_function

    def compile(tokenizer, tokenizer_config = {})
      profile = audit(tokenizer, tokenizer_config)
      tokens = vocabulary(tokenizer)
      validate_added_token_ids(profile.fetch(:added_tokens), tokens)
      [profile.freeze, tokens]
    end

    def runtime_meta(profile, tokens)
      prefix = profile.fetch(:continuing_subword_prefix)
      {
        tokenizer_type: Format::TOKENIZER_BERT_WORDPIECE_V1,
        tokenizer_profile: TOKENIZER_PROFILE,
        do_lower_case: profile.fetch(:lowercase),
        strip_accents: profile.fetch(:strip_accents),
        handle_chinese_chars: profile.fetch(:handle_chinese_chars),
        clean_text: profile.fetch(:clean_text),
        added_token_mask: profile.fetch(:added_token_mask),
        max_input_chars_per_word: profile.fetch(:max_input_chars_per_word),
        max_token_chars: max_token_chars(tokens, prefix),
        subword_prefix: prefix,
        pad_id: token_id(tokens, "[PAD]", 0),
        unk_id: token_id(tokens, profile.fetch(:unk_token)),
        cls_id: token_id(tokens, "[CLS]", 0),
        sep_id: token_id(tokens, "[SEP]", 0),
        mask_id: token_id(tokens, "[MASK]", 0)
      }.freeze
    end

    def audit(tokenizer, tokenizer_config)
      unsupported!("tokenizer.json is not an object") unless tokenizer.is_a?(Hash)
      unsupported!("tokenizer_config.json is not an object") unless tokenizer_config.is_a?(Hash)
      model = tokenizer["model"] || unsupported!("tokenizer.json has no model section")
      normalizer = tokenizer["normalizer"] || unsupported!("tokenizer has no normalizer")
      pre_tokenizer = tokenizer["pre_tokenizer"]
      unsupported!("tokenizer model section is not an object") unless model.is_a?(Hash)
      unsupported!("tokenizer normalizer is not an object") unless normalizer.is_a?(Hash)

      audit_model(model)
      audit_normalizer(normalizer)
      audit_pre_tokenizer(pre_tokenizer)
      added_tokens = audit_added_tokens(tokenizer["added_tokens"] || [])
      lowercase = flag(normalizer, "lowercase", tokenizer_config["do_lower_case"], true)
      clean_text = flag(normalizer, "clean_text", nil, true)
      max_input_chars = Integer(model.fetch("max_input_chars_per_word", 100))
      unsupported!("max_input_chars_per_word must be positive") unless max_input_chars.positive?
      unsupported!("clean_text=false is not supported by the runtime") unless clean_text

      {
        lowercase: lowercase,
        strip_accents: strip_accents(normalizer, lowercase),
        clean_text: clean_text,
        handle_chinese_chars: flag(normalizer, "handle_chinese_chars",
                                   tokenizer_config["tokenize_chinese_chars"], true),
        continuing_subword_prefix: model.fetch("continuing_subword_prefix", "##"),
        unk_token: model.fetch("unk_token", "[UNK]"),
        max_input_chars_per_word: max_input_chars,
        tokenizer_class: tokenizer_config["tokenizer_class"],
        added_tokens: added_tokens,
        added_token_mask: added_tokens.reduce(0) do |mask, token|
          mask | STANDARD_SPECIAL_TOKENS.fetch(token.fetch("content"))
        end
      }
    end

    def audit_model(model)
      unsupported!("tokenizer model.type is #{model['type'].inspect}, expected WordPiece") unless model["type"] == "WordPiece"

      prefix = model.fetch("continuing_subword_prefix", "##")
      unsupported!("continuing_subword_prefix #{prefix.inspect} is not supported") unless prefix == "##"

      unk_token = model.fetch("unk_token", "[UNK]")
      unsupported!("unk_token #{unk_token.inspect} is not supported") unless unk_token == "[UNK]"
    end

    def audit_normalizer(normalizer)
      unsupported!("normalizer type #{normalizer['type'].inspect} is not BertNormalizer") unless normalizer["type"] == "BertNormalizer"

      unknown = normalizer.keys - ALLOWED_NORMALIZER_KEYS
      unsupported!("normalizer has unsupported keys #{unknown.inspect}") unless unknown.empty?
    end

    def audit_pre_tokenizer(pre_tokenizer)
      return if pre_tokenizer.is_a?(Hash) && pre_tokenizer["type"] == "BertPreTokenizer"

      unsupported!("pre_tokenizer type #{pre_tokenizer && pre_tokenizer['type'].inspect} is not BertPreTokenizer")
    end

    def audit_added_tokens(tokens)
      unsupported!("added_tokens is not an array") unless tokens.is_a?(Array)
      unsupported!("added_tokens contains a non-object entry") unless tokens.all? { |token| token.is_a?(Hash) }
      bad = tokens.reject { |token| standard_special?(token) }
      unsupported!("tokenizer declares non-standard added_tokens #{bad.map { |token| token['content'] }.inspect}") unless bad.empty?

      tokens.each do |token|
        content = token["content"].to_s
        unsupported!("added token #{content.inspect} contains whitespace") if content.match?(/\s/)
        unsupported!("added token #{content.inspect} uses lstrip/rstrip/single_word") if token["lstrip"] || token["rstrip"] || token["single_word"]
        unsupported!("added token #{content.inspect} must use normalized=false") unless token["normalized"] == false
      end

      duplicate = tokens.group_by { |token| token["content"] }.find { |_, rows| rows.length > 1 }
      unsupported!("duplicate added token #{duplicate.first.inspect}") if duplicate
      tokens.freeze
    end

    def vocabulary(tokenizer)
      vocab = tokenizer.dig("model", "vocab") || unsupported!("tokenizer.json has no model.vocab")
      unsupported!("tokenizer model.vocab is not an object") unless vocab.is_a?(Hash)
      tokens = Array.new(vocab.length)
      vocab.each { |token, id| assign_vocab(tokens, token, id) }
      missing = tokens.index(nil)
      invalid!("vocab has a hole at id #{missing}") if missing
      tokens
    end

    def assign_vocab(tokens, token, id)
      unsupported!("vocab id #{id.inspect} is not an integer") unless id.is_a?(Integer)
      invalid!("vocab id #{id} for #{token.inspect} is outside 0...#{tokens.length}") unless id.between?(0, tokens.length - 1)
      invalid!("duplicate vocab id #{id}") unless tokens[id].nil?
      tokens[id] = token
    end

    def validate_added_token_ids(added_tokens, tokens)
      added_tokens.each do |token|
        content = token.fetch("content")
        id = token["id"]
        unsupported!("added token #{content.inspect} has non-integer id #{id.inspect}") unless id.is_a?(Integer)
        valid = id.between?(0, tokens.length - 1) && tokens[id] == content
        unsupported!("added token #{content.inspect} id #{id} does not match model.vocab") unless valid
      end
    end

    def flag(hash, key, fallback, default)
      return !!hash[key] if hash.key?(key) && !hash[key].nil?
      return !!fallback unless fallback.nil?

      default
    end

    def strip_accents(normalizer, lowercase)
      value = normalizer["strip_accents"]
      normalizer.key?("strip_accents") && !value.nil? ? !!value : lowercase
    end

    def standard_special?(token)
      token["special"] && STANDARD_SPECIAL_TOKENS.key?(token["content"])
    end

    def max_token_chars(tokens, prefix)
      tokens.reduce(1) do |maximum, token|
        body = token.start_with?(prefix) ? token[prefix.length..] : token
        [maximum, body.each_char.count].max
      end
    end

    def token_id(tokens, token, fallback = nil)
      id = tokens.index(token)
      return id unless id.nil?
      return fallback unless fallback.nil?

      invalid!("vocabulary has no #{token.inspect}")
    end

    def unsupported!(message)
      raise UnsupportedModelError,
            "#{message}. Supported tokenizer profile: #{TOKENIZER_PROFILE}; " \
            "other tokenizer behaviour requires a separate runtime capability."
    end

    def invalid!(message)
      raise ConversionError, message
    end
  end
end
