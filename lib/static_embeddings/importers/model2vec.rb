module StaticEmbeddings
  module Importers
    module Model2Vec
      SOURCE_FILES = %w[tokenizer.json config.json tokenizer_config.json model.safetensors].freeze
      DEFAULT_MAX_TOKENS = 512
      DEFAULT_TRUNCATION = :raw_ids
      FAMILY = "model2vec"
      ORACLE = "model2vec.StaticModel"

      module_function

      def call(root, model_id: nil, max_tokens: nil, dimensions: nil,
               source_revision: nil, trained_mrl_dims: nil, truncation: nil)
        tokenizer = Support.load_object(File.join(root, "tokenizer.json"))
        config = Support.load_object(File.join(root, "config.json"), optional: true) || {}
        tokenizer_config = Support.load_object(File.join(root, "tokenizer_config.json"), optional: true) || {}
        profile, tokens = BertWordPiece.compile(tokenizer, tokenizer_config)
        payload, native_dim, baked = Support.extract_matrix(File.join(root, "model.safetensors"), tokens.length,
                                                            side_tensors: Support::MODEL2VEC_SIDE_TENSORS)
        output_dim = Support.resolve_dimensions(dimensions, native_dim)

        Canonical.model(
          tokens: tokens,
          matrix: Support.slice_matrix(payload, native_dim, output_dim),
          dimensions: Canonical.dimensions(
            native: native_dim,
            output: output_dim,
            trained: Support.normalize_mrl_dims(trained_mrl_dims, native_dim: native_dim)
          ),
          runtime: Canonical.runtime(
            normalization: normalization(config),
            unk_policy: Format::UNK_DROP,
            empty_policy: Format::EMPTY_ZERO_VECTOR,
            max_tokens: Support.resolve_max_tokens(max_tokens, default_max_tokens(config)),
            truncation: Support.resolve_truncation(truncation, DEFAULT_TRUNCATION)
          ),
          tokenizer: BertWordPiece.runtime_meta(profile, tokens),
          source: Canonical.source(
            family: FAMILY,
            model: model_id || File.basename(root),
            revision: source_revision,
            oracle: ORACLE,
            files_sha256: Support.digest_files(SOURCE_FILES.map { |name| File.join(root, name) }),
            tokenizer_class: profile[:tokenizer_class],
            config_seq_length: config["seq_length"],
            baked: baked
          )
        )
      end

      def default_max_tokens(config)
        return DEFAULT_MAX_TOKENS unless config.key?("max_length")

        value = config["max_length"]
        return 0 if value.nil?
        unless value.is_a?(Integer) && value.between?(1, Format::UINT32_MAX)
          raise InvalidSourceError,
                "config.json max_length must be an integer in 1..#{Format::UINT32_MAX} or null, got #{value.inspect}"
        end
        value
      end

      def normalization(config)
        config.fetch("normalize", false) ? Format::NORMALIZATION_L2 : Format::NORMALIZATION_NONE
      end
    end
  end
end
