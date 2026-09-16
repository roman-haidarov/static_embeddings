require "pathname"

module StaticEmbeddings
  module Importers
    module SentenceTransformersStatic
      TYPE = "sentence_transformers.models.StaticEmbedding"
      FAMILY = "sentence_transformers_static"
      ORACLE = "sentence_transformers.SentenceTransformer.encode"
      DEFAULT_MAX_TOKENS = 0

      module_function

      def call(root, model_id: nil, max_tokens: nil, dimensions: nil,
               source_revision: nil, trained_mrl_dims: nil)
        module_dir = module_directory(root)
        tokenizer = Support.load_object(File.join(module_dir, "tokenizer.json"))
        tokenizer_config = Support.load_object(File.join(module_dir, "tokenizer_config.json"), optional: true) || {}
        config = Support.load_object(File.join(module_dir, "config.json"), optional: true) || {}
        profile, tokens = BertWordPiece.compile(tokenizer, tokenizer_config)
        payload, native_dim = Support.extract_matrix(File.join(module_dir, "model.safetensors"), tokens.length)
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
            normalization: Format::NORMALIZATION_NONE,
            unk_policy: Format::UNK_INCLUDE,
            empty_policy: Format::EMPTY_ZERO_VECTOR,
            max_tokens: Support.resolve_max_tokens(max_tokens, DEFAULT_MAX_TOKENS)
          ),
          tokenizer: BertWordPiece.runtime_meta(profile, tokens),
          source: Canonical.source(
            family: FAMILY,
            model: model_id || File.basename(root),
            revision: source_revision,
            oracle: ORACLE,
            files_sha256: source_digests(root, module_dir),
            tokenizer_class: profile[:tokenizer_class],
            config_seq_length: config["seq_length"]
          )
        )
      end

      def module_directory(root)
        spec = module_spec(root)
        contained_directory(root, spec.fetch("path"))
      end

      def module_spec(root)
        modules = Support.load_json(File.join(root, "modules.json"))
        reject_source("modules.json is not an array") unless modules.is_a?(Array)
        reject_source("modules.json declares #{modules.length} modules, expected exactly one StaticEmbedding") unless modules.length == 1

        spec = modules.first
        reject_source("modules.json[0] is not an object") unless spec.is_a?(Hash)
        reject_source("modules.json[0].type is #{spec['type'].inspect}, expected #{TYPE}") unless spec["type"] == TYPE
        reject_source("modules.json[0] has no path") unless spec.key?("path")
        reject_source("modules.json[0].path is not a string") unless spec["path"].is_a?(String)
        spec
      end

      def contained_directory(root, relative)
        relative = relative.to_s
        reject_source("module path contains a NUL byte") if relative.include?("\0")
        reject_source("module path #{relative.inspect} is absolute") if Pathname.new(relative).absolute?

        root_real = File.realpath(root)
        candidate = relative.empty? ? root : File.expand_path(relative, root)
        reject_source("module path #{relative.inspect} is not a directory") unless File.directory?(candidate)

        real = File.realpath(candidate)
        prefix = root_real.end_with?(File::SEPARATOR) ? root_real : "#{root_real}#{File::SEPARATOR}"
        reject_source("module path #{relative.inspect} escapes the source directory") unless real == root_real || real.start_with?(prefix)
        real
      rescue Errno::ENOENT
        reject_source("module path #{relative.inspect} does not exist")
      end

      def source_digests(root, module_dir)
        paths = [
          File.join(root, "modules.json"),
          File.join(module_dir, "tokenizer.json"),
          File.join(module_dir, "tokenizer_config.json"),
          File.join(module_dir, "config.json"),
          File.join(module_dir, "model.safetensors")
        ]
        Support.digest_files(paths, root: root)
      end

      def reject_source(message)
        raise UnsupportedModelError,
              "#{message}. Sentence Transformers conversion accepts exactly one #{TYPE} module " \
              "using #{BertWordPiece::TOKENIZER_PROFILE}."
      end
    end
  end
end
