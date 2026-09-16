require "static_embeddings/errors"
require "static_embeddings/bert_wordpiece"
require "static_embeddings/canonical"
require "static_embeddings/importers/support"
require "static_embeddings/importers/model2vec"
require "static_embeddings/importers/sentence_transformers_static"

module StaticEmbeddings
  module Importers
    module_function

    def import(source_dir, **options)
      root = source_root(source_dir)
      case detect(root)
      when :model2vec then Model2Vec.call(root, **options)
      when :sentence_transformers_static then SentenceTransformersStatic.call(root, **options)
      end
    end

    def detect(root)
      config = Support.load_object(File.join(root, "config.json"), optional: true) || {}
      tokenizer = File.join(root, "tokenizer.json")
      weights = File.join(root, "model.safetensors")
      modules = File.join(root, "modules.json")

      return :model2vec if model2vec_config?(config) && File.file?(tokenizer) && File.file?(weights)
      return :sentence_transformers_static if File.file?(modules)

      if File.file?(tokenizer) && File.file?(weights)
        raise InvalidSourceError,
              "cannot prove source family in #{root}: config.json is not Model2Vec and modules.json is absent"
      end

      raise InvalidSourceError,
            "cannot detect a static embedding source in #{root}: need a Model2Vec config with " \
            "tokenizer.json + model.safetensors, or modules.json for Sentence Transformers StaticEmbedding"
    end

    def source_root(source_dir)
      root = File.expand_path(source_dir.to_s)
      raise InvalidSourceError, "source directory #{root} does not exist" unless File.directory?(root)

      root
    end

    def model2vec_config?(config)
      config["model_type"].to_s == "model2vec" || Array(config["architectures"]).include?("StaticModel")
    end
  end
end
