require "static_embeddings/version"

begin
  require "static_embeddings/static_embeddings"
rescue LoadError
  ext_dir = File.expand_path("static_embeddings", __dir__)
  extension = %w[.so .bundle].lazy.map { |suffix| File.join(ext_dir, "static_embeddings#{suffix}") }
                              .find { |path| File.file?(path) }
  unless extension
    raise LoadError, "Could not find the compiled StaticEmbeddings extension. Run: bundle exec rake compile"
  end
  require extension
end

require "static_embeddings/errors"
require "static_embeddings/paths"
require "static_embeddings/format/constants"
require "static_embeddings/model"

module StaticEmbeddings
  class << self
    def load(path, verify: false)
      expanded = File.expand_path(path.to_s)
      raise ModelNotFound, "no model at #{expanded}" unless File.file?(expanded)
      raise InvalidModelError, "checksum mismatch for #{expanded}" if verify && !self.verify(expanded)[:ok]

      Model.new(expanded)
    end

    def load_builtin(name = :demo, verify: false)
      load(Paths.builtin_path(name), verify: verify)
    end

    def builtin_available?(name = :demo)
      Paths.builtin_available?(name)
    end

    def cache_dir
      Paths.cache_dir
    end

    def model_path(model_id)
      Paths.model_path(model_id)
    end

    def load_model(model_id, verify: false)
      path = model_path(model_id)
      unless File.file?(path)
        raise ModelNotFound,
              "model #{model_id.inspect} is not installed. Convert it first: " \
              "static_embeddings convert <hf-dir> --id #{model_id}"
      end
      load(path, verify: verify)
    end

    def convert(source_dir, output_path:, model_id: nil, max_tokens: nil, dimensions: nil,
                source_revision: nil, trained_mrl_dims: nil)
      require "static_embeddings/conversion"
      Conversion.call(
        source_dir,
        output_path: output_path,
        model_id: model_id,
        max_tokens: max_tokens,
        dimensions: dimensions,
        source_revision: source_revision,
        trained_mrl_dims: trained_mrl_dims
      )
    end

    def verify(path)
      require "static_embeddings/format/verifier"
      Format::Verifier.call(File.expand_path(path.to_s))
    end

    def unpack(blob, dim, format: :f32)
      require "static_embeddings/codec"
      Codec.unpack(blob, dim, format: format)
    end

    def pack(rows, format: :f32)
      require "static_embeddings/codec"
      Codec.pack(rows, format: format)
    end

    def normalize_format(format)
      require "static_embeddings/codec"
      Codec.normalize_format(format)
    end
  end
end
