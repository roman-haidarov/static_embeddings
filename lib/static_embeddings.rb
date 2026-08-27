require "json"
require "static_embeddings/version"

begin
  require "static_embeddings/static_embeddings"
rescue LoadError
  ext_dir = File.expand_path("static_embeddings", __dir__)
  so_path = %w[.so .bundle].lazy.map { |ext| File.join(ext_dir, "static_embeddings#{ext}") }
                                .find { |path| File.file?(path) }
  unless so_path
    raise LoadError, "Could not find the compiled StaticEmbeddings extension. " \
                     "Run: bundle exec rake compile"
  end

  require so_path
end

require "static_embeddings/errors"
require "static_embeddings/paths"
require "static_embeddings/format"
require "static_embeddings/unicode_tables"
require "static_embeddings/safetensors"
require "static_embeddings/converter"
require "static_embeddings/reference"
require "static_embeddings/model"

module StaticEmbeddings
  class << self
    def load(path, verify: false)
      expanded = File.expand_path(path.to_s)
      raise ModelNotFound, "no model at #{expanded}" unless File.file?(expanded)

      if verify
        result = Format.verify(expanded)
        raise InvalidModelError, "checksum mismatch for #{expanded}" unless result[:ok]
      end

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
              "model #{model_id.inspect} is not installed. " \
              "Convert it first: static_embeddings convert <hf-dir> --id #{model_id}"
      end
      load(path, verify: verify)
    end

    def convert(source_dir, output_path:, model_id: nil, max_tokens: Converter::REFERENCE_MAX_TOKENS)
      converter = Converter.new(source_dir)
      converter.convert(output_path: output_path, model_id: model_id, max_tokens: max_tokens)
      converter.report
    end

    def verify(path)
      Format.verify(File.expand_path(path.to_s))
    end

    def unpack(blob, dim, format: :f32)
      raise ArgumentError, "dim must be positive" unless dim.to_i.positive?

      floats =
        case normalize_format(format)
        when :f32
          raise ArgumentError, "f32 blob byte size must be a multiple of 4" unless (blob.bytesize % 4).zero?

          blob.unpack("e*")
        when :f16
          raise ArgumentError, "f16 blob byte size must be a multiple of 2" unless (blob.bytesize % 2).zero?

          blob.unpack("S<*").map { |half| half_to_float(half) }
        end

      raise ArgumentError, "blob is not a multiple of dim" unless (floats.length % dim).zero?

      floats.each_slice(dim).to_a
    end

    def normalize_format(format)
      case format&.to_sym
      when nil, :f32, :float32
        :f32
      when :f16, :float16
        :f16
      else
        raise ArgumentError, "unsupported embedding format #{format.inspect} (expected :f32 or :f16)"
      end
    end

    def half_to_float(half)
      sign = (half & 0x8000) << 16
      exp = (half >> 10) & 0x1f
      mant = half & 0x03ff

      bits = if exp.zero?
        if mant.zero?
          sign
        else
          exp = 1
          while (mant & 0x0400).zero?
            mant <<= 1
            exp -= 1
          end
          mant &= 0x03ff
          sign | ((exp + (127 - 15)) << 23) | (mant << 13)
        end
      elsif exp == 31
        sign | 0x7f800000 | (mant << 13)
      else
        sign | ((exp + (127 - 15)) << 23) | (mant << 13)
      end

      [bits].pack("L<").unpack1("e")
    end
  end
end
