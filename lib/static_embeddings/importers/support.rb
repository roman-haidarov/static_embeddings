require "digest"
require "json"
require "static_embeddings/errors"
require "static_embeddings/format/constants"
require "static_embeddings/safetensors"
require "static_embeddings/row_prefix_payload"

module StaticEmbeddings
  module Importers
    module Support
      module_function

      def load_json(path, optional: false)
        return nil if optional && !File.file?(path)
        raise InvalidSourceError, "missing #{File.basename(path)} in #{File.dirname(path)}" unless File.file?(path)

        JSON.parse(File.binread(path))
      rescue JSON::ParserError => e
        raise InvalidSourceError, "invalid JSON in #{path}: #{e.message}"
      end

      def load_object(path, optional: false)
        value = load_json(path, optional: optional)
        return nil if value.nil?
        raise InvalidSourceError, "#{path} must contain a JSON object" unless value.is_a?(Hash)

        value
      end

      def digest_files(paths, root: nil)
        paths.each_with_object({}) do |path, digests|
          next unless File.file?(path)

          digests[digest_key(path, root)] = Digest::SHA256.file(path).hexdigest
        end
      end

      def extract_matrix(path, vocab_size)
        raise InvalidSourceError, "missing model.safetensors in #{File.dirname(path)}" unless File.file?(path)

        name, tensor = sole_matrix_tensor(Safetensors.describe(path).fetch(:tensors))
        rows, dim = tensor.fetch(:shape)
        unless rows == vocab_size
          raise ConversionError,
                "embedding matrix #{name} has #{rows} rows but the tokenizer has #{vocab_size} tokens"
        end
        unless dim.positive? && dim <= Format::UINT32_MAX
          raise ConversionError, "embedding matrix #{name} has unsupported dimension #{dim}"
        end

        [Safetensors.f32_payload(path, tensor), dim]
      end

      def resolve_max_tokens(requested, default)
        return default if requested.nil?
        return 0 if requested == false || requested == :unlimited || requested == 0

        value = Integer(requested)
        unless value.between?(1, Format::UINT32_MAX)
          raise InvalidOptionError, "max_tokens must be 1..#{Format::UINT32_MAX}, false, or :unlimited"
        end
        value
      end

      def resolve_dimensions(requested, native_dim)
        return native_dim if requested.nil?

        value = Integer(requested)
        unless value.between?(1, native_dim)
          raise InvalidOptionError, "dimensions must be between 1 and native_dim #{native_dim}, got #{value}"
        end
        value
      end

      def normalize_mrl_dims(value, native_dim:)
        return nil if value.nil?

        dims = value.is_a?(Array) ? value : value.to_s.split(",").map(&:strip)
        dims = dims.map { |item| Integer(item) }
        if dims.empty? || dims.any? { |dim| !dim.between?(1, native_dim) }
          raise InvalidOptionError, "trained_mrl_dims must contain dimensions between 1 and #{native_dim}"
        end
        dims.uniq.freeze
      end

      def slice_matrix(payload, native_dim, output_dim)
        return payload if output_dim == native_dim

        RowPrefixPayload.new(payload, native_dim: native_dim, output_dim: output_dim)
      end

      def sole_matrix_tensor(tensors)
        matrices = tensors.select { |_, tensor| tensor.fetch(:shape).length == 2 }
        raise ConversionError, "model.safetensors contains no 2-D tensor" if matrices.empty?
        return matrices.first if matrices.length == 1

        raise ConversionError,
              "model.safetensors contains several 2-D tensors (#{matrices.keys.inspect}); expected exactly one"
      end

      def digest_key(path, root)
        return File.basename(path) unless root

        root_real = File.realpath(root)
        real = File.realpath(path)
        prefix = root_real.end_with?(File::SEPARATOR) ? root_real : "#{root_real}#{File::SEPARATOR}"
        real.start_with?(prefix) ? real.delete_prefix(prefix) : File.basename(real)
      end
    end
  end
end
