module StaticEmbeddings
  module Codec
    module_function

    def unpack(blob, dim, format: :f32)
      dim = Integer(dim)
      raise InvalidOptionError, "dim must be positive" unless dim.positive?

      values = decode(blob, normalize_format(format))
      raise InvalidOptionError, "blob is not a multiple of dim" unless (values.length % dim).zero?

      values.each_slice(dim).to_a
    end

    def pack(rows, format: :f32)
      values = rows.first.is_a?(Array) ? rows.flatten(1) : rows
      floats = values.map(&:to_f)

      case normalize_format(format)
      when :f32 then floats.pack("e*")
      when :f16 then StaticEmbeddings.encode_f16(floats)
      end
    end

    def decode(blob, format)
      case format
      when :f32
        raise InvalidOptionError, "f32 blob byte size must be a multiple of 4" unless (blob.bytesize % 4).zero?
        blob.unpack("e*")
      when :f16
        raise InvalidOptionError, "f16 blob byte size must be a multiple of 2" unless (blob.bytesize % 2).zero?
        StaticEmbeddings.decode_f16(blob)
      end
    end

    def normalize_format(format)
      case format&.to_sym
      when nil, :f32, :float32 then :f32
      when :f16, :float16 then :f16
      else
        raise InvalidOptionError, "unsupported embedding format #{format.inspect} (expected :f32 or :f16)"
      end
    end
  end
end
