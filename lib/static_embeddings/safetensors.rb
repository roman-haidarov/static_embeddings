require "json"

module StaticEmbeddings
  module Safetensors
    MAX_HEADER_BYTES = 100 * 1024 * 1024
    SUPPORTED_DTYPES = { "F32" => 4, "F16" => 2, "BF16" => 2 }.freeze
    CONVERT_CHUNK_ELEMENTS = 64 * 1024
    IO_CHUNK_BYTES = 1024 * 1024

    class F32Payload
      attr_reader :bytesize

      def initialize(path, tensor)
        @path = path
        @offset = tensor.fetch(:absolute_offset)
        @source_bytes = tensor.fetch(:source_bytes)
        @dtype = tensor.fetch(:dtype)
        @bytesize = tensor.fetch(:shape).reduce(1) { |count, dim| count * dim } * 4
      end

      def each_chunk
        return enum_for(__method__) unless block_given?

        File.open(@path, "rb") do |io|
          io.seek(@offset, IO::SEEK_SET)
          remaining = @source_bytes
          while remaining.positive?
            read_size = [remaining, IO_CHUNK_BYTES].min
            # F16/BF16 elements are two bytes; never split one between chunks.
            read_size -= 1 if @dtype != "F32" && read_size.odd?
            read_size = remaining if read_size.zero?
            chunk = io.read(read_size)
            unless chunk&.bytesize == read_size
              raise ConversionError, "tensor data is truncated while streaming #{@path}"
            end
            chunk.force_encoding(Encoding::BINARY)
            yield case @dtype
                  when "F32" then chunk
                  when "F16" then Safetensors.convert_f16_to_f32(chunk)
                  when "BF16" then Safetensors.convert_bf16_to_f32(chunk)
                  else
                    raise ConversionError, "tensor dtype #{@dtype} is not supported"
                  end
            remaining -= read_size
          end
        end
      end
    end

    module_function

    def describe(path)
      File.open(path, "rb") do |io|
        file_size = io.stat.size
        header, body_offset = read_header(io, file_size)
        body_size = file_size - body_offset

        tensors = header.each_with_object({}) do |(name, spec), acc|
          next if name == "__metadata__"

          dtype = spec["dtype"]
          shape = spec["shape"]
          begin_off, end_off = checked_offsets(name, spec["data_offsets"], body_size)
          expected = checked_tensor_bytes(name, dtype, shape)
          actual = end_off - begin_off
          if actual != expected
            raise ConversionError,
                  "tensor #{name}: shape #{shape.inspect} implies #{expected} bytes, offsets span #{actual}"
          end

          acc[name] = {
            dtype: dtype,
            shape: shape,
            absolute_offset: body_offset + begin_off,
            source_bytes: actual
          }
        end

        { metadata: header["__metadata__"] || {}, tensors: tensors }
      end
    end

    def read(path)
      description = describe(path)
      tensors = description.fetch(:tensors).transform_values do |tensor|
        File.open(path, "rb") do |io|
          io.seek(tensor.fetch(:absolute_offset), IO::SEEK_SET)
          bytes = io.read(tensor.fetch(:source_bytes))
          unless bytes&.bytesize == tensor.fetch(:source_bytes)
            raise ConversionError, "tensor data is truncated while reading #{path}"
          end
          tensor.merge(bytes: bytes.force_encoding(Encoding::BINARY))
        end
      end
      { metadata: description.fetch(:metadata), tensors: tensors }
    end

    def f32_payload(path, tensor)
      F32Payload.new(path, tensor)
    end

    def write(path, name, shape, floats)
      body = floats.pack("e*")
      write_raw(path, name, shape, "F32", body)
    end

    def write_raw(path, name, shape, dtype, body)
      checked_tensor_bytes(name, dtype, shape).tap do |expected|
        raise ArgumentError, "body size does not match #{dtype} shape" unless body.bytesize == expected
      end

      header = JSON.generate(name => { "dtype" => dtype, "shape" => shape, "data_offsets" => [0, body.bytesize] })
      padded = header << (" " * ((8 - (header.bytesize % 8)) % 8))
      File.open(path, "wb") do |io|
        io.write([padded.bytesize].pack("Q<"))
        io.write(padded)
        io.write(body)
      end
    end

    def f32_bytes(tensor)
      case tensor.fetch(:dtype)
      when "F32"
        tensor.fetch(:bytes)
      when "F16"
        convert_f16_to_f32(tensor.fetch(:bytes))
      when "BF16"
        convert_bf16_to_f32(tensor.fetch(:bytes))
      else
        raise ConversionError, "tensor dtype #{tensor[:dtype]} is not supported"
      end
    end

    def read_header(io, file_size)
      raise ConversionError, "safetensors file is shorter than its length prefix" if file_size < 8

      prefix = io.read(8)
      raise ConversionError, "safetensors file is shorter than its length prefix" unless prefix&.bytesize == 8

      header_len = prefix.unpack1("Q<")
      unless header_len.positive? && header_len <= MAX_HEADER_BYTES
        raise ConversionError, "implausible safetensors header length #{header_len}"
      end

      body_offset = 8 + header_len
      raise ConversionError, "safetensors header runs past end of file" if body_offset > file_size

      raw_header = io.read(header_len)
      unless raw_header&.bytesize == header_len
        raise ConversionError, "safetensors header runs past end of file"
      end
      raise ConversionError, "safetensors header is not a JSON object" unless raw_header.lstrip.start_with?("{")

      [JSON.parse(raw_header), body_offset]
    rescue JSON::ParserError => e
      raise ConversionError, "invalid safetensors JSON header: #{e.message}"
    end

    def checked_offsets(name, offsets, body_size)
      unless offsets.is_a?(Array) && offsets.length == 2
        raise ConversionError, "tensor #{name}: malformed data_offsets"
      end

      begin_off, end_off = offsets
      unless begin_off.is_a?(Integer) && end_off.is_a?(Integer) && begin_off >= 0 &&
             end_off >= begin_off && end_off <= body_size
        raise ConversionError, "tensor #{name}: data_offsets out of bounds"
      end

      [begin_off, end_off]
    end

    def checked_tensor_bytes(name, dtype, shape)
      element_size = SUPPORTED_DTYPES[dtype]
      unless element_size
        raise ConversionError, "tensor #{name}: dtype #{dtype} is not supported (expected F32, F16, or BF16)"
      end
      unless shape.is_a?(Array) && shape.all? { |dim| dim.is_a?(Integer) && dim >= 0 }
        raise ConversionError, "tensor #{name}: malformed shape #{shape.inspect}"
      end

      shape.reduce(1) { |count, dim| count * dim } * element_size
    end

    def convert_f16_to_f32(bytes)
      out = String.new(capacity: (bytes.bytesize / 2) * 4, encoding: Encoding::BINARY)
      offset = 0
      chunk_bytes = CONVERT_CHUNK_ELEMENTS * 2
      while offset < bytes.bytesize
        chunk = bytes.byteslice(offset, [chunk_bytes, bytes.bytesize - offset].min)
        floats = if StaticEmbeddings.respond_to?(:decode_f16)
                   StaticEmbeddings.decode_f16(chunk)
                 else
                   chunk.unpack("v*").map { |bits| half_to_float(bits) }
                 end
        out << floats.pack("e*")
        offset += chunk.bytesize
      end
      out
    end

    def convert_bf16_to_f32(bytes)
      out = String.new(capacity: (bytes.bytesize / 2) * 4, encoding: Encoding::BINARY)
      offset = 0
      chunk_bytes = CONVERT_CHUNK_ELEMENTS * 2
      while offset < bytes.bytesize
        chunk = bytes.byteslice(offset, [chunk_bytes, bytes.bytesize - offset].min)
        words = chunk.unpack("v*")
        out << words.map { |bits| bits << 16 }.pack("V*")
        offset += chunk.bytesize
      end
      out
    end

    def half_to_float(bits)
      sign = (bits >> 15) & 1
      exponent = (bits >> 10) & 0x1F
      fraction = bits & 0x3FF

      value =
        if exponent.zero?
          fraction.zero? ? 0.0 : Math.ldexp(fraction.to_f, -24)
        elsif exponent == 0x1F
          fraction.zero? ? Float::INFINITY : Float::NAN
        else
          Math.ldexp(1.0 + fraction.to_f / 1024.0, exponent - 15)
        end
      sign.zero? ? value : -value
    end
  end
end
