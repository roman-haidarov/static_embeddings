require "json"

module StaticEmbeddings
  module Safetensors
    MAX_HEADER_BYTES = 100 * 1024 * 1024
    SUPPORTED_DTYPES = { "F32" => 4 }.freeze

    module_function

    def read(path)
      data = File.binread(path)
      header, body_offset = parse_header(data)
      body_size = data.bytesize - body_offset

      tensors = header.each_with_object({}) do |(name, spec), acc|
        next if name == "__metadata__"

        acc[name] = tensor_from(data, body_offset, body_size, name, spec)
      end

      { metadata: header["__metadata__"] || {}, tensors: tensors }
    end

    def write(path, name, shape, floats)
      body = floats.pack("e*")
      header = JSON.generate(name => { "dtype" => "F32", "shape" => shape, "data_offsets" => [0, body.bytesize] })
      padded = header << (" " * ((8 - (header.bytesize % 8)) % 8))
      File.binwrite(path, [padded.bytesize].pack("Q<") << padded << body)
    end

    def parse_header(data)
      raise ConversionError, "safetensors file is shorter than its length prefix" if data.bytesize < 8

      header_len = data.byteslice(0, 8).unpack1("Q<")
      unless header_len.positive? && header_len <= MAX_HEADER_BYTES
        raise ConversionError, "implausible safetensors header length #{header_len}"
      end

      body_offset = 8 + header_len
      raise ConversionError, "safetensors header runs past end of file" if body_offset > data.bytesize

      raw_header = data.byteslice(8, header_len)
      raise ConversionError, "safetensors header is not a JSON object" unless raw_header.lstrip.start_with?("{")

      [JSON.parse(raw_header), body_offset]
    end

    def tensor_from(data, body_offset, body_size, name, spec)
      dtype = spec["dtype"]
      shape = spec["shape"]
      begin_off, end_off = checked_offsets(name, spec["data_offsets"], body_size)
      expected = checked_tensor_bytes(name, dtype, shape)
      actual = end_off - begin_off

      if actual != expected
        raise ConversionError,
              "tensor #{name}: shape #{shape.inspect} implies #{expected} bytes, offsets span #{actual}"
      end

      { dtype: dtype, shape: shape, bytes: data.byteslice(body_offset + begin_off, actual) }
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
      raise ConversionError, "tensor #{name}: dtype #{dtype} is not supported (F32 only)" unless element_size
      unless shape.is_a?(Array) && shape.all? { |dim| dim.is_a?(Integer) && dim >= 0 }
        raise ConversionError, "tensor #{name}: malformed shape #{shape.inspect}"
      end

      shape.reduce(1, :*) * element_size
    end
  end
end
