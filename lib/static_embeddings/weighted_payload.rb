require "static_embeddings/errors"
require "static_embeddings/format/constants"

module StaticEmbeddings
  class WeightedPayload
    FLOAT_BYTES = 4
    ROWS_PER_CHUNK = 1024

    attr_reader :bytesize

    def initialize(inner, dim:, source_rows:, output_rows:, weights: nil, mapping: nil)
      raise InvalidOptionError, "weights or mapping is required" if weights.nil? && mapping.nil?
      unless inner.bytesize == source_rows * dim * FLOAT_BYTES
        raise ConversionError, "embedding payload does not hold #{source_rows} rows of #{dim} floats"
      end

      @inner = inner
      @row_bytes = dim * FLOAT_BYTES
      @output_rows = output_rows
      @weights = weights
      @mapping = mapping
      @bytesize = output_rows * @row_bytes
    end

    def each_chunk(&block)
      return enum_for(__method__) unless block_given?

      @mapping ? each_mapped_chunk(&block) : each_streamed_chunk(&block)
    end

    private

    def each_mapped_chunk
      table = Format.binary_string(@inner.bytesize)
      @inner.each_chunk { |chunk| table << chunk }

      index = 0
      while index < @output_rows
        count = [ROWS_PER_CHUNK, @output_rows - index].min
        out = Format.binary_string(count * @row_bytes)
        count.times do
          row = table.byteslice(@mapping[index] * @row_bytes, @row_bytes)
          out << (@weights ? scale(row, @weights[index]) : row)
          index += 1
        end
        yield out
      end
    end

    def each_streamed_chunk
      remainder = Format.binary_string
      index = 0
      @inner.each_chunk do |chunk|
        data = remainder.empty? ? chunk : remainder + chunk
        whole = (data.bytesize / @row_bytes) * @row_bytes
        remainder = (data.byteslice(whole, data.bytesize - whole) || Format.binary_string).force_encoding(Encoding::BINARY)
        next if whole.zero?

        out = Format.binary_string(whole)
        (whole / @row_bytes).times do |r|
          out << scale(data.byteslice(r * @row_bytes, @row_bytes), @weights[index])
          index += 1
        end
        yield out
      end
      raise ConversionError, "truncated embedding matrix while applying weights" unless remainder.empty?
    end

    def scale(row, weight)
      row.unpack("e*").map! { |x| x * weight }.pack("e*")
    end
  end
end
