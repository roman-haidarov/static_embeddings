require "static_embeddings/errors"
require "static_embeddings/format/constants"

module StaticEmbeddings
  class RowPrefixPayload
    FLOAT_BYTES = 4

    attr_reader :bytesize

    def initialize(inner, native_dim:, output_dim:)
      raise InvalidOptionError, "native_dim must be positive" unless native_dim.positive?
      unless output_dim.between?(1, native_dim)
        raise InvalidOptionError, "output_dim must be between 1 and #{native_dim}"
      end

      @inner = inner
      @source_row_bytes = native_dim * FLOAT_BYTES
      @output_row_bytes = output_dim * FLOAT_BYTES
      unless (inner.bytesize % @source_row_bytes).zero?
        raise ConversionError, "embedding payload is not a whole number of #{native_dim}-d rows"
      end
      @bytesize = (inner.bytesize / @source_row_bytes) * @output_row_bytes
    end

    def each_chunk
      return enum_for(__method__) unless block_given?

      remainder = Format.binary_string
      @inner.each_chunk do |chunk|
        data = remainder.empty? ? chunk : remainder + chunk
        complete_rows, remainder = split_complete_rows(data)
        yield slice_rows(complete_rows) unless complete_rows.empty?
      end

      raise ConversionError, "truncated embedding matrix while slicing dimensions" unless remainder.empty?
    end

    private

    def split_complete_rows(data)
      bytes = (data.bytesize / @source_row_bytes) * @source_row_bytes
      complete = data.byteslice(0, bytes) || Format.binary_string
      remainder = data.byteslice(bytes, data.bytesize - bytes) || Format.binary_string
      remainder.force_encoding(Encoding::BINARY)
      [complete, remainder]
    end

    def slice_rows(data)
      rows = data.bytesize / @source_row_bytes
      out = Format.binary_string(rows * @output_row_bytes)
      offset = 0
      rows.times do
        out << data.byteslice(offset, @output_row_bytes)
        offset += @source_row_bytes
      end
      out
    end
  end
end
