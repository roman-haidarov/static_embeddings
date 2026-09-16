require "digest"
require "static_embeddings/errors"
require "static_embeddings/format/constants"

module StaticEmbeddings
  module Format
    module Verifier
      CHUNK_BYTES = 1024 * 1024

      module_function

      def call(path)
        raise InvalidModelError, "file too small" if File.size(path) < HEADER_SIZE

        File.open(path, "rb") do |io|
          header = io.read(HEADER_SIZE)
          raise InvalidModelError, "bad magic" unless header.byteslice(0, 8) == MAGIC.b

          stored = header.byteslice(CHECKSUM_OFFSET, CHECKSUM_SIZE)
          actual = checksum(io, header)
          { ok: stored == actual, expected: actual.unpack1("H*"), stored: stored.unpack1("H*") }
        end
      end

      def checksum(io, header)
        digest = Digest::SHA256.new
        zeroed = header.dup
        zeroed[CHECKSUM_OFFSET, CHECKSUM_SIZE] = "\0".b * CHECKSUM_SIZE
        digest << zeroed

        buffer = String.new(capacity: CHUNK_BYTES)
        digest << buffer while io.read(CHUNK_BYTES, buffer)
        digest.digest
      end
    end

    def self.verify(path)
      Verifier.call(path)
    end
  end
end
