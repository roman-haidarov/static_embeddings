require "static_embeddings/format/constants"

module StaticEmbeddings
  module Format
    module HashTable
      module_function

      def build(tokens)
        size = next_power_of_two((tokens.length / LOAD_FACTOR).ceil + 1)
        slots = Array.new(size)
        strings = Format.binary_string
        max_probe = 0

        tokens.each_with_index do |token, id|
          bytes = token.b
          offset = strings.bytesize
          strings << bytes
          probe = insert(slots, tokens, size, bytes, hash_bytes(bytes), offset, id)
          max_probe = probe if probe > max_probe
        end

        [size, strings, pack(slots), max_probe]
      end

      def hash_bytes(string, seed = HASH_SEED)
        string.each_byte.reduce(seed) { |hash, byte| ((hash ^ byte) * 16_777_619) & UINT32_MAX }
      end

      def next_power_of_two(value)
        1 << (value - 1).bit_length
      end

      def insert(slots, tokens, size, bytes, hash, offset, id)
        position = hash & (size - 1)
        probe = 1

        loop do
          slot = slots[position]
          unless slot
            slots[position] = [hash, offset, bytes.bytesize, id]
            return probe
          end

          if slot[0] == hash && slot[2] == bytes.bytesize && tokens.fetch(slot[3]).b == bytes
            raise ArgumentError, "duplicate token in vocabulary: #{tokens.fetch(slot[3]).inspect}"
          end

          position = (position + 1) & (size - 1)
          probe += 1
        end
      end

      def pack(slots)
        empty = [0, 0, 0, SLOT_EMPTY].pack("V4")
        slots.each_with_object(Format.binary_string(slots.length * 16)) do |slot, packed|
          packed << (slot ? slot.pack("V4") : empty)
        end
      end
    end

    def self.hash_bytes(string, seed = HASH_SEED)
      HashTable.hash_bytes(string, seed)
    end

    def self.next_power_of_two(value)
      HashTable.next_power_of_two(value)
    end
  end
end
