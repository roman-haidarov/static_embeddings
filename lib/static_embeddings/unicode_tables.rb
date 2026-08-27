module StaticEmbeddings

  module UnicodeTables

    MAP_SCAN_LIMIT = 0x2FFFF
    CODESPACE_LIMIT = 0x10FFFF
    SURROGATES = (0xD800..0xDFFF)

    ENTRY_PACK = "V6"

    RE_MN = /\A\p{Mn}\z/
    RE_PUNCT = /\A\p{P}\z/
    RE_CONTROL = /\A(?:\p{Cc}|\p{Cf}|\p{Co})\z/
    RE_WHITESPACE = /\A(?:\p{Zs}|[\u0085\u2028\u2029])\z/

    class RangeAccumulator
      def initialize
        @ranges = []
      end

      def add(codepoint)
        last = @ranges.last
        if last && last[1] + 1 == codepoint
          last[1] = codepoint
        else
          @ranges << [codepoint, codepoint]
        end
      end

      def to_a
        @ranges
      end
    end

    module_function

    def build
      lower = []
      nfd = []
      mn = RangeAccumulator.new
      punct = RangeAccumulator.new
      control = RangeAccumulator.new
      whitespace = RangeAccumulator.new

      (0..CODESPACE_LIMIT).each do |cp|
        if SURROGATES.cover?(cp)
          control.add(cp)
          next
        end

        ch = begin
          cp.chr(Encoding::UTF_8)
        rescue RangeError
          next
        end

        mn.add(cp) if RE_MN.match?(ch)
        punct.add(cp) if RE_PUNCT.match?(ch)
        control.add(cp) if RE_CONTROL.match?(ch)
        whitespace.add(cp) if RE_WHITESPACE.match?(ch)

        next if cp > MAP_SCAN_LIMIT

        down = ch.downcase
        lower << [cp, down.codepoints] if down != ch

        decomposed = ch.unicode_normalize(:nfd)
        nfd << [cp, decomposed.codepoints] if decomposed != ch
      end

      { "lowercase" => lower, "NFD" => nfd }.each do |name, table|
        long = table.find { |(_, out)| out.length > 4 }
        next unless long

        raise "#{name} mapping for U+#{format('%04X', long[0])} needs #{long[1].length} slots"
      end

      {
        lower: lower,
        nfd: nfd,
        mn: mn.to_a,
        punct: punct.to_a,
        control: control.to_a,
        whitespace: whitespace.to_a
      }
    end

    def pack(tables)
      blob = +""
      blob.force_encoding(Encoding::BINARY)
      blob << [
        tables[:lower].length,
        tables[:nfd].length,
        tables[:mn].length,
        tables[:punct].length,
        tables[:control].length,
        tables[:whitespace].length,
        0, 0
      ].pack("V8")

      %i[lower nfd].each do |key|
        tables[key].sort_by(&:first).each do |(cp, out)|
          padded = out + [0] * (4 - out.length)
          blob << ([cp, out.length] + padded).pack(ENTRY_PACK)
        end
      end

      %i[mn punct control whitespace].each do |key|
        tables[key].sort_by(&:first).each { |(lo, hi)| blob << [lo, hi].pack("V2") }
      end

      blob
    end

    def packed
      @packed ||= pack(build)
    end

    def reset!
      @packed = nil
    end

    def source_stamp
      "ruby-#{RUBY_VERSION}p#{RUBY_PATCHLEVEL}"
    end
  end
end
