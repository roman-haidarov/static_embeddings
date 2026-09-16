require "static_embeddings/importers"
require "static_embeddings/format/constants"

module StaticEmbeddings
  class Reference
    CJK_RANGES = [
      0x4E00..0x9FFF, 0x3400..0x4DBF, 0x20000..0x2A6DF, 0x2A700..0x2B73F,
      0x2B740..0x2B81F, 0x2B920..0x2CEAF, 0xF900..0xFAFF, 0x2F800..0x2FA1F
    ].freeze

    ASCII_PUNCT = [33..47, 58..64, 91..96, 123..126].freeze
    ASCII_SPACES = [" ", "\t", "\n", "\r"].freeze
    RE_PUNCT = /\A\p{P}\z/
    RE_CONTROL = /\A(?:\p{Cc}|\p{Cf}|\p{Co})\z/
    RE_MN = /\A\p{Mn}\z/
    RE_WHITESPACE = /\A(?:\p{Zs}|[\u0085\u2028\u2029])\z/

    attr_reader :meta

    def initialize(tokens:, matrix:, meta:)
      @tokens = tokens
      @vocab = tokens.each_with_index.to_h
      @matrix = matrix
      @meta = meta
      @dim = meta.fetch(:dim)
    end

    def self.from_source_dir(dir, max_tokens: nil, dimensions: nil)
      from_canonical(Importers.import(dir, max_tokens: max_tokens, dimensions: dimensions))
    end

    def self.from_canonical(model)
      dim = model.dimensions.output
      floats = matrix_bytes(model.matrix).unpack("e*")
      rows = model.tokens.length
      new(
        tokens: model.tokens,
        matrix: Array.new(rows) { |index| floats[index * dim, dim] },
        meta: meta_from_canonical(model)
      )
    end

    def self.matrix_bytes(matrix)
      return matrix unless matrix.respond_to?(:each_chunk)

      matrix.each_chunk.each_with_object(Format.binary_string) { |chunk, packed| packed << chunk }
    end

    def self.meta_from_canonical(model)
      tokenizer = model.tokenizer
      runtime = model.runtime
      {
        dim: model.dimensions.output,
        lowercase: tokenizer.fetch(:do_lower_case),
        strip_accents: tokenizer.fetch(:strip_accents),
        clean_text: tokenizer.fetch(:clean_text),
        handle_chinese_chars: tokenizer.fetch(:handle_chinese_chars),
        max_input_chars_per_word: tokenizer.fetch(:max_input_chars_per_word),
        unk_token: model.tokens.fetch(tokenizer.fetch(:unk_id)),
        normalize: runtime.normalization == Format::NORMALIZATION_L2,
        max_tokens: runtime.max_tokens,
        unk_policy: runtime.unk_policy,
        added_tokens: added_tokens(model)
      }
    end

    def self.added_tokens(model)
      mask = model.tokenizer.fetch(:added_token_mask)
      BertWordPiece::STANDARD_SPECIAL_TOKENS.each_with_object({}) do |(content, bit), tokens|
        next if (mask & bit).zero?

        id = model.tokens.index(content)
        tokens[content] = id unless id.nil?
      end
    end

    def normalize_text(text)
      chars = text.chars
      chars = clean_chars(chars) if @meta[:clean_text]
      chars = split_chinese(chars) if @meta[:handle_chinese_chars]
      chars = strip_accents(chars) if @meta[:strip_accents]
      chars = lowercase(chars) if @meta[:lowercase]
      chars.join
    end

    def pre_tokenize(normalized)
      normalized.split(" ").flat_map { |word| split_punctuation(word) }
    end

    def tokenize(text, max_tokens: @meta[:max_tokens])
      ids = tokenize_with_added_tokens(text)
      max_tokens && max_tokens.positive? && ids.length > max_tokens ? ids.first(max_tokens) : ids
    end

    def embed(text, max_tokens: @meta[:max_tokens])
      ids = apply_unk_policy(tokenize(text, max_tokens: false))
      ids = ids.first(max_tokens) if max_tokens && max_tokens.positive? && ids.length > max_tokens
      return Array.new(@dim, 0.0) if ids.empty?

      vector = pooled(ids)
      @meta[:normalize] ? l2_normalize(vector) : vector
    end

    def apply_unk_policy(ids)
      case @meta.fetch(:unk_policy)
      when Format::UNK_DROP then ids.reject { |id| id == unk_id }
      when Format::UNK_INCLUDE then ids
      else raise InvalidModelError, "unknown unk_policy #{@meta[:unk_policy].inspect}"
      end
    end

    private

    def tokenize_with_added_tokens(text)
      added = @meta[:added_tokens]
      return tokenize_plain(text) if added.empty?

      ids = []
      cursor = 0
      binary = text.b
      while cursor < text.bytesize
        match = added.keys.filter_map do |literal|
          index = binary.index(literal.b, cursor)
          index && [index, -literal.bytesize, literal]
        end.min

        unless match
          ids.concat(tokenize_plain(text.byteslice(cursor, text.bytesize - cursor)))
          break
        end

        index, _, literal = match
        ids.concat(tokenize_plain(text.byteslice(cursor, index - cursor))) if index > cursor
        ids << added.fetch(literal)
        cursor = index + literal.bytesize
      end
      ids
    end

    def tokenize_plain(text)
      return [] if text.nil? || text.empty?

      pre_tokenize(normalize_text(text)).flat_map { |word| wordpiece(word) }
    end

    def clean_chars(chars)
      chars.filter_map do |ch|
        cp = ch.ord
        next nil if cp.zero? || cp == 0xFFFD || control?(ch)

        whitespace?(ch) ? " " : ch
      end
    end

    def split_chinese(chars)
      chars.flat_map { |ch| cjk?(ch.ord) ? [" ", ch, " "] : [ch] }
    end

    def strip_accents(chars)
      chars.flat_map { |ch| ch.unicode_normalize(:nfd).chars }.reject { |ch| RE_MN.match?(ch) }
    end

    def lowercase(chars)
      chars.flat_map { |ch| ch.downcase.chars }
    end

    def split_punctuation(word)
      out = []
      current = +""
      word.each_char do |ch|
        if punctuation?(ch)
          out << current unless current.empty?
          out << ch
          current = +""
        else
          current << ch
        end
      end
      out << current unless current.empty?
      out
    end

    def wordpiece(word)
      chars = word.chars
      return [unk_id] if chars.length > @meta[:max_input_chars_per_word]

      ids = []
      start = 0
      while start < chars.length
        found, finish = longest_piece(chars, start)
        return [unk_id] if found.nil?

        ids << found
        start = finish
      end
      ids
    end

    def longest_piece(chars, start)
      finish = chars.length
      while start < finish
        piece = chars[start...finish].join
        piece = "###{piece}" if start.positive?
        id = @vocab[piece]
        return [id, finish] unless id.nil?

        finish -= 1
      end
      [nil, nil]
    end

    def pooled(ids)
      acc = Array.new(@dim, 0.0)
      ids.each do |id|
        row = @matrix[id]
        @dim.times { |i| acc[i] += row[i] }
      end
      scale = 1.0 / ids.length
      acc.map { |v| v * scale }
    end

    def l2_normalize(vec)
      norm = Math.sqrt(vec.sum { |v| v * v })
      norm.positive? ? vec.map { |v| v / norm } : vec
    end

    def unk_id
      @vocab.fetch(@meta[:unk_token])
    end

    def cjk?(codepoint)
      CJK_RANGES.any? { |range| range.cover?(codepoint) }
    end

    def punctuation?(char)
      cp = char.ord
      ASCII_PUNCT.any? { |range| range.cover?(cp) } || RE_PUNCT.match?(char)
    end

    def control?(char)
      !ASCII_SPACES.include?(char) && RE_CONTROL.match?(char)
    end

    def whitespace?(char)
      ASCII_SPACES.include?(char) || RE_WHITESPACE.match?(char)
    end
  end
end
