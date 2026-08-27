module StaticEmbeddings
  class Reference
    CJK_RANGES = [
      0x4E00..0x9FFF, 0x3400..0x4DBF, 0x20000..0x2A6DF, 0x2A700..0x2B73F,
      0x2B740..0x2B81F, 0x2B820..0x2CEAF, 0xF900..0xFAFF, 0x2F800..0x2FA1F
    ].freeze

    ASCII_PUNCT = [33..47, 58..64, 91..96, 123..126].freeze
    ASCII_SPACES = [" ", "\t", "\n", "\r"].freeze
    RE_PUNCT = /\A\p{P}\z/
    RE_CONTROL = /\A(?:\p{Cc}|\p{Cf}|\p{Co}|\p{Cs})\z/
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

    def self.from_source_dir(dir, max_tokens: Converter::REFERENCE_MAX_TOKENS)
      tokenizer = JSON.parse(File.binread(File.join(dir, "tokenizer.json")))
      config_path = File.join(dir, "config.json")
      config = File.file?(config_path) ? JSON.parse(File.binread(config_path)) : {}
      tokens = tokens_from(tokenizer)
      tensor = Safetensors.read(File.join(dir, "model.safetensors"))[:tensors].values.first
      rows, dim = tensor[:shape]
      floats = tensor[:bytes].unpack("e*")
      normalizer = tokenizer["normalizer"] || {}

      new(tokens: tokens,
          matrix: Array.new(rows) { |i| floats[i * dim, dim] },
          meta: meta_from(tokenizer, config, normalizer, dim, max_tokens))
    end

    def self.tokens_from(tokenizer)
      vocab = tokenizer.dig("model", "vocab")
      vocab.each_with_object(Array.new(vocab.length)) { |(token, id), tokens| tokens[id] = token }
    end

    def self.meta_from(tokenizer, config, normalizer, dim, max_tokens)
      lowercase = normalizer.fetch("lowercase", true)
      {
        dim: dim,
        lowercase: lowercase,
        strip_accents: normalizer["strip_accents"].nil? ? lowercase : normalizer["strip_accents"],
        clean_text: normalizer.fetch("clean_text", true),
        handle_chinese_chars: normalizer.fetch("handle_chinese_chars", true),
        max_input_chars_per_word: tokenizer.dig("model", "max_input_chars_per_word") || 100,
        unk_token: tokenizer.dig("model", "unk_token") || "[UNK]",
        normalize: config.key?("normalize") ? config["normalize"] : true,
        max_tokens: max_tokens
      }
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
      ids = []
      pre_tokenize(normalize_text(text)).each { |word| ids.concat(wordpiece(word)) }
      max_tokens && max_tokens.positive? && ids.length > max_tokens ? ids.first(max_tokens) : ids
    end

    def embed(text, max_tokens: @meta[:max_tokens])
      used = tokenize(text, max_tokens: max_tokens).reject { |id| id == unk_id }
      return Array.new(@dim, 0.0) if used.empty?

      vector = pooled(used)
      @meta[:normalize] ? l2_normalize(vector) : vector
    end

    private

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
