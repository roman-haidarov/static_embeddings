require_relative "test_helper"

class AsciiParityTest < Minitest::Test
  BOUNDARY_CODEPOINTS = [
    0x00, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x1F, 0x20,
    0x2F, 0x30, 0x39, 0x3A, 0x40, 0x41, 0x5A, 0x5B, 0x60, 0x61, 0x7A, 0x7B, 0x7E, 0x7F,
    0x80, 0x85, 0x9F, 0xA0, 0xAD, 0x0301, 0x200B, 0x2028, 0x2029, 0x3000, 0x4E2D, 0xFEFF, 0xFFFD
  ].freeze

  FUZZ_ALPHABET = (
    ("a".."z").to_a + ("A".."Z").to_a + ("0".."9").to_a +
    %w[. , ! ? - : ; ' " ( ) / # @ % & * + = _ ~ ^ $ ` | < > [ ] { }] +
    ["\t", "\n", "\r", " ", "\u0000", "\u007F", "\u0085", "\u00A0", "\u00AD",
     "é", "ü", "ß", "ﬁ", "и", "ы", "中", "文", "🧬", "\uFFFD",
     "\u0301", "\u200B", "\u2028", "\u3000"]
  ).freeze

  def setup
    @model = TestSupport.model
    @reference = TestSupport.reference
  end

  def test_every_ascii_byte_matches_reference
    divergent = []

    (0x00..0x7F).each do |cp|
      char = cp.chr(Encoding::UTF_8)
      contexts_for(char).each do |text|
        native = @model.tokenize(text)
        expected = @reference.tokenize(text)
        divergent << format("0x%02X %p: %p != %p", cp, text, native, expected) if native != expected
      end
    end

    assert_empty divergent, -> { divergent.join("\n") }
  end

  def test_del_is_dropped_like_any_other_control_character
    assert_equal @reference.tokenize("hello\u007Fworld"), @model.tokenize("hello\u007Fworld")
    assert_equal @model.tokenize("helloworld"), @model.tokenize("hello\u007Fworld")
    assert_equal @model.tokenize("hello world"), @model.tokenize("hello\u007F world")
    assert_equal @reference.tokenize("\u007F"), @model.tokenize("\u007F")
  end

  def test_boundary_codepoints_match_reference
    BOUNDARY_CODEPOINTS.each do |cp|
      text = "hello#{cp.chr(Encoding::UTF_8)}world static"
      assert_equal @reference.tokenize(text), @model.tokenize(text),
                   "token ids differ around U+#{format('%04X', cp)}"
    end
  end

  def test_differential_fuzz_against_reference
    rng = Random.new(Integer(ENV.fetch("FUZZ_SEED", 20_260_829)))
    cases = Integer(ENV.fetch("FUZZ_N", 2_000))
    mismatches = []

    cases.times do
      text = Array.new(rng.rand(0..80)) { FUZZ_ALPHABET.sample(random: rng) }.join
      next unless text.valid_encoding?

      native = @model.tokenize(text)
      expected = @reference.tokenize(text)
      mismatches << format("%p: %p != %p", text, native, expected) if native != expected
    end

    assert_empty mismatches, -> { mismatches.first(5).join("\n") }
  end

  def test_truncation_boundaries_match_reference
    text = (1..400).map { |i| "token#{i}" }.join(" ")
    full = @reference.tokenize(text)

    [1, 7, 63, 64, 511, 512, 513].each do |max_tokens|
      assert_equal full.first(max_tokens), @model.tokenize(text, max_tokens: max_tokens),
                   "token ids differ at max_tokens: #{max_tokens}"
    end
  end

  def test_del_near_the_prefix_window_boundary
    filler = "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    (3_900..4_400).step(37) do |offset|
      prefix = (filler * ((offset / filler.bytesize) + 2)).byteslice(0, offset)
      text = "#{prefix}\u007Fkilo lima mike#{filler * 4}"

      assert_equal @reference.tokenize(text).first(@model.max_tokens),
                   @model.tokenize(text),
                   "token ids differ with DEL at byte offset #{offset}"
      assert_equal @model.embed_token_ids(@model.tokenize(text, max_tokens: false)), @model.embed(text),
                   "embed disagrees with its own tokenization at byte offset #{offset}"
    end
  end

  def test_embed_agrees_with_reference_when_del_crosses_the_window
    filler = "static embedding vector search local ruby postgres pipeline mode "
    text = "#{filler * 70}\u007Ftail word#{filler * 70}"

    assert_equal @reference.tokenize(text).first(@model.max_tokens), @model.tokenize(text)
    assert_equal @model.embed_token_ids(@model.tokenize(text, max_tokens: false)), @model.embed(text)
  end

  private

  def contexts_for(char)
    ["hello#{char}world", "#{char}hello", "hello#{char}", "a#{char}b c#{char}d", char,
     "hello #{char} world"]
  end
end
