require_relative "test_helper"

class TokenizerTest < Minitest::Test
  def setup
    @model = TestSupport.model
    @reference = TestSupport.reference
  end

  def test_matches_reference_on_tricky_texts
    TestSupport::TRICKY_TEXTS.each do |text|
      assert_equal @reference.tokenize(text), @model.tokenize(text),
                   "token ids differ for #{text[0, 40].inspect}"
    end
  end

  def test_fuzz_against_reference
    rng = Random.new(20_260_827)
    alphabet = (0x20..0x7E).to_a + (0x400..0x45F).to_a +
               [0x300, 0x301, 0x308, 0x200B, 0x00A0, 0x2028, 0x3000, 0x4E2D, 0x1F9EC, 0xFFFD]

    300.times do
      length = rng.rand(0..40)
      text = Array.new(length) { alphabet.sample(random: rng) }.pack("U*")
      assert_equal @reference.tokenize(text), @model.tokenize(text),
                   "token ids differ for #{text.inspect}"
    end
  end

  def test_lowercasing_is_not_case_folding

    assert_equal @model.tokenize("ß"), @model.tokenize("ß".downcase)
    refute_equal @model.tokenize("ss"), @model.tokenize("ß")
  end

  def test_truncation_is_applied_before_pooling
    text = "слово " * 400
    assert_equal 512, @model.tokenize(text).length
    assert_equal 700, @model.tokenize(text, max_tokens: 700).length
    assert_operator @model.tokenize(text, max_tokens: false).length, :>, 700
  end

  def test_unknown_words_become_single_unk

    ids = @model.tokenize("\u03be\u03c8\u03c9")
    assert_equal [@model.unk_id], ids
  end

  def test_stats_expose_unk_ratio
    stats = @model.embed_with_stats("\u03be\u03c8\u03c9 \u03b1\u03b2\u03b3 hello")
    assert_equal 3, stats[:token_count]
    assert_equal 2, stats[:unk_count]
    assert_in_delta 2.0 / 3.0, stats[:unk_count].to_f / stats[:token_count], 1e-9
    refute stats[:truncated]
  end

  def test_truncated_flag
    stats = @model.embed_with_stats("слово " * 400)
    assert stats[:truncated]
  end

  def test_rejects_non_utf8_encoding
    assert_raises(StaticEmbeddings::EncodingError) do
      @model.embed("привет".encode("Windows-1251"))
    end
  end

  def test_rejects_invalid_utf8
    assert_raises(StaticEmbeddings::EncodingError) do
      @model.embed("\xC3(".dup.force_encoding("UTF-8"))
    end
  end

  def test_accepts_us_ascii
    assert_equal @model.embed("hello"), @model.embed("hello".encode("US-ASCII"))
  end
end
