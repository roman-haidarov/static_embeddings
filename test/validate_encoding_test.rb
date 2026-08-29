require_relative "test_helper"

class ValidateEncodingTest < Minitest::Test
  BASE = "postgres pipeline mode in ruby with local static embeddings ".freeze
  BAD_SEQUENCE = "\xC3\x28".b.freeze

  def setup
    @model = TestSupport.model
  end

  def unscanned(text)
    copy = text.dup
    copy.force_encoding(Encoding::BINARY)
    copy.force_encoding(Encoding::UTF_8)
    copy
  end

  def test_modes_agree_on_valid_input
    text = BASE * 200

    assert_equal @model.embed(unscanned(text)),
                 @model.embed(unscanned(text), validate_encoding: :prefix)
    assert_equal @model.tokenize(unscanned(text)),
                 @model.tokenize(unscanned(text), validate_encoding: :prefix)
    assert_equal @model.embed_batch([unscanned(text), unscanned(text)]),
                 @model.embed_batch([unscanned(text), unscanned(text)], validate_encoding: :prefix)
  end

  def test_default_is_full
    text = unscanned("#{BASE * 400}#{BAD_SEQUENCE}")

    assert_raises(StaticEmbeddings::EncodingError) { @model.embed(text) }
  end

  def test_invalid_bytes_inside_the_window_raise_in_both_modes
    %i[full prefix].each do |mode|
      text = unscanned("hello #{BAD_SEQUENCE} world")

      assert_raises(StaticEmbeddings::EncodingError, "mode: #{mode}") do
        @model.embed(text, validate_encoding: mode)
      end
    end
  end

  def test_invalid_bytes_past_the_window_only_raise_under_full
    build = -> { unscanned("#{BASE * 4000}#{BAD_SEQUENCE}") }

    assert_raises(StaticEmbeddings::EncodingError) do
      @model.embed(build.call, validate_encoding: :full)
    end

    vector = @model.embed(build.call, validate_encoding: :prefix)
    assert_equal @model.dim * 4, vector.bytesize
  end

  def test_a_cached_broken_coderange_raises_under_prefix
    text = unscanned("hello #{BAD_SEQUENCE}")
    refute_predicate text, :valid_encoding?

    assert_raises(StaticEmbeddings::EncodingError) do
      @model.embed(text, validate_encoding: :prefix)
    end
  end

  def test_non_utf8_encodings_are_rejected_in_both_modes
    %i[full prefix].each do |mode|
      text = "hello".dup.force_encoding(Encoding::SHIFT_JIS)

      assert_raises(StaticEmbeddings::EncodingError, "mode: #{mode}") do
        @model.embed(text, validate_encoding: mode)
      end
    end
  end

  def test_unknown_mode_is_rejected
    [:window, :none, "prefix", 1, true].each do |value|
      assert_raises(ArgumentError, "value: #{value.inspect}") do
        @model.embed("hello world", validate_encoding: value)
      end
    end
  end

  def test_batch_reports_the_offending_index
    texts = ["fine", unscanned("bad #{BAD_SEQUENCE}"), "also fine"]

    error = assert_raises(StaticEmbeddings::EncodingError) do
      @model.embed_batch(texts, validate_encoding: :prefix)
    end
    assert_includes error.message, "input[1]"
  end
end
