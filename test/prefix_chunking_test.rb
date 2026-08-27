require_relative "test_helper"

class PrefixChunkingTest < Minitest::Test
  LARGE_BYTES = 200_000

  def setup
    @model = TestSupport.model
  end

  def assert_prefix_path_matches(text, label)
    expected = @model.embed_token_ids(@model.tokenize(text))
    assert_equal expected, @model.embed(text), "prefix path diverged for #{label}"
  end

  def test_ordinary_large_document
    text = ("postgres pipeline mode in ruby static embeddings vector search " * 4000)
    assert_operator text.bytesize, :>, LARGE_BYTES
    assert_prefix_path_matches(text, "repeated ascii prose")
  end

  def test_document_with_no_ascii_boundary_at_all
    text = "a" * LARGE_BYTES
    assert_prefix_path_matches(text, "one giant word")
  end

  def test_document_whose_prefix_yields_too_few_tokens
    text = ("word" + (" " * 2000)) * 200
    assert_operator text.bytesize, :>, LARGE_BYTES
    assert_prefix_path_matches(text, "separator heavy text")
  end

  def test_document_with_only_non_ascii_separators
    text = "слово\u00a0" * 20_000
    assert_operator text.bytesize, :>, LARGE_BYTES
    assert_prefix_path_matches(text, "nbsp separated cyrillic")
  end

  def test_document_that_is_punctuation_separated
    text = ("alpha,beta;gamma.delta!" * 12_000)
    assert_operator text.bytesize, :>, LARGE_BYTES
    assert_prefix_path_matches(text, "punctuation separated")
  end

  def test_multibyte_document_never_cuts_inside_a_codepoint
    text = ("中文测试 café привет ascii " * 6000)
    assert_operator text.bytesize, :>, LARGE_BYTES
    assert_prefix_path_matches(text, "mixed width utf-8")
  end

  def test_unlimited_max_tokens_reads_the_whole_document
    text = ("postgres pipeline mode in ruby " * 8000)
    expected = @model.embed_token_ids(@model.tokenize(text, max_tokens: false),
                                      max_tokens: false)
    assert_equal expected, @model.embed(text, max_tokens: false)
  end

  def test_truncated_flag_is_reported_from_the_prefix
    text = ("postgres pipeline mode in ruby " * 8000)
    stats = @model.embed_with_stats(text)

    assert stats.fetch(:truncated), "a document this large must report truncation"
    assert_equal @model.max_tokens, stats.fetch(:token_count)
  end

  def test_batch_applies_the_same_prefix_logic_as_single
    texts = [
      "short one",
      ("postgres pipeline mode in ruby static embeddings " * 5000),
      "another short one",
      "a" * LARGE_BYTES,
      ("word" + (" " * 2000)) * 200
    ]

    batch = @model.embed_batch(texts)
    assert_equal texts.length * @model.dim * 4, batch.bytesize

    texts.each_with_index do |text, i|
      row = batch.byteslice(i * @model.dim * 4, @model.dim * 4)
      assert_equal @model.embed(text), row, "batch row #{i} diverged from single embed"
    end
  end

  def test_batch_mixing_retrying_and_non_retrying_rows    
    fast = "postgres pipeline mode in ruby static embeddings " * 5000
    slow = ("word" + (" " * 2000)) * 200

    batch = @model.embed_batch([fast, slow, fast])
    stride = @model.dim * 4

    assert_equal @model.embed(fast), batch.byteslice(0, stride)
    assert_equal @model.embed(slow), batch.byteslice(stride, stride)
    assert_equal @model.embed(fast), batch.byteslice(2 * stride, stride)
  end
  
  def test_every_printable_ascii_separator_survives_the_prefix_window
    mismatches = []

    (0x20..0x7e).each do |code|
      sep = code.chr
      text = ("zz#{sep}" * 4000)
      next if text.bytesize < 10_000

      expected = @model.embed_token_ids(@model.tokenize(text))
      mismatches << format("0x%02x %s", code, sep.inspect) if @model.embed(text) != expected
    end

    assert_empty mismatches,
                 "prefix cut disagrees with word splitting for: #{mismatches.join(', ')}"
  end

  def test_small_max_tokens_still_matches
    text = ("postgres pipeline mode in ruby static embeddings " * 5000)
    [1, 7, 64].each do |cap|
      expected = @model.embed_token_ids(@model.tokenize(text, max_tokens: cap), max_tokens: cap)
      assert_equal expected, @model.embed(text, max_tokens: cap), "max_tokens=#{cap} diverged"
    end
  end

  def test_prefix_path_is_deterministic
    text = ("postgres pipeline mode in ruby static embeddings " * 5000)
    first = @model.embed(text)
    5.times { assert_equal first, @model.embed(text) }
  end
end
