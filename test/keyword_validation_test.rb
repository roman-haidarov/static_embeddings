require_relative "test_helper"

class KeywordValidationTest < Minitest::Test
  def setup
    @model = TestSupport.model
    @vector = @model.embed("hello world")
    @matrix = (@vector * 40).freeze
    @ids = @model.tokenize("hello world")
  end

  def cases
    {
      "embed" => [
        %i[max_tokens format validate_encoding threads],
        ->(kw) { @model.embed("hello world", **kw) }
      ],
      "embed_batch" => [
        %i[max_tokens format validate_encoding threads],
        ->(kw) { @model.embed_batch(["hello world"], **kw) }
      ],
      "embed_with_stats" => [
        %i[max_tokens format validate_encoding threads],
        ->(kw) { @model.embed_with_stats("hello world", **kw) }
      ],
      "tokenize" => [
        %i[max_tokens validate_encoding],
        ->(kw) { @model.tokenize("hello world", **kw) }
      ],
      "embed_token_ids" => [
        %i[max_tokens format threads],
        ->(kw) { @model.embed_token_ids(@ids, **kw) }
      ],
      "embed_token_ids_with_stats" => [
        %i[max_tokens format threads],
        ->(kw) { @model.embed_token_ids_with_stats(@ids, **kw) }
      ],
      "dot_top_k" => [
        %i[dim format allow_unfrozen],
        ->(kw) { StaticEmbeddings.dot_top_k(@vector, @matrix, 3, dim: @model.dim, **kw) }
      ],
      "cosine_top_k" => [
        %i[dim format allow_unfrozen],
        ->(kw) { StaticEmbeddings.cosine_top_k(@vector, @matrix, 3, dim: @model.dim, **kw) }
      ]
    }
  end

  def test_every_method_rejects_an_unknown_keyword
    cases.each do |name, (_accepted, call)|
      error = assert_raises(ArgumentError, "#{name} accepted an unknown keyword") do
        call.call(definitely_not_a_real_keyword: 1)
      end
      assert_includes error.message, "unknown keyword: :definitely_not_a_real_keyword"
    end
  end

  def test_the_error_lists_the_accepted_keywords
    cases.each do |name, (accepted, call)|
      error = assert_raises(ArgumentError) { call.call(nope: 1) }
      accepted.each do |keyword|
        assert_includes error.message, ":#{keyword}", "#{name} did not offer :#{keyword}"
      end
    end
  end

  def test_plausible_typos_are_rejected
    assert_raises(ArgumentError) { @model.embed("hello", fromat: :f16) }
    assert_raises(ArgumentError) { @model.embed("hello", max_token: 3) }
    assert_raises(ArgumentError) { @model.embed("hello", validate_encodings: :prefix) }
    assert_raises(ArgumentError) { @model.embed_batch(["hello"], threds: 4) }
    assert_raises(ArgumentError) do
      StaticEmbeddings.dot_top_k(@vector, @matrix, 3, dim: @model.dim, dimension: 8)
    end
  end

  def test_keywords_are_rejected_where_they_do_not_apply
    assert_raises(ArgumentError) { @model.tokenize("hello", format: :f16) }
    assert_raises(ArgumentError) { @model.embed_token_ids(@ids, validate_encoding: :prefix) }
    assert_raises(ArgumentError) { @model.embed("hello", dim: 8) }
  end

  def test_non_symbol_keys_are_rejected
    error = assert_raises(ArgumentError) { @model.embed("hello", **{ "format" => :f16 }) }
    assert_includes error.message, "Symbol"
  end

  def test_accepted_keywords_still_work
    assert_equal @model.dim * 2, @model.embed("hello", format: :f16).bytesize
    assert_equal @model.dim * 4, @model.embed("hello", max_tokens: false).bytesize
    assert_equal @model.dim * 4, @model.embed("hello", validate_encoding: :prefix).bytesize
    assert_equal @model.dim * 4, @model.embed_batch(["hello"], threads: 1).bytesize
    assert_equal @model.dim * 4, @model.embed("hello").bytesize
    assert_equal 3, StaticEmbeddings.dot_top_k(@vector, @matrix, 3, dim: @model.dim).length
  end

  def test_known_keywords_still_get_their_own_validation
    error = assert_raises(ArgumentError) { @model.embed("hello", format: :f64) }
    refute_includes error.message, "unknown keyword"

    error = assert_raises(ArgumentError) { @model.embed_batch(["hello"], threads: 4) }
    assert_includes error.message, "threads:"
  end
end
