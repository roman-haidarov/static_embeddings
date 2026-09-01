require_relative "test_helper"

class EmbeddingTest < Minitest::Test
  COSINE_FLOOR = 1.0 - 1e-6
  MAX_ABS_DIFF = 1e-5

  def setup
    @model = TestSupport.model
    @reference = TestSupport.reference
  end

  def test_matches_reference_within_tolerance
    TestSupport::TRICKY_TEXTS.each do |text|
      actual = @model.embed_array(text)
      expected = @reference.embed(text)

      max_diff = actual.zip(expected).map { |a, b| (a - b).abs }.max
      assert_operator max_diff, :<=, MAX_ABS_DIFF, "max abs diff too large for #{text[0, 40].inspect}"

      next if expected.all?(&:zero?)

      assert_operator cosine(actual, expected), :>=, COSINE_FLOOR,
                      "cosine too low for #{text[0, 40].inspect}"
    end
  end

  def test_output_is_binary_float32
    blob = @model.embed("hello world")
    assert_equal Encoding::BINARY, blob.encoding
    assert_equal @model.dim * 4, blob.bytesize
  end

  def test_output_format_f16
    f32 = @model.embed("hello world", format: :f32)
    f16 = @model.embed("hello world", format: :f16)

    assert_equal @model.dim * 4, f32.bytesize
    assert_equal @model.dim * 2, f16.bytesize
    assert_equal Encoding::BINARY, f16.encoding

    decoded = StaticEmbeddings.unpack(f16, @model.dim, format: :f16).first
    assert_operator cosine(f32.unpack("e*"), decoded), :>=, 0.999
  end

  def test_output_format_accepts_string_aliases
    assert_equal @model.dim * 4, @model.embed("hello", format: "float32").bytesize
    assert_equal @model.dim * 2, @model.embed("hello", format: "float16").bytesize
  end

  def test_output_format_rejects_unknown_format
    assert_raises(ArgumentError) { @model.embed("hello", format: :wat) }
  end

  def test_embed_array_decodes_f16
    f32 = @model.embed_array("hello world", format: :f32)
    f16 = @model.embed_array("hello world", format: :f16)

    assert_equal @model.dim, f16.length
    assert_operator cosine(f32, f16), :>=, 0.999
  end

  def test_embed_batch_format_f16
    texts = ["hello", "world", "ruby postgres"]
    f16 = @model.embed_batch(texts, format: :f16)

    assert_equal texts.length * @model.dim * 2, f16.bytesize
    decoded = StaticEmbeddings.unpack(f16, @model.dim, format: :f16)
    expected = @model.embed_batch_arrays(texts)

    decoded.zip(expected).each do |actual, reference|
      assert_operator cosine(actual, reference), :>=, 0.999
    end
  end

  def test_embed_with_stats_uses_requested_format
    stats = @model.embed_with_stats("hello world", format: :f16)
    assert_equal @model.dim * 2, stats.fetch(:vector).bytesize
  end

  def test_embed_token_ids_uses_requested_format
    ids = @model.tokenize("hello world")
    f16 = @model.embed_token_ids(ids, format: :f16)

    assert_equal @model.dim * 2, f16.bytesize
    decoded = StaticEmbeddings.unpack(f16, @model.dim, format: :f16).first
    assert_operator cosine(decoded, @model.embed_array("hello world")), :>=, 0.999
  end

  def test_top_k_accepts_f16_storage_format
    corpus = ["hello world", "привет мир", "ruby gem", "static embedding vector"]
    matrix = @model.embed_batch(corpus, format: :f16)
    query = @model.embed("hello world", format: :f16)

    results = StaticEmbeddings.cosine_top_k(query, matrix, 2, dim: @model.dim, format: :f16)
    assert_equal 2, results.length
    assert_equal 0, results.first[0]
    assert_in_delta 1.0, results.first[1], 1e-3
    assert_operator results[0][1], :>=, results[1][1]
  end

  def test_max_tokens_zero_is_rejected
    assert_raises(ArgumentError) { @model.embed("hello world", max_tokens: 0) }
  end

  def test_max_tokens_too_large_is_rejected
    assert_raises(ArgumentError) { @model.embed("hello world", max_tokens: 2**32 + 5) }
  end

  def test_max_tokens_false_means_unlimited
    blob = @model.embed("hello world", max_tokens: false)
    assert_equal @model.dim * 4, blob.bytesize
  end

  def test_vectors_are_l2_normalized
    vec = @model.embed_array("hello world ruby")
    assert_in_delta 1.0, Math.sqrt(vec.sum { |v| v * v }), 1e-5
  end

  def test_empty_input_yields_zero_vector
    vec = @model.embed_array("")
    assert_equal Array.new(@model.dim, 0.0), vec
  end

  def test_all_unknown_input_yields_zero_vector

    vec = @model.embed_array("\u03be\u03c8\u03c9 \u03b1\u03b2\u03b3")
    assert_equal Array.new(@model.dim, 0.0), vec
  end

  def test_identical_text_is_deterministic
    a = @model.embed("локальный поиск")
    100.times { assert_equal a, @model.embed("локальный поиск") }
  end


  def test_large_embed_releases_the_gvl
    skip "timing-sensitive GVL progress probe; set STATIC_EMBEDDINGS_TIMING_TESTS=1 to run" unless TestSupport.timing_sensitive_tests?

    text = "локальный поиск по тексту " * 80_000
    ticks = TestSupport.thread_ticks_during { @model.embed(text, max_tokens: false) }

    assert_operator ticks, :>, 0,
                    "no other Ruby thread ran during large embed — the GVL was never released"
  end

  def test_cosine_top_k
    corpus = ["hello world", "привет мир", "ruby gem", "static embedding vector"]
    matrix = @model.embed_batch(corpus)
    query = @model.embed("hello world")

    results = StaticEmbeddings.cosine_top_k(query, matrix, 2, dim: @model.dim)
    assert_equal 2, results.length
    assert_equal 0, results.first[0]
    assert_in_delta 1.0, results.first[1], 1e-5
    assert_operator results[0][1], :>=, results[1][1]
  end

  def test_top_k_empty_matrix_returns_empty_array
    query = [1.0, 0.0].pack("e*")
    assert_equal [], StaticEmbeddings.dot_top_k(query, "".b, 10, dim: 2)
    assert_equal [], StaticEmbeddings.cosine_top_k(query, "".b, 10, dim: 2)
  end

  def test_top_k_accepts_unaligned_matrix_string
    query = [1.0, 0.0].pack("e*")
    raw = "x".b + [1.0, 0.0, 0.0, 1.0].pack("e*")
    matrix = raw.byteslice(1, raw.bytesize - 1)

    assert_equal [[0, 1.0], [1, 0.0]], StaticEmbeddings.dot_top_k(query, matrix, 2, dim: 2)
  end

  def test_dot_top_k_keeps_scores_below_minus_two
    query = [-1.0].pack("e")
    matrix = [3.0, 2.0, 1.0].pack("e*")
    results = StaticEmbeddings.dot_top_k(query, matrix, 3, dim: 1)

    assert_equal [[2, -1.0], [1, -2.0], [0, -3.0]], results
  end

  def test_dot_top_k_is_a_raw_dot_product
    query = [3.0, 4.0].pack("e*")
    matrix = [3.0, 4.0].pack("e*")
    assert_in_delta 25.0, StaticEmbeddings.dot_top_k(query, matrix, 1, dim: 2).first[1], 1e-5
  end

  def test_cosine_top_k_normalizes_unnormalized_input
    query = [3.0, 4.0].pack("e*")
    matrix = [30.0, 40.0, -4.0, 3.0].pack("e*")
    results = StaticEmbeddings.cosine_top_k(query, matrix, 2, dim: 2)

    assert_equal 0, results.first[0]
    assert_in_delta 1.0, results.first[1], 1e-6
    assert_in_delta 0.0, results.last[1], 1e-6
  end

  def test_cosine_top_k_treats_a_zero_row_as_zero_similarity
    query = [1.0, 0.0].pack("e*")
    matrix = [0.0, 0.0, 1.0, 0.0].pack("e*")
    results = StaticEmbeddings.cosine_top_k(query, matrix, 2, dim: 2)

    assert_equal 1, results.first[0]
    assert_in_delta 1.0, results.first[1], 1e-6
    assert_in_delta 0.0, results.last[1], 1e-6
  end

  def test_cosine_top_k_rejects_a_zero_query
    query = [0.0, 0.0].pack("e*")
    matrix = [1.0, 0.0].pack("e*")
    assert_raises(ArgumentError) { StaticEmbeddings.cosine_top_k(query, matrix, 1, dim: 2) }
  end

  def test_top_k_is_not_poisoned_by_nan_rows
    nan = [Float::NAN].pack("e")
    query = [1.0].pack("e")
    matrix = ([2.0].pack("e") + nan + [5.0].pack("e") + [1.0].pack("e"))
    results = StaticEmbeddings.dot_top_k(query, matrix, 3, dim: 1)

    assert_equal [2, 0, 3], results.map(&:first)
    refute results.any? { |(_, score)| score.nan? }, "a NaN row leaked into the top-k"
  end

  def test_unk_is_filtered_before_the_usable_token_cap
    known = @model.tokenize("hello", max_tokens: false).fetch(0)
    unk = @model.unk_id
    ids = [known, unk, known, unk, known, unk, known, unk, known]

    expected = @model.embed_token_ids([known, known, known, known], max_tokens: false)
    assert_equal expected, @model.embed_token_ids(ids, max_tokens: 4)

    stats = @model.embed_token_ids_with_stats(ids, max_tokens: 4)
    assert_equal 7, stats.fetch(:token_count)
    assert_equal 3, stats.fetch(:unk_count)
    assert_equal 4, stats.fetch(:pooled_count)
    assert stats.fetch(:truncated)
  end

  def test_text_embedding_scans_past_unk_to_fill_the_usable_cap
    text = "hello [UNK] hello [UNK] hello [UNK] hello [UNK] world"
    raw = @model.tokenize(text, max_tokens: false)

    assert_includes raw, @model.unk_id
    assert_equal @model.embed_token_ids(raw, max_tokens: 4), @model.embed(text, max_tokens: 4)

    stats = @model.embed_with_stats(text, max_tokens: 4)
    assert_equal 4, stats.fetch(:pooled_count)
    assert_operator stats.fetch(:token_count), :>, 4
    assert stats.fetch(:truncated)
  end

  def test_embed_token_ids_debug_path
    ids = @model.tokenize("hello world")
    assert_equal @model.embed("hello world"), @model.embed_token_ids(ids)
  end

  def test_embed_token_ids_caps_usable_ids_before_pooling
    ids = @model.tokenize("hello world ruby postgres vector")
    skip "fixture text is too short to truncate" if ids.length < 3

    capped = @model.embed_token_ids(ids, max_tokens: 2)
    assert_equal @model.embed_token_ids(ids.first(2)), capped
  end

  def test_embed_token_ids_with_stats
    ids = @model.tokenize("hello world ruby postgres vector")
    skip "fixture text is too short to truncate" if ids.length < 3

    stats = @model.embed_token_ids_with_stats(ids, max_tokens: 2)
    assert_equal 2, stats.fetch(:token_count)
    assert_equal 2, stats.fetch(:pooled_count)
    assert stats.fetch(:truncated)
    assert_equal @model.dim * 4, stats.fetch(:vector).bytesize

    full = @model.embed_token_ids_with_stats(ids)
    assert_equal ids.length, full.fetch(:token_count)
    refute full.fetch(:truncated)
  end

  def test_embed_token_ids_rejects_out_of_range_ids
    assert_raises(ArgumentError) { @model.embed_token_ids([@model.vocab_size]) }
  end

  def test_embed_token_ids_rejects_non_integer_ids
    assert_raises(TypeError) { @model.embed_token_ids([1, "x", 3]) }
  end

  private

  def cosine(a, b)
    dot = a.zip(b).sum { |x, y| x * y }
    na = Math.sqrt(a.sum { |x| x * x })
    nb = Math.sqrt(b.sum { |x| x * x })
    return 1.0 if na.zero? || nb.zero?

    dot / (na * nb)
  end
end
