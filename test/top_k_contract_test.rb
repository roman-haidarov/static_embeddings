require "test_helper"

class TopKContractTest < Minitest::Test
  LARGE_DIM = 64
  LARGE_ROWS = 8_000

  def setup
    @model = TestSupport.model
  end

  def large_matrix
    @large_matrix ||= Array.new(LARGE_ROWS * LARGE_DIM) { |i| ((i * 37) % 101) / 101.0 }
                           .pack("e*").freeze
  end

  def large_query
    @large_query ||= Array.new(LARGE_DIM) { |i| ((i * 17) % 31) / 31.0 }.pack("e*")
  end

  def test_large_frozen_matrix_is_searchable_from_many_threads
    assert_operator large_matrix.bytesize, :>, 1024 * 1024

    expected = StaticEmbeddings.dot_top_k(large_query, large_matrix, 5, dim: LARGE_DIM)
    errors = Queue.new
    results = Queue.new

    threads = 4.times.map do
      Thread.new do
        100.times do
          results << StaticEmbeddings.dot_top_k(large_query, large_matrix, 5, dim: LARGE_DIM)
        rescue StandardError => e
          errors << "#{e.class}: #{e.message}"
          break
        end
      end
    end
    threads.each(&:join)

    assert_empty errors.size.times.map { errors.pop }.uniq
    assert_equal 400, results.size
    results.size.times { assert_equal expected, results.pop }
  end

  def test_large_unfrozen_matrix_is_rejected_with_an_actionable_message
    matrix = large_matrix.dup
    refute_predicate matrix, :frozen?

    error = assert_raises(ArgumentError) do
      StaticEmbeddings.dot_top_k(large_query, matrix, 5, dim: LARGE_DIM)
    end
    assert_match(/freeze/, error.message)
  end

  def test_large_unfrozen_matrix_is_allowed_explicitly_under_the_gvl
    matrix = large_matrix.dup

    assert_equal StaticEmbeddings.dot_top_k(large_query, large_matrix, 5, dim: LARGE_DIM),
                 StaticEmbeddings.dot_top_k(large_query, matrix, 5, dim: LARGE_DIM,
                                                                   allow_unfrozen: true)
  end

  def test_small_unfrozen_matrix_does_not_need_freezing
    query = [1.0, 0.0].pack("e*")
    matrix = [1.0, 0.0, 0.0, 1.0].pack("e*")

    assert_equal [[0, 1.0], [1, 0.0]], StaticEmbeddings.dot_top_k(query, matrix, 2, dim: 2)
  end

  def test_large_unaligned_matrix_is_rejected_instead_of_silently_copied
    raw = "x".b + large_matrix
    matrix = raw.byteslice(1, raw.bytesize - 1).freeze

    begin
      StaticEmbeddings.dot_top_k(large_query, matrix, 5, dim: LARGE_DIM)
    rescue ArgumentError => e
      assert_match(/aligned/, e.message)
      return
    end

    assert_equal StaticEmbeddings.dot_top_k(large_query, large_matrix, 5, dim: LARGE_DIM),
                 StaticEmbeddings.dot_top_k(large_query, matrix, 5, dim: LARGE_DIM)
  end

  def test_dim_is_required
    query = [1.0, 0.0].pack("e*")
    matrix = [1.0, 0.0].pack("e*")

    assert_raises(ArgumentError) { StaticEmbeddings.dot_top_k(query, matrix, 1) }
    assert_raises(ArgumentError) { StaticEmbeddings.cosine_top_k(query, matrix, 1) }
  end

  def test_f32_query_against_an_f16_matrix_raises_instead_of_returning_garbage
    corpus = Array.new(8) { |i| "document number #{i} about hello world" }
    matrix_f16 = @model.embed_batch(corpus, format: :f16)
    query_f32 = @model.embed("hello world")

    error = assert_raises(ArgumentError) do
      StaticEmbeddings.cosine_top_k(query_f32, matrix_f16, 3, dim: @model.dim, format: :f16)
    end
    assert_match(/query is/, error.message)
  end

  def test_f16_query_against_an_f32_matrix_raises
    corpus = Array.new(8) { |i| "document number #{i} about hello world" }
    matrix_f32 = @model.embed_batch(corpus)
    query_f16 = @model.embed("hello world", format: :f16)

    assert_raises(ArgumentError) do
      StaticEmbeddings.cosine_top_k(query_f16, matrix_f32, 3, dim: @model.dim)
    end
  end

  def test_matrix_that_is_not_a_whole_number_of_rows_raises
    query = [1.0, 0.0].pack("e*")
    matrix = [1.0, 0.0, 1.0].pack("e*")

    assert_raises(ArgumentError) { StaticEmbeddings.dot_top_k(query, matrix, 1, dim: 2) }
  end

  def test_model_level_wrappers_inject_dim
    corpus = ["hello world", "привет мир", "ruby gem"]
    matrix = @model.embed_batch(corpus)
    query = @model.embed("hello world")

    assert_equal StaticEmbeddings.cosine_top_k(query, matrix, 2, dim: @model.dim),
                 @model.cosine_top_k(query, matrix, 2)
    assert_equal StaticEmbeddings.dot_top_k(query, matrix, 2, dim: @model.dim),
                 @model.dot_top_k(query, matrix, 2)
  end

  def test_model_level_wrapper_rejects_a_mismatched_blob
    matrix = @model.embed_batch(["hello world", "ruby gem"], format: :f16)
    query = @model.embed("hello world")

    assert_raises(ArgumentError) { @model.cosine_top_k(query, matrix, 1, format: :f16) }
  end

  def test_model_level_wrapper_rejects_explicit_dim_override
    corpus = ["hello world", "ruby gem"]
    matrix = @model.embed_batch(corpus)
    query = @model.embed("hello world")

    assert_raises(ArgumentError) { @model.cosine_top_k(query, matrix, 1, dim: 999) }
    assert_raises(ArgumentError) { @model.dot_top_k(query, matrix, 1, dim: 999) }
  end

  def test_simd_backend_reports_a_known_kernel
    assert_includes %w[neon-fp16 f16c lut], StaticEmbeddings.simd_backend
  end

  def test_cosine_top_k_drops_a_nan_row_like_dot_top_k_does
    query = [1.0, 1.0, 1.0, 1.0]
    rows = ([1.0] * 4) + ([Float::NAN] * 4) + ([2.0] * 4)

    %i[f32 f16].each do |format|
      q = StaticEmbeddings.pack(query, format: format)
      m = StaticEmbeddings.pack(rows, format: format)

      cosine = StaticEmbeddings.cosine_top_k(q, m, 3, dim: 4, format: format)
      dot = StaticEmbeddings.dot_top_k(q, m, 3, dim: 4, format: format)

      refute_includes cosine.map(&:first), 1, "#{format}: NaN row leaked into cosine_top_k"
      refute_includes dot.map(&:first), 1, "#{format}: NaN row leaked into dot_top_k"
      assert_equal [0, 2], cosine.map(&:first).sort
    end
  end

  def test_cosine_top_k_still_scores_a_zero_row_as_zero
    query = StaticEmbeddings.pack([1.0, 0.0], format: :f32)
    matrix = StaticEmbeddings.pack([0.0, 0.0, 1.0, 0.0], format: :f32)
    results = StaticEmbeddings.cosine_top_k(query, matrix, 2, dim: 2)

    assert_equal [[1, 1.0], [0, 0.0]], results
  end
end
