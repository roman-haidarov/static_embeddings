require_relative "test_helper"
require "objspace"

class GcCompactionTest < Minitest::Test
  LARGE_DIM = 64
  LARGE_ROWS = 8_000
  TOPK_GVL_THRESHOLD = 1024 * 1024
  EMBED_GVL_THRESHOLD = 2048

  def setup
    @model = TestSupport.model
    @corpus = Array.new(200) { |i| "row #{i} hello world ruby postgres embeddings" }
    @topk_matrix = large_matrix
    @topk_query = large_query
  end

  def compact
    GC.start
    GC.compact if GC.respond_to?(:compact)
  end

  def test_top_k_agrees_with_itself_across_compaction
    before = StaticEmbeddings.dot_top_k(@topk_query, @topk_matrix, 5, dim: LARGE_DIM)
    compact
    after = StaticEmbeddings.dot_top_k(@topk_query, @topk_matrix, 5, dim: LARGE_DIM)

    assert_equal before, after
  end

  def test_top_k_survives_compaction_running_concurrently
    assert_operator @topk_matrix.bytesize, :>=, TOPK_GVL_THRESHOLD,
                    "matrix is under the top_k GVL threshold; the scan would not release it"
    expected = StaticEmbeddings.dot_top_k(@topk_query, @topk_matrix, 5, dim: LARGE_DIM)

    compactor = Thread.new { 40.times { compact } }
    results = Array.new(100) { StaticEmbeddings.dot_top_k(@topk_query, @topk_matrix, 5, dim: LARGE_DIM) }
    compactor.join

    assert_equal [expected], results.uniq
  end

  def test_cosine_top_k_survives_compaction_running_concurrently
    assert_operator @topk_matrix.bytesize, :>=, TOPK_GVL_THRESHOLD,
                    "matrix is under the top_k GVL threshold; the scan would not release it"
    expected = StaticEmbeddings.cosine_top_k(@topk_query, @topk_matrix, 5, dim: LARGE_DIM)

    compactor = Thread.new { 40.times { compact } }
    results = Array.new(100) { StaticEmbeddings.cosine_top_k(@topk_query, @topk_matrix, 5, dim: LARGE_DIM) }
    compactor.join

    assert_equal [expected], results.uniq
  end

  def test_embed_and_batch_survive_compaction_running_concurrently
    assert_operator @corpus.sum(&:bytesize), :>=, EMBED_GVL_THRESHOLD,
                    "corpus is under the embed GVL threshold; embed_batch would not release it"

    single = @model.embed(@corpus.first)
    batch = @model.embed_batch(@corpus)

    compactor = Thread.new { 40.times { compact } }
    50.times do
      assert_equal single, @model.embed(@corpus.first)
      assert_equal batch, @model.embed_batch(@corpus)
    end
    compactor.join
  end

  def test_large_single_embed_survives_compaction_running_concurrently
    text = (@corpus.join(" ") * 2).freeze
    assert_operator text.bytesize, :>=, EMBED_GVL_THRESHOLD
    expected = @model.embed(text)

    compactor = Thread.new { 40.times { compact } }
    results = Array.new(100) { @model.embed(text) }
    compactor.join

    assert_equal [expected], results.uniq
  end

  def test_the_model_keeps_working_after_compaction
    mapped = @model.mapped_bytes
    dim = @model.dim
    vector = @model.embed("hello world")

    5.times { compact }

    assert_equal mapped, @model.mapped_bytes
    assert_equal dim, @model.dim
    assert_equal vector, @model.embed("hello world")
    assert_equal @model.tokenize("hello world"), @model.tokenize("hello world")
  end

  def test_a_matrix_built_before_compaction_still_scans_correctly
    rows = StaticEmbeddings.unpack(@topk_matrix, LARGE_DIM)
    compact

    top = StaticEmbeddings.dot_top_k(@topk_query, @topk_matrix, 1, dim: LARGE_DIM).first
    index, score = top
    expected = rows[index].each_with_index.sum { |value, i| value * unpacked_query[i] }

    assert_in_delta expected, score, 1e-4
  end

  private

  def large_matrix
    Array.new(LARGE_ROWS * LARGE_DIM) { |i| ((i * 37) % 101) / 101.0 }.pack("e*").freeze
  end

  def large_query
    Array.new(LARGE_DIM) { |i| ((i * 17) % 31) / 31.0 }.pack("e*")
  end

  def unpacked_query
    @unpacked_query ||= StaticEmbeddings.unpack(@topk_query, LARGE_DIM).first
  end
end
