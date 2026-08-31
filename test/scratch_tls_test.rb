require_relative "test_helper"

class ScratchTlsTest < Minitest::Test
  GENERATIONS = 8
  THREADS = 16

  def setup
    @model = TestSupport.model
  end

  def test_scratch_plateaus_across_thread_generations
    skip "allocation stats build was not loaded" unless StaticEmbeddings.respond_to?(:__alloc_stats__)

    samples = Array.new(GENERATIONS) do
      threads = Array.new(THREADS) { Thread.new { @model.embed("hello world") } }
      threads.each(&:join)
      scratch_current_bytes
    end

    later = samples.drop(1)
    peak = later.max
    floor = later.min
    assert_operator peak, :<=, samples.first + 65_536,
                    "scratch kept growing across thread generations: #{samples.inspect}"
    assert_operator peak - floor, :<=, 65_536,
                    "scratch did not plateau: #{samples.inspect}"
  end

  def test_large_document_does_not_pin_scratch
    skip "allocation stats build was not loaded" unless StaticEmbeddings.respond_to?(:__alloc_stats__)

    @model.embed("hi")
    baseline = scratch_current_bytes
    huge = ("alpha bravo charlie " * 80_000).freeze
    refute_operator huge.bytesize, :<, 1_000_000

    @model.embed(huge, max_tokens: false)
    @model.embed("hi")
    after = scratch_current_bytes

    assert_operator after, :<=, [baseline * 2, 65_536].max,
                    "large input pinned scratch: baseline=#{baseline} after=#{after}"
  end

  def test_nested_calls_on_one_thread_still_work
    ids = @model.tokenize("hello world")
    blob = @model.embed("hello world")
    assert_equal @model.dim * 4, blob.bytesize
    assert_operator ids.length, :>=, 1
  end

  private

  def scratch_current_bytes
    stats = StaticEmbeddings.__alloc_stats__
    stats.fetch(:scratch).fetch(:current_bytes)
  end
end
