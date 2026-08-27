require_relative "test_helper"

class BatchTest < Minitest::Test
  def setup
    @model = TestSupport.model
  end

  def test_batch_layout_is_row_major
    texts = ["hello", "привет мир", "ruby gem"]
    blob = @model.embed_batch(texts)
    assert_equal texts.length * @model.dim * 4, blob.bytesize

    rows = StaticEmbeddings.unpack(blob, @model.dim)
    texts.each_with_index do |text, i|
      assert_equal @model.embed_array(text), rows[i]
    end
  end

  def test_empty_batch
    assert_equal "", @model.embed_batch([])
  end

  def test_threads_option_is_rejected
    error = assert_raises(ArgumentError) do
      @model.embed_batch(["hello"], threads: 2)
    end
    assert_match(/threads:/, error.message)
  end

  def test_batches_are_safe_under_ruby_threads
    texts = (1..100).map { |i| "text number #{i}" }
    expected = @model.embed_batch(texts)

    results = 8.times.map do
      Thread.new { @model.embed_batch(texts) }
    end.map(&:value)

    results.each { |r| assert_equal expected, r }
  end

  def test_error_inside_batch_is_reported
    assert_raises(StaticEmbeddings::EncodingError) do
      @model.embed_batch(["fine", "привет".encode("Windows-1251")])
    end
  end

  def test_batch_releases_the_gvl

    texts = (1..4000).map { |i| "локальный поиск по тексту номер #{i}" }
    ticks = 0
    ticker = Thread.new do
      loop do
        ticks += 1
        sleep 0.001
      end
    end

    sleep 0.02
    before = ticks
    @model.embed_batch(texts)
    ticker.kill

    assert_operator ticks - before, :>, 0,
                    "no other Ruby thread ran during embed_batch — the GVL was never released"
  end

  def test_works_under_a_nonblocking_fiber_scheduler_if_available
    skip "Fiber scheduler is unavailable" unless Fiber.respond_to?(:set_scheduler)
    skip "Fiber.blocking? is unavailable" unless Fiber.respond_to?(:blocking?)

    scheduler = RecordingScheduler.new
    texts = (1..3000).map { |i| "fiber scheduler embedding text #{i}" }
    expected = @model.embed_batch(texts)
    actual = nil

    previous = Fiber.scheduler
    Fiber.set_scheduler(scheduler)
    fiber = Fiber.new(blocking: false) do
      actual = @model.embed_batch(texts)
    end

    fiber.resume
    scheduler.drain_until_done(fiber)

    assert_equal expected, actual
    assert_operator scheduler.blocks, :>, 0
  ensure
    Fiber.set_scheduler(previous) if Fiber.respond_to?(:set_scheduler)
  end

  class RecordingScheduler
    attr_reader :blocks

    def initialize
      @blocks = 0
      @ready = Queue.new
    end

    def block(_blocker, _timeout = nil)
      @blocks += 1
      Fiber.yield
      true
    end

    def unblock(_blocker, fiber)
      @ready << fiber
    end

    def kernel_sleep(_duration = nil)
      @blocks += 1
      Fiber.yield
      0
    end

    def io_wait(_io, _events, _timeout = nil)
      @blocks += 1
      0
    end

    def close
    end

    def drain_until_done(fiber)
      while fiber.alive?
        ready = @ready.pop
        ready.resume if ready.alive?
      end
    end
  end
end
