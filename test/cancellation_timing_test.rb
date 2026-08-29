require_relative "test_helper"
require "timeout"

class CancellationTimingTest < Minitest::Test
  UNIT = "alpha bravo charlie delta echo foxtrot golf hotel ".freeze
  PROBE_BYTES = 2 * 1024 * 1024
  MAX_BYTES = 96 * 1024 * 1024

  def setup
    @model = TestSupport.model
  end

  def test_ascii_fast_path_stays_interruptible
    budget = Float(ENV.fetch("CANCEL_BUDGET", "0.05"))
    text = ascii_document_taking_at_least(budget * 12)
    skip "machine is too fast to size an interruptible document" if text.nil?

    full = measure { @model.embed(text, max_tokens: false) }

    elapsed = measure do
      assert_raises(Timeout::Error) do
        Timeout.timeout(budget) { @model.embed(text, max_tokens: false) }
      end
    end

    assert_operator elapsed, :<, full / 3.0,
                    "embed ignored the deadline for #{elapsed.round(3)}s of a #{full.round(3)}s call"
  end

  private

  def ascii_document_taking_at_least(target)
    probe = UNIT * (PROBE_BYTES / UNIT.bytesize)
    elapsed = measure { @model.embed(probe, max_tokens: false) }
    return probe if elapsed >= target

    scale = (target / elapsed).ceil
    bytes = probe.bytesize * scale
    return nil if bytes > MAX_BYTES

    UNIT * (bytes / UNIT.bytesize)
  end

  def measure
    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    yield
    Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
  end
end
