require "test_helper"

class InputSnapshotTest < Minitest::Test
  def setup
    @model = TestSupport.model
  end

  def test_replacing_array_elements_during_embed_batch_is_survivable
    big = "привет мир " * 40_000
    texts = [big, big.dup, big.dup]
    expected = @model.embed_batch(texts)

    stop = false
    mutator = Thread.new do
      until stop
        texts[0] = 12_345
        texts[0] = nil
        texts[0] = big
        Thread.pass
      end
    end

    begin
      30.times { assert_equal expected.bytesize, @model.embed_batch(texts).bytesize }
    ensure
      stop = true
      mutator.join
    end
  end

  def test_shrinking_the_strings_during_embed_batch_is_survivable
    big = "hello world " * 40_000
    victim = big.dup
    texts = [victim, "stable text"]

    stop = false
    mutator = Thread.new do
      until stop
        victim.replace("tiny")
        victim.replace(big)
        Thread.pass
      end
    end

    begin
      30.times { refute_empty @model.embed_batch(texts) }
    ensure
      stop = true
      mutator.join
    end
  end

  def test_growing_the_array_during_embed_batch_does_not_change_the_row_count
    texts = ["one", "two"]
    stop = false
    mutator = Thread.new do
      until stop
        texts << "three"
        texts.pop
        Thread.pass
      end
    end

    begin
      50.times { assert_equal 2 * @model.dim * 4, @model.embed_batch(texts).bytesize }
    ensure
      stop = true
      mutator.join
    end
  end

  def test_non_string_elements_still_raise_a_ruby_error
    assert_raises(TypeError) { @model.embed_batch(["ok", 42]) }
    assert_raises(TypeError) { @model.embed_batch([nil]) }
  end
end
