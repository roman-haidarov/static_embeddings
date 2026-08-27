require_relative "test_helper"

class BuiltinTest < Minitest::Test
  def test_builtin_dir_is_where_the_demo_model_is_written
    expected = File.expand_path("../lib/models", __dir__)
    assert_equal expected, StaticEmbeddings::Paths::BUILTIN_DIR
  end

  def test_builtin_model_ships_and_loads
    skip "run `rake demo_model` first" unless StaticEmbeddings.builtin_available?

    model = StaticEmbeddings.load_builtin
    assert_operator model.dim, :>, 0
    assert_equal model.dim * 4, model.embed("hello world").bytesize
    assert_equal "demo/tiny-wordpiece", model.model_id
  end

  def test_warmup_is_idempotent
    skip "run `rake demo_model` first" unless StaticEmbeddings.builtin_available?

    model = StaticEmbeddings.load_builtin
    before = model.embed("hello world")
    3.times { model.warmup! }
    assert_equal before, model.embed("hello world")
  end

  def test_cache_dir_is_overridable
    original = ENV["STATIC_EMBEDDINGS_CACHE"]
    ENV["STATIC_EMBEDDINGS_CACHE"] = "/tmp/se-cache-test"
    assert_equal "/tmp/se-cache-test", StaticEmbeddings.cache_dir
    assert_equal "/tmp/se-cache-test/models/foo.semb", StaticEmbeddings.model_path("foo")
  ensure
    ENV["STATIC_EMBEDDINGS_CACHE"] = original
  end

  def test_missing_installed_model_raises_rather_than_downloading
    ENV["STATIC_EMBEDDINGS_CACHE"] = "/tmp/se-cache-empty"
    error = assert_raises(StaticEmbeddings::ModelNotFound) do
      StaticEmbeddings.load_model("potion-retrieval-32m")
    end
    assert_match(/Convert it first/, error.message)
  ensure
    ENV.delete("STATIC_EMBEDDINGS_CACHE")
  end
end
