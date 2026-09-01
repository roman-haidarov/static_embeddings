require_relative "test_helper"

class PathsTest < Minitest::Test
  def setup
    @env = { "XDG_CACHE_HOME" => "/tmp/static-embeddings-path-test" }
  end

  def test_model_id_can_use_namespace_segments
    assert_equal "/tmp/static-embeddings-path-test/static_embeddings/models/minishlab/potion-retrieval-32M.semb",
                 StaticEmbeddings::Paths.model_path("minishlab/potion-retrieval-32M", @env)
  end

  def test_model_id_rejects_parent_and_absolute_paths
    ["../etc/passwd", "x/../../etc/passwd", "/etc/passwd", "x/../y", "x\\..\\y", "C:/Windows/System32"].each do |id|
      assert_raises(ArgumentError, id) { StaticEmbeddings::Paths.model_path(id, @env) }
    end
  end

  def test_model_id_rejects_ambiguous_or_invalid_segments
    ["", ".", "x//y", "x/./y", "x/", "x\0y"].each do |id|
      assert_raises(ArgumentError, id.inspect) { StaticEmbeddings::Paths.model_path(id, @env) }
    end
  end
end
