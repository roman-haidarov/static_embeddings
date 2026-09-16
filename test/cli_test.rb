require_relative "test_helper"
require "open3"
require "rbconfig"

class CliTest < Minitest::Test
  def test_convert_command_loads_offline_stack_lazily
    Dir.mktmpdir("static-embeddings-cli") do |dir|
      out = File.join(dir, "cli.semb")
      stdout, stderr, status = Open3.capture3(
        RbConfig.ruby,
        "-Ilib",
        "exe/static_embeddings",
        "convert",
        TestSupport.source_dir,
        "--out",
        out,
        "--id",
        "fixture/cli"
      )

      assert status.success?, "CLI failed:\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
      assert File.file?(out)
      assert_includes stdout, "wrote #{out}"
      assert_empty stderr
    end
  end

  def test_convert_unlimited_and_dimensions
    Dir.mktmpdir("static-embeddings-cli") do |dir|
      out = File.join(dir, "cli.semb")
      stdout, stderr, status = Open3.capture3(
        RbConfig.ruby,
        "-Ilib",
        "exe/static_embeddings",
        "convert",
        TestSupport.source_dir,
        "--out", out,
        "--id", "fixture/cli-unlimited",
        "--max-tokens", "unlimited",
        "--dimensions", "4"
      )

      assert status.success?, "CLI failed:\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
      assert_includes stdout, "max_tokens unlimited"
      assert_includes stdout, "dim        4 (from native 8)"
      model = StaticEmbeddings.load(out)
      assert_equal false, model.max_tokens
      assert_equal 4, model.dim
    ensure
      model&.close
    end
  end
end
