require_relative "test_helper"
require "open3"
require "rbconfig"

class CliTest < Minitest::Test
  def test_convert_command_loads_converter_lazily
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
end
