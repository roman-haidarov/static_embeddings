require_relative "test_helper"
require "json"
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

  def test_convert_truncation_flag
    Dir.mktmpdir("static-embeddings-cli") do |dir|
      out = File.join(dir, "cli.semb")
      stdout, stderr, status = Open3.capture3(
        RbConfig.ruby, "-Ilib", "exe/static_embeddings", "convert", TestSupport.source_dir,
        "--out", out, "--id", "fixture/cli-usable", "--truncation", "usable-ids"
      )
      assert status.success?, "CLI failed:\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
      assert_includes stdout, "truncation usable_ids"
      model = StaticEmbeddings.load(out)
      assert_equal :usable_ids, model.truncation
      model.close
    end
  end

  def test_eval_retrieval_flags_trivial_k
    stdout, stderr, status = Open3.capture3(
      RbConfig.ruby, "-Ilib", "tools/eval_retrieval.rb",
      "--model", TestSupport.model_path, "--dataset", "test/fixtures/russian_faq_eval.json",
      "--k", "10", "--bootstrap", "100"
    )
    assert status.success?, "eval failed:\n#{stderr}"
    result = JSON.parse(stdout)
    assert_match(/cannot distinguish rankings/, stderr)
    assert_in_delta 0.293, result.fetch("random_baseline").fetch("mrr"), 0.03
    low, high = result.fetch("bootstrap_95").fetch("mrr")
    assert_operator low, :<=, result.fetch("metrics").fetch("mrr")
    assert_operator high, :>=, result.fetch("metrics").fetch("mrr")
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
