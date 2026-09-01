require_relative "test_helper"
require "json"

class ConverterTest < Minitest::Test
  def with_modified_tokenizer(&block)
    Dir.mktmpdir do |dir|
      FileUtils.cp_r(Dir[File.join(TestSupport.source_dir, "*")], dir)
      tokenizer = JSON.parse(File.read(File.join(dir, "tokenizer.json")))
      block.call(tokenizer)
      File.write(File.join(dir, "tokenizer.json"), JSON.generate(tokenizer))
      yield_convert(dir)
    end
  end

  def yield_convert(dir)
    Dir.mktmpdir do |out|
      StaticEmbeddings.convert(dir, output_path: File.join(out, "m.semb"))
    end
  end

  def test_rejects_non_wordpiece_model
    assert_raises(StaticEmbeddings::UnsupportedModelError) do
      with_modified_tokenizer { |t| t["model"]["type"] = "Unigram" }
    end
  end

  def test_rejects_unknown_normalizer
    assert_raises(StaticEmbeddings::UnsupportedModelError) do
      with_modified_tokenizer { |t| t["normalizer"]["type"] = "Sequence" }
    end
  end

  def test_rejects_unknown_normalizer_keys
    assert_raises(StaticEmbeddings::UnsupportedModelError) do
      with_modified_tokenizer { |t| t["normalizer"]["nmt"] = true }
    end
  end

  def test_rejects_clean_text_false
    error = assert_raises(StaticEmbeddings::UnsupportedModelError) do
      with_modified_tokenizer { |t| t["normalizer"]["clean_text"] = false }
    end
    assert_match(/clean_text=false/, error.message)
  end

  def test_rejects_unknown_pre_tokenizer
    assert_raises(StaticEmbeddings::UnsupportedModelError) do
      with_modified_tokenizer { |t| t["pre_tokenizer"] = { "type" => "Whitespace" } }
    end
  end

  def test_rejects_non_standard_added_tokens
    assert_raises(StaticEmbeddings::UnsupportedModelError) do
      with_modified_tokenizer do |t|
        t["added_tokens"] << {
          "id" => 5, "content" => "new york", "single_word" => false,
          "lstrip" => false, "rstrip" => false, "normalized" => true, "special" => false
        }
      end
    end
  end

  def test_rejects_standard_added_token_that_is_normalized
    error = assert_raises(StaticEmbeddings::UnsupportedModelError) do
      with_modified_tokenizer { |t| t["added_tokens"].last["normalized"] = true }
    end
    assert_match(/normalized=false/, error.message)
  end

  def test_rejects_standard_added_token_with_wrong_vocab_id
    error = assert_raises(StaticEmbeddings::UnsupportedModelError) do
      with_modified_tokenizer { |t| t["added_tokens"].last["id"] = 3 }
    end
    assert_match(/model\.vocab/, error.message)
  end

  def test_rejects_added_token_with_lstrip
    assert_raises(StaticEmbeddings::UnsupportedModelError) do
      with_modified_tokenizer { |t| t["added_tokens"].first["lstrip"] = true }
    end
  end

  def test_rejects_unsupported_subword_prefix
    assert_raises(StaticEmbeddings::UnsupportedModelError) do
      with_modified_tokenizer { |t| t["model"]["continuing_subword_prefix"] = "@@" }
    end
  end

  def test_rejects_vocab_matrix_mismatch
    Dir.mktmpdir do |dir|
      FileUtils.cp_r(Dir[File.join(TestSupport.source_dir, "*")], dir)
      tokenizer = JSON.parse(File.read(File.join(dir, "tokenizer.json")))
      tokenizer["model"]["vocab"]["extra_token_not_in_matrix"] = tokenizer["model"]["vocab"].length
      File.write(File.join(dir, "tokenizer.json"), JSON.generate(tokenizer))

      error = assert_raises(StaticEmbeddings::ConversionError) { yield_convert(dir) }
      assert_match(/rows but the tokenizer has/, error.message)
    end
  end

  def test_rejects_truncated_safetensors
    Dir.mktmpdir do |dir|
      FileUtils.cp_r(Dir[File.join(TestSupport.source_dir, "*")], dir)
      path = File.join(dir, "model.safetensors")
      File.binwrite(path, File.binread(path).byteslice(0, 64))
      assert_raises(StaticEmbeddings::ConversionError) { yield_convert(dir) }
    end
  end

  def test_max_tokens_default_is_the_reference_value_not_model_max_length

    assert_equal 512, TestSupport.model.max_tokens
    assert_equal 1_000_000, TestSupport.model.provenance["config_seq_length"]
  end
end
