require_relative "test_helper"
require "json"

class ConversionTest < Minitest::Test
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

  def test_wordpiece_conversion_stays_format_v3
    Dir.mktmpdir do |dir|
      path = File.join(dir, "wp.semb")
      StaticEmbeddings.convert(TestSupport.source_dir, output_path: path, model_id: "wp-v3")
      header = File.binread(path, 32)
      assert_equal 3, header.byteslice(8, 4).unpack1("V")
      assert_equal StaticEmbeddings::Format::TOKENIZER_BERT_WORDPIECE_V1, header.byteslice(28, 4).unpack1("V")
    end
  end

  def test_max_tokens_default_is_the_reference_value_not_model_max_length
    assert_equal 512, TestSupport.model.max_tokens
    assert_equal 1_000_000, TestSupport.model.provenance["config_seq_length"]
  end

  def test_model2vec_missing_normalize_defaults_to_none
    Dir.mktmpdir do |dir|
      FileUtils.cp_r(Dir[File.join(TestSupport.source_dir, "*")], dir)
      config_path = File.join(dir, "config.json")
      config = JSON.parse(File.read(config_path))
      config.delete("normalize")
      File.write(config_path, JSON.generate(config))

      Dir.mktmpdir do |out|
        path = File.join(out, "unnormalized.semb")
        StaticEmbeddings.convert(dir, output_path: path)
        model = StaticEmbeddings.load(path)
        refute model.normalized?
        assert_equal false, model.provenance["normalize"]
      ensure
        model&.close
      end
    end
  end

  def test_max_tokens_above_u32_is_rejected_before_write
    error = assert_raises(StaticEmbeddings::InvalidOptionError) do
      Dir.mktmpdir do |out|
        StaticEmbeddings.convert(
          TestSupport.source_dir,
          output_path: File.join(out, "overflow.semb"),
          max_tokens: StaticEmbeddings::Format::UINT32_MAX + 1
        )
      end
    end
    assert_match(/max_tokens/, error.message)
  end

  def test_unknown_source_family_is_rejected_fail_closed
    Dir.mktmpdir do |dir|
      FileUtils.cp_r(Dir[File.join(TestSupport.source_dir, "*")], dir)
      config_path = File.join(dir, "config.json")
      config = JSON.parse(File.read(config_path))
      config["model_type"] = "unknown"
      config.delete("architectures")
      File.write(config_path, JSON.generate(config))

      assert_raises(StaticEmbeddings::InvalidSourceError) { yield_convert(dir) }
    end
  end

  def test_model2vec_semantic_sections_are_stable
    Dir.mktmpdir do |dir|
      a = File.join(dir, "a.semb")
      b = File.join(dir, "b.semb")
      StaticEmbeddings.convert(TestSupport.source_dir, output_path: a, model_id: "x")
      StaticEmbeddings.convert(TestSupport.source_dir, output_path: b, model_id: "x")
      assert_equal semantic_sections(a), semantic_sections(b)
    end
  end

  def test_dimensions_prefix_slice_with_st_none_matches_row_prefix
    with_st_layout(TestSupport.source_dir) do |st_dir|
      Dir.mktmpdir do |out|
        full = File.join(out, "full.semb")
        sliced = File.join(out, "sliced.semb")
        StaticEmbeddings.convert(st_dir, output_path: full, model_id: "st-full")
        StaticEmbeddings.convert(st_dir, output_path: sliced, model_id: "st-sliced", dimensions: 4)

        full_model = StaticEmbeddings.load(full)
        sliced_model = StaticEmbeddings.load(sliced)
        assert_equal 8, full_model.dim
        assert_equal 4, sliced_model.dim
        assert_equal false, sliced_model.normalized?
        assert_equal false, sliced_model.max_tokens

        text = "hello world ruby"
        expected = StaticEmbeddings.unpack(full_model.embed(text), full_model.dim).first.first(4)
        actual = StaticEmbeddings.unpack(sliced_model.embed(text), sliced_model.dim).first
        assert_equal expected, actual
      ensure
        full_model&.close
        sliced_model&.close
      end
    end
  end

  def test_st_importer_uses_unk_include_and_no_l2
    with_st_layout(TestSupport.source_dir) do |st_dir|
      Dir.mktmpdir do |out|
        path = File.join(out, "st.semb")
        StaticEmbeddings.convert(st_dir, output_path: path, model_id: "st")
        model = StaticEmbeddings.load(path)
        reference = StaticEmbeddings::Reference.from_source_dir(st_dir)

        assert_equal "sentence_transformers_static", model.provenance["source_family"]
        assert_equal "include", model.provenance["unk_policy"]
        assert_equal false, model.normalized?
        assert_equal false, model.max_tokens
        refute model.provenance["add_special_tokens"]

        oov = "\u03be\u03c8\u03c9 \u03b1\u03b2\u03b3"
        unk_row = model.embed_token_ids([model.unk_id])
        assert_equal unk_row, model.embed(oov)
        assert_equal reference.embed(oov), StaticEmbeddings.unpack(model.embed(oov), model.dim).first
        refute_equal Array.new(model.dim, 0.0), model.embed_array(oov)
      ensure
        model&.close
      end
    end
  end

  def test_model2vec_all_unk_is_still_a_zero_vector
    oov = "\uE000\uE001\uE002"
    model = TestSupport.model
    assert(model.tokenize(oov, max_tokens: false).all? { |id| id == model.unk_id })
    assert_equal Array.new(model.dim, 0.0), model.embed_array(oov)
    assert_equal Array.new(model.dim, 0.0), TestSupport.reference.embed(oov)
  end

  def test_model2vec_default_caps_usable_ids_at_512
    text = ("hello " * 600).strip
    model = TestSupport.model
    unbounded = model.tokenize(text, max_tokens: false)
    assert_operator unbounded.length, :>, 512
    stats = model.embed_with_stats(text)
    assert stats[:truncated]
    assert_equal 512, stats[:pooled_count]
  end

  def test_st_unlimited_does_not_cap_past_512
    with_st_layout(TestSupport.source_dir) do |st_dir|
      Dir.mktmpdir do |out|
        path = File.join(out, "st.semb")
        StaticEmbeddings.convert(st_dir, output_path: path, model_id: "st-long")
        model = StaticEmbeddings.load(path)
        text = ("hello " * 600).strip
        unbounded = model.tokenize(text, max_tokens: false)
        assert_operator unbounded.length, :>, 512
        stats = model.embed_with_stats(text)
        refute stats[:truncated]
        assert_equal unbounded.length, stats[:pooled_count]
      ensure
        model&.close
      end
    end
  end

  def test_model2vec_config_wins_over_st_modules_json
    Dir.mktmpdir do |dir|
      FileUtils.cp_r(Dir[File.join(TestSupport.source_dir, "*")], dir)
      File.write(File.join(dir, "modules.json"), JSON.generate([
        { "idx" => 0, "name" => "0", "path" => ".",
          "type" => "sentence_transformers.models.StaticEmbedding" },
        { "idx" => 1, "name" => "1", "path" => "1_Normalize",
          "type" => "sentence_transformers.models.Normalize" }
      ]))
      out = File.join(dir, "out.semb")
      StaticEmbeddings.convert(dir, output_path: out, model_id: "m2v-wrapped")
      model = StaticEmbeddings.load(out)
      assert_equal "model2vec", model.provenance["source_family"]
      assert_equal "drop", model.provenance["unk_policy"]
      assert model.normalized?
      assert_equal 512, model.max_tokens
    ensure
      model&.close
    end
  end

  def test_st_importer_rejects_extra_modules
    with_st_layout(TestSupport.source_dir) do |st_dir|
      modules = [
        { "idx" => 0, "name" => "0", "path" => "0_StaticEmbedding",
          "type" => "sentence_transformers.models.StaticEmbedding" },
        { "idx" => 1, "name" => "1", "path" => "1_Normalize",
          "type" => "sentence_transformers.models.Normalize" }
      ]
      File.write(File.join(st_dir, "modules.json"), JSON.generate(modules))
      error = assert_raises(StaticEmbeddings::UnsupportedModelError) { yield_convert(st_dir) }
      assert_match(/exactly one StaticEmbedding/, error.message)
    end
  end

  def test_st_importer_reads_path_from_modules_json
    with_st_layout(TestSupport.source_dir, module_path: "static_embedding") do |st_dir|
      Dir.mktmpdir do |out|
        path = File.join(out, "st.semb")
        report = StaticEmbeddings.convert(st_dir, output_path: path, model_id: "st-path")
        assert_equal 8, report[:dim]
        assert File.file?(path)
      end
    end
  end

  def test_st_importer_rejects_escaped_module_path
    Dir.mktmpdir do |root|
      outside = File.join(root, "outside")
      src = File.join(root, "src")
      FileUtils.mkdir_p(outside)
      FileUtils.mkdir_p(src)
      FileUtils.cp_r(Dir[File.join(TestSupport.source_dir, "*")], outside)
      File.write(File.join(src, "modules.json"), JSON.generate([{
        "idx" => 0, "name" => "0", "path" => "../outside",
        "type" => "sentence_transformers.models.StaticEmbedding"
      }]))
      error = assert_raises(StaticEmbeddings::UnsupportedModelError) { yield_convert(src) }
      assert_match(/escapes the source directory/, error.message)
    end
  end

  def test_st_importer_rejects_symlink_escape
    skip "symlink escape probe requires POSIX" if Gem.win_platform?

    with_st_layout(TestSupport.source_dir) do |st_dir|
      Dir.mktmpdir do |outside|
        FileUtils.rm_rf(File.join(st_dir, "0_StaticEmbedding"))
        File.symlink(outside, File.join(st_dir, "0_StaticEmbedding"))
        error = assert_raises(StaticEmbeddings::UnsupportedModelError) { yield_convert(st_dir) }
        assert_match(/escapes the source directory/, error.message)
      end
    end
  end

  def test_convert_max_tokens_unlimited
    Dir.mktmpdir do |out|
      path = File.join(out, "unlimited.semb")
      StaticEmbeddings.convert(TestSupport.source_dir, output_path: path, model_id: "u", max_tokens: false)
      model = StaticEmbeddings.load(path)
      assert_equal false, model.max_tokens
      assert_equal 0, model.provenance["reference_max_tokens"]
    ensure
      model&.close
    end
  end

  def test_dimensions_out_of_range_is_rejected
    error = assert_raises(ArgumentError) do
      Dir.mktmpdir do |out|
        StaticEmbeddings.convert(TestSupport.source_dir, output_path: File.join(out, "x.semb"),
                                 dimensions: 64)
      end
    end
    assert_match(/native_dim/, error.message)
  end

  def with_st_layout(src, module_path: "0_StaticEmbedding")
    Dir.mktmpdir do |dir|
      mod = File.join(dir, module_path)
      FileUtils.mkdir_p(mod)
      FileUtils.cp_r(Dir[File.join(src, "*")], mod)
      File.write(File.join(dir, "modules.json"), JSON.generate([{
        "idx" => 0,
        "name" => "0",
        "path" => module_path,
        "type" => "sentence_transformers.models.StaticEmbedding"
      }]))
      yield dir
    end
  end

  def semantic_sections(path)
    data = File.binread(path)
    StaticEmbeddings::Format::SECTION_FIELDS.each_with_object({}) do |(name, offset), acc|
      next if name == :provenance

      section_off = read_u64(data, offset)
      section_size = read_u64(data, offset + 8)
      acc[name] = data.byteslice(section_off, section_size)
    end
  end

  def read_u64(data, offset)
    lo, hi = data.byteslice(offset, 8).unpack("V2")
    lo + (hi << 32)
  end
end
