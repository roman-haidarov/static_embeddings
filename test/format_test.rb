require_relative "test_helper"
require "tmpdir"
require "fileutils"

class FormatTest < Minitest::Test
  def setup
    @path = TestSupport.model_path
  end

  def test_checksum_verifies
    result = StaticEmbeddings.verify(@path)
    assert result[:ok], "checksum mismatch: #{result.inspect}"
    assert_equal result[:expected], result[:stored]
  end

  def test_verify_detects_a_flipped_byte_in_the_matrix
    Dir.mktmpdir do |dir|
      tampered = File.join(dir, "tampered.semb")
      FileUtils.cp(@path, tampered)
      File.open(tampered, "r+b") do |file|
        file.seek(-64, IO::SEEK_END)
        byte = file.read(1)
        file.seek(-1, IO::SEEK_CUR)
        file.write([byte.getbyte(0) ^ 0xFF].pack("C"))
      end

      result = StaticEmbeddings.verify(tampered)
      refute result[:ok], "a flipped matrix byte was not detected"
      refute_equal result[:expected], result[:stored]
    end
  end

  def test_verify_rejects_truncated_and_foreign_files
    Dir.mktmpdir do |dir|
      short = File.join(dir, "short.semb")
      File.binwrite(short, "x" * 16)
      assert_raises(StaticEmbeddings::InvalidModelError) { StaticEmbeddings.verify(short) }

      foreign = File.join(dir, "foreign.semb")
      File.binwrite(foreign, "Z" * 1024)
      assert_raises(StaticEmbeddings::InvalidModelError) { StaticEmbeddings.verify(foreign) }
    end
  end

  def test_verify_does_not_hold_the_file_in_memory
    before = GC.stat(:total_allocated_objects)
    StaticEmbeddings.verify(@path)
    allocated = GC.stat(:total_allocated_objects) - before

    assert_operator allocated, :<, 1_000,
                    "verify allocated #{allocated} objects; it should stream"
  end

  def test_metadata_round_trip
    model = TestSupport.model
    assert_equal 8, model.dim
    assert_operator model.vocab_size, :>, 100
    assert_equal 512, model.max_tokens
    assert model.normalized?
    assert model.lowercase?
    assert_equal File.size(@path), model.mapped_bytes
  end

  def test_provenance_is_mandatory_and_complete
    provenance = TestSupport.model.provenance
    %w[format_version converter_version source_model_id source_files_sha256
       reference_impl reference_max_tokens unicode_source tokenizer_profile
       vocab_size dim].each do |key|
      assert provenance.key?(key), "provenance is missing #{key}"
    end
    assert_equal "model2vec.StaticModel", provenance["reference_impl"]
    assert_equal 512, provenance["reference_max_tokens"]
    assert_equal 4, provenance["source_files_sha256"].length
  end

  def test_conversion_is_deterministic
    Dir.mktmpdir do |dir|
      a = File.join(dir, "a.semb")
      b = File.join(dir, "b.semb")
      StaticEmbeddings.convert(TestSupport.source_dir, output_path: a, model_id: "x")
      StaticEmbeddings.convert(TestSupport.source_dir, output_path: b, model_id: "x")
      assert_equal File.binread(a), File.binread(b), "converter output is not byte-identical"
    end
  end

  def test_missing_file_raises_model_not_found
    assert_raises(StaticEmbeddings::ModelNotFound) { StaticEmbeddings.load("/nonexistent.semb") }
  end

  def test_truncated_file_is_rejected
    Dir.mktmpdir do |dir|
      path = File.join(dir, "short.semb")
      File.binwrite(path, File.binread(@path).byteslice(0, 128))
      assert_raises(StaticEmbeddings::InvalidModelError) { StaticEmbeddings.load(path) }
    end
  end

  def test_bad_magic_is_rejected
    Dir.mktmpdir do |dir|
      path = File.join(dir, "bad.semb")
      data = File.binread(@path).dup
      data[0, 4] = "XXXX"
      File.binwrite(path, data)
      assert_raises(StaticEmbeddings::InvalidModelError) { StaticEmbeddings.load(path) }
    end
  end

  def test_unsupported_version_is_rejected
    Dir.mktmpdir do |dir|
      path = File.join(dir, "future.semb")
      data = File.binread(@path).dup
      data[8, 4] = [99].pack("V")
      File.binwrite(path, data)
      assert_raises(StaticEmbeddings::UnsupportedModelError) { StaticEmbeddings.load(path) }
    end
  end

  def test_out_of_range_section_offset_is_rejected
    Dir.mktmpdir do |dir|
      path = File.join(dir, "bad_offset.semb")
      data = File.binread(@path).dup
      data[160, 8] = [0xFFFFFF00, 0].pack("V2")
      File.binwrite(path, data)
      assert_raises(StaticEmbeddings::InvalidModelError) { StaticEmbeddings.load(path) }
    end
  end


  def test_overlapping_sections_are_rejected
    Dir.mktmpdir do |dir|
      path = File.join(dir, "overlap.semb")
      data = File.binread(@path).dup
      data[144, 16] = data[128, 16]
      File.binwrite(path, data)
      assert_raises(StaticEmbeddings::InvalidModelError) { StaticEmbeddings.load(path) }
    end
  end

  def test_invalid_header_policy_is_rejected
    Dir.mktmpdir do |dir|
      path = File.join(dir, "bad_policy.semb")
      data = File.binread(@path).dup
      data[52, 4] = [1].pack("V")
      File.binwrite(path, data)
      assert_raises(StaticEmbeddings::InvalidModelError) { StaticEmbeddings.load(path) }
    end
  end

  def test_invalid_normalizer_flag_is_rejected
    Dir.mktmpdir do |dir|
      path = File.join(dir, "bad_flag.semb")
      data = File.binread(@path).dup
      data[64, 4] = [2].pack("V")
      File.binwrite(path, data)
      assert_raises(StaticEmbeddings::InvalidModelError) { StaticEmbeddings.load(path) }
    end
  end

  def test_norm_map_len_above_four_is_rejected
    Dir.mktmpdir do |dir|
      path = File.join(dir, "bad_norm_len.semb")
      data = File.binread(@path).dup
      norm_off, = section(data, 176)
      lower_count = read_u32(data, norm_off)
      skip "fixture has no lower table" if lower_count.zero?

      data[norm_off + 32 + 4, 4] = [5].pack("V")
      File.binwrite(path, data)
      assert_raises(StaticEmbeddings::InvalidModelError) { StaticEmbeddings.load(path) }
    end
  end

  def test_norm_ranges_must_be_sorted
    Dir.mktmpdir do |dir|
      path = File.join(dir, "bad_norm_ranges.semb")
      data = File.binread(@path).dup
      norm_off, = section(data, 176)
      lower_count = read_u32(data, norm_off)
      nfd_count = read_u32(data, norm_off + 4)
      mn_count = read_u32(data, norm_off + 8)
      skip "fixture has fewer than two mark ranges" if mn_count < 2

      first_range = norm_off + 32 + (lower_count + nfd_count) * 24
      a = data.byteslice(first_range, 8)
      b = data.byteslice(first_range + 8, 8)
      data[first_range, 8] = b
      data[first_range + 8, 8] = a
      File.binwrite(path, data)
      assert_raises(StaticEmbeddings::InvalidModelError) { StaticEmbeddings.load(path) }
    end
  end

  def test_random_corruption_never_segfaults
    original = File.binread(@path)
    rng = Random.new(20_260_827)
    Dir.mktmpdir do |dir|
      path = File.join(dir, "fuzz.semb")
      80.times do
        data = original.dup
        5.times do
          offset = rng.rand(data.bytesize)
          data.setbyte(offset, rng.rand(256))
        end
        File.binwrite(path, data)
        begin
          model = StaticEmbeddings.load(path)
          model.embed("hello world")
          model.close
        rescue StaticEmbeddings::Error
        end
      end
    end
  end

  def test_closed_model_raises
    model = StaticEmbeddings.load(@path)
    model.close
    assert model.closed?
    assert_raises(StaticEmbeddings::Error) { model.embed("hello") }
  end

  private

  def read_u32(data, offset)
    data.byteslice(offset, 4).unpack1("V")
  end

  def read_u64(data, offset)
    lo, hi = data.byteslice(offset, 8).unpack("V2")
    lo + (hi << 32)
  end

  def section(data, offset)
    [read_u64(data, offset), read_u64(data, offset + 8)]
  end
end
