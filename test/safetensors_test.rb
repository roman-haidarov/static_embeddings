require_relative "test_helper"
require "static_embeddings/safetensors"

class SafetensorsTest < Minitest::Test
  def test_reads_and_converts_f16_without_loading_the_whole_source_file
    Dir.mktmpdir do |dir|
      path = File.join(dir, "f16.safetensors")
      floats = [1.0, -2.0, 0.5, 0.0]
      body = StaticEmbeddings.encode_f16(floats)
      StaticEmbeddings::Safetensors.write_raw(path, "embeddings", [2, 2], "F16", body)

      description = StaticEmbeddings::Safetensors.describe(path)
      source = description.fetch(:tensors).fetch("embeddings")
      refute source.key?(:bytes)
      payload = StaticEmbeddings::Safetensors.f32_payload(path, source)
      assert_equal 16, payload.bytesize
      assert_equal floats, payload.each_chunk.to_a.join.unpack("e*")

      tensor = StaticEmbeddings::Safetensors.read(path).fetch(:tensors).fetch("embeddings")
      assert_equal "F16", tensor.fetch(:dtype)
      assert_equal [2, 2], tensor.fetch(:shape)
      assert_equal floats, StaticEmbeddings::Safetensors.f32_bytes(tensor).unpack("e*")
    end
  end

  def test_reads_and_converts_bf16
    Dir.mktmpdir do |dir|
      path = File.join(dir, "bf16.safetensors")
      floats = [1.0, -2.0, 0.5, 4.0]
      body = floats.map { |value| value.to_f.then { |v| [v].pack("e").unpack1("V") >> 16 } }.pack("v*")
      StaticEmbeddings::Safetensors.write_raw(path, "embeddings", [2, 2], "BF16", body)

      tensor = StaticEmbeddings::Safetensors.read(path).fetch(:tensors).fetch("embeddings")
      assert_equal "BF16", tensor.fetch(:dtype)
      assert_equal floats, StaticEmbeddings::Safetensors.f32_bytes(tensor).unpack("e*")
    end
  end

  def test_rejects_dtype_outside_the_converter_contract
    Dir.mktmpdir do |dir|
      path = File.join(dir, "bad.safetensors")
      header = JSON.generate("embeddings" => { "dtype" => "I8", "shape" => [1], "data_offsets" => [0, 1] })
      padded = header + (" " * ((8 - (header.bytesize % 8)) % 8))
      File.binwrite(path, [padded.bytesize].pack("Q<") + padded + "\0")

      error = assert_raises(StaticEmbeddings::ConversionError) { StaticEmbeddings::Safetensors.read(path) }
      assert_match(/dtype I8 is not supported/, error.message)
    end
  end

  def test_reference_reads_f16_source_matrix
    Dir.mktmpdir do |dir|
      source = File.join(dir, "model")
      FileUtils.cp_r(TestSupport.source_dir, source)
      tensor = StaticEmbeddings::Safetensors.read(File.join(source, "model.safetensors")).fetch(:tensors).values.first
      floats = StaticEmbeddings::Safetensors.f32_bytes(tensor).unpack("e*")
      StaticEmbeddings::Safetensors.write_raw(
        File.join(source, "model.safetensors"), "embeddings", tensor.fetch(:shape), "F16",
        StaticEmbeddings.encode_f16(floats)
      )

      reference = StaticEmbeddings::Reference.from_source_dir(source)
      assert_equal TestSupport.reference.tokenize("hello world"), reference.tokenize("hello world")
      assert_equal TestSupport.reference.embed("hello world").length, reference.embed("hello world").length
    end
  end
end
