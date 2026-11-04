require_relative "test_helper"
require "json"

class Model2VecSideTensorsTest < Minitest::Test
  TEXTS = ["hello world", "привет мир", "test run", "postgres pipeline mode in ruby"].freeze

  def source_matrix
    tensor = StaticEmbeddings::Safetensors.read(File.join(TestSupport.source_dir, "model.safetensors"))
                                          .fetch(:tensors).fetch("embeddings")
    rows, dim = tensor.fetch(:shape)
    floats = tensor.fetch(:bytes).unpack("e*")
    [floats.each_slice(dim).to_a, rows, dim]
  end

  def write_tensors(path, tensors)
    offset = 0
    header = {}
    body = +"".b
    tensors.each do |name, (dtype, shape, bytes)|
      header[name] = { "dtype" => dtype, "shape" => shape, "data_offsets" => [offset, offset + bytes.bytesize] }
      body << bytes
      offset += bytes.bytesize
    end
    json = JSON.generate(header)
    json << (" " * ((8 - (json.bytesize % 8)) % 8))
    File.binwrite(path, [json.bytesize].pack("Q<") + json + body)
  end

  def with_source(tensors)
    Dir.mktmpdir do |dir|
      FileUtils.cp_r(Dir[File.join(TestSupport.source_dir, "*")], dir)
      write_tensors(File.join(dir, "model.safetensors"), tensors)
      Dir.mktmpdir do |out|
        path = File.join(out, "m.semb")
        StaticEmbeddings.convert(dir, output_path: path)
        model = StaticEmbeddings.load(path)
        begin
          yield model
        ensure
          model.close
        end
      end
    end
  end

  def expected(model, text, rows, weights:, mapping: nil)
    ids = model.tokenize(text, max_tokens: false).reject { |id| id == model.unk_id }
    return Array.new(rows.first.length, 0.0) if ids.empty?

    acc = Array.new(rows.first.length, 0.0)
    ids.each do |id|
      row = rows[mapping ? mapping[id] : id]
      row.each_with_index { |x, j| acc[j] += x * weights[id] }
    end
    norm = Math.sqrt(acc.sum { |x| x * x })
    acc.map { |x| x / norm }
  end

  def assert_vectors_close(expected, actual)
    expected.zip(actual).each { |e, a| assert_in_delta e, a, 1e-5 }
  end

  def test_weights_are_applied_not_silently_ignored
    rows, vocab, dim = source_matrix
    weights = Array.new(vocab) { |i| 0.1 + ((i * 37) % 29) / 7.0 }

    with_source("embeddings" => ["F32", [vocab, dim], rows.flatten.pack("e*")],
                "weights" => ["F64", [vocab], weights.pack("E*")]) do |model|
      assert_equal true, model.provenance["baked_token_weights"]
      TEXTS.each do |text|
        assert_vectors_close expected(model, text, rows, weights: weights), model.embed_array(text)
      end
    end
  end

  def test_vocabulary_quantized_mapping_is_expanded
    rows, vocab, dim = source_matrix
    clusters = rows.first(16)
    mapping = Array.new(vocab) { |i| i % clusters.length }
    weights = Array.new(vocab) { |i| 0.5 + (i % 5) }

    with_source("embeddings" => ["F32", [clusters.length, dim], clusters.flatten.pack("e*")],
                "weights" => ["F32", [vocab], weights.pack("e*")],
                "mapping" => ["I64", [vocab], mapping.pack("q<*")]) do |model|
      assert_equal 16, model.provenance["baked_vocabulary_quantization"]
      TEXTS.each do |text|
        assert_vectors_close expected(model, text, clusters, weights: weights, mapping: mapping),
                             model.embed_array(text)
      end
    end
  end

  def test_unknown_tensor_is_rejected
    rows, vocab, dim = source_matrix
    error = assert_raises(StaticEmbeddings::ConversionError) do
      with_source("embeddings" => ["F32", [vocab, dim], rows.flatten.pack("e*")],
                  "bias" => ["F32", [dim], Array.new(dim, 0.0).pack("e*")]) { |_| nil }
    end
    assert_match(/does not understand.*bias/, error.message)
  end

  def test_out_of_range_mapping_is_rejected
    rows, vocab, dim = source_matrix
    assert_raises(StaticEmbeddings::ConversionError) do
      with_source("embeddings" => ["F32", [4, dim], rows.first(4).flatten.pack("e*")],
                  "mapping" => ["I32", [vocab], Array.new(vocab, 4).pack("l<*")]) { |_| nil }
    end
  end
end
