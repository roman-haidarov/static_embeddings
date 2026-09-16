require_relative "test_helper"
require "open3"
require "rbconfig"

class RuntimeRequireTest < Minitest::Test
  OFFLINE_FEATURES = %w[
    static_embeddings/conversion.rb
    static_embeddings/codec.rb
    static_embeddings/reference.rb
    static_embeddings/safetensors.rb
    static_embeddings/unicode_tables.rb
    static_embeddings/canonical.rb
    static_embeddings/provenance.rb
    static_embeddings/bert_wordpiece.rb
    static_embeddings/importers.rb
    static_embeddings/importers/model2vec.rb
    static_embeddings/importers/support.rb
    static_embeddings/importers/sentence_transformers_static.rb
    static_embeddings/row_prefix_payload.rb
    static_embeddings/format/writer.rb
    static_embeddings/format/hash_table.rb
    static_embeddings/format/trie.rb
  ].freeze

  def test_plain_runtime_require_does_not_load_offline_stack
    forbidden = OFFLINE_FEATURES.inspect
    code = <<~RUBY
      require "static_embeddings"
      forbidden = #{forbidden}
      loaded = $LOADED_FEATURES.select { |feature| forbidden.any? { |suffix| feature.end_with?(suffix) } }
      abort loaded.join("\n") unless loaded.empty?
    RUBY

    stdout, stderr, status = Open3.capture3(RbConfig.ruby, "-Ilib", "-e", code, chdir: TestSupport::ROOT)
    assert status.success?, "runtime require pulled offline code:\n#{stdout}#{stderr}"
  end
end
