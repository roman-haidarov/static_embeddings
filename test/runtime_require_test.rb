require_relative "test_helper"
require "open3"
require "rbconfig"

class RuntimeRequireTest < Minitest::Test
  OFFLINE_FEATURES = %w[
    static_embeddings/converter.rb
    static_embeddings/reference.rb
    static_embeddings/safetensors.rb
    static_embeddings/unicode_tables.rb
  ].freeze

  def test_plain_runtime_require_does_not_load_converter_stack
    code = <<~'CODE'
      require "static_embeddings"
      forbidden = %w[
        static_embeddings/converter.rb
        static_embeddings/reference.rb
        static_embeddings/safetensors.rb
        static_embeddings/unicode_tables.rb
      ]
      loaded = $LOADED_FEATURES.select { |feature| forbidden.any? { |suffix| feature.end_with?(suffix) } }
      abort loaded.join("\n") unless loaded.empty?
    CODE

    stdout, stderr, status = Open3.capture3(RbConfig.ruby, "-Ilib", "-e", code, chdir: TestSupport::ROOT)
    assert status.success?, "runtime require pulled offline code:\n#{stdout}#{stderr}"
  end
end
