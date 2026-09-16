module StaticEmbeddings
  module Canonical
    Dimensions = Struct.new(:native, :output, :trained, keyword_init: true)
    Runtime = Struct.new(:normalization, :unk_policy, :empty_policy, :max_tokens, :add_special_tokens,
                         keyword_init: true)
    Source = Struct.new(:family, :model, :revision, :oracle, :files_sha256, :tokenizer_class,
                        :config_seq_length, keyword_init: true)
    Model = Struct.new(:tokens, :matrix, :dimensions, :runtime, :tokenizer, :source, keyword_init: true)

    module_function

    def dimensions(native:, output:, trained: nil)
      Dimensions.new(native: native, output: output, trained: trained&.freeze).freeze
    end

    def runtime(normalization:, unk_policy:, empty_policy:, max_tokens:, add_special_tokens: false)
      Runtime.new(
        normalization: normalization,
        unk_policy: unk_policy,
        empty_policy: empty_policy,
        max_tokens: max_tokens,
        add_special_tokens: add_special_tokens
      ).freeze
    end

    def source(family:, model:, oracle:, files_sha256:, revision: nil, tokenizer_class: nil,
               config_seq_length: nil)
      Source.new(
        family: family,
        model: model,
        revision: revision,
        oracle: oracle,
        files_sha256: files_sha256.freeze,
        tokenizer_class: tokenizer_class,
        config_seq_length: config_seq_length
      ).freeze
    end

    def model(tokens:, matrix:, dimensions:, runtime:, tokenizer:, source:)
      Model.new(
        tokens: tokens.freeze,
        matrix: matrix,
        dimensions: dimensions,
        runtime: runtime,
        tokenizer: tokenizer.freeze,
        source: source
      ).freeze
    end
  end
end
