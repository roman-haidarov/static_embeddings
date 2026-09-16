require "json"
require "static_embeddings/format/constants"
require "static_embeddings/bert_wordpiece"

module StaticEmbeddings
  module Provenance
    MODEL2VEC_VERSION = "0.9.0"
    TOKENIZERS_VERSION = "0.23.1"
    UNICODE_CATEGORIES_VERSION = "0.1.1"
    NOTES = "Vectors are only reference-compatible if docs/MODEL_AUDIT.md records a passing oracle run for this source revision."

    module_function

    def json(model, unicode_source:)
      JSON.generate(payload(model, unicode_source: unicode_source).sort.to_h)
    end

    def payload(model, unicode_source:)
      dimensions = model.dimensions
      runtime = model.runtime
      source = model.source
      tokenizer = model.tokenizer

      data = {
        "add_special_tokens" => runtime.add_special_tokens,
        "config_seq_length" => source.config_seq_length,
        "converter_version" => StaticEmbeddings::VERSION,
        "dim" => dimensions.output,
        "format_version" => Format::VERSION,
        "native_dim" => dimensions.native,
        "normalize" => runtime.normalization == Format::NORMALIZATION_L2,
        "notes" => NOTES,
        "oracle" => source.oracle,
        "output_dim" => dimensions.output,
        "reference_impl" => source.oracle,
        "reference_max_tokens" => runtime.max_tokens,
        "reference_tokenizers_version" => TOKENIZERS_VERSION,
        "source_family" => source.family,
        "source_files_sha256" => source.files_sha256,
        "source_model_id" => source.model,
        "source_revision" => source.revision,
        "tokenizer_class" => source.tokenizer_class,
        "tokenizer_profile" => tokenizer.fetch(:tokenizer_profile, BertWordPiece::TOKENIZER_PROFILE),
        "unicode_source" => unicode_source,
        "unk_policy" => runtime.unk_policy == Format::UNK_DROP ? "drop" : "include",
        "vocab_size" => model.tokens.length
      }
      data["trained_mrl_dims"] = dimensions.trained if dimensions.trained
      add_model2vec_versions(data) if source.family == "model2vec"
      data
    end

    def add_model2vec_versions(data)
      data["reference_model2vec_version"] = MODEL2VEC_VERSION
      data["reference_unicode_categories_version"] = UNICODE_CATEGORIES_VERSION
    end
  end
end
