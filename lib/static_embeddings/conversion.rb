require "json"
require "static_embeddings/format/writer"
require "static_embeddings/importers"
require "static_embeddings/provenance"
require "static_embeddings/unicode_tables"

module StaticEmbeddings
  module Conversion
    module_function

    def call(source_dir, output_path:, **options)
      model = Importers.import(source_dir, **options)
      provenance = Provenance.json(model, unicode_source: UnicodeTables.source_stamp)
      result = Format::Writer.call(
        path: output_path,
        meta: format_meta(model),
        tokens: model.tokens,
        matrix: model.matrix,
        norm_tables: UnicodeTables.packed,
        provenance: provenance
      )

      result.merge(
        vocab_size: model.tokens.length,
        dim: model.dimensions.output,
        native_dim: model.dimensions.native,
        max_tokens: model.runtime.max_tokens,
        provenance: JSON.parse(provenance)
      )
    end

    def format_meta(model)
      tokenizer = model.tokenizer
      runtime = model.runtime
      {
        dim: model.dimensions.output,
        normalization_type: runtime.normalization,
        max_tokens_default: runtime.max_tokens,
        add_special_tokens: runtime.add_special_tokens,
        unk_policy: runtime.unk_policy,
        empty_policy: runtime.empty_policy,
        do_lower_case: tokenizer.fetch(:do_lower_case),
        strip_accents: tokenizer.fetch(:strip_accents),
        handle_chinese_chars: tokenizer.fetch(:handle_chinese_chars),
        clean_text: tokenizer.fetch(:clean_text),
        added_token_mask: tokenizer.fetch(:added_token_mask),
        max_input_chars_per_word: tokenizer.fetch(:max_input_chars_per_word),
        max_token_chars: tokenizer.fetch(:max_token_chars),
        subword_prefix: tokenizer.fetch(:subword_prefix),
        pad_id: tokenizer.fetch(:pad_id),
        unk_id: tokenizer.fetch(:unk_id),
        cls_id: tokenizer.fetch(:cls_id),
        sep_id: tokenizer.fetch(:sep_id),
        mask_id: tokenizer.fetch(:mask_id)
      }
    end
  end
end
