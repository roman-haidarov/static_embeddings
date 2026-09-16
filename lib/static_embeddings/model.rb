require "json"

module StaticEmbeddings
  class Model
    attr_reader :path

    if method_defined?(:max_tokens) && !method_defined?(:max_tokens_limit)
      alias_method :max_tokens_limit, :max_tokens

      def max_tokens
        limit = max_tokens_limit
        limit.zero? ? false : limit
      end
    end

    def provenance
      @provenance ||= parse_provenance.freeze
    end

    def model_id
      provenance["source_model_id"]
    end

    def embed_array(text, **options)
      format = options.fetch(:format, :f32)
      StaticEmbeddings.unpack(embed(text, **options), dim, format: format).first
    end

    def embed_batch_arrays(texts, **options)
      format = options.fetch(:format, :f32)
      StaticEmbeddings.unpack(embed_batch(texts, **options), dim, format: format)
    end

    def cosine_top_k(query_blob, matrix_blob, k, **options)
      reject_runtime_dim!(options)
      StaticEmbeddings.cosine_top_k(query_blob, matrix_blob, k, **options.merge(dim: dim))
    end

    def dot_top_k(query_blob, matrix_blob, k, **options)
      reject_runtime_dim!(options)
      StaticEmbeddings.dot_top_k(query_blob, matrix_blob, k, **options.merge(dim: dim))
    end

    def to_s
      "#<StaticEmbeddings::Model #{model_id || path} dim=#{dim} vocab=#{vocab_size}>"
    end

    alias inspect to_s

    private

    def parse_provenance
      raw = provenance_json
      raw.nil? ? {} : JSON.parse(raw)
    rescue JSON::ParserError, EncodingError => e
      raise InvalidModelError, "invalid provenance JSON: #{e.message}"
    end

    def reject_runtime_dim!(options)
      raise InvalidOptionError, "dim: is set by the model" if options.key?(:dim)
    end
  end
end
