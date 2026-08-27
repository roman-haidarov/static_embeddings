require "json"

module StaticEmbeddings
  class Model
    attr_reader :path

    def provenance
      raw = provenance_json
      return {} if raw.nil?

      JSON.parse(raw)
    rescue JSON::ParserError, EncodingError => e
      raise InvalidModelError, "invalid provenance JSON: #{e.message}"
    end

    def model_id
      provenance["source_model_id"]
    end

    def embed_array(text, **opts)
      format = opts.key?(:format) ? opts[:format] : :f32
      StaticEmbeddings.unpack(embed(text, **opts), dim, format: format).first
    end

    def embed_batch_arrays(texts, **opts)
      format = opts.key?(:format) ? opts[:format] : :f32
      StaticEmbeddings.unpack(embed_batch(texts, **opts), dim, format: format)
    end

    def to_s
      "#<StaticEmbeddings::Model #{model_id || path} dim=#{dim} vocab=#{vocab_size}>"
    end

    alias inspect to_s
  end
end
