module StaticEmbeddings
  module Paths
    BUILTIN_DIR = File.expand_path("../models", __dir__)
    BUILTIN_MODELS = { demo: "demo.semb" }.freeze

    module_function

    def cache_dir(env = ENV)
      env["STATIC_EMBEDDINGS_CACHE"] ||
        File.join(env["XDG_CACHE_HOME"] || File.join(Dir.home, ".cache"), "static_embeddings")
    end

    def model_path(model_id, env = ENV)
      id = model_id.to_s
      raise ArgumentError, "model_id must not be empty" if id.empty?
      raise ArgumentError, "model_id contains a NUL byte" if id.include?("\0")

      normalized = id.tr("\\", "/")
      parts = normalized.split("/", -1)
      if normalized.start_with?("/") ||
         parts.any? { |part| part.empty? || part == "." || part == ".." || part.include?(":") }
        raise ArgumentError, "model_id must be a relative slash-separated identifier"
      end

      base = File.expand_path(File.join(cache_dir(env), "models"))
      path = File.expand_path(File.join(base, "#{normalized}.semb"))
      prefix = base.end_with?(File::SEPARATOR) ? base : "#{base}#{File::SEPARATOR}"
      unless path.start_with?(prefix)
        raise ArgumentError, "model_id escapes the model cache"
      end
      path
    end

    def builtin_path(name)
      file = BUILTIN_MODELS.fetch(name) do
        raise ModelNotFound, "unknown builtin model #{name.inspect}"
      end
      File.join(BUILTIN_DIR, file)
    end

    def builtin_available?(name = :demo)
      file = BUILTIN_MODELS[name]
      !file.nil? && File.file?(File.join(BUILTIN_DIR, file))
    end
  end
end
