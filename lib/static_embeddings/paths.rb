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
      File.join(cache_dir(env), "models", "#{model_id}.semb")
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
