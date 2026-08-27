require_relative "lib/static_embeddings/version"

Gem::Specification.new do |spec|
  spec.name    = "static_embeddings"
  spec.version = StaticEmbeddings::VERSION
  spec.authors = ["Roman Haydarov"]
  spec.email   = ["romnhajdarov@gmail.com"]

  spec.summary     = "Fast local text embeddings for Ruby — no ONNX, no Rust, no network"
  spec.description = "A small C-extension runtime for Model2Vec-style static embedding " \
                     "models. Models are converted offline into a flat mmap-able .semb " \
                     "file; at runtime the gem tokenizes (BERT WordPiece), looks up rows " \
                     "and mean-pools them. Releases the GVL on large native work, rejects internal " \
                     "thread fan-out, and links nothing but libc."
  spec.homepage = "https://github.com/roman-haidarov/static_embeddings"
  spec.license  = "MIT"
  spec.required_ruby_version = ">= 3.1.0"

  spec.metadata["homepage_uri"]    = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"]   = "#{spec.homepage}/blob/main/CHANGELOG.md"

  spec.files = Dir[
    "lib/**/*.rb",
    "lib/models/*.semb",
    "exe/*",
    "ext/**/*.{c,h,rb}",
    "docs/**/*.md",
    "tools/*.rb",
    "static_embeddings.gemspec",
    "README.md",
    "CHANGELOG.md",
    "LICENSE.txt"
  ].reject { |path| File.basename(path) == ".DS_Store" }

  spec.bindir        = "exe"
  spec.executables   = ["static_embeddings"]
  spec.require_paths = ["lib"]
  spec.extensions    = ["ext/static_embeddings/extconf.rb"]

  spec.add_development_dependency "bundler", "~> 2.0"
  spec.add_development_dependency "minitest", "~> 5.0"
  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rake-compiler", "~> 1.2"
end
