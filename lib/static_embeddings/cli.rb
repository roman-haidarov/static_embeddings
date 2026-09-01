require "json"
require "optparse"
require "fileutils"

module StaticEmbeddings
  class CLI
    COMMANDS = {
      "convert" => :convert,
      "verify" => :verify,
      "inspect" => :inspect_model,
      "tokenize" => :tokenize,
      "embed" => :embed,
      "cache-path" => :cache_path,
      "help" => :usage,
      "-h" => :usage,
      "--help" => :usage,
      nil => :usage
    }.freeze

    HELP = <<~TEXT
      static_embeddings <command> [options]

        convert SOURCE_DIR    Convert a HuggingFace/Model2Vec directory to .semb
          --out PATH          Output file (default: <cache>/models/<id>.semb)
          --id ID             Model id recorded in provenance
          --max-tokens N      Truncation limit baked into the file (default 512)

        verify PATH           Recompute the SHA-256 embedded in the header
        inspect PATH          Print header fields and provenance
        tokenize PATH TEXT    Print token ids
        embed PATH TEXT       Print the vector, token count and UNK ratio
        cache-path            Print the model cache directory
    TEXT

    def self.run(argv)
      new.run(argv)
    end

    def run(argv)
      method = COMMANDS[argv.shift]
      return unknown unless method

      send(method, argv)
    rescue StaticEmbeddings::Error => e
      warn "#{e.class.name.split('::').last}: #{e.message}"
      1
    end

    private

    def usage(*)
      puts HELP
      0
    end

    def unknown
      warn "unknown command"
      usage
      1
    end

    def cache_path(*)
      puts StaticEmbeddings.cache_dir
      0
    end

    def convert(argv)
      require "static_embeddings/converter"
      options = parse_convert_options(argv)
      source = required_arg(argv, "usage: static_embeddings convert SOURCE_DIR [--out PATH]")
      model_id = options[:id] || File.basename(File.expand_path(source))
      out = options[:out] || StaticEmbeddings.model_path(model_id)
      FileUtils.mkdir_p(File.dirname(out))

      report = StaticEmbeddings.convert(source, output_path: out, model_id: model_id,
                                                max_tokens: options[:max_tokens])
      puts conversion_report(out, report, options[:max_tokens])
      0
    end

    def parse_convert_options(argv)
      options = { max_tokens: Converter::REFERENCE_MAX_TOKENS }
      OptionParser.new do |parser|
        parser.on("--out PATH") { |value| options[:out] = value }
        parser.on("--id ID") { |value| options[:id] = value }
        parser.on("--max-tokens N", Integer) { |value| options[:max_tokens] = value }
      end.parse!(argv)
      options
    end

    def conversion_report(path, report, max_tokens)
      [
        "wrote #{path}",
        "  vocab      #{report[:vocab_size]}",
        "  dim        #{report[:dim]}",
        "  bytes      #{report[:bytes]}",
        "  sha256     #{report[:sha256]}",
        "  max_tokens #{max_tokens}",
        "",
        "Record this conversion in docs/MODEL_AUDIT.md before trusting the vectors."
      ].join("\n")
    end

    def verify(argv)
      result = StaticEmbeddings.verify(required_arg(argv, "usage: static_embeddings verify PATH"))
      return puts("ok #{result[:expected]}") || 0 if result[:ok]

      warn "CHECKSUM MISMATCH"
      warn "  stored   #{result[:stored]}"
      warn "  computed #{result[:expected]}"
      1
    end

    def inspect_model(argv)
      model = StaticEmbeddings.load(required_arg(argv, "usage: static_embeddings inspect PATH"))
      puts JSON.pretty_generate(model_summary(model))
      0
    end

    def model_summary(model)
      {
        "path" => model.path,
        "dim" => model.dim,
        "vocab_size" => model.vocab_size,
        "max_tokens" => model.max_tokens,
        "normalized" => model.normalized?,
        "lowercase" => model.lowercase?,
        "unk_id" => model.unk_id,
        "mapped_bytes" => model.mapped_bytes,
        "provenance" => model.provenance
      }
    end

    def tokenize(argv)
      model, text = model_and_text(argv, "usage: static_embeddings tokenize PATH TEXT")
      ids = model.tokenize(text)
      puts JSON.generate("ids" => ids, "count" => ids.length, "unk" => ids.count(model.unk_id))
      0
    end

    def embed(argv)
      model, text = model_and_text(argv, "usage: static_embeddings embed PATH TEXT")
      stats = model.embed_with_stats(text)
      warn_high_unk(stats) if high_unk?(stats)
      puts JSON.generate(stats_payload(model, stats))
      0
    end

    def stats_payload(model, stats)
      {
        "token_count" => stats[:token_count],
        "unk_count" => stats[:unk_count],
        "truncated" => stats[:truncated],
        "vector" => StaticEmbeddings.unpack(stats[:vector], model.dim).first.map { |v| v.round(6) }
      }
    end

    def high_unk?(stats)
      stats[:token_count].positive? && stats[:unk_count].to_f / stats[:token_count] > 0.3
    end

    def warn_high_unk(stats)
      ratio = stats[:unk_count].to_f / stats[:token_count]
      warn "warning: #{(ratio * 100).round}% of tokens are [UNK] — wrong model for this language?"
    end

    def model_and_text(argv, usage)
      path = argv.shift
      text = argv.join(" ")
      abort usage if path.nil? || text.empty?

      [StaticEmbeddings.load(path), text]
    end

    def required_arg(argv, usage)
      argv.shift || abort(usage)
    end
  end
end
