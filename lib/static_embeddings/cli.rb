require "fileutils"
require "json"
require "optparse"

module StaticEmbeddings
  module CLI
    COMMANDS = {
      "convert" => :convert_command,
      "verify" => :verify_command,
      "inspect" => :inspect_command,
      "tokenize" => :tokenize_command,
      "embed" => :embed_command,
      "cache-path" => :cache_path_command,
      "help" => :usage,
      "-h" => :usage,
      "--help" => :usage,
      nil => :usage
    }.freeze

    HELP = <<~TEXT
      static_embeddings <command> [options]

        convert SOURCE_DIR    Convert Model2Vec or Sentence Transformers StaticEmbedding to .semb
          --out PATH          Output file (default: <cache>/models/<id>.semb)
          --id ID             Model id recorded in provenance
          --max-tokens N      Model2Vec default: 512; Sentence Transformers default: unlimited
          --max-tokens unlimited
          --dimensions N      Keep the first N dims (Matryoshka prefix slice)
          --revision SHA      Source revision recorded in provenance
          --trained-mrl-dims  1024,512,256,... training dims recorded in provenance

        verify PATH           Recompute the SHA-256 embedded in the header
        inspect PATH          Print header fields and provenance
        tokenize PATH TEXT    Print token ids
        embed PATH TEXT       Print the vector, token count and UNK ratio
        cache-path            Print the model cache directory
    TEXT

    module_function

    def run(argv)
      command = COMMANDS[argv.shift]
      return unknown_command unless command

      public_send(command, argv)
    rescue StaticEmbeddings::Error, ArgumentError, OptionParser::ParseError => e
      warn "#{e.class.name.split('::').last}: #{e.message}"
      1
    end

    def usage(*)
      puts HELP
      0
    end

    def unknown_command
      warn "unknown command"
      usage
      1
    end

    def cache_path_command(*)
      puts StaticEmbeddings.cache_dir
      0
    end

    def convert_command(argv)
      options = convert_options(argv)
      source = required_arg(argv, "usage: static_embeddings convert SOURCE_DIR [--out PATH]")
      model_id = options[:id] || File.basename(File.expand_path(source))
      output = options[:out] || StaticEmbeddings.model_path(model_id)
      FileUtils.mkdir_p(File.dirname(output))

      report = StaticEmbeddings.convert(source, output_path: output, model_id: model_id, **conversion_options(options))
      puts conversion_report(output, report)
      0
    end

    def convert_options(argv)
      {}.tap do |options|
        OptionParser.new do |parser|
          parser.on("--out PATH") { |value| options[:out] = value }
          parser.on("--id ID") { |value| options[:id] = value }
          parser.on("--max-tokens N") { |value| options[:max_tokens] = parse_max_tokens(value) }
          parser.on("--dimensions N", Integer) { |value| options[:dimensions] = value }
          parser.on("--revision SHA") { |value| options[:source_revision] = value }
          parser.on("--trained-mrl-dims LIST") { |value| options[:trained_mrl_dims] = value }
        end.parse!(argv)
      end
    end

    def conversion_options(options)
      options.select { |key, _| %i[max_tokens dimensions source_revision trained_mrl_dims].include?(key) }
    end

    def parse_max_tokens(value)
      return :unlimited if value == "unlimited" || value == "0"

      integer = Integer(value)
      raise OptionParser::InvalidArgument, "--max-tokens must be positive or unlimited" unless integer.positive?
      integer
    end

    def conversion_report(path, report)
      max_tokens = report[:max_tokens].to_i.zero? ? "unlimited" : report[:max_tokens]
      dim = report[:native_dim] == report[:dim] ? report[:dim].to_s : "#{report[:dim]} (from native #{report[:native_dim]})"
      [
        "wrote #{path}",
        "  vocab      #{report[:vocab_size]}",
        "  dim        #{dim}",
        "  bytes      #{report[:bytes]}",
        "  sha256     #{report[:sha256]}",
        "  max_tokens #{max_tokens}",
        "",
        "Record this conversion in docs/MODEL_AUDIT.md before trusting the vectors."
      ].join("\n")
    end

    def verify_command(argv)
      result = StaticEmbeddings.verify(required_arg(argv, "usage: static_embeddings verify PATH"))
      if result[:ok]
        puts "ok #{result[:expected]}"
        0
      else
        warn "CHECKSUM MISMATCH"
        warn "  stored   #{result[:stored]}"
        warn "  computed #{result[:expected]}"
        1
      end
    end

    def inspect_command(argv)
      model = StaticEmbeddings.load(required_arg(argv, "usage: static_embeddings inspect PATH"))
      puts JSON.pretty_generate(model_summary(model))
      0
    ensure
      model&.close
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

    def tokenize_command(argv)
      with_model_and_text(argv, "usage: static_embeddings tokenize PATH TEXT") do |model, text|
        ids = model.tokenize(text)
        puts JSON.generate("ids" => ids, "count" => ids.length, "unk" => ids.count(model.unk_id))
      end
      0
    end

    def embed_command(argv)
      with_model_and_text(argv, "usage: static_embeddings embed PATH TEXT") do |model, text|
        stats = model.embed_with_stats(text)
        warn_high_unk(stats) if high_unk?(stats)
        puts JSON.generate(stats_payload(model, stats))
      end
      0
    end

    def stats_payload(model, stats)
      {
        "token_count" => stats[:token_count],
        "unk_count" => stats[:unk_count],
        "truncated" => stats[:truncated],
        "vector" => StaticEmbeddings.unpack(stats[:vector], model.dim).first.map { |value| value.round(6) }
      }
    end

    def high_unk?(stats)
      stats[:token_count].positive? && stats[:unk_count].to_f / stats[:token_count] > 0.3
    end

    def warn_high_unk(stats)
      ratio = stats[:unk_count].to_f / stats[:token_count]
      warn "warning: #{(ratio * 100).round}% of tokens are [UNK] — wrong model for this language?"
    end

    def with_model_and_text(argv, usage)
      path = argv.shift
      text = argv.join(" ")
      raise InvalidOptionError, usage if path.nil? || text.empty?

      model = StaticEmbeddings.load(path)
      yield model, text
    ensure
      model&.close
    end

    def required_arg(argv, usage)
      argv.shift || raise(InvalidOptionError, usage)
    end
  end
end
