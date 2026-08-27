$stdout.sync = true
$stderr.sync = true
$LOAD_PATH.unshift(File.expand_path("../../lib", __dir__))

require "static_embeddings"

module StaticEmbeddingsSample
  module_function

  NATIVE_GREP = "StaticEmbeddings|static_embeddings|se_|model_embed|embed_one|embed_batch|batch_execute|topk_execute|warmup_execute|wordpiece|normalize|vocab|hash|tokenize|l2|cosine|unblock_cancel"

  def root
    @root ||= File.expand_path("../..", __dir__)
  end

  def monotonic
    Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  def page_size
    require "etc"
    Etc.respond_to?(:sysconf) ? Etc.sysconf(Etc::SC_PAGESIZE) : 4096
  rescue StandardError
    4096
  end

  def env_float(name, default)
    Float(ENV.fetch(name, default.to_s))
  end

  def env_int(name, default)
    Integer(ENV.fetch(name, default.to_s))
  end

  def gc_snapshot
    stat = GC.stat
    {
      total_allocated_objects: stat.fetch(:total_allocated_objects),
      minor_gc_count: stat.fetch(:minor_gc_count),
      major_gc_count: stat.fetch(:major_gc_count)
    }
  end

  def gc_delta(before, after)
    before.each_with_object({}) { |(key, value), out| out[key] = after.fetch(key) - value }
  end

  def candidate_model_paths
    paths = []
    paths << ARGV[0] if ARGV[0] && !ARGV[0].empty?
    paths << ENV["MODEL"] if ENV["MODEL"] && !ENV["MODEL"].empty?
    paths << File.expand_path("~/.cache/static_embeddings/models/potion-retrieval-32m.semb")
    paths << File.join(root, "tmp/test-tiny.semb")
    paths << File.join(root, "lib/models/demo.semb")
    paths
  end

  def load_model
    path = candidate_model_paths.find { |candidate| File.file?(candidate) }
    unless path
      abort "model not found\nrun: bundle exec rake demo_model\nor pass a model path: bundle exec ruby samples/<sample>.rb /path/to/model.semb\nor set MODEL=/path/to/model.semb"
    end

    StaticEmbeddings.load(path, verify: ENV.fetch("VERIFY", "0") == "1")
  end

  def text_bytes!(value, label)
    raise "#{label}: expected String, got #{value.class}" unless value.is_a?(String)
    raise "#{label}: empty text" if value.empty?
    value.freeze
  end

  def embedding_format
    StaticEmbeddings.normalize_format(ENV.fetch("FORMAT", "f32"))
  end

  def bytes_per_component(format = embedding_format)
    case StaticEmbeddings.normalize_format(format)
    when :f32 then 4
    when :f16 then 2
    else raise "unsupported embedding format #{format.inspect}"
    end
  end

  def format_label(format = embedding_format)
    StaticEmbeddings.normalize_format(format).to_s
  end

  def embedding_blob!(blob, model, rows, label, format: embedding_format)
    raise "#{label}: expected binary String, got #{blob.class}" unless blob.is_a?(String)
    raise "#{label}: expected BINARY encoding" unless blob.encoding == Encoding::BINARY

    expected = rows * model.dim * bytes_per_component(format)
    raise "#{label}: expected #{expected} bytes for format=#{format_label(format)}, got #{blob.bytesize}" unless blob.bytesize == expected
  end

  def result_dir
    configured = ENV["RESULT_DIR"]
    return File.expand_path(configured, root) if configured && !configured.empty?

    File.expand_path("../results", __dir__)
  end

  def capture_paths(sample_name)
    sample_file = "/tmp/#{sample_name}.sample"
    txt_file = File.join(result_dir, "#{sample_name}.txt")
    [sample_file, txt_file, File.dirname(txt_file)]
  end

  def print_header(sample_name:, model:, call:, duration:, sleep_before_hot_loop:, preheat_iterations:, native_grep:, extra: {})
    extra = extra.dup
    expected_hot_symbols = Array(extra.delete(:expected_hot_symbols))
    sample_seconds = (sleep_before_hot_loop + duration + 15).ceil
    sample_file, txt_file, txt_dir = capture_paths(sample_name)

    puts "pid=#{Process.pid}"
    puts "ruby=#{RUBY_DESCRIPTION}"
    puts "platform=#{RUBY_PLATFORM}"
    puts "mode=#{sample_name}"
    puts "model_path=#{model.path}"
    puts "model_id=#{model.model_id || "unknown"}"
    puts "dim=#{model.dim}"
    puts "vocab=#{model.vocab_size}"
    puts "max_tokens=#{model.max_tokens}"
    puts "mapped_bytes=#{model.mapped_bytes}"
    puts "call=#{call}"
    extra.each { |key, value| puts "#{key}=#{value}" }
    puts "duration=#{duration}"
    puts "preheat_iterations=#{preheat_iterations}"
    puts "sample_seconds=#{sample_seconds}"
    puts "sample_file=#{sample_file}"
    puts "txt_file=#{txt_file}"
    puts
    puts "Copy this one-line capture command:"
    puts %(mkdir -p "#{txt_dir}"; OUT="#{txt_file}"; SAMPLE="#{sample_file}"; { sample #{Process.pid} #{sample_seconds} -f "$SAMPLE"; echo; echo "===== focused native symbols ====="; filtercalltree "$SAMPLE" | grep -E "#{native_grep}" | head -320; echo; echo "===== filtercalltree head -320 ====="; filtercalltree "$SAMPLE" | head -320; } 2>&1 | tee "$OUT")
    puts
    puts "Expected hot native symbols:"
    expected_hot_symbols.each { |line| puts "  #{line}" }
    puts "sleep=#{sleep_before_hot_loop} seconds before hot loop"
    puts
    sample_seconds
  end

  def preheat(iterations)
    iterations.times { yield }
  end

  def run_loop(duration)
    count = 0
    started = monotonic
    deadline = started + duration
    last = nil

    while monotonic < deadline
      last = yield
      count += 1
    end

    elapsed = monotonic - started
    [count, elapsed, last]
  end

  def gc_disabled?
    ENV.fetch("GC_DISABLE", "0") == "1"
  end

  def run_measured(duration)
    GC.start
    disabled = gc_disabled?
    GC.disable if disabled
    before_gc = gc_snapshot
    count, elapsed, last = run_loop(duration) { yield }
    after_gc = gc_snapshot
    [count, elapsed, last, gc_delta(before_gc, after_gc).merge(gc_disabled: disabled)]
  ensure
    GC.enable if disabled
  end

  def ascii_rag_text(target_bytes)
    seed = "postgres pipeline mode in ruby static embeddings local rag search vector database latency throughput "
    repeat_to_bytes(seed, target_bytes)
  end

  def hash_text(count)
    (0...count).map { |i| "%064x" % ((i + 1) * 0x9e3779b97f4a7c15) }.join(" ")
  end

  def base64_text(count)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    (0...count).map do |i|
      (0...88).map { |j| alphabet.getbyte((i * 17 + j * 31) & 63).chr }.join
    end.join(" ")
  end

  def identifier_text(count)
    (0...count).map { |i| "StaticEmbeddingsTokenizerHotPath#{i.to_s(36)}LongIdentifierWithoutNaturalBreaks" }.join(" ")
  end

  def long_ascii_word(chars)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    (0...chars).map { |i| alphabet.getbyte((i * 13 + 7) % alphabet.bytesize).chr }.join
  end

  def unicode_corner_text(target_bytes)
    seed = "Русский текст\tс табами\nпереносами café café 中文測試 emoji 🧬🚀 zero\u200bwidth nbsp\u00a0word punctuation!!! "
    repeat_to_bytes(seed, target_bytes)
  end

  def corpus(mode, count, bytes_per_text)
    Array.new(count) do |i|
      case mode
      when "ascii"
        ascii_rag_text(bytes_per_text).sub("latency", "latency #{i}")
      when "hashes"
        hash_text([1, bytes_per_text / 66].max)
      when "base64"
        base64_text([1, bytes_per_text / 90].max)
      when "identifiers"
        identifier_text([1, bytes_per_text / 72].max)
      when "long_words"
        Array.new([1, bytes_per_text / 104].max) { |j| long_ascii_word(100 - ((i + j) % 7)) }.join(" ")
      when "unicode"
        unicode_corner_text(bytes_per_text).sub("emoji", "emoji #{i}")
      else
        raise "unknown MODE=#{mode.inspect}; use ascii, hashes, base64, identifiers, long_words, unicode"
      end.freeze
    end.freeze
  end

  def random_id_sets(vocab_size, tokens, sets, seed)
    rng = Random.new(seed)
    Array.new(sets) { Array.new(tokens) { rng.rand(vocab_size) }.freeze }.freeze
  end

  def repeat_to_bytes(seed, target_bytes)
    out = +""
    out.force_encoding(Encoding::UTF_8)
    out << seed while out.bytesize < target_bytes
    return out if out.bytesize == target_bytes

    trimmed = +""
    trimmed.force_encoding(Encoding::UTF_8)
    out.each_char do |char|
      break if trimmed.bytesize + char.bytesize > target_bytes

      trimmed << char
    end
    trimmed
  end
end
