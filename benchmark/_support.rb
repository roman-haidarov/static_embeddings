# frozen_string_literal: true

require "rbconfig"

ROOT = File.expand_path("..", __dir__)
$LOAD_PATH.unshift(File.join(ROOT, "lib")) unless $LOAD_PATH.include?(File.join(ROOT, "lib"))
$stdout.sync = true
$stderr.sync = true

begin
  require "static_embeddings"
rescue LoadError
  warn "Native extension is not built. Run: bundle exec rake compile"
  raise
end

module SEBench
  module_function

  DEFAULT_REPEATS = 7
  DEFAULT_MIN_SECONDS = 0.25
  DEFAULT_WARMUP_SECONDS = 0.05
  MAX_ITERATIONS = 100_000_000
  CANDIDATE_WORDS = %w[
    hello world ruby gem static embedding vector search text token
    the a of and for local fast model cafe naive
    postgres pipeline database query batch matrix cosine kernel window prefix runtime memory latency
  ].freeze
  CANDIDATE_SUFFIXES = (0..15).map(&:to_s).freeze

  def env_integer(name, default, min: 1)
    value = Integer(ENV.fetch(name, default.to_s))
    raise ArgumentError, "#{name} must be >= #{min}, got #{value}" if value < min

    value
  end

  def env_float(name, default, min: 0.0)
    value = Float(ENV.fetch(name, default.to_s))
    raise ArgumentError, "#{name} must be >= #{min}, got #{value}" if value < min

    value
  end

  def repeats = env_integer("BENCH_REPEATS", DEFAULT_REPEATS)
  def min_seconds = env_float("BENCH_MIN_SECONDS", DEFAULT_MIN_SECONDS)
  def warmup_seconds = env_float("BENCH_WARMUP_SECONDS", DEFAULT_WARMUP_SECONDS)
  def adaptive? = ENV.fetch("BENCH_ADAPTIVE", "1") != "0"
  def kv? = ENV["BENCH_FORMAT"] == "kv"
  def monotonic = Process.clock_gettime(Process::CLOCK_MONOTONIC)

  def realtime
    started = monotonic
    yield
    monotonic - started
  end

  def median(values)
    sorted = values.sort
    return 0.0 if sorted.empty?

    mid = sorted.length / 2
    sorted.length.odd? ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0
  end

  def model
    @model ||= begin
      path = ENV["BENCH_MODEL"]
      m =
        if path
          StaticEmbeddings.load(path)
        elsif StaticEmbeddings.builtin_available?
          StaticEmbeddings.load_builtin
        else
          abort "No model. Set BENCH_MODEL=/path/to/model.semb or run: bundle exec rake demo_model"
        end
      m.warmup!
      m
    end
  end

  def candidate_pool
    @candidate_pool ||= begin
      suffixed = CANDIDATE_WORDS.flat_map do |word|
        CANDIDATE_SUFFIXES.map { |suffix| "#{word}#{suffix}" }
      end
      (CANDIDATE_WORDS + suffixed).uniq.freeze
    end
  end

  def nonzero_vector?(blob)
    blob.unpack("e*").any? { |value| value != 0.0 }
  end

  def in_vocabulary?(word)
    stats = model.embed_with_stats(word)
    stats[:token_count].positive? && stats[:unk_count].zero? && nonzero_vector?(stats[:vector])
  end

  def words
    @words ||= begin
      pool = candidate_pool.select { |word| in_vocabulary?(word) }
      if pool.empty?
        abort "No benchmark candidate word is in this model's vocabulary with a non-zero vector. " \
              "Set BENCH_MODEL to a larger model or extend SEBench::CANDIDATE_WORDS."
      end
      pool.freeze
    end
  end

  def text(word_count, rng)
    Array.new(word_count) { words[rng.rand(words.length)] }.join(" ")
  end

  def corpus(count, word_count, seed: 20_260_829)
    rng = Random.new(seed)
    Array.new(count) { text(word_count, rng) }.each(&:valid_encoding?).freeze
  end

  def calibrate(initial, &block)
    iterations = initial
    return iterations unless adaptive?
    return iterations if min_seconds <= 0.0

    loop do
      GC.start
      elapsed = realtime { block.call(iterations) }
      break iterations if elapsed >= min_seconds || iterations >= MAX_ITERATIONS

      scale = elapsed <= 0.0 ? 10 : (min_seconds / elapsed * 1.5).ceil
      iterations = [[iterations * scale, iterations * 2].max, MAX_ITERATIONS].min
    end
  end

  def measure(initial_iterations: 1, operations_per_iteration: 1, &block)
    iterations = calibrate(initial_iterations, &block)

    deadline = monotonic + warmup_seconds
    block.call(iterations) while monotonic < deadline

    samples = Array.new(repeats) do
      GC.start
      before = GC.stat(:total_allocated_objects)
      seconds = realtime { block.call(iterations) }
      [seconds, GC.stat(:total_allocated_objects) - before]
    end

    times = samples.map(&:first)
    best = times.min
    worst = times.max
    operations = iterations * operations_per_iteration

    {
      iterations: iterations,
      repeats: repeats,
      operations: operations,
      median_sec: median(times),
      best_sec: best,
      worst_sec: worst,
      spread_pct: best.positive? ? ((worst - best) / best * 100.0) : 0.0,
      ops_per_sec: operations / median(times),
      ns_per_op: median(times) / operations * 1e9,
      allocations_per_op: median(samples.map(&:last)) / operations.to_f
    }
  end

  def header(title)
    m = model
    puts
    puts "== #{title} =="
    puts "ruby=#{RUBY_VERSION} platform=#{RUBY_PLATFORM} backend=#{StaticEmbeddings.simd_backend}"
    puts "model=#{m.model_id} dim=#{m.dim} vocab=#{m.vocab_size} max_tokens=#{m.max_tokens}"
    puts "repeats=#{repeats} min_seconds=#{min_seconds} adaptive=#{adaptive? ? "yes" : "no"}"
    puts "corpus_vocabulary=#{words.length} distinct in-vocabulary words " \
         "(small pools keep matrix rows cached and overstate throughput)"
    puts "input_coderange=cached unless a row says otherwise"
    puts "spread_pct is best-to-worst across repeats; treat smaller differences as noise."
    puts
    return if kv?

    printf("%-34s %12s %12s %10s %10s\n", "case", "ns/op", "ops/sec", "spread%", "alloc/op")
  end

  def report(name, stats, **fields)
    if kv?
      payload = { benchmark: name, ruby: RUBY_VERSION, platform: RUBY_PLATFORM,
                  backend: StaticEmbeddings.simd_backend, dim: model.dim }
                .merge(fields).merge(stats)
      puts payload.map { |k, v| "#{k}=#{v.is_a?(Float) ? format("%.6g", v) : v}" }.join(" ")
    else
      label = fields.empty? ? name : "#{name} #{fields.map { |k, v| "#{k}=#{v}" }.join(" ")}"
      printf("%-34s %12.1f %12.0f %9.1f%% %10.2f\n", label[0, 34], stats[:ns_per_op],
             stats[:ops_per_sec], stats[:spread_pct], stats[:allocations_per_op])
    end
  end
end
