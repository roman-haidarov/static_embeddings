require_relative "support/hot_path"
require "timeout"

model = StaticEmbeddingsSample.load_model
duration = StaticEmbeddingsSample.env_float("DURATION", 25.0)
sleep_before_hot_loop = StaticEmbeddingsSample.env_float("SLEEP_BEFORE_HOT_LOOP", 7.0)
preheat_iterations = StaticEmbeddingsSample.env_int("PREHEAT_ITERATIONS", 1)
full_scan = ENV.fetch("FULL_SCAN", "0") == "1"
sample_name = full_scan ? "static_embeddings_cancellation_full_scan" : "static_embeddings_cancellation_gvl_fairness"
timeout_default = full_scan ? 0.005 : 0.05
timeout_seconds = StaticEmbeddingsSample.env_float("TIMEOUT", timeout_default)
input_bytes = StaticEmbeddingsSample.env_int("INPUT_BYTES", 3_000_000)
texts_count = StaticEmbeddingsSample.env_int("TEXTS", 5_000)
text = StaticEmbeddingsSample.text_bytes!(StaticEmbeddingsSample.ascii_rag_text(input_bytes), "input")
texts = StaticEmbeddingsSample.corpus("ascii", texts_count, 192)
native_grep = "StaticEmbeddings|static_embeddings|model_embed|model_embed_batch|embed_one|batch_execute|unblock_cancel|cleanup_batch_job|se_embed_one|se_tokenize|append_wordpiece"

StaticEmbeddingsSample.preheat(preheat_iterations) do
  begin
    Timeout.timeout(timeout_seconds) { model.embed(text, max_tokens: (full_scan ? false : nil)) }
  rescue Timeout::Error
  end
  begin
    Timeout.timeout(timeout_seconds) { model.embed_batch(texts, max_tokens: (full_scan ? false : nil)) }
  rescue Timeout::Error
  end
end

StaticEmbeddingsSample.print_header(
  sample_name: sample_name,
  model: model,
  call: "Timeout.timeout around model.embed(text) and model.embed_batch(texts)",
  duration: duration,
  sleep_before_hot_loop: sleep_before_hot_loop,
  preheat_iterations: preheat_iterations,
  native_grep: native_grep,
  extra: {
    timeout_seconds: timeout_seconds,
    full_scan: full_scan,
    input_bytes: text.bytesize,
    batch_texts: texts.length,
    expected_hot_symbols: full_scan ? [
      "FULL_SCAN disables truncation so work outlives the deadline",
      "single_timeouts and batch_timeouts must both be > 0 or this run proves nothing",
      "unblock_cancel may appear in the profiler, but timeouts/RSS are the proof"
    ] : [
      "this mode measures GVL fairness, NOT cancellation",
      "thread_ticks close to duration/0.001 means other Ruby threads kept running",
      "timeouts are expected to stay at 0 here; run FULL_SCAN=1 to exercise unblock_cancel"
    ]
  }
)

mapped_kb = model.mapped_bytes / 1024
rss_before = `ps -o rss= -p #{Process.pid}`.to_i
single_timeouts = 0
batch_timeouts = 0
thread_ticks = 0
stop = false
ticker = Thread.new do
  until stop
    thread_ticks += 1
    sleep 0.001
  end
end

sleep sleep_before_hot_loop
GC.start
started = StaticEmbeddingsSample.monotonic
deadline = started + duration
iterations = 0
while StaticEmbeddingsSample.monotonic < deadline
  begin
    Timeout.timeout(timeout_seconds) { model.embed(text, max_tokens: (full_scan ? false : nil)) }
  rescue Timeout::Error
    single_timeouts += 1
  end

  begin
    Timeout.timeout(timeout_seconds) { model.embed_batch(texts, max_tokens: (full_scan ? false : nil)) }
  rescue Timeout::Error
    batch_timeouts += 1
  end
  iterations += 1
end
elapsed = StaticEmbeddingsSample.monotonic - started
stop = true
ticker.join
GC.start
rss_after = `ps -o rss= -p #{Process.pid}`.to_i

puts "full_scan=#{full_scan}"
puts "iterations=#{iterations}"
puts "single_timeouts=#{single_timeouts}"
puts "batch_timeouts=#{batch_timeouts}"
puts "thread_ticks=#{thread_ticks}"
puts "elapsed=#{format('%.6f', elapsed)}"
puts "rss_before_kb=#{rss_before}"
puts "rss_after_kb=#{rss_after}"
puts "rss_delta_kb=#{rss_after - rss_before}"
puts "mapped_kb=#{mapped_kb}"
puts "rss_delta_excluding_mapped_kb=#{[(rss_after - rss_before) - mapped_kb, 0].max}"
