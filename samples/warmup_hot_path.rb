require_relative "support/hot_path"

model = StaticEmbeddingsSample.load_model
sample_name = "static_embeddings_warmup_hot_path"
duration = StaticEmbeddingsSample.env_float("DURATION", 25.0)
sleep_before_hot_loop = StaticEmbeddingsSample.env_float("SLEEP_BEFORE_HOT_LOOP", 7.0)
preheat_iterations = StaticEmbeddingsSample.env_int("PREHEAT_ITERATIONS", 1)
native_grep = "StaticEmbeddings|static_embeddings|model_warmup|warmup_execute|se_model_warmup"

StaticEmbeddingsSample.preheat(preheat_iterations) { model.warmup! }

StaticEmbeddingsSample.print_header(
  sample_name: sample_name,
  model: model,
  call: "model.warmup!",
  duration: duration,
  sleep_before_hot_loop: sleep_before_hot_loop,
  preheat_iterations: preheat_iterations,
  native_grep: native_grep,
  extra: {
    expected_hot_symbols: [
      "model_warmup / warmup_execute / se_model_warmup",
      "this sample isolates WARM mmap re-touch cost and GVL release",
      "it does NOT measure cold page faults; cold_start.rb does"
    ]
  }
)

sleep sleep_before_hot_loop
count, elapsed, last, gc_delta = StaticEmbeddingsSample.run_measured(duration) { model.warmup! }
raise "warmup did not return self" unless last.equal?(model)

puts "count=#{count}"
puts "elapsed=#{format('%.6f', elapsed)}"
puts "ops_per_sec=#{format('%.6f', count / elapsed)}"
puts "sec_per_op=#{format('%.6f', elapsed / [count, 1].max)}"
puts "mapped_bytes=#{model.mapped_bytes}"
page_size = StaticEmbeddingsSample.page_size
pages_per_warmup = (model.mapped_bytes.to_f / page_size).ceil
puts "page_size=#{page_size}"
puts "pages_per_warmup=#{pages_per_warmup}"
puts "pages_touched_per_sec=#{format('%.0f', (count * pages_per_warmup) / elapsed)}"
puts "bytes_actually_read_per_sec=#{format('%.0f', (count * pages_per_warmup) / elapsed)}"
puts "mapped_range_gb_per_sec=#{format('%.3f', (count * model.mapped_bytes) / elapsed / 1_000_000_000.0)}"
puts "warmup_metric_note=warmup reads ONE byte per page. mapped_range_gb_per_sec is address-range coverage in GB/s, NOT memory bandwidth; bytes_actually_read_per_sec is the real figure."
puts "warmup_scope_note=this loop re-touches already resident pages. For the number that matters at boot see samples/cold_start.rb."
puts "gc_delta=#{gc_delta}"
