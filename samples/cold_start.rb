require_relative "support/hot_path"

path = StaticEmbeddingsSample.candidate_model_paths.find { |candidate| File.file?(candidate) }
abort "model not found; pass a path or set MODEL=/path/to/model.semb" unless path

verify = ENV.fetch("VERIFY", "0") == "1"
text = ENV.fetch("TEXT", "postgres pipeline mode in ruby static embeddings")

def ms(seconds)
  format("%.3f", seconds * 1000.0)
end

def rss_kb
  `ps -o rss= -p #{Process.pid}`.to_i
end

rss_start = rss_kb
StaticEmbeddingsSample.alloc_stats_reset

t0 = StaticEmbeddingsSample.monotonic
model = StaticEmbeddings.load(path, verify: verify)
t_load = StaticEmbeddingsSample.monotonic - t0
rss_after_load = rss_kb
load_alloc_stats = StaticEmbeddingsSample.alloc_stats_snapshot

t0 = StaticEmbeddingsSample.monotonic
first_blob = model.embed(text)
t_first_embed = StaticEmbeddingsSample.monotonic - t0
StaticEmbeddingsSample.embedding_blob!(first_blob, model, 1, "first_embed")
rss_after_first = rss_kb

t0 = StaticEmbeddingsSample.monotonic
model.warmup!
t_warmup = StaticEmbeddingsSample.monotonic - t0
rss_after_warmup = rss_kb

t0 = StaticEmbeddingsSample.monotonic
model.embed(text)
t_warm_embed = StaticEmbeddingsSample.monotonic - t0

t0 = StaticEmbeddingsSample.monotonic
model.warmup!
t_second_warmup = StaticEmbeddingsSample.monotonic - t0

puts "pid=#{Process.pid}"
puts "ruby=#{RUBY_DESCRIPTION}"
puts "platform=#{RUBY_PLATFORM}"
puts "mode=static_embeddings_cold_start"
puts "model_path=#{model.path}"
puts "model_id=#{model.model_id || "unknown"}"
puts "dim=#{model.dim}"
puts "vocab=#{model.vocab_size}"
puts "mapped_bytes=#{model.mapped_bytes}"
puts "verify=#{verify}"
puts "page_size=#{StaticEmbeddingsSample.page_size}"
puts
puts "load_ms=#{ms(t_load)}"
puts "first_embed_ms=#{ms(t_first_embed)}"
puts "warmup_ms=#{ms(t_warmup)}"
puts "warm_embed_ms=#{ms(t_warm_embed)}"
puts "second_warmup_ms=#{ms(t_second_warmup)}"
puts "time_to_first_query_ms=#{ms(t_load + t_first_embed)}"
puts "time_to_ready_ms=#{ms(t_load + t_warmup)}"
puts
puts "rss_start_kb=#{rss_start}"
puts "rss_after_load_kb=#{rss_after_load}"
puts "rss_after_first_embed_kb=#{rss_after_first}"
puts "rss_after_warmup_kb=#{rss_after_warmup}"
puts "rss_growth_kb=#{rss_after_warmup - rss_start}"
puts
residency_ratio = t_second_warmup.zero? ? 0.0 : t_warmup / t_second_warmup
bytes_per_sec = t_warmup.zero? ? 0.0 : model.mapped_bytes / t_warmup

puts "warmup_residency_ratio=#{format('%.2f', residency_ratio)}"
puts "warmup_mapped_bytes_per_sec=#{format('%.0f', bytes_per_sec)}"
puts "warmup_mapped_gb_per_sec=#{format('%.2f', bytes_per_sec / 1e9)}"
StaticEmbeddingsSample.print_alloc_stats_table(load_alloc_stats, prefix: "c_alloc.load")
StaticEmbeddingsSample.print_alloc_stats(prefix: "c_alloc.total")
if StaticEmbeddingsSample.alloc_stats_enabled?
  puts "c_alloc_note=model bytes are mmapped on POSIX, so the model_file category " \
       "stays at zero there by design; on Windows the loader reads the file into the heap"
end
puts
puts "page_cache_state=unknown"
puts "page_cache_note=a true cold-start number requires dropping the OS page cache first " \
     "(macOS: sudo purge, Linux: echo 3 > /proc/sys/vm/drop_caches) and running this sample once"
