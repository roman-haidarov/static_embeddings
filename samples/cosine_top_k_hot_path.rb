require_relative "support/hot_path"

model = StaticEmbeddingsSample.load_model
metric = ENV.fetch("METRIC", "dot")
raise "unknown METRIC=#{metric.inspect}; use dot or cosine" unless %w[dot cosine].include?(metric)
mode = ENV.fetch("MODE", "ascii")
format = StaticEmbeddingsSample.embedding_format
sample_name = "static_embeddings_#{metric}_top_k_#{mode}_#{StaticEmbeddingsSample.format_label(format)}_hot_path"
duration = StaticEmbeddingsSample.env_float("DURATION", 25.0)
sleep_before_hot_loop = StaticEmbeddingsSample.env_float("SLEEP_BEFORE_HOT_LOOP", 7.0)
preheat_iterations = StaticEmbeddingsSample.env_int("PREHEAT_ITERATIONS", 3)
rows = StaticEmbeddingsSample.env_int("ROWS", 50_000)
k = StaticEmbeddingsSample.env_int("K", 10)
bytes_per_text = StaticEmbeddingsSample.env_int("BYTES_PER_TEXT", 96)
texts = StaticEmbeddingsSample.corpus(mode, rows, bytes_per_text)
query = model.embed("postgres ruby static embeddings vector search hot path", format: format)
matrix = model.embed_batch(texts, format: format).freeze # frozen: required for lock-free concurrent scans
StaticEmbeddingsSample.embedding_blob!(query, model, 1, "query", format: format)
StaticEmbeddingsSample.embedding_blob!(matrix, model, rows, "matrix", format: format)
native_grep = "StaticEmbeddings|static_embeddings|top_k_impl|se_topk_execute|se_dot_product|se_dot_and_row_sq"

top_k = metric == "cosine" ? StaticEmbeddings.method(:cosine_top_k) : StaticEmbeddings.method(:dot_top_k)

StaticEmbeddingsSample.preheat(preheat_iterations) do
  result = top_k.call(query, matrix, k, dim: model.dim, format: format)
  raise "preheat: empty result" if result.empty?
end

StaticEmbeddingsSample.print_header(
  sample_name: sample_name,
  model: model,
  call: "StaticEmbeddings.#{metric}_top_k(query, matrix, #{k}, dim: #{model.dim}, format: :#{StaticEmbeddingsSample.format_label(format)})",
  duration: duration,
  sleep_before_hot_loop: sleep_before_hot_loop,
  preheat_iterations: preheat_iterations,
  native_grep: native_grep,
  extra: {
    rows: rows,
    k: k,
    metric: metric,
    mode: mode,
    format: StaticEmbeddingsSample.format_label(format),
    bytes_per_component: StaticEmbeddingsSample.bytes_per_component(format),
    matrix_bytes: matrix.bytesize,
    expected_hot_symbols: [
      "top_k_impl / se_topk_execute",
      "no tokenizer symbols should dominate this sample",
      "this isolates row scanning, dot products and top-k maintenance"
    ]
  }
)

sleep sleep_before_hot_loop
count, elapsed, last, gc_delta = StaticEmbeddingsSample.run_measured(duration) do
  top_k.call(query, matrix, k, dim: model.dim, format: format)
end
raise "last_hot_loop: empty result" if last.nil? || last.empty?

puts "count=#{count}"
puts "elapsed=#{format('%.6f', elapsed)}"
puts "ops_per_sec=#{format('%.6f', count / elapsed)}"
puts "sec_per_op=#{format('%.6f', elapsed / [count, 1].max)}"
puts "rows_scanned_per_sec=#{format('%.3f', (count * rows) / elapsed)}"
puts "gb_scanned_per_sec=#{format('%.3f', (count * matrix.bytesize) / elapsed / 1_000_000_000.0)}"
puts "first_result=#{last.first.inspect}"
puts "gc_delta=#{gc_delta}"
