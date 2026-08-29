require_relative "support/hot_path"

model = StaticEmbeddingsSample.load_model
format = StaticEmbeddingsSample.embedding_format
sample_name = "static_embeddings_embed_single_large_#{StaticEmbeddingsSample.format_label(format)}_hot_path"
duration = StaticEmbeddingsSample.env_float("DURATION", 25.0)
sleep_before_hot_loop = StaticEmbeddingsSample.env_float("SLEEP_BEFORE_HOT_LOOP", 7.0)
preheat_iterations = StaticEmbeddingsSample.env_int("PREHEAT_ITERATIONS", 2)
input_bytes = StaticEmbeddingsSample.env_int("INPUT_BYTES", 3_000_000)
text = StaticEmbeddingsSample.text_bytes!(StaticEmbeddingsSample.ascii_rag_text(input_bytes), "input")
processed_bytes = StaticEmbeddingsSample.processed_input_bytes(model, text, format: format)
native_grep = "StaticEmbeddings|static_embeddings|model_embed|embed_one|se_embed_one|se_tokenize|normalize|append_wordpiece|wordpiece|se_vocab_lookup|se_hash_bytes|se_l2_normalize|unblock_cancel"

StaticEmbeddingsSample.preheat(preheat_iterations) do
  blob = model.embed(text, format: format)
  StaticEmbeddingsSample.embedding_blob!(blob, model, 1, "preheat", format: format)
end

StaticEmbeddingsSample.print_header(
  sample_name: sample_name,
  model: model,
  call: "model.embed(text, format: :#{StaticEmbeddingsSample.format_label(format)})",
  duration: duration,
  sleep_before_hot_loop: sleep_before_hot_loop,
  preheat_iterations: preheat_iterations,
  native_grep: native_grep,
  extra: {
    input_bytes: text.bytesize,
    processed_bytes: processed_bytes || text.bytesize,
    format: StaticEmbeddingsSample.format_label(format),
    bytes_per_component: StaticEmbeddingsSample.bytes_per_component(format),
    expected_hot_symbols: [
      "model_embed / embed_one_value / embed_one_via_batch",
      "se_embed_one / se_tokenize / normalize / append_wordpiece",
      "se_vocab_lookup / se_hash_bytes / se_l2_normalize",
      "unblock_cancel when input is above the GVL release threshold"
    ]
  }
)

sleep sleep_before_hot_loop
count, elapsed, last, gc_delta = StaticEmbeddingsSample.run_measured(duration) { model.embed(text, format: format) }
StaticEmbeddingsSample.embedding_blob!(last, model, 1, "last_hot_loop", format: format) if last

puts "count=#{count}"
puts "elapsed=#{format('%.6f', elapsed)}"
puts "ops_per_sec=#{format('%.6f', count / elapsed)}"
puts "sec_per_op=#{format('%.6f', elapsed / [count, 1].max)}"
puts "truncation_active=true"
StaticEmbeddingsSample.print_input_throughput(
  name: "logical_input_mb_per_sec",
  bytes: text.bytesize,
  count: count,
  elapsed: elapsed,
  truncation_active: true,
  processed_bytes: processed_bytes
)
puts "output_bytes=#{last&.bytesize}"
puts "gc_delta=#{gc_delta}"
