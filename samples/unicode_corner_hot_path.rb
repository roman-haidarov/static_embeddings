require_relative "support/hot_path"

model = StaticEmbeddingsSample.load_model
format = StaticEmbeddingsSample.embedding_format
sample_name = "static_embeddings_unicode_corner_#{StaticEmbeddingsSample.format_label(format)}_hot_path"
duration = StaticEmbeddingsSample.env_float("DURATION", 25.0)
sleep_before_hot_loop = StaticEmbeddingsSample.env_float("SLEEP_BEFORE_HOT_LOOP", 7.0)
preheat_iterations = StaticEmbeddingsSample.env_int("PREHEAT_ITERATIONS", 3)
input_bytes = StaticEmbeddingsSample.env_int("INPUT_BYTES", 16_384)
text = StaticEmbeddingsSample.text_bytes!(StaticEmbeddingsSample.unicode_corner_text(input_bytes), "input")
native_grep = "StaticEmbeddings|static_embeddings|embed_with_stats|se_embed_one|se_tokenize|normalize|se_map_lookup|se_range_contains|se_is_cjk|append_wordpiece|wordpiece|se_vocab_lookup|se_l2_normalize"

StaticEmbeddingsSample.preheat(preheat_iterations) do
  stats = model.embed_with_stats(text, format: format)
  StaticEmbeddingsSample.embedding_blob!(stats.fetch(:vector), model, 1, "preheat", format: format)
end

StaticEmbeddingsSample.print_header(
  sample_name: sample_name,
  model: model,
  call: "model.embed_with_stats(text, format: :#{StaticEmbeddingsSample.format_label(format)})",
  duration: duration,
  sleep_before_hot_loop: sleep_before_hot_loop,
  preheat_iterations: preheat_iterations,
  native_grep: native_grep,
  extra: {
    input_bytes: text.bytesize,
    format: StaticEmbeddingsSample.format_label(format),
    bytes_per_component: StaticEmbeddingsSample.bytes_per_component(format),
    expected_hot_symbols: [
      "model_embed_with_stats / se_embed_one",
      "se_tokenize / normalize / se_map_lookup / se_range_contains",
      "se_is_cjk / append_wordpiece / se_vocab_lookup",
      "se_l2_normalize after pooling"
    ]
  }
)

sleep sleep_before_hot_loop
count, elapsed, last, gc_delta = StaticEmbeddingsSample.run_measured(duration) { model.embed_with_stats(text, format: format) }
StaticEmbeddingsSample.embedding_blob!(last.fetch(:vector), model, 1, "last_hot_loop", format: format) if last
ratio = last.fetch(:unk_count).to_f / [last.fetch(:token_count), 1].max

puts "count=#{count}"
puts "elapsed=#{format('%.6f', elapsed)}"
puts "ops_per_sec=#{format('%.6f', count / elapsed)}"
puts "sec_per_op=#{format('%.6f', elapsed / [count, 1].max)}"
puts "logical_input_mb_per_sec=#{format('%.3f', (count * text.bytesize) / elapsed / 1_000_000.0)}"
puts "truncation_active=true"
puts "token_count=#{last.fetch(:token_count)}"
puts "unk_count=#{last.fetch(:unk_count)}"
puts "unk_ratio=#{format('%.6f', ratio)}"
puts "truncated=#{last.fetch(:truncated)}"
puts "gc_delta=#{gc_delta}"
