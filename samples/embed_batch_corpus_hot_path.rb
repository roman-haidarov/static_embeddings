require_relative "support/hot_path"

model = StaticEmbeddingsSample.load_model
mode = ENV.fetch("MODE", "ascii")
format = StaticEmbeddingsSample.embedding_format
sample_name = "static_embeddings_embed_batch_#{mode}_#{StaticEmbeddingsSample.format_label(format)}_hot_path"
duration = StaticEmbeddingsSample.env_float("DURATION", 25.0)
sleep_before_hot_loop = StaticEmbeddingsSample.env_float("SLEEP_BEFORE_HOT_LOOP", 7.0)
preheat_iterations = StaticEmbeddingsSample.env_int("PREHEAT_ITERATIONS", 2)
count_texts = StaticEmbeddingsSample.env_int("TEXTS", 5_000)
bytes_per_text = StaticEmbeddingsSample.env_int("BYTES_PER_TEXT", 192)
texts = StaticEmbeddingsSample.corpus(mode, count_texts, bytes_per_text)
total_input_bytes = texts.sum(&:bytesize)
native_grep = "StaticEmbeddings|static_embeddings|model_embed_batch|embed_batch|batch_execute|se_embed_one|se_tokenize|normalize|append_wordpiece|wordpiece|se_vocab_lookup|se_hash_bytes|se_l2_normalize|unblock_cancel"

StaticEmbeddingsSample.preheat(preheat_iterations) do
  blob = model.embed_batch(texts, format: format)
  StaticEmbeddingsSample.embedding_blob!(blob, model, texts.length, "preheat", format: format)
end

StaticEmbeddingsSample.print_header(
  sample_name: sample_name,
  model: model,
  call: "model.embed_batch(texts, format: :#{StaticEmbeddingsSample.format_label(format)})",
  duration: duration,
  sleep_before_hot_loop: sleep_before_hot_loop,
  preheat_iterations: preheat_iterations,
  native_grep: native_grep,
  extra: {
    mode: mode,
    format: StaticEmbeddingsSample.format_label(format),
    bytes_per_component: StaticEmbeddingsSample.bytes_per_component(format),
    texts: texts.length,
    total_input_bytes: total_input_bytes,
    bytes_per_text: bytes_per_text,
    expected_hot_symbols: [
      "model_embed_batch / embed_batch_internal / batch_execute",
      "se_embed_one / se_tokenize / normalize / append_wordpiece",
      "se_vocab_lookup / se_hash_bytes / se_l2_normalize",
      "unblock_cancel for Timeout, Thread#kill and Ctrl-C"
    ]
  }
)

sleep sleep_before_hot_loop
batch_count, elapsed, last, gc_delta = StaticEmbeddingsSample.run_measured(duration) { model.embed_batch(texts, format: format) }
StaticEmbeddingsSample.embedding_blob!(last, model, texts.length, "last_hot_loop", format: format) if last
embedded_texts = batch_count * texts.length
embedded_bytes = batch_count * total_input_bytes

puts "count=#{batch_count}"
puts "elapsed=#{format('%.6f', elapsed)}"
puts "batches_per_sec=#{format('%.6f', batch_count / elapsed)}"
puts "texts_per_sec=#{format('%.3f', embedded_texts / elapsed)}"
puts "input_mb_per_sec=#{format('%.3f', embedded_bytes / elapsed / 1_000_000.0)}"
puts "output_bytes_per_batch=#{last&.bytesize}"
puts "gc_delta=#{gc_delta}"
