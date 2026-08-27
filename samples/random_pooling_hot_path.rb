require_relative "support/hot_path"

model = StaticEmbeddingsSample.load_model
format = StaticEmbeddingsSample.embedding_format
sample_name = "static_embeddings_random_pooling_#{StaticEmbeddingsSample.format_label(format)}_hot_path"
duration = StaticEmbeddingsSample.env_float("DURATION", 25.0)
sleep_before_hot_loop = StaticEmbeddingsSample.env_float("SLEEP_BEFORE_HOT_LOOP", 7.0)
preheat_iterations = StaticEmbeddingsSample.env_int("PREHEAT_ITERATIONS", 2)
token_count = StaticEmbeddingsSample.env_int("TOKENS", 512)
seed = StaticEmbeddingsSample.env_int("SEED", 20_260_827)
id_sets_count = StaticEmbeddingsSample.env_int("ID_SETS", 256)
id_sets = StaticEmbeddingsSample.random_id_sets(model.vocab_size, token_count, id_sets_count, seed)
distinct_rows = id_sets.flatten.uniq.length
footprint_bytes = distinct_rows * model.dim * 4
cursor = 0
native_grep = "StaticEmbeddings|static_embeddings|embed_token_ids|ids_execute|se_embed_ids|se_embed_one|add_row|se_l2_normalize"

StaticEmbeddingsSample.preheat(preheat_iterations) do
  id_sets.each do |set|
    blob = model.embed_token_ids(set, format: format)
    StaticEmbeddingsSample.embedding_blob!(blob, model, 1, "preheat", format: format)
  end
end

StaticEmbeddingsSample.print_header(
  sample_name: sample_name,
  model: model,
  call: "model.embed_token_ids(random_ids, format: :#{StaticEmbeddingsSample.format_label(format)}) over #{id_sets_count} rotating id sets",
  duration: duration,
  sleep_before_hot_loop: sleep_before_hot_loop,
  preheat_iterations: preheat_iterations,
  native_grep: native_grep,
  extra: {
    token_count: token_count,
    format: StaticEmbeddingsSample.format_label(format),
    bytes_per_component: StaticEmbeddingsSample.bytes_per_component(format),
    seed: seed,
    id_sets: id_sets_count,
    distinct_rows: distinct_rows,
    touched_matrix_bytes: footprint_bytes,
    expected_hot_symbols: [
      "embed_token_ids / ids_execute / se_embed_ids",
      "add_row and se_l2_normalize should dominate random embedding row access",
      "this bypasses tokenizer to isolate random embedding matrix locality",
      "touched_matrix_bytes must exceed the last level cache or the number is cache bandwidth"
    ]
  }
)

sleep sleep_before_hot_loop
count, elapsed, last, gc_delta = StaticEmbeddingsSample.run_measured(duration) do
  set = id_sets[cursor]
  cursor += 1
  cursor = 0 if cursor >= id_sets.length
  model.embed_token_ids(set, format: format)
end
StaticEmbeddingsSample.embedding_blob!(last, model, 1, "last_hot_loop", format: format) if last

puts "count=#{count}"
puts "elapsed=#{format('%.6f', elapsed)}"
puts "ops_per_sec=#{format('%.6f', count / elapsed)}"
puts "sec_per_op=#{format('%.6f', elapsed / [count, 1].max)}"
puts "token_ids_per_sec=#{format('%.3f', (count * token_count) / elapsed)}"
puts "distinct_rows=#{distinct_rows}"
puts "touched_matrix_mb=#{format('%.1f', footprint_bytes / 1_000_000.0)}"
puts "row_bytes_per_sec=#{format('%.3f', (count * token_count * model.dim * 4) / elapsed)}"
puts "output_bytes=#{last&.bytesize}"
puts "gc_delta=#{gc_delta}"
