require_relative "support/hot_path"

model = StaticEmbeddingsSample.load_model
sample_name = "static_embeddings_tokenize_wordpiece_stress_hot_path"
duration = StaticEmbeddingsSample.env_float("DURATION", 25.0)
sleep_before_hot_loop = StaticEmbeddingsSample.env_float("SLEEP_BEFORE_HOT_LOOP", 7.0)
preheat_iterations = StaticEmbeddingsSample.env_int("PREHEAT_ITERATIONS", 5)
mode = ENV.fetch("MODE", "hashes")
units = StaticEmbeddingsSample.env_int("UNITS", 40)
text = case mode
       when "hashes" then StaticEmbeddingsSample.hash_text(units)
       when "base64" then StaticEmbeddingsSample.base64_text(units)
       when "identifiers" then StaticEmbeddingsSample.identifier_text(units)
       when "long_words" then Array.new(units) { |i| StaticEmbeddingsSample.long_ascii_word(100 - (i % 5)) }.join(" ")
       else raise "unknown MODE=#{mode.inspect}; use hashes, base64, identifiers, long_words"
       end.freeze
native_grep = "StaticEmbeddings|static_embeddings|model_tokenize|se_tokenize|normalize|append_wordpiece|wordpiece|se_vocab_lookup|se_hash_bytes|se_utf8_encode|se_range_contains|se_map_lookup|is_whitespace|is_punct|is_control"

StaticEmbeddingsSample.preheat(preheat_iterations) do
  ids = model.tokenize(text)
  raise "preheat: tokenize returned no ids" if ids.empty?
end

StaticEmbeddingsSample.print_header(
  sample_name: sample_name,
  model: model,
  call: "model.tokenize(text)",
  duration: duration,
  sleep_before_hot_loop: sleep_before_hot_loop,
  preheat_iterations: preheat_iterations,
  native_grep: native_grep,
  extra: {
    mode: mode,
    units: units,
    input_bytes: text.bytesize,
    expected_hot_symbols: [
      "model_tokenize / se_tokenize",
      "append_wordpiece / wordpiece / se_vocab_lookup / se_hash_bytes",
      "se_utf8_encode only if candidate encoding still dominates",
      "se_map_lookup / se_range_contains when Unicode classification dominates"
    ]
  }
)

sleep sleep_before_hot_loop
count, elapsed, last, gc_delta = StaticEmbeddingsSample.run_measured(duration) { model.tokenize(text) }
raise "last_hot_loop: tokenize returned no ids" if last.nil? || last.empty?

puts "count=#{count}"
puts "elapsed=#{format('%.6f', elapsed)}"
puts "ops_per_sec=#{format('%.6f', count / elapsed)}"
puts "sec_per_op=#{format('%.6f', elapsed / [count, 1].max)}"
puts "logical_input_mb_per_sec=#{format('%.3f', (count * text.bytesize) / elapsed / 1_000_000.0)}"
puts "token_count=#{last.length}"
puts "first_tokens=#{last.first(16).inspect}"
puts "gc_delta=#{gc_delta}"
