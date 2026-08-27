require_relative "support/hot_path"

model = StaticEmbeddingsSample.load_model
rows = StaticEmbeddingsSample.env_int("ROWS", 256)
bytes_per_text = StaticEmbeddingsSample.env_int("BYTES_PER_TEXT", 160)
modes = ENV.fetch("MODES", "ascii,hashes,base64,identifiers,long_words,unicode").split(",")
texts = modes.flat_map do |mode|
  StaticEmbeddingsSample.corpus(mode, [rows / modes.length, 1].max, bytes_per_text)
end.first(rows)
texts << "" if texts.empty?

sum_cos = 0.0
min_cos = Float::INFINITY
max_abs_all = 0.0
max_norm_delta = 0.0
worst_text = nil

texts.each do |text|
  f32 = model.embed_array(text, format: :f32)
  f16 = model.embed_array(text, format: :f16)

  dot = 0.0
  n32 = 0.0
  n16 = 0.0
  max_abs = 0.0

  f32.zip(f16) do |a, b|
    dot += a * b
    n32 += a * a
    n16 += b * b
    diff = (a - b).abs
    max_abs = diff if diff > max_abs
  end

  cos = if n32.zero? && n16.zero?
    1.0
  elsif n32.zero? || n16.zero?
    0.0
  else
    dot / Math.sqrt(n32 * n16)
  end

  norm_delta = (Math.sqrt(n16) - Math.sqrt(n32)).abs
  sum_cos += cos
  max_abs_all = max_abs if max_abs > max_abs_all
  max_norm_delta = norm_delta if norm_delta > max_norm_delta

  if cos < min_cos
    min_cos = cos
    worst_text = text
  end
end

mean_cos = sum_cos / texts.length
puts "mode=static_embeddings_f16_quality_check"
puts "model_path=#{model.path}"
puts "model_id=#{model.model_id || "unknown"}"
puts "dim=#{model.dim}"
puts "rows=#{texts.length}"
puts "bytes_per_text=#{bytes_per_text}"
puts "modes=#{modes.join(',')}"
puts "f32_bytes_per_vector=#{model.dim * 4}"
puts "f16_bytes_per_vector=#{model.dim * 2}"
puts "storage_ratio=#{format('%.3f', (model.dim * 2).to_f / (model.dim * 4))}"
puts "min_cosine=#{format('%.12f', min_cos)}"
puts "mean_cosine=#{format('%.12f', mean_cos)}"
puts "max_abs_all=#{format('%.10g', max_abs_all)}"
puts "max_norm_delta=#{format('%.10g', max_norm_delta)}"
puts "worst_text_bytes=#{worst_text&.bytesize}"
puts "worst_text=#{worst_text.to_s.byteslice(0, 96).inspect}"
puts "quality_status=#{min_cos >= 0.9999 ? "pass" : "review"}"
