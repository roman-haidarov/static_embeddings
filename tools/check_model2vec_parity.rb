require "json"
require "optparse"
require "static_embeddings"

options = {
  min_cosine: 1.0 - 1e-6,
  max_abs: 1e-5
}

parser = OptionParser.new do |opts|
  opts.on("--model PATH") { |value| options[:model] = value }
  opts.on("--oracle PATH") { |value| options[:oracle] = value }
  opts.on("--min-cosine N", Float) { |value| options[:min_cosine] = value }
  opts.on("--max-abs N", Float) { |value| options[:max_abs] = value }
end
parser.parse!(ARGV)
unless options[:model] && options[:oracle]
  abort "usage: ruby -Ilib tools/check_model2vec_parity.rb --model MODEL.semb --oracle oracle.json"
end

model = StaticEmbeddings.load(options[:model], verify: true)
payload = JSON.parse(File.read(options[:oracle], encoding: "UTF-8"))
abort "unsupported oracle schema #{payload["schema_version"].inspect}" unless payload["schema_version"] == 2

reference = payload.fetch("reference")
max_length = Integer(reference.fetch("max_length"))
if max_length != model.max_tokens
  abort "oracle max_length=#{max_length} but model.max_tokens=#{model.max_tokens}"
end
if reference["unk_token_id"] && Integer(reference["unk_token_id"]) != model.unk_id
  abort "oracle unk_token_id=#{reference["unk_token_id"]} but model.unk_id=#{model.unk_id}"
end

rows = payload.fetch("rows")

def dot(a, b)
  a.zip(b).sum { |x, y| x * y }
end

def norm(a)
  Math.sqrt(a.sum { |x| x * x })
end

def vector_metrics(reference, got)
  max_abs = reference.zip(got).map { |a, b| (a - b).abs }.max || 0.0
  ref_zero = reference.all?(&:zero?)
  got_zero = got.all?(&:zero?)
  cosine =
    if ref_zero && got_zero
      1.0
    elsif ref_zero || got_zero
      0.0
    else
      dot(reference, got) / (norm(reference) * norm(got))
    end
  [cosine, max_abs]
end

raw_failures = []
usable_failures = []
vector_failures = []
invariant_failures = []
intentional_deviations = []
min_cosine = 1.0
max_abs_all = 0.0
vectors_checked = 0

rows.each_with_index do |row, i|
  text = row.fetch("text")
  label = row.fetch("label", i.to_s)
  problems = []

  expected_raw = row.fetch("hf_raw_token_ids")
  got_raw = model.tokenize(text)
  if got_raw != expected_raw
    raw_failures << i
    first = got_raw.zip(expected_raw).index { |a, b| a != b } || [got_raw.length, expected_raw.length].min
    problems << "raw ids differ at #{first} (got #{got_raw.length}, ref #{expected_raw.length})"
  end

  full_raw = model.tokenize(text, max_tokens: false)
  expected_usable = row.fetch("static_usable_token_ids")
  got_usable = full_raw.reject { |id| id == model.unk_id }.first(max_length)
  if got_usable != expected_usable
    usable_failures << i
    first = got_usable.zip(expected_usable).index { |a, b| a != b } || [got_usable.length, expected_usable.length].min
    problems << "usable ids differ at #{first} (got #{got_usable.length}, ref #{expected_usable.length})"
  end

  got_vector_blob = model.embed(text)
  pooled_blob = model.embed_token_ids(full_raw)
  unless got_vector_blob == pooled_blob
    invariant_failures << i
    problems << "embed(text) != embed_token_ids(unbounded tokenize(text))"
  end

  model2vec_ids = row.fetch("model2vec_token_ids")
  declared_deviation = row.fetch("model2vec_character_pretruncate_changes_ids")
  actual_deviation = model2vec_ids != expected_usable
  if actual_deviation != declared_deviation
    problems << "oracle character-pretruncate flag is inconsistent"
    usable_failures << i unless usable_failures.include?(i)
  elsif actual_deviation
    intentional_deviations << i
  else
    reference_vector = row.fetch("model2vec_vector")
    got_vector = got_vector_blob.unpack("e*")
    cosine, max_abs = vector_metrics(reference_vector, got_vector)
    vectors_checked += 1
    min_cosine = [min_cosine, cosine].min
    max_abs_all = [max_abs_all, max_abs].max
    unless cosine >= options[:min_cosine] && max_abs <= options[:max_abs]
      vector_failures << i
      problems << format("vector out of tolerance cos=%.10f max_abs=%.8g", cosine, max_abs)
    end
  end

  status = problems.empty? ? "ok" : "FAIL"
  suffix = actual_deviation ? " intentional-character-pretruncate-deviation" : ""
  puts "#{status} idx=#{format('%03d', i)} label=#{label.inspect} bytes=#{text.bytesize}#{suffix}" +
       (problems.empty? ? "" : " [#{problems.join('; ')}]")
end

puts "rows=#{rows.length}"
puts "vectors_checked=#{vectors_checked}"
puts "intentional_character_pretruncate_deviations=#{intentional_deviations.length}"
puts "min_cosine=#{min_cosine}" if vectors_checked.positive?
puts "max_abs_all=#{max_abs_all}" if vectors_checked.positive?
puts "raw_token_id_failures=#{raw_failures.inspect}"
puts "usable_token_id_failures=#{usable_failures.inspect}"
puts "embed_invariant_failures=#{invariant_failures.inspect}"
puts "vector_failures=#{vector_failures.inspect}"

unless raw_failures.empty? && usable_failures.empty? && invariant_failures.empty? && vector_failures.empty?
  abort "parity failed"
end

puts "corpus parity OK (#{rows.length}/#{rows.length}); intentional Model2Vec character pre-truncation deviations are reported separately"
