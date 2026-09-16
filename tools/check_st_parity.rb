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
  abort "usage: ruby -Ilib tools/check_st_parity.rb --model MODEL.semb --oracle oracle.json"
end

model = StaticEmbeddings.load(options[:model], verify: true)
payload = JSON.parse(File.read(options[:oracle], encoding: "UTF-8"))
abort "unsupported oracle schema #{payload["schema_version"].inspect}" unless payload["schema_version"] == 1

reference = payload.fetch("reference")
abort "ST oracle must use add_special_tokens=false" unless reference["add_special_tokens"] == false
abort "ST oracle must use normalize_embeddings=false" unless reference["normalize_embeddings"] == false
if model.normalized?
  abort "ST source-faithful .semb must not L2-normalize; got normalized?=true"
end
unless model.max_tokens == false
  abort "ST source-faithful .semb must bake unlimited tokens; got max_tokens=#{model.max_tokens.inspect}"
end
if model.provenance["unk_policy"] != "include"
  abort "ST source-faithful .semb must use UNK_INCLUDE; got #{model.provenance["unk_policy"].inspect}"
end

def prefix_dims(vector, dim)
  return vector if vector.length == dim
  abort "oracle vector shorter than model dim (#{vector.length} < #{dim})" if vector.length < dim

  vector.first(dim)
end

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
vector_failures = []
diag_failures = []
min_cosine = 1.0
max_abs_all = 0.0
vectors_checked = 0

payload.fetch("rows").each_with_index do |row, i|
  text = row.fetch("text")
  label = row.fetch("label", i.to_s)
  problems = []

  expected_raw = row.fetch("hf_raw_token_ids")
  got_raw = model.tokenize(text, max_tokens: false)
  if got_raw != expected_raw
    raw_failures << i
    first = got_raw.zip(expected_raw).index { |a, b| a != b } || [got_raw.length, expected_raw.length].min
    problems << "raw ids differ at #{first} (got #{got_raw.length}, ref #{expected_raw.length})"
  end

  got_vector = model.embed_array(text)
  reference_vector = prefix_dims(row.fetch("st_encode_vector"), got_vector.length)
  cosine, max_abs = vector_metrics(reference_vector, got_vector)
  vectors_checked += 1
  min_cosine = [min_cosine, cosine].min
  max_abs_all = [max_abs_all, max_abs].max
  unless cosine >= options[:min_cosine] && max_abs <= options[:max_abs]
    vector_failures << i
    problems << format("encode vector out of tolerance cos=%.10f max_abs=%.8g", cosine, max_abs)
  end

  if row.key?("static_embedding_vector")
    diag = prefix_dims(row.fetch("static_embedding_vector"), got_vector.length)
    dcos, dabs = vector_metrics(diag, got_vector)
    unless dcos >= options[:min_cosine] && dabs <= options[:max_abs]
      diag_failures << i
      problems << format("StaticEmbedding diagnostic out of tolerance cos=%.10f max_abs=%.8g", dcos, dabs)
    end
  end

  status = problems.empty? ? "ok" : "FAIL"
  puts "#{status} idx=#{format('%03d', i)} label=#{label.inspect} bytes=#{text.bytesize}" +
       (problems.empty? ? "" : " [#{problems.join('; ')}]")
end

puts "rows=#{payload.fetch("rows").length}"
puts "vectors_checked=#{vectors_checked}"
puts "min_cosine=#{min_cosine}"
puts "max_abs_all=#{max_abs_all}"
puts "raw_token_id_failures=#{raw_failures.inspect}"
puts "vector_failures=#{vector_failures.inspect}"
puts "static_embedding_failures=#{diag_failures.inspect}"

unless raw_failures.empty? && vector_failures.empty? && diag_failures.empty?
  abort "ST parity failed"
end

puts "ST corpus parity OK (#{payload.fetch("rows").length}/#{payload.fetch("rows").length})"
