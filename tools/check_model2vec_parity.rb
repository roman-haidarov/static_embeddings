require "json"
require "optparse"
require "static_embeddings"

options = {
  min_cosine: 1.0 - 1e-6,
  max_abs: 1e-5,
  ids: true
}

parser = OptionParser.new do |opts|
  opts.on("--model PATH") { |value| options[:model] = value }
  opts.on("--oracle PATH") { |value| options[:oracle] = value }
  opts.on("--min-cosine N", Float) { |value| options[:min_cosine] = value }
  opts.on("--max-abs N", Float) { |value| options[:max_abs] = value }
  opts.on("--[no-]ids") { |value| options[:ids] = value }
end
parser.parse!(ARGV)
unless options[:model] && options[:oracle]
  abort "usage: ruby -Ilib tools/check_model2vec_parity.rb --model MODEL.semb --oracle oracle.json"
end

model = StaticEmbeddings.load(options[:model], verify: true)
payload = JSON.parse(File.read(options[:oracle], encoding: "UTF-8"))
rows = payload.is_a?(Hash) ? payload.fetch("rows") : payload
oracle_max_length = payload.is_a?(Hash) ? payload["max_length"] : nil

if oracle_max_length && oracle_max_length != model.max_tokens
  warn "WARNING oracle max_length=#{oracle_max_length} but model.max_tokens=#{model.max_tokens}"
end

def dot(a, b)
  a.zip(b).sum { |x, y| x * y }
end

def norm(a)
  Math.sqrt(a.sum { |x| x * x })
end

min_cosine = 1.0
max_abs_all = 0.0
vector_failures = []
id_failures = []
id_rows_checked = 0

rows.each_with_index do |row, i|
  text = row.fetch("text")
  ref = row.fetch("vector")
  ref_ids = row["token_ids"]

  problems = []

  if options[:ids] && ref_ids
    id_rows_checked += 1
    got_ids = model.tokenize(text)
    if got_ids != ref_ids
      id_failures << i
      first = got_ids.zip(ref_ids).index { |a, b| a != b } || [got_ids.length, ref_ids.length].min
      problems << "ids differ at #{first} (got #{got_ids.length}, ref #{ref_ids.length})"
    end
  end

  got = model.embed(text).unpack("e*")
  max_abs = ref.zip(got).map { |a, b| (a - b).abs }.max || 0.0
  ref_zero = ref.all?(&:zero?)
  got_zero = got.all?(&:zero?)
  cosine =
    if ref_zero && got_zero
      1.0
    elsif ref_zero || got_zero
      0.0
    else
      dot(ref, got) / (norm(ref) * norm(got))
    end

  min_cosine = [min_cosine, cosine].min
  max_abs_all = [max_abs_all, max_abs].max

  unless cosine >= options[:min_cosine] && max_abs <= options[:max_abs]
    vector_failures << i
    problems << "vector out of tolerance"
  end

  status = problems.empty? ? "ok" : "FAIL"
  label = text.bytesize > 64 ? "#{text[0, 32].inspect}...(#{text.bytesize}B)" : text.inspect
  line = format("%s idx=%02d cos=%.10f max_abs=%.8g bytes=%d text=%s",
                status, i, cosine, max_abs, text.bytesize, label)
  line += "  [#{problems.join('; ')}]" unless problems.empty?
  puts line
end

puts "rows=#{rows.length}"
puts "id_rows_checked=#{id_rows_checked}"
puts "min_cosine=#{min_cosine}"
puts "max_abs_all=#{max_abs_all}"
puts "token_id_failures=#{id_failures.inspect}"
puts "vector_failures=#{vector_failures.inspect}"

if id_rows_checked.zero? && options[:ids]
  warn "WARNING oracle has no token_ids; regenerate it with the current tools/model2vec_oracle.py"
end

unless id_failures.empty? && vector_failures.empty?
  abort "parity failed: token ids #{id_failures.inspect}, vectors #{vector_failures.inspect}"
end

puts "parity OK"
