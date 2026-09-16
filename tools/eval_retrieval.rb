require "json"
require "optparse"
require "static_embeddings"

options = { k: 10 }
parser = OptionParser.new do |opts|
  opts.on("--model PATH") { |value| options[:model] = value }
  opts.on("--dataset PATH") { |value| options[:dataset] = value }
  opts.on("--k N", Integer) { |value| options[:k] = value }
end
parser.parse!(ARGV)
abort "usage: ruby -Ilib tools/eval_retrieval.rb --model MODEL.semb --dataset dataset.json" unless options[:model] && options[:dataset]

payload = JSON.parse(File.read(options[:dataset], encoding: "UTF-8"))
docs = payload.fetch("documents")
queries = payload.fetch("queries")
k = options[:k]

model = StaticEmbeddings.load(options[:model], verify: true)
matrix = model.embed_batch(docs.map { |doc| doc.fetch("text") })

def dcg(gains)
  gains.each_with_index.sum { |gain, i| gain / Math.log2(i + 2) }
end

mrr = 0.0
ndcg = 0.0
hits = 0

queries.each do |query|
  relevant = query.fetch("relevant").map(&:to_s)
  blob = model.embed(query.fetch("text"))
  ranked = model.cosine_top_k(blob, matrix, [k, docs.length].min)
  ids = ranked.map { |index, _| docs[index].fetch("id").to_s }
  rank = ids.index { |id| relevant.include?(id) }
  mrr += rank ? 1.0 / (rank + 1) : 0.0
  hits += 1 if rank && rank < k
  gains = ids.map { |id| relevant.include?(id) ? 1.0 : 0.0 }
  ideal = [1.0] * [relevant.length, k].min + [0.0] * [k - relevant.length, 0].max
  ideal = ideal.first(gains.length)
  denom = dcg(ideal)
  ndcg += denom.positive? ? dcg(gains) / denom : 0.0
end

n = queries.length.to_f
result = {
  "model_id" => model.model_id,
  "dim" => model.dim,
  "normalized" => model.normalized?,
  "queries" => queries.length,
  "documents" => docs.length,
  "k" => k,
  "mrr" => mrr / n,
  "ndcg_at_k" => ndcg / n,
  "hit_at_k" => hits / n
}
puts JSON.pretty_generate(result)
model.close
