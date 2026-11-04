require "json"
require "optparse"
require "static_embeddings"

options = { k: 10, bootstrap: 2000, seed: 42, per_query: false }
parser = OptionParser.new do |opts|
  opts.on("--model PATH") { |value| options[:model] = value }
  opts.on("--dataset PATH") { |value| options[:dataset] = value }
  opts.on("--k N", Integer) { |value| options[:k] = value }
  opts.on("--bootstrap N", Integer, "bootstrap resamples (0 disables)") { |value| options[:bootstrap] = value }
  opts.on("--seed N", Integer) { |value| options[:seed] = value }
  opts.on("--per-query", "include each query's first relevant rank") { options[:per_query] = true }
end
parser.parse!(ARGV)
abort "usage: ruby -Ilib tools/eval_retrieval.rb --model MODEL.semb --dataset dataset.json" unless options[:model] && options[:dataset]

payload = JSON.parse(File.read(options[:dataset], encoding: "UTF-8"))
docs = payload.fetch("documents")
queries = payload.fetch("queries")
k = options[:k]
abort "--k must be positive" unless k.positive?
abort "--bootstrap must be nonnegative" if options[:bootstrap].negative?
abort "dataset has no documents" if docs.empty?
abort "dataset has no queries" if queries.empty?

doc_ids = docs.map { |doc| doc.fetch("id").to_s }
unknown = queries.flat_map { |q| q.fetch("relevant").map(&:to_s) } - doc_ids
abort "queries reference unknown document ids: #{unknown.uniq.inspect}" unless unknown.empty?

def dcg(gains)
  gains.each_with_index.sum { |gain, i| gain / Math.log2(i + 2) }
end

def query_metrics(ranking, relevant, k)
  first = ranking.index { |id| relevant.include?(id) }
  top = ranking.first(k)
  gains = top.map { |id| relevant.include?(id) ? 1.0 : 0.0 }
  ideal = Array.new([relevant.length, top.length].min, 1.0)
  ideal_dcg = dcg(ideal)
  {
    first_rank: first && first + 1,
    rr: first ? 1.0 / (first + 1) : 0.0,
    rr_at_k: first && first < k ? 1.0 / (first + 1) : 0.0,
    ndcg_at_k: ideal_dcg.positive? ? dcg(gains) / ideal_dcg : 0.0,
    recall_at_k: relevant.empty? ? 0.0 : gains.sum / relevant.length,
    hit_at_k: first && first < k ? 1.0 : 0.0
  }
end

METRICS = %i[rr rr_at_k ndcg_at_k recall_at_k hit_at_k].freeze
NAMES = { rr: "mrr", rr_at_k: "mrr_at_k", ndcg_at_k: "ndcg_at_k", recall_at_k: "recall_at_k", hit_at_k: "hit_at_k" }.freeze

def mean(values)
  values.sum / values.length.to_f
end

def summarize(per_query)
  METRICS.to_h { |m| [NAMES.fetch(m), mean(per_query.map { |q| q.fetch(m) })] }
end

def bootstrap(per_query, resamples, rng)
  return nil if resamples.zero?

  n = per_query.length
  samples = METRICS.to_h { |m| [m, []] }
  resamples.times do
    draw = Array.new(n) { per_query[rng.rand(n)] }
    METRICS.each { |m| samples[m] << mean(draw.map { |q| q.fetch(m) }) }
  end
  METRICS.to_h do |m|
    sorted = samples[m].sort
    [NAMES.fetch(m), [sorted[(0.025 * (resamples - 1)).round], sorted[(0.975 * (resamples - 1)).round]]]
  end
end

model = StaticEmbeddings.load(options[:model], verify: true)
matrix = model.embed_batch(docs.map { |doc| doc.fetch("text") }).freeze
rng = Random.new(options[:seed])

zero_queries = []
per_query = queries.map do |query|
  relevant = query.fetch("relevant").map(&:to_s)
  blob = model.embed(query.fetch("text"))
  ranking =
    if blob.unpack("e*").all?(&:zero?)
      zero_queries << query.fetch("id", query.fetch("text")).to_s
      []
    else
      model.cosine_top_k(blob, matrix, docs.length).map { |index, _| doc_ids[index] }
    end

  query_metrics(ranking, relevant, k).merge(id: query.fetch("id", nil))
end

random_runs = 200
random = Array.new(random_runs) do
  summarize(queries.map { |query| query_metrics(doc_ids.shuffle(random: rng), query.fetch("relevant").map(&:to_s), k) })
end
random_baseline = random.first.keys.to_h { |key| [key, mean(random.map { |run| run.fetch(key) })] }

warnings = []
if k >= docs.length
  warnings << "k=#{k} >= documents=#{docs.length}: hit_at_k and recall_at_k cannot distinguish rankings when queries have relevant documents"
end
if queries.length < 50
  warnings << "only #{queries.length} queries: compare models by the bootstrap intervals, not the point estimates"
end
warnings << "#{zero_queries.length} queries embedded to a zero vector: #{zero_queries.inspect}" unless zero_queries.empty?

result = {
  "model_id" => model.model_id,
  "dim" => model.dim,
  "normalized" => model.normalized?,
  "truncation" => model.truncation,
  "queries" => queries.length,
  "documents" => docs.length,
  "k" => k,
  "metrics" => summarize(per_query),
  "bootstrap_95" => bootstrap(per_query, options[:bootstrap], rng),
  "random_baseline" => random_baseline,
  "warnings" => warnings
}
result["per_query"] = per_query.map { |q| { "id" => q[:id], "first_relevant_rank" => q[:first_rank] } } if options[:per_query]
warnings.each { |warning| warn "warning: #{warning}" }
puts JSON.pretty_generate(result)
model.close
