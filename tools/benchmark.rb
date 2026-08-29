#!/usr/bin/env ruby

$LOAD_PATH.unshift(File.expand_path("../lib", __dir__))
require "static_embeddings"
require "benchmark"

path = ARGV[0] || File.expand_path("../lib/models/demo.semb", __dir__)
model = StaticEmbeddings.load(path)
model.warmup!

CORPUS = (1..5_000).map do |i|
  "локальный поиск по тексту номер #{i} hello world static embedding vector search"
end

stats = CORPUS.map { |t| model.embed_with_stats(t) }
tokens = stats.sum { |s| s[:token_count] }
truncated = stats.count { |s| s[:truncated] }
logical_bytes = CORPUS.sum(&:bytesize)
processed_bytes = truncated.zero? ? logical_bytes : nil

def timed(label, iterations)
  elapsed = Benchmark.realtime { iterations.times { yield } }
  [label, elapsed]
end

single = Benchmark.realtime { 2_000.times { model.embed(CORPUS.first) } }
batch1 = Benchmark.realtime { model.embed_batch(CORPUS) }
tok_only = Benchmark.realtime { CORPUS.each { |t| model.tokenize(t) } }

per_token_ns = (batch1 / tokens) * 1e9
per_token_per_100dim = per_token_ns / (model.dim / 100.0)

puts "model            #{model.model_id} dim=#{model.dim} vocab=#{model.vocab_size}"
puts "corpus           #{CORPUS.length} texts, #{tokens} tokens, #{logical_bytes} logical bytes"
puts "truncation       #{truncated} of #{CORPUS.length} texts truncated at max_tokens=#{model.max_tokens}"
puts
puts "single embed     #{((single / 2_000) * 1e6).round(1)} us/call"
puts "batch           #{(CORPUS.length / batch1).round} texts/s, #{(tokens / batch1).round} tokens/s"
if processed_bytes
  puts "tokenize only    #{((tok_only / processed_bytes) * 1e9).round(1)} ns/processed byte"
else
  puts "tokenize only    #{(tok_only * 1e3).round(1)} ms total (no ns/byte: truncation makes"
  puts "                 processed bytes smaller than the #{logical_bytes} logical bytes)"
end
puts
puts "normalised       #{per_token_ns.round(1)} ns/token"
puts "                 #{per_token_per_100dim.round(1)} ns/token/100dim"
