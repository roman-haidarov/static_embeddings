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

tokens = CORPUS.sum { |t| model.tokenize(t).length }
bytes = CORPUS.sum(&:bytesize)

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
puts "corpus           #{CORPUS.length} texts, #{tokens} tokens, #{bytes} bytes"
puts
puts "single embed     #{((single / 2_000) * 1e6).round(1)} us/call"
puts "batch           #{(CORPUS.length / batch1).round} texts/s, #{(tokens / batch1).round} tokens/s"
puts "tokenize only    #{((tok_only / bytes) * 1e9).round(1)} ns/input byte"
puts
puts "normalised       #{per_token_ns.round(1)} ns/token"
puts "                 #{per_token_per_100dim.round(1)} ns/token/100dim"
