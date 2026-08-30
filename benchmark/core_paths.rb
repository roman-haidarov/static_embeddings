# frozen_string_literal: true
require_relative "_support"

SEBench.header("core_paths")

model = SEBench.model
short = SEBench.corpus(2_000, 20)
one = short.first
ids = model.tokenize(one)

SEBench.report("tokenize", SEBench.measure(initial_iterations: 500) { |n| n.times { model.tokenize(one) } },
               words: 20, coderange: "cached")

SEBench.report("embed", SEBench.measure(initial_iterations: 500) { |n| n.times { model.embed(one) } },
               words: 20, coderange: "cached")

SEBench.report("embed rotating corpus",
               SEBench.measure(initial_iterations: 1, operations_per_iteration: short.length) { |n|
                 n.times { short.each { |t| model.embed(t) } }
               }, texts: short.length, coderange: "cached")

SEBench.report("embed_batch",
               SEBench.measure(initial_iterations: 1, operations_per_iteration: short.length) { |n|
                 n.times { model.embed_batch(short) }
               }, texts: short.length, coderange: "cached")

SEBench.report("embed_batch f16",
               SEBench.measure(initial_iterations: 1, operations_per_iteration: short.length) { |n|
                 n.times { model.embed_batch(short, format: :f16) }
               }, texts: short.length, coderange: "cached")

SEBench.report("embed_token_ids",
               SEBench.measure(initial_iterations: 500) { |n| n.times { model.embed_token_ids(ids) } },
               tokens: ids.length)

query = model.embed(one)
abort "Benchmark query pooled to a zero vector; cosine_top_k requires a non-zero query" unless SEBench.nonzero_vector?(query)
matrix = model.embed_batch(short).freeze
rows = matrix.bytesize / (model.dim * 4)

SEBench.report("dot_top_k", SEBench.measure(initial_iterations: 5) { |n| n.times { model.dot_top_k(query, matrix, 10) } },
               rows: rows)
SEBench.report("cosine_top_k", SEBench.measure(initial_iterations: 5) { |n| n.times { model.cosine_top_k(query, matrix, 10) } },
               rows: rows)

query16 = model.embed(one, format: :f16)
matrix16 = model.embed_batch(short, format: :f16).freeze
SEBench.report("dot_top_k f16",
               SEBench.measure(initial_iterations: 5) { |n|
                 n.times { StaticEmbeddings.dot_top_k(query16, matrix16, 10, dim: model.dim, format: :f16) }
               }, rows: rows)

large = (SEBench.corpus(1, 200_000).first).freeze
large.valid_encoding?
SEBench.report("embed large truncated",
               SEBench.measure(initial_iterations: 20) { |n| n.times { model.embed(large) } },
               logical_bytes: large.bytesize, coderange: "cached")
