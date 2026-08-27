# Get started

## 1. Build it

```bash
bundle install
bundle exec rake            # compile + fixtures + test
bundle exec rake demo_model # build the tiny model shipped in the gem
```

## 2. Try it without any model download

```ruby
require "static_embeddings"

model = StaticEmbeddings.load_builtin   # after bundle exec rake demo_model

model.embed_array("hello world")
model.tokenize("Привет, мир!")
model.embed_with_stats("hello world")
```

## 3. Convert a real model

```bash
git lfs clone https://huggingface.co/minishlab/potion-retrieval-32M
bundle exec static_embeddings convert ./potion-retrieval-32M --id potion-retrieval-32m
```

The converter will refuse the model if its tokenizer is outside the supported
profile. That refusal is the feature: read `docs/MODEL_AUDIT.md` before
working around it.

## 4. Wire it into retrieval

```ruby
model = StaticEmbeddings.load(ENV.fetch("EMBEDDING_MODEL"))

# index
db.transaction do
  chunks.each_slice(500) do |slice|
    blob = model.embed_batch(slice.map(&:text), format: :f16)
    rows = StaticEmbeddings.unpack(blob, model.dim, format: :f16)
    slice.each_with_index { |chunk, i| store(chunk, rows[i]) }
  end
end

# query
query_vector = model.embed(question, format: :f16) # 512-dim model => 1024-byte blob
```


`format: :f32` is the default and returns `dim * 4` bytes. `format: :f16`
returns `dim * 2` bytes for compact storage. Keep one format per index/table.

Run multiple application workers/jobs when offline indexing should use more
than one CPU core. The runtime intentionally does not expose internal
`threads:` parallelism.

For anything real, fuse this with BM25 (SQLite FTS5 gives it to you for free)
via Reciprocal Rank Fusion — dense and lexical retrieval miss different things,
and the combination recovers most of the quality gap against a transformer
encoder.

## 5. Watch the UNK ratio

```ruby
stats = model.embed_with_stats(text)
ratio = stats[:unk_count].to_f / [stats[:token_count], 1].max
```

An English model fed Russian returns a valid vector built from nothing. This
is the single most likely way to get a silently bad index.
