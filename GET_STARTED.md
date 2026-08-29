# Get started

## 1. Build it

```bash
bundle install
bundle exec rake            # compile + fixtures + test
bundle exec rake demo_model # tiny synthetic model, checkout only
```

## 2. Try it without downloading a model

```ruby
require "static_embeddings"

model = StaticEmbeddings.load_builtin

model.embed_array("hello world")
model.tokenize("Привет, мир!")
model.embed_with_stats("hello world")
```

The demo model is a fixture, not a quality baseline.

## 3. Convert a real model

```bash
git lfs clone https://huggingface.co/minishlab/potion-retrieval-32M
bundle exec static_embeddings convert ./potion-retrieval-32M --id potion-retrieval-32m
```

If the converter refuses the model, its tokenizer is outside the supported
profile. That refusal is the feature; read `docs/MODEL_AUDIT.md` before working
around it.

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
query_vector = model.embed(question, format: :f16)
```

`format: :f32` is the default and returns `dim * 4` bytes; `format: :f16`
returns `dim * 2` for compact storage. Keep one format per index.

Offline indexing scales by running more application workers. The runtime
deliberately does not expose internal `threads:` parallelism.

For anything real, fuse this with BM25 — SQLite FTS5 gives it to you for free —
via Reciprocal Rank Fusion. Dense and lexical retrieval miss different things,
and the combination recovers most of the quality gap against a transformer
encoder.

## 5. Watch the UNK ratio

```ruby
stats = model.embed_with_stats(text)
ratio = stats[:unk_count].to_f / [stats[:token_count], 1].max
```

An English model fed Russian returns a valid vector built from nothing. This is
the single most likely way to get a silently bad index.
