# static_embeddings

`static_embeddings` is a Ruby runtime for converted Model2Vec / potion static embedding models.
The current production target is `minishlab/potion-retrieval-32M` converted once into a local `.semb` file and loaded through a small C extension.

Runtime goals:

- no ONNX Runtime;
- no Rust or Python dependency at runtime;
- no runtime network access;
- no HuggingFace parsing in the hot path;
- one mmap-able model file;
- binary float32 output suitable for storing or passing to a vector index.

```ruby
require "static_embeddings"

model = StaticEmbeddings.load_model("potion-retrieval-32m")

blob = model.embed("postgres pipeline mode in Ruby")
blob.bytesize == model.dim * 4

batch = model.embed_batch([
  "postgres pipeline mode in Ruby",
  "local static embeddings without ONNX Runtime"
])

batch.bytesize == 2 * model.dim * 4
```

## Runtime contract

The runtime does not load arbitrary HuggingFace models. It loads only `.semb` files produced by this repository's converter.

A `.semb` file contains:

- validated model metadata;
- a BERT WordPiece tokenizer profile;
- a mmap-ready vocabulary lookup table;
- float32 embedding rows;
- provenance and checksum data;
- the Model2Vec inference decisions needed by the C runtime.

The first supported tokenizer profile is `BERT_WORDPIECE_V1`. The converter must reject unsupported tokenizer features instead of approximating them.

## Current model workflow

The expected workflow for `potion-retrieval-32M` is explicit.

```bash
git lfs install
git clone https://huggingface.co/minishlab/potion-retrieval-32M

bundle exec rake compile

bundle exec ruby -Ilib exe/static_embeddings convert ./potion-retrieval-32M \
  --id potion-retrieval-32m
```

The converted model is written to:

```text
~/.cache/static_embeddings/models/potion-retrieval-32m.semb
```

Inspect and verify it:

```bash
bundle exec ruby -Ilib exe/static_embeddings inspect \
  ~/.cache/static_embeddings/models/potion-retrieval-32m.semb

bundle exec ruby -Ilib exe/static_embeddings verify \
  ~/.cache/static_embeddings/models/potion-retrieval-32m.semb
```

Load it from Ruby:

```ruby
model = StaticEmbeddings.load_model("potion-retrieval-32m")
```

Or load the file directly:

```ruby
model = StaticEmbeddings.load(
  File.expand_path("~/.cache/static_embeddings/models/potion-retrieval-32m.semb")
)
```

`load` does not verify the SHA-256 by default: hashing a 135 MB file on every
boot would undo the point of mmapping it lazily. Structural validation always
runs, so a malformed container is rejected, but a bit flip inside the matrix is
not detected and silently changes the vectors. Verify once where the artifact
enters your system - during the image build, in CI, or right after conversion -
and load without verification at runtime:

```ruby
StaticEmbeddings.load(path, verify: true)   # image build / CI
StaticEmbeddings.load(path)                 # hot path, artifact already trusted
```

## Demo model

A tiny demo model can be generated for smoke tests without network access or external model files.

```bash
bundle exec rake demo_model
```

```ruby
model = StaticEmbeddings.load_builtin
```

`load_builtin` only works inside a checkout after `rake demo_model` has run. The
demo model is a build artifact, it is not shipped in the published gem, and
`StaticEmbeddings.builtin_available?` returns `false` when it is missing. Treat
it only as a test fixture and API demo. Do not use it as a retrieval quality
baseline.

## API

```ruby
model.path
model.dim
model.vocab_size
model.max_tokens
model.normalized?
model.lowercase?
model.unk_id
model.mapped_bytes
model.provenance
model.model_id

vector_blob = model.embed("postgres pipeline mode in Ruby")               # f32 by default
small_blob  = model.embed("postgres pipeline mode in Ruby", format: :f16) # half-size storage
array       = model.embed_array("postgres pipeline mode in Ruby")

batch_blob  = model.embed_batch(texts)
small_batch = model.embed_batch(texts, format: :f16)
arrays      = model.embed_batch_arrays(texts)

stats = model.embed_with_stats("postgres pipeline mode in Ruby")
stats[:vector]
stats[:token_count]
stats[:unk_count]
stats[:truncated]

ids = model.tokenize("postgres pipeline mode in Ruby")

vector_blob = model.embed_token_ids(ids)
stats       = model.embed_token_ids_with_stats(ids)

model.cosine_top_k(query_blob, matrix_blob, 10)
model.dot_top_k(query_blob, matrix_blob, 10)

StaticEmbeddings.cosine_top_k(query_blob, matrix_blob, 10, dim: model.dim)
StaticEmbeddings.cosine_top_k(query_f16, matrix_f16, 10, dim: model.dim, format: :f16)

StaticEmbeddings.pack(rows, format: :f16)     # Array(s) of Float -> blob
StaticEmbeddings.unpack(blob, model.dim)      # blob -> Array of Array(Float)
```

Default embedding output is a binary `String` containing little-endian float32 values in row-major order.
Pass `format: :f16` to `embed`, `embed_batch`, `embed_with_stats`, `embed_token_ids`,
or `embed_token_ids_with_stats` when you want IEEE float16 storage instead. For a
512-dimensional model this changes one vector from `512 * 4 = 2048` bytes to
`512 * 2 = 1024` bytes. The model still computes in float32; `format:` only
controls the returned storage encoding.

`embed_array` and `embed_batch_arrays` are convenience methods and allocate Ruby `Float` objects.
They accept the same `format:` option and decode the returned storage format for debugging or application code.

`f16` top-k scans decode half rows with native conversion where the CPU has it:
`FCVTL` on AArch64 (baseline, no `-mcpu` needed) and `VCVTPH2PS` on x86 when
CPUID reports F16C with OS support for AVX state. Everything else falls back to
a 65,536-entry lookup table, which is populated only when that fallback is
actually selected.

`StaticEmbeddings.simd_backend` reports which kernel is live and returns
`"neon-fp16"`, `"f16c"` or `"lut"`. Check it before drawing conclusions from an
`f16` benchmark - the difference between the native and fallback kernels is
several-fold.

With a native kernel `f16` search can be faster on memory-bound corpora, but it
is not guaranteed: half rows still have to be decoded before scoring. Treat
`f16` primarily as a storage/RAM trade-off and benchmark on the target CPU. On
the lookup-table fallback it is usually slower than `f32`.

`embed_token_ids` skips the tokenizer and pools the rows you give it. It applies
the same `max_tokens` truncation as `embed`, so `embed_token_ids(model.tokenize(text))`
equals `embed(text)`. Pass `max_tokens: false` to pool every id. It exists for
debugging, for reusing a cached tokenization, and for benchmarking the pooling
loop in isolation; it is not a faster path for ordinary text.

### Similarity helpers

`cosine_top_k` divides by both vector norms and returns cosine similarity in
`[-1, 1]`. It raises `ArgumentError` on a zero-norm query and scores zero-norm
rows as `0.0`.

`dot_top_k` returns the raw dot product with no normalization. Use it when your
vectors are already unit length, including output from a model whose
`normalized?` is `true`, because then the two agree and the dot product is
cheaper. Use `cosine_top_k` in every other case.

Both skip rows whose score is `NaN` rather than letting them into the result.
That includes a row that itself contains `NaN`: `cosine_top_k` returns `0.0`
only for a genuinely zero-norm row, so a corrupt row is never reported as if it
were merely empty.

```ruby
if model.normalized?
  model.dot_top_k(query_blob, matrix_blob, 10)
else
  model.cosine_top_k(query_blob, matrix_blob, 10)
end
```

#### `dim:` is required on the module-level form

An embedding blob is bare bytes. It carries no dimension and no format tag, so
a query and a matrix stored in different formats can divide out to a consistent
row count and produce plausible nonsense. The module-level helpers therefore
require `dim:` and check the query against it:

```ruby
StaticEmbeddings.cosine_top_k(query_f32, matrix_f16, 10, dim: 512, format: :f16)
# ArgumentError: query is 2048 bytes but dim: 512 with format :f16 needs 1024 bytes
```

`model.cosine_top_k` / `model.dot_top_k` fill `dim:` in from the model and are
the recommended form.

#### The matrix must be frozen

A matrix at or above 1 MiB is scanned with the GVL released, so other Ruby
threads keep running while the scan is in flight. Nothing stops one of them from
reallocating the blob mid-scan, and copying a corpus-sized matrix on every query
would defeat the point of the API. A frozen `String` cannot be mutated at all,
so freezing is the contract:

```ruby
MATRIX = model.embed_batch(corpus).freeze

model.dot_top_k(query, MATRIX, 10)          # lock-free, concurrent, GVL released
```

Passing a large unfrozen matrix raises `ArgumentError`. If you cannot freeze it,
`allow_unfrozen: true` scans it while holding the GVL: correct, but it blocks
every other thread in the process for the duration of the scan.

A large matrix must also be 4-byte aligned when `format: :f32`. A blob produced
by `pack`, `embed_batch` or `File.binread` always is; a `byteslice` at an odd
offset may not be, and is rejected rather than silently copied.

## Language safety

`potion-retrieval-32M` is an English retrieval model. Russian text can produce a high `[UNK]` ratio while still returning a valid vector.

Check unknown-token pressure before trusting a corpus:

```ruby
stats = model.embed_with_stats(text)
ratio = stats[:token_count].zero? ? 0.0 : stats[:unk_count].to_f / stats[:token_count]

warn "high [UNK] ratio: #{ratio.round(3)}" if ratio > 0.3
```

Use `tokenize` when vector quality looks wrong:

```bash
bundle exec ruby -Ilib exe/static_embeddings tokenize \
  ~/.cache/static_embeddings/models/potion-retrieval-32m.semb \
  "postgres pipeline mode in Ruby"
```

## Correctness contract

The reference implementation is `model2vec.StaticModel`.

The converter records these decisions in the `.semb` file:

- default truncation is 512 tokens;
- truncation happens after tokenization and before pooling;
- `[UNK]` tokens are dropped;
- input with no usable tokens returns a zero vector;
- vectors are L2-normalized when the source model requires it.

`docs/MODEL_AUDIT.md` records the source model revision, file digests, tokenizer facts, added-token audit, edge-case behavior and reference parity result for every trusted converted model. The audited `potion-retrieval-32m` conversion is recorded there with `.semb` bytes, SHA256, token-id parity and vector parity. New conversions must add their own parity record before they are treated as trusted.

Run the parity check with:

```bash
python3 -m venv .venv-model2vec && . .venv-model2vec/bin/activate
pip install -U model2vec numpy
python tools/model2vec_oracle.py minishlab/potion-retrieval-32M --out tmp/model2vec_oracle.json
bundle exec rake parity \
  MODEL="$HOME/.cache/static_embeddings/models/potion-retrieval-32m.semb" \
  ORACLE=tmp/model2vec_oracle.json
```

The oracle records token ids as well as vectors, and includes inputs on both sides of the large-input prefix window.

Token ids must match the reference exactly. Float vectors are checked with tolerance because floating-point addition order is not bit-stable across implementations.

Accepted vector tolerance:

```text
cosine >= 1 - 1e-6
max_abs_diff < 1e-5
```

## Large inputs

Ruby strings have to be copied into C-owned memory before the GVL can be
released, and that copy is charged to the calling thread while it still holds
the GVL. When truncation is active the runtime therefore does not copy the whole
string: it copies a leading slice sized from `max_tokens`, cut immediately after
an ASCII byte that ends a word, and only accepts the result once that slice has
actually reached `max_tokens`. If it has not, the budget doubles and the text is
tokenized again.

The result is identical to tokenizing the whole document — WordPiece
segmentation is word-local, and the cut is always on both a codepoint and a word
boundary — but the cost of `embed` stops growing with input size:

```text
10 KB    43-47 us
100 KB   43-47 us
1 MB     43-47 us
3 MB     43-47 us
```

A cut is legal where the tokenizer would have started a fresh word: after
whitespace, after punctuation, and on either side of a CJK codepoint, using the
model's own Unicode tables rather than an ASCII-only rule. Text with no legal
cut anywhere in the scan window - one enormous word, or a run of combining marks
- falls back to copying the whole input, which is the only correct thing to do
there.

`embed_batch` applies the same strategy per text. With `max_tokens: false` there
is nothing to truncate, so the whole input is copied and processed.

`embed_batch` builds its result in C memory and then copies it into a Ruby
`String`, so peak memory for one call is about twice the size of the returned
blob. Split very large corpora into chunks if that matters.

## Concurrency

`embed_batch` copies Ruby input into C-owned memory, then releases the GVL for large C computations. That is the intended web-runtime behaviour: one request can run a CPU-bound embedding batch while other Puma threads in the same process continue serving work.

The runtime does not expose internal parallelism through `threads:`. For offline indexing, split work at the application/job level and run multiple Ruby workers explicitly.

Do not mutate an Array while it is being embedded. `embed_batch` snapshots the
elements it was given before it releases the GVL, so a concurrent mutation
cannot corrupt memory, but the call embeds the snapshot and ignores anything
that happened afterwards.

`cosine_top_k` and `dot_top_k` scan a frozen matrix without any lock, so any
number of threads can search one shared corpus at the same time. See the
similarity helpers section for the frozen-matrix contract.

When a `Fiber::Scheduler` is installed and the current fiber is non-blocking,
work above 2 KB is handed to a real thread that the fiber joins, so the
scheduler keeps running other fibers. Note that this creates one OS thread per
call: under high concurrency prefer batching over many small `embed` calls.

For Puma with `preload_app!`, load and warm the model before workers fork:

```ruby
preload_app!

before_fork do
  MODEL = StaticEmbeddings.load(ENV.fetch("EMBEDDING_MODEL"))
  MODEL.warmup!
end
```

## Docker / production

Convert models during image build or deploy preparation. Production runtime should only see `.semb` files.

```dockerfile
RUN bundle exec ruby -Ilib exe/static_embeddings convert ./potion-retrieval-32M \
  --id potion-retrieval-32m
```

Then set the path explicitly:

```bash
export EMBEDDING_MODEL=/root/.cache/static_embeddings/models/potion-retrieval-32m.semb
```

```ruby
model = StaticEmbeddings.load(ENV.fetch("EMBEDDING_MODEL"))
```

## Development

```bash
bundle install
bundle exec rake clean compile
bundle exec rake test
```

Build the synthetic fixture model:

```bash
ruby tools/make_fixture_model.rb
```

Run the benchmark:

```bash
ruby tools/benchmark.rb
```

Run the C memory smoke binary locally after building a fixture:

```bash
ruby tools/make_fixture_model.rb
ruby -Ilib -e 'require "static_embeddings"; StaticEmbeddings.convert("test/fixtures/tiny-wordpiece", output_path: "tmp/test-tiny.semb", model_id: "fixture/tiny-wordpiece")'
cc -O2 -std=c99 -Wall -Wextra -Iext/static_embeddings \
  tools/memory_smoke.c \
  ext/static_embeddings/se_format.c \
  ext/static_embeddings/se_unicode.c \
  ext/static_embeddings/se_tokenizer.c \
  ext/static_embeddings/se_embed.c \
  -lm -o tmp/memory_smoke
./tmp/memory_smoke tmp/test-tiny.semb
```

## Notes

The runtime is intentionally narrow. If a model requires unsupported tokenizer behavior, fix the converter whitelist and audit first; do not make the C runtime guess.

## License

MIT. See `LICENSE.txt`.
