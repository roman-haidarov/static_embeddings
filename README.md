# static_embeddings

A Ruby runtime for converted Model2Vec / potion static embedding models. A model
is converted once into a local `.semb` file and loaded through a small C
extension. No ONNX Runtime, no Rust or Python at runtime, no network access, one
mmap-able file, binary float32 output.

The current production target is `minishlab/potion-retrieval-32M`.

```ruby
require "static_embeddings"

model = StaticEmbeddings.load_model("potion-retrieval-32m")

blob = model.embed("postgres pipeline mode in Ruby")
blob.bytesize == model.dim * 4

batch = model.embed_batch([
  "postgres pipeline mode in Ruby",
  "local static embeddings without ONNX Runtime"
])
```

New to the gem? Start with `GET_STARTED.md`. Design rationale is in
`docs/ARCHITECTURE.md`, and `docs/LIMITATIONS.md` is the short answer to "does
it do X".

## Runtime contract

The runtime loads only `.semb` files produced by this repository's converter,
never arbitrary HuggingFace models. A `.semb` file carries validated metadata, a
BERT WordPiece tokenizer profile, an mmap-ready vocabulary lookup, float32
embedding rows, and provenance plus a checksum.

The only supported tokenizer profile is `BERT_WORDPIECE_V1`, pinned for
compatibility to `tokenizers 0.23.1`. The converter rejects unsupported
tokenizer features rather than approximating them. `.semb` format v3 also
records which of the five standard BERT added tokens are active. Files produced
by 0.1.4 and earlier use format v2 and must be reconverted.

## Converting a model

```bash
git lfs install
git clone https://huggingface.co/minishlab/potion-retrieval-32M

bundle exec rake compile
bundle exec ruby -Ilib exe/static_embeddings convert ./potion-retrieval-32M \
  --id potion-retrieval-32m
```

The result lands in `~/.cache/static_embeddings/models/potion-retrieval-32m.semb`.
Inspect or verify it with the `inspect` and `verify` subcommands, then load it:

```ruby
model = StaticEmbeddings.load_model("potion-retrieval-32m")
model = StaticEmbeddings.load(ENV.fetch("EMBEDDING_MODEL"))   # explicit path
```

Convert during image build or deploy preparation; production should only ever
see `.semb` files.

The offline converter accepts F32, F16 and BF16 safetensors embedding matrices;
all three are converted to float32 rows in `.semb`.

### Verify once, not on every boot

`load` does not check the SHA-256 by default: hashing a 135 MB file at boot
would undo the point of mmapping it lazily. `verify` streams the file, so it
costs one chunk of memory rather than a copy of the model. Structural validation always runs,
so a malformed container is rejected, but a bit flip inside the matrix is not
detected and silently changes vectors. Verify where the artifact enters your
system:

```ruby
StaticEmbeddings.load(path, verify: true)   # image build / CI
StaticEmbeddings.load(path)                 # hot path, artifact already trusted
```

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
bounded     = model.embed(huge_text, validate_encoding: :prefix)          # see "Large inputs"
array       = model.embed_array("postgres pipeline mode in Ruby")

batch_blob  = model.embed_batch(texts)
small_batch = model.embed_batch(texts, format: :f16)
arrays      = model.embed_batch_arrays(texts)

stats = model.embed_with_stats("postgres pipeline mode in Ruby")
stats[:vector]
stats[:token_count]
stats[:unk_count]
stats[:pooled_count]  # rows actually eligible for pooling
stats[:truncated]

ids         = model.tokenize("postgres pipeline mode in Ruby")
vector_blob = model.embed_token_ids(ids)
stats       = model.embed_token_ids_with_stats(ids)

model.cosine_top_k(query_blob, matrix_blob, 10)
model.dot_top_k(query_blob, matrix_blob, 10)

StaticEmbeddings.cosine_top_k(query_blob, matrix_blob, 10, dim: model.dim)
StaticEmbeddings.pack(rows, format: :f16)     # Array(s) of Float -> blob
StaticEmbeddings.unpack(blob, model.dim)      # blob -> Array of Array(Float)
```

Output is a binary `String` of little-endian float32 values in row-major order.
`embed_array` and `embed_batch_arrays` decode that into Ruby `Float` objects and
exist for debugging and application code, not for the hot path.

`tokenize` exposes raw WordPiece ids, including `[UNK]`, and its own
`max_tokens` cap is therefore a raw-id cap. Embedding has a different, upstream
pooling rule: for `UNK_DROP`, `[UNK]` is removed first and `max_tokens` is then
applied to usable ids. To cache tokenization without changing the embedding,
cache it unbounded:

```ruby
ids = model.tokenize(text, max_tokens: false)
model.embed_token_ids(ids) == model.embed(text)
```

`embed_token_ids` applies that same usable-id cap. This distinction matters only
when `[UNK]` appears around the truncation boundary.

### `format: :f16`

Unknown keywords raise. `embed(text, fromat: :f16)` is an `ArgumentError`
naming the accepted keywords, not a silent f32 vector.

`format:` selects the returned storage encoding only. The model always computes
in float32. For a 512-dimensional model one vector goes from 2048 to 1024 bytes.
It is accepted by `embed`, `embed_batch`, `embed_with_stats`, `embed_token_ids`,
`embed_token_ids_with_stats`, `pack`, `unpack`, and the top-k helpers.

Choose `f16` for the RAM and storage it saves; speed is a property of the
machine and of the decode kernel. It halves the bytes a top-k scan streams, but
every row still has to be decoded before scoring — this repository's own
samples show `f16` winning on x86-64 F16C and, as of 0.1.4, on M1 Pro neon-fp16
as well. On the lookup-table fallback it usually loses. See
`docs/PERFORMANCE.md` before assuming either.

`StaticEmbeddings.simd_backend` reports the live kernel: `"neon-fp16"`,
`"f16c"`, or `"lut"` for the lookup-table fallback, which is several-fold
slower. Check it before drawing any conclusion from an `f16` benchmark.

### Similarity helpers

`cosine_top_k` divides by both norms and returns similarity in `[-1, 1]`. It
raises `ArgumentError` on a zero-norm query and scores zero-norm rows as `0.0`.
`dot_top_k` returns the raw dot product; use it when your vectors are already
unit length, including output from a model whose `normalized?` is `true`.

Both skip rows scoring `NaN` rather than admitting them to the result, so a
corrupt row is never reported as if it were merely empty.

Three contracts, each of which raises rather than guessing:

- **`dim:` is required on the module-level form.** A blob is bare bytes with no
  dimension and no format tag, so an f32 query against an f16 matrix would
  otherwise divide out to a plausible row count and return nonsense.
  `model.cosine_top_k` and `model.dot_top_k` fill `dim:` in and are the
  recommended form.
- **A matrix at or above 1 MiB must be frozen.** It is scanned with the GVL
  released so other threads keep running, and a frozen `String` cannot be
  mutated mid-scan. `allow_unfrozen: true` scans while holding the GVL: correct,
  but it blocks every other thread for the duration.
- **A large `format: :f32` matrix must be 4-byte aligned.** Anything from
  `pack`, `embed_batch` or `File.binread` is; a `byteslice` at an odd offset may
  not be.

```ruby
MATRIX = model.embed_batch(corpus).freeze
model.dot_top_k(query, MATRIX, 10)          # lock-free, concurrent, GVL released
```

## Large inputs

When truncation is active the runtime starts with a leading slice sized from
`max_tokens`, cut on a word boundary, and grows the budget until it has reached
`max_tokens` **usable** ids. For ordinary in-vocabulary text this keeps the
tokenized prefix to a few kilobytes even when the document is megabytes long.
For OOV-heavy input it may scan farther — or the whole string — because `[UNK]`
no longer consumes the usable-token budget. That is required for the corrected
UNK-before-truncate contract.

That bounds the tokenizer, not the whole call. `embed` also has to establish
that the Ruby `String` is valid UTF-8, and when Ruby has not computed the
string's coderange yet that scan is O(total bytes) and happens before any
prefix is chosen:

```ruby
text = File.read(path)   # coderange unknown
model.embed(text)        # full UTF-8 scan, then a few KB of tokenizing
model.embed(text)        # coderange cached: prefix window only
```

Three ways out, in order of preference. Reuse the `String`, since Ruby caches
the coderange after the first scan. Force the scan once outside the hot path
with `text.valid_encoding?`. Or ask for validation to be bounded the same way
tokenizing is:

```ruby
model.embed(text, validate_encoding: :prefix)
```

`:prefix` skips the up-front scan and lets the tokenizer validate the bytes it
actually reads; malformed UTF-8 inside the window still raises. On a 3 MB
document that turned 442 µs into 78 µs, against 71 µs for the same string with
its coderange already cached.

The trade is real and worth stating: under `:prefix`, invalid bytes *past* the
truncation window are never looked at and no longer raise. `:full` is the
default and keeps the whole-string guarantee. A coderange Ruby has already
computed is honoured either way, so a `String` already known to be broken
raises in both modes, and one already known to be valid costs nothing in
either. The option is accepted by `embed`, `embed_batch`, `embed_with_stats`
and `tokenize`.

Text with no legal cut anywhere in the scan window — one enormous word, a run of
combining marks — falls back to copying the whole input. With
`max_tokens: false` there is nothing to truncate, so the whole input is copied.
`docs/ARCHITECTURE.md` explains what makes a cut legal.

`embed_batch` builds its result in C memory and then copies it into a Ruby
`String`, so peak memory for one call is about twice the returned blob. Chunk
very large corpora.

## Concurrency

`embed_batch` snapshots its input, then releases the GVL for the C work, so one
request can compute embeddings while other Puma threads keep serving. Because
the snapshot is taken up front, mutating the Array afterwards does not corrupt
anything but is ignored.

`cosine_top_k` and `dot_top_k` scan a frozen matrix without any lock, so any
number of threads can search one shared corpus at once.

The runtime does not expose internal parallelism through `threads:`. For offline
indexing, split work at the job level and run multiple Ruby workers.

With a `Fiber::Scheduler` installed and the current fiber non-blocking, work
above 2 KB is handed to a real thread the fiber joins, so the scheduler keeps
running other fibers. That is one OS thread per call: under high concurrency,
prefer batching over many small `embed` calls.

For Puma with `preload_app!`, load and warm before workers fork:

```ruby
preload_app!

before_fork do
  MODEL = StaticEmbeddings.load(ENV.fetch("EMBEDDING_MODEL"))
  MODEL.warmup!
end
```

## Language safety

`potion-retrieval-32M` is an English retrieval model. Russian text can produce a
high `[UNK]` ratio while still returning a valid vector, which is the most
likely way to end up with a silently bad index. Check the ratio before trusting
a corpus:

```ruby
stats = model.embed_with_stats(text)
ratio = stats[:token_count].zero? ? 0.0 : stats[:unk_count].to_f / stats[:token_count]

warn "high [UNK] ratio: #{ratio.round(3)}" if ratio > 0.3
```

Out-of-vocabulary text is also slower, because every word falls through to
subword splitting instead of hitting the vocabulary directly.

## Correctness contract

0.1.5 separates contracts that 0.1.4 accidentally mixed together:

- `tokenize` matches the pinned Hugging Face `tokenizers 0.23.1` raw
  `BertNormalizer + BertPreTokenizer + WordPiece` ids, including the supported
  `normalized: false` standard added tokens and `[UNK]`.
- Embedding drops `[UNK]` when the model records `UNK_DROP`, then applies
  `max_tokens` to the remaining usable ids, mean-pools, and performs source-model
  L2 normalization.
- The gem intentionally does **not** reproduce Model2Vec 0.9.0's character
  pre-cut (`max_length * median_token_length`). Here `max_tokens` means actual
  usable tokenizer ids. The oracle reports rows affected by this difference
  separately instead of calling them parity.

Where the execution contracts are the same, vectors are compared to
`model2vec.StaticModel` with:

```text
cosine >= 1 - 1e-6
max_abs_diff < 1e-5
```

The committed boundary generator currently produces 434 systematic rows. CI
generates their oracle from pinned Python packages both on the deterministic
fixture and on the pinned `potion-retrieval-32M` snapshot. The Ruby `Reference`
fuzz is kept as a separate C-vs-Ruby implementation check and is not treated as
proof of upstream compatibility. `docs/MODEL_AUDIT.md` records per-model audits;
a corpus pass means exactly that corpus passed, not an exhaustive proof of all
Unicode/tokenizer behavior.

## Development

```bash
bundle install
bundle exec rake            # compile + fixtures + test
bundle exec rake demo_model # tiny synthetic model for smoke tests
```

`StaticEmbeddings.load_builtin` loads that demo model. It only works inside a
checkout, it is not shipped in the published gem, and
`StaticEmbeddings.builtin_available?` returns `false` when it is missing. It is
a test fixture and API demo, not a retrieval quality baseline.

```bash
ruby tools/benchmark.rb                  # normalised performance budget
bundle exec rake benchmark               # benchmark/, see docs/BENCHMARKING.md
bundle exec rake gc_compaction           # GC.compact hardening
bundle exec rake cancellation_timing     # timing-sensitive, excluded from rake test
samples/run_all.sh                       # native hot-path probes; see samples/README.md
```

The C memory smoke binary runs without Ruby:

```bash
ruby tools/make_fixture_model.rb
ruby -Ilib -e 'require "static_embeddings"; StaticEmbeddings.convert("test/fixtures/tiny-wordpiece", output_path: "tmp/test-tiny.semb", model_id: "fixture/tiny-wordpiece")'
cc -O2 -std=c99 -Wall -Wextra -Iext/static_embeddings \
  tools/memory_smoke.c \
  ext/static_embeddings/se_f16.c \
  ext/static_embeddings/se_topk.c \
  ext/static_embeddings/se_format.c \
  ext/static_embeddings/se_unicode.c \
  ext/static_embeddings/se_tokenizer.c \
  ext/static_embeddings/se_embed.c \
  -lm -pthread -o tmp/memory_smoke
./tmp/memory_smoke tmp/test-tiny.semb
```

The runtime is intentionally narrow. If a model needs unsupported tokenizer
behaviour, fix the converter whitelist and audit it; do not make the C runtime
guess.

## License

MIT. See `LICENSE.txt`.
