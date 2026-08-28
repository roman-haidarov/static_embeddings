# Changelog

## 0.1.1 (unreleased)

Safety and correctness release. Everything here was found by re-reviewing 0.1.0
against a built extension rather than against the source, and every item has a
regression test that fails on 0.1.0.

### Fixed

- **Concurrent `top_k` on a shared matrix (blocker).** 0.1.0 wrapped the matrix
  in `rb_str_locktmp` before releasing the GVL. That is an exclusive lock, so a
  second thread searching the same corpus got
  `RuntimeError: temporal locking already locked string` - 900 failures out of
  1200 calls with four threads over a 51 MB matrix. The lock is gone. A matrix
  at or above 1 MiB must now be `frozen`, which makes mutation impossible and
  lock-free concurrent scanning safe. A large unfrozen matrix raises
  `ArgumentError`; `allow_unfrozen: true` scans it while holding the GVL.
- **`embed_batch` read the caller's Array after releasing the GVL (blocker).**
  Elements were validated once and then re-read on every round, so another
  thread replacing an element with a non-String turned a later `RSTRING_LEN`
  into undefined behaviour - a reproducible `[BUG] Segmentation fault` from
  ordinary Ruby. Input is now snapshotted into a private Array before any GVL
  release, and the caller's Array is never read again.
- **Silent `f32`/`f16` mixing.** An `f32` query against an `f16` matrix used to
  divide out to a consistent row count and return plausible garbage.
  `StaticEmbeddings.cosine_top_k` / `dot_top_k` now require `dim:` and check the
  query length against it. `Model#cosine_top_k` / `Model#dot_top_k` supply
  `dim:` automatically and are the recommended form.
- **Prefix cutting only worked on ASCII.** The cut had to land on an ASCII
  whitespace or punctuation byte, so CJK and NBSP-separated documents grew the
  budget to the full length and copied the whole input under the GVL - cost
  scaled linearly with input size again. `se_prefix_boundary_len` now uses the
  tokenizer's own predicates: ASCII and Unicode whitespace, ASCII and Unicode
  punctuation, and either side of a CJK codepoint, skipping any codepoint the
  normaliser rewrites. A 24 MB CJK document went from 2.24 ms to 0.10 ms and is
  byte-identical to `embed_token_ids(tokenize(text))`.
- A large unaligned matrix is rejected instead of being silently copied in full
  while holding the GVL.
- **`f16` top-k was scalar and unusably slow on large corpora.** Half-precision
  rows are now decoded with AArch64 `FCVTL` / x86 `VCVTPH2PS` kernels when
  available, and with a lookup-table fallback otherwise. Measured on x86 with
  F16C at dim 512 over 20k rows: 55 -> 978 ops/s, which is also 1.8x the `f32`
  scan since it moves half the bytes. The fallback alone is 220 ops/s.
- The AArch64 kernel was gated on `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`, which
  is only defined with an explicit `-mcpu`. Default builds - including
  `rake compile` on Apple silicon - therefore fell back to the lookup table.
  `vcvt_f32_f16` is baseline AArch64, so the gate is now just `__aarch64__`.
- The half-precision lookup table (256 KB) is populated only when the fallback
  kernel is selected; native builds never touch those pages.
- `cosine_top_k` scored a row containing `NaN` as `0.0`, making a corrupt row
  indistinguishable from a zero-norm one, while `dot_top_k` dropped it. Both
  now drop it.
- `cold_start` reported `cold_page_cache=likely` from a ratio that cannot tell a
  page-cache read from a device read - it said "cold" while faulting 132 MB in
  4.3 ms. It now reports `warmup_mapped_bytes_per_sec` and
  `page_cache_state=unknown` with instructions for a real cold measurement.
- The valgrind job no longer passes `--undef-value-errors=no`. The harness is
  pure C with no Ruby noise, so the flag only hid the class of bug most likely
  in a parser reading an mmapped file.

### Added

- `StaticEmbeddings.encode_f16` / `decode_f16`: the storage codec that backs
  `format: :f16`, exposed directly and covered by table tests for zero, negative
  zero, subnormals, the smallest normal, the largest finite half, overflow to
  infinity, mantissa carry and NaN. `Ruby`'s duplicate half decoder in `unpack`
  was deleted; there is now one implementation.
- `StaticEmbeddings.pack`, the inverse of `unpack`.
- `StaticEmbeddings.simd_backend`, reporting the live `f16` kernel.
- `Model#cosine_top_k`, `Model#dot_top_k`.
- `test/top_k_contract_test.rb`, `test/input_snapshot_test.rb`,
  `test/f16_codec_test.rb`, and Unicode-boundary cases in
  `test/prefix_chunking_test.rb`.

### Changed

- `se_scratch_reserve` no longer takes an `input_bytes` argument it ignored.
- README documents the frozen-matrix contract, the required `dim:`, the
  non-ASCII prefix fallback, `verify: true` at artifact-entry time, the
  `embed_batch` peak-memory factor of two, and that `load_builtin` is a
  checkout-only fixture.

## 0.1.0

First working cut of the runtime and the offline converter.

- `.semb` v2 container: 320-byte header, 64-byte aligned sections, SHA-256
  over the file with the checksum field zeroed, mandatory provenance section.
- C runtime: mmap loader with full bounds validation, read-only
  open-addressing vocabulary hash table straight out of the mapping,
  `BERT_WORDPIECE_V1` tokenizer, f32 mean pooling with optional L2
  normalisation, binary vector output with `format: :f32` by default and compact
  `format: :f16` storage for single embeddings, batches, and token-id pooling.
- Model semantics (`max_tokens`, unk policy, empty policy, normalizer flags)
  are header fields, not hardcoded assumptions. Default truncation is 512
  tokens, matching `model2vec.StaticModel` rather than `model_max_length`.
- Unicode tables (NFD, simple lowercase, Mn/P/C/Zs ranges) are generated by
  the converter and shipped inside the model file. No utf8proc, no vendored
  dependency of any kind.
- Concurrency: GVL released for batches above 2 KB, real-thread hand-off when
  a `Fiber::Scheduler` is installed; internal `threads:` fan-out is intentionally rejected.
- Large inputs are not copied whole. When truncation is active `embed` and
  `embed_batch` copy a leading slice sized from `max_tokens`, cut on a word
  boundary, and grow it only if that slice did not reach `max_tokens`. Cost per
  call stops scaling with input size (3 MB behaves like 10 KB).
- `cosine_top_k` returns real cosine similarity; `dot_top_k` returns the raw dot
  product. They are separate kernels, not aliases. Both skip `NaN` rows and
  accept `format: :f16` when query and matrix blobs are half-precision encoded.
- `embed_token_ids` / `embed_token_ids_with_stats` pool a caller-supplied id
  array, honouring `max_tokens` truncation.
- Strict fail-closed converter: refuses non-WordPiece models, unknown
  normalizer or pre-tokenizer types, non-standard `added_tokens`, and any
  vocabulary/matrix size mismatch.
- CLI: `convert`, `verify`, `inspect`, `tokenize`, `embed`, `cache-path`.
- Tiny demo model is generated by `bundle exec rake demo_model` for smoke tests.
- Test suite runs offline against a synthetic source model: format fuzzing over
  the whole file, tokenizer parity and fuzzing against a pure-Ruby reference,
  vector tolerance, converter determinism, batch/thread determinism, GVL
  release, and the large-input prefix path (including a sweep over every
  printable ASCII separator).
- `tools/model2vec_oracle.py` and `tools/check_model2vec_parity.rb` compare
  token ids and vectors against Python `model2vec.StaticModel`. Run via
  `rake parity`.

### Known gaps

- `format: :f16` is a storage encoding, not a new model. The runtime computes in
  float32, converts the returned blob to IEEE float16, and decodes half-precision
  top-k inputs on the fly. Use `format: :f32` when exact float32 parity is needed.

### Not in this release

- int8 weights (the format has the slot, the runtime has no path)
- SIMD distance kernels
- SentencePiece / Unigram tokenizers
