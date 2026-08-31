# Changelog

## 0.1.4 (unreleased)

Hot-path work in the C runtime. Token ids, the pooling contract and f16
rounding are unchanged: the ASCII parity sweep and the differential fuzz
digest against `StaticEmbeddings::Reference` still match, and the 0.1.3
`model2vec.StaticModel` oracle still covers this runtime. L2 still
accumulates sum-of-squares in double; 0.1.4 does that with SIMD pairwise
adds, so a bit-identical match against a 0.1.3 vector is not promised.

Measured on an M1 Pro, `potion-retrieval-32m`, `samples/run_all.sh`
`DURATION=25`, production build (no alloc-stats). The 0.1.3 comparison run
had `STATIC_EMBEDDINGS_ALLOC_STATS=1`, which the docs cost at about 3% on
`embed` and 13% on `tokenize`; differences smaller than that, or inside the
usual 5–10% laptop spread, are not claimed.

### Changed

- **L2 normalisation is a SIMD double reduction plus one scale into `out`.**
  It used to `scale_copy` with `1.0` and then walk the vector twice in scalar
  double. On this model `dim=512` and short in-vocabulary English, that L2
  slot was about 19% of an ASCII `embed_batch` sample tree. The fused path
  has a NEON/`__aarch64__` kernel and an SSE2 kernel; the scalar tail is
  unchanged. ASCII batch went 307k → 338k texts/s on the M1 Pro (+10%). The
  SSE2 path is compiled, not timed.

- **The AArch64 f16 top-k kernel decodes 16 halves per iteration, not 8.**
  The x86 F16C kernel is untouched. Same 50 000-row, `k=10`, ASCII matrix:

  ```text
                       f32        f16
  x86-64, f16c       3.00 ms    1.65 ms    f16 1.8x faster   (0.1.2, F16C)
  M1 Pro, neon-fp16  2.36 ms    1.81 ms    f16 1.3x faster   (0.1.4)
  ```

  The M1 Pro row used to read 2.29 / 2.88 ms, f16 1.3x *slower*. `docs/PERFORMANCE.md`
  and the README follow the new numbers. On the lookup-table fallback f16 is
  still usually slower than f32.

- **Scratch buffers are reused per OS thread, and freed when the thread
  exits.** `embed`, `tokenize`, `embed_batch` and `embed_token_ids` used to
  `malloc`/`free` a scratch set on every call. A first cut of 0.1.4 used
  `__thread` storage with no destructor, which leaked the heap buffers inside
  the slot — bounded for a Puma pool, unbounded on the fiber-scheduler path
  that `rb_thread_create`s one OS thread per large call. The slot is now a
  `pthread_key` / `FlsAlloc` value whose destructor frees it. On release the
  slot is trimmed back to the reserve sizes, so one `max_tokens: false`
  document does not pin megabytes on that thread until process exit.
  `tokenize` and the small `embed` path release the slot through `rb_ensure`,
  so `rb_ary_push` raising cannot leave `in_use` stuck. `memory_smoke` calls
  acquire/release so ASan/valgrind actually see this code.

- **ASCII WordPiece copies an all-ASCII word as bytes** instead of
  encoding UTF-8 through the codepoint buffer, then hashes and splits on the
  trie with identity offsets. Output is the same ids. On the OOV tokenize
  probes this is inside the run-to-run floor.

- **Latin-1 (`U+0080..U+00FF`) lower/NFD/Mn/P/C/Zs tables are built at
  load.** Codepoints below 256 skip the binary search into the mmap'd maps.
  ASCII (`< 0x80`) still uses the existing 128-entry class table. Not visible
  as texts/s on the English batch corpus.

- **Vocabulary compare is an inline unaligned equality test** instead of
  libc `memcmp`. Trie child lookup special-cases one- and two-edge nodes
  before the linear/binary search. Same ids; no measured throughput claim.

Two things were measured and not shipped. Unrolling `add_row` to 32 floats
and stretching the prefetch distance to 8 cost about 13% on
`random_pooling_hot_path` — that loop is already memory-bound on random 2 KiB
row gathers. Walking the ASCII trie without `cps2` (`start += matched_len`)
cost about 38% on the OOV tokenize probes.

### Added

- **Batch samples print `mean_tokens_per_text`, `unk_ratio` and
  `tokens_per_sec`.** 307k vs 103k texts/s is not readable without token
  density: on this corpus ASCII is 31 tokens/text and 0 unk, hashes 73 and 0,
  unicode 68 and 4.4% unk.

- **`test/scratch_tls_test.rb`.** When the instrumented allocator is loaded:
  scratch bytes plateau across eight generations of helper threads, and a
  large `max_tokens: false` embed must not leave the calling thread's slot
  pinned. The `alloc_stats` CI job is what actually runs it.

## 0.1.3 (unreleased)

Hardening and tooling, mostly borrowed from the sibling `tg_geometry` gem after
reading its C extension side by side with this one. No behaviour changes to
tokenization, pooling or search: the differential fuzz digest against
`StaticEmbeddings::Reference` is unchanged.

### Fixed

- **Unknown keywords are rejected instead of ignored.** `rb_hash_lookup2`
  returns the default for an absent key, so every misspelled keyword used to be
  silently dropped and the caller got a plausible wrong answer. Two cases were
  actively dangerous: `embed(text, fromat: :f16)` returned a full-width f32
  vector, and `embed_batch(texts, threds: 4)` slipped past the guard whose only
  job is to say that `threads:` buys nothing. All eight keyword-accepting
  methods now validate against a whitelist and name the accepted keywords in the
  error.

  ```ruby
  model.embed("hi", fromat: :f16)
  # ArgumentError: unknown keyword: :fromat
  #   (accepted: :max_tokens, :format, :validate_encoding, :threads)
  ```

- **`memsize` no longer charges the mapped model to the Ruby heap.** A mapped
  `.semb` is file-backed, shared between forked workers and reclaimable by the
  kernel, but `ObjectSpace.memsize_of(model)` reported its full size, which fed
  GC heuristics memory the VM could neither free nor account for. It now reports
  only process-owned bytes; `mapped_bytes` still reports the mapping. The
  Windows heap fallback really is process memory and is still counted.

### Added

- **`benchmark/` with a shared harness.** Iteration counts are calibrated so
  every measurement runs for at least `BENCH_MIN_SECONDS`, repeats are taken,
  the median is reported, and every row carries `spread_pct` — the best-to-worst
  distance across repeats. This release exists partly because a 6% throughput
  claim in 0.1.2 turned out to be inside that spread. `BENCH_FORMAT=kv` gives
  machine-readable output. Run with `rake benchmark`. Inputs are
  `valid_encoding?`-primed, so rows report `coderange=cached` and measure the
  hot path rather than a string fresh from a file or a database.

- **`benchmark/gvl_threshold.rb`.** `SE_GVL_UNLOCK_THRESHOLD` has never been
  justified by data. The sweep measures both halves of the trade: on one x86-64
  run the step at 2048 bytes cost about 32% latency and four extra allocations,
  and it is also the point where a second Ruby thread starts making progress at
  all. The constant is unchanged; now there is something to argue from.

- **`benchmark/core_paths.rb`,** covering tokenize, single embed, batch embed,
  pooling in isolation and top-k in both formats. It reports single-text embed
  twice, once repeating one string and once rotating a corpus, because the gap
  between them is the cache effect that makes hot-loop numbers unreproducible.

- **`test/gc_compaction_test.rb` and a `gc_compaction` CI job.** `top_k`,
  `embed` and `embed_batch` hand raw pointers into C and release the GVL, so
  compaction moving Ruby objects would show up as wrong numbers rather than a
  crash.

  Getting this test to test anything took two corrections. It first used a
  64-row matrix, which is 2 KiB against a 1 MiB `SE_TOPK_GVL_UNLOCK_THRESHOLD`,
  so the scan never released the GVL and compaction could not interleave with
  it. The same mistake then turned out to be hiding in the embed case: 2038
  bytes of corpus against a 2048-byte `SE_GVL_UNLOCK_THRESHOLD`, ten bytes
  under. Measured with a spinning neighbour thread, both old cases produced zero
  ticks during the call and both fixed ones produce hundreds of millions. Every
  concurrency case now asserts it is over the relevant threshold, because
  without that assertion the test silently stops testing the moment a constant
  or a fixture changes.

- **`docs/LIMITATIONS.md`.** One page answering "does it do X", split into
  decisions and unfinished work: no ANN index, no batched queries, no candidate
  filtering, no int8, no Ractor, no internal parallelism, and so on.

- **`docs/BENCHMARKING.md`,** covering how to read `spread_pct` and the four
  measurement traps this repository has already fallen into.

## 0.1.2 (unreleased)

A tokenizer correctness fix, the ASCII fast path it made safe to write, and an
optional build that accounts for the runtime's own C heap.

External parity against `model2vec.StaticModel` has **not** been re-run for this
version. The DEL fix below changes token ids for any input containing `U+007F`,
so the 0.1.1 audit record no longer covers the shipping runtime. See
`docs/MODEL_AUDIT.md`; that re-run is the release blocker.

### Fixed

- **`U+007F DEL` was not treated as a control character.** `is_control()`
  matched `cp < 0x20` only, so DEL survived `clean_text` and stayed a word byte
  while the reference normalizer drops it. The effect was not one token: the
  word carrying DEL fell out of the vocabulary and collapsed to `[UNK]`.
  A 20 000-case differential fuzz against `StaticEmbeddings::Reference` produced
  4 084 mismatches before the fix and 0 after; an exhaustive sweep of
  `0x00..0x7F` shows DEL was the only divergent ASCII byte.

- **`embed_batch(format: :f16)` allocated a second output buffer.** The f32
  batch buffer was encoded into a freshly allocated half buffer, so an f16 call
  peaked at 1.5x the f32 output and cost more transient C heap than f32 for the
  same work. It now encodes in place. Output is byte-identical; on 5000 x 512
  the transient peak drops from 16.4 MB to 11.3 MB, the same peak the f32 call
  has. Throughput is unchanged: the run-to-run spread on that sample is wider
  than any difference the change makes.

### Added

- **ASCII fast path in the tokenizer.** For `cp < 0x80` the normalizer chain
  resolves the CJK, NFD, combining-mark and case-folding branches identically
  every time, so ASCII runs now go through a 128-entry classification table and
  reserve their codepoint buffer once per word instead of once per character.
  Output is unchanged: the token-stream digest over 20 000 fuzz cases is
  identical with and without it. Roughly 1.7x on `tokenize` and 1.2x on `embed`
  for short English text, 2x on `tokenize` with `max_tokens: false`. Non-ASCII
  input is unaffected.

- **`test/ascii_parity_test.rb`.** Exhaustive `0x00..0x7F` sweep in six
  contexts, boundary codepoints across the ASCII/Unicode edge, deterministic
  differential fuzz against `Reference` (`FUZZ_N`, `FUZZ_SEED`), truncation
  boundaries, and DEL walked across the prefix window so the `embed` path is
  covered and not just `tokenize`. Four of its cases fail on 0.1.1.

- **`test/validate_encoding_test.rb`** and verify coverage in
  `test/format_test.rb`: mode agreement on valid input, invalid bytes inside
  and past the window, cached-broken coderange, non-UTF-8 encodings, batch
  index reporting, tamper detection, truncated and foreign files, and an
  allocation bound proving `verify` does not hold the file in memory.

- **`test/cancellation_timing_test.rb`.** Measures how far past a `Timeout`
  deadline `embed` keeps running. It is excluded from `rake test` because it is
  timing-sensitive; run it with `rake cancellation_timing`.

- **`validate_encoding: :full | :prefix`** on `embed`, `embed_batch`,
  `embed_with_stats` and `tokenize`. `embed` has to establish that its `String`
  is valid UTF-8, and when Ruby has not computed the coderange yet that scan is
  O(total bytes) and runs before the prefix window is chosen — so the
  large-input path was not actually bounded on a freshly read document.
  `:prefix` skips the up-front scan and lets the tokenizer validate the bytes it
  reads: 442 µs to 78 µs on a 3 MB string, against 71 µs for the same string
  with its coderange cached. `:full` stays the default, because `:prefix`
  changes behaviour — invalid bytes past the truncation window are no longer
  seen. A coderange Ruby has already computed is honoured in both modes.

- **Optional C allocation accounting.** Building with
  `STATIC_EMBEDDINGS_ALLOC_STATS=1` wraps the runtime's own allocations and
  exposes per-category bytes and counts, which the sample harness prints as
  `c_alloc.<category>.<metric>`. Default builds do not define the internal
  methods and do not pay for the counters. See `docs/PERFORMANCE.md`.

- **CI jobs `parity_fuzz`, `timing`, `alloc_stats` and `windows_loader`,** plus
  a sanitizer build of the instrumented allocator inside the existing `memory`
  job. The Windows job cross-compiles with mingw-w64 and runs the result under
  wine, because that branch of `se_model_open` is the one path no other job
  builds. `alloc_stats` is separate because it has to `rake clobber` first, and
  clobber removes `tmp/`.

### Changed

- **The ASCII fast path checks the cancellation flag** on the same cadence as
  the main loop. A run of ASCII bytes is consumed inside one C call, so without
  the check a `max_tokens: false` call on a large document ignored its deadline
  until the whole document was tokenized: 0.33 s against a 0.05 s `Timeout` on a
  16 MB input, versus 0.05 s now.

- **`ext/static_embeddings/static_embeddings.c` was split.** The f16 codec and
  kernels moved to `se_f16.c`, the top-k kernels to `se_topk.c`, and the
  allocation counters live in `se_alloc_stats.c`. Shared overflow-checked size
  helpers, `static_assert`s on the mmapped struct layouts, and an explicit
  rejection of big-endian hosts moved into `se_internal.h`.

- **`Format.verify` streams the file.** It read the whole model and then
  duplicated it to zero the checksum field, so verifying a 135 MB artifact
  needed roughly 270 MB of transient `String` before hashing started. Peak is
  now one 1 MiB chunk: measured on a 62 MB model, 433 ms and 280 MB of peak RSS
  down to 249 ms and 30 MB.

- **Windows maps the model instead of reading it into the heap.** The POSIX
  loader has always used `mmap`; Windows did `fopen` plus `fread` into a
  private allocation, so every process paid a full read before the first query
  and a private copy of the whole model. It now uses
  `CreateFileMapping`/`MapViewOfFile`, with the old read path kept as a
  fallback for filesystems that refuse to map. Verified by cross-compiling with
  mingw-w64 and running under wine against a real `.semb`: same `map_size`,
  same vectors as the POSIX build, and `mapped=1`.

- **Throughput figures separate logical from processed bytes.**
  `tools/benchmark.rb` reports how many texts were truncated and refuses a
  ns/byte figure when truncation makes the two differ; the large-input samples
  print a measured `processed_input_mb_per_sec` alongside the logical one.

- **`samples/run_all.sh` captures `rake test`** into `00_test.log`, so a run
  directory can back both "samples green" and "tests green". `TEST=0` skips it.

- **Documentation.** The README no longer implies `embed` is constant-time in
  input size (the prefix window bounds the tokenizer, not the UTF-8 validity
  scan) and no longer claims `f16` is faster, since this repository's own
  samples show it winning on x86-64 F16C and losing on M1 Pro.

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
