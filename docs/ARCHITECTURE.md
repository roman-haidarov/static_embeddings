# Architecture

## The shape of the thing

```
HuggingFace / Model2Vec files          offline, once, on your machine
  tokenizer.json, config.json,
  model.safetensors
        |
        v
  Converter (pure Ruby, strict)   <--- all parsing, all validation, all
        |                              Unicode table generation happens here
        v
  model.semb                       <--- flat, versioned, mmap-able
        |
        v
  C runtime                        <--- mmap + bounds checks + tokenize +
  (no JSON, no safetensors,             lookup + pool. Nothing else.
   no Unicode database, no net)
```

The single most important decision: **the runtime reads only our own format.**
Everything expensive, fragile or security-sensitive about reading third-party
model files happens once, offline, in a language where it is easy to get
right. What remains in C is a bounds-checked mmap and three loops.

The C side is `se_format.c` (mmap and validation), `se_tokenizer.c`,
`se_embed.c`, `se_unicode.c`, `se_f16.c` (half-precision codec and kernels),
`se_topk.c`, `se_alloc_stats.c` (optional allocation counters), and
`static_embeddings.c` for the Ruby bindings. Shared size arithmetic,
`static_assert`s on the mmapped struct layouts and the big-endian rejection live
in `se_internal.h`: the header fields are decoded little-endian explicitly, but
the mmapped structures and the float matrix are read in native order, so a
big-endian host is refused at load rather than silently misread.

## `.semb` v2

Little-endian throughout. 320-byte header, then sections aligned to 64 bytes.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `SEMBv1\0\0` |
| 8 | 4 | format_version |
| 12 | 4 | header_size (320) |
| 16 | 4 | flags |
| 20 | 4 | dim |
| 24 | 4 | vocab_size |
| 28 | 4 | tokenizer_type (1 = BERT_WORDPIECE_V1) |
| 32 | 4 | embedding_dtype (1 = F32) |
| 36 | 4 | pooling_type (1 = MEAN) |
| 40 | 4 | normalization_type (0 none, 1 = L2) |
| 44 | 4 | **max_tokens_default** |
| 48 | 4 | truncation_policy |
| 52 | 4 | add_special_tokens |
| 56 | 4 | unk_policy (0 include, 1 drop) |
| 60 | 4 | empty_policy (0 zero vector, 1 raise) |
| 64 | 4 | do_lower_case |
| 68 | 4 | strip_accents |
| 72 | 4 | handle_chinese_chars |
| 76 | 4 | clean_text |
| 80 | 4 | max_input_chars_per_word |
| 84–100 | 20 | pad/unk/cls/sep/mask ids |
| 104 | 4 | hash_table_size (power of two) |
| 108 | 4 | hash_seed |
| 112 | 4 | subword_prefix_len |
| 116 | 8 | subword_prefix bytes |
| 124 | 4 | max_token_chars |
| 128–207 | 80 | five (offset, size) u64 pairs: `vocab_strings`, `vocab_hash`, `embeddings`, `norm_tables`, `provenance` |
| 208–239 | 32 | two (offset, size) u64 pairs: `root_trie`, `continuation_trie` |
| 240 | 32 | sha256 of the file with these 32 bytes zeroed |
| 304 | 4 | max_probe observed in the vocabulary hash |

Sections, in fixed order: `vocab_strings`, `vocab_hash`, `embeddings`,
`norm_tables`, `provenance`, `root_trie`, `continuation_trie`.

Extension points are the enums and `format_version`, not stub code. `dtype`
already has a slot for I8; there is no dead I8 path in C.

### Why the semantics live in the header

`max_tokens_default`, `unk_policy`, `empty_policy` and the normalizer flags
are properties of the *model*, not options the caller guesses. The reference
implementation truncates at 512 tokens; a model card's `model_max_length` of
1,000,000 is a different number entirely, and confusing them silently changes
every vector for long chunks. Anything that can silently change output is a
recorded field.

## Vocabulary lookup and WordPiece trie

The converter still writes a finished open-addressing hash table into the
file: 16-byte slots of `{hash, str_off, str_len, token_id}`, power-of-two size,
load factor ≤ 0.70, linear probing, FNV-1a with a fixed seed. The hash table is
kept for validation, direct lookup and debugging.

The hot WordPiece path no longer performs repeated hash lookups for every
`end--` candidate. Format v2 also stores two converter-built, mmap-readable
tries inspired by double-array trie libraries such as libdatrie:

- `root_trie` for tokens that can start a word;
- `continuation_trie` for `##token` entries stored without the `##` prefix.

At runtime `append_wordpiece` first tries the hash table for the whole word,
which is the common case for in-vocabulary text, and falls back to walking the
relevant trie once, recording the longest terminal node. Both halves earn their
place: removing the hash pre-check made tokenization several times slower on
ordinary text, and replacing the trie with repeated hash lookups by decreasing
length made out-of-vocabulary words several times slower still.

Normalization has an ASCII fast path. Below `0x80` the CJK, NFD, combining-mark
and case-folding branches always resolve the same way, so those runs go through
a 128-entry classification table instead of the full
`emit_cleaned -> emit_stripped -> emit_lowered -> feed_token_cp` chain, and a
word reserves its codepoint buffer once rather than once per character. The
table is generated to agree with `is_control`, `se_is_ascii_whitespace` and
`se_is_ascii_punct`, and `test/ascii_parity_test.rb` sweeps every byte in
`0x00..0x7F` against the Ruby reference to keep it that way. That test exists
because `U+007F DEL` was misclassified through 0.1.1 and every sampled test in
the suite passed anyway.

Loading a model still performs **zero runtime insertions**. It is `mmap` plus a
validation pass over the hash table, trie node ranges and trie edge ordering.
The vocabulary structures live in the page cache, shared between every forked
Puma worker.

Rejected alternatives:

- **khash / any runtime hash map.** Would rebuild 250k entries at every boot
  and allocate outside the mapping — exactly what the offline format exists
  to avoid.
- **Sorted table + binary search.** ~18 cache misses per probe at 250k
  entries versus one or two for open addressing.
- **Vendored trie runtime.** MARISA, cedar and libdatrie are useful references,
  but the runtime needs a small `.semb` section that can be bounds-checked and
  mmap-read directly, not an external object format or system dependency.

## Unicode without a Unicode library

`BertNormalizer` needs NFD, simple lowercase, and the Mn / P / C / Zs
categories. Rather than vendoring utf8proc, the converter generates exactly
those tables from the Ruby VM's own Unicode data and writes them into the
model file; the runtime binary-searches them.

This removes the last third-party dependency, and it sidesteps a specific
trap: utf8proc's case *folding* is not the same as case *lowering*. Folding
maps `ß` to `ss`; `str::to_lowercase`, which HF uses, does not. A gem that
picked the wrong one would produce plausible, slightly wrong vectors forever.

The Ruby build that generated the tables is stamped into provenance, so a
model file and its normalizer can never drift apart unnoticed.

## The embedding kernel

```
tokenize -> ids
truncate to max_tokens          (before pooling — the reference contract)
drop [UNK] if unk_policy = DROP
acc[j] += row[j] for each id, in token order
divide by the number of pooled rows
L2 normalize if the model says so
```

Deliberately absent:

- **Deduplicating repeated token ids** and multiplying by a count. It saves
  memory traffic and changes the float rounding, in the one place where we are
  trying to stay inside 1e-6 of a reference. Not worth it.

The runtime computes embeddings in float32. Returned storage is selected by the
caller's `format:` option: `:f32` returns the native 4-byte components and `:f16`
encodes the same vector as little-endian IEEE float16 components. This is a
storage/transport choice, not a different model. `embed_array` and
`embed_batch_arrays` decode whichever storage format was requested back to Ruby
`Float` values.

The f32 compute path uses small explicit SIMD kernels where they keep the format
simple: row accumulation and scaling use NEON/SSE when the compiler target
exposes them, with scalar fallback everywhere else. `dot_top_k`/`cosine_top_k`
share the SIMD dot kernel for `format: :f32`; `cosine_top_k` additionally
accumulates each row's sum of squares so it can divide by the true norms.
`format: :f16` top-k decodes half components on the fly. What is still
deliberately absent is a dependency on BLAS: pooling is gathering random
embedding rows, not a dense matrix multiply.

A matrix at or above `SE_TOPK_GVL_UNLOCK_THRESHOLD` (1 MiB) is scanned with the
GVL released and must therefore be frozen. The alternatives were both bad:
copying a corpus-sized blob on every query defeats the purpose of the API, and
`rb_str_locktmp` is an exclusive lock, so it turns a read-only shared corpus
into something exactly one thread at a time may search. Freezing removes the
mutation hazard outright and keeps the scan lock-free. Because a raw blob has no
dimension or format tag, `dim:` is required and the query length is checked
against it; that is what stops an f32 query and an f16 matrix from dividing out
to a consistent-looking row count.

Top-k insertion uses `!(score > worst)` rather than `score <= worst` so that a
`NaN` score is skipped. With the naive comparison a single `NaN` row would fall
through the early exit, settle at the last slot and then make every subsequent
comparison false, turning an O(1) reject into an O(k) insert for the rest of the
scan and leaking `NaN` into the result.

## Large inputs and the prefix window

Ruby strings must be copied into C-owned memory before the GVL is released, and
that copy runs with the GVL held. Copying a whole multi-megabyte document
therefore serialised the process on work that the tokenizer would discard after
`max_tokens` anyway.

`embed` and `embed_batch` instead copy a leading slice. The budget starts at
`max_tokens * 16` bytes clamped to `[4096, 65536]`, and the slice is cut at a
position where the tokenizer would have started a fresh word. A result is
accepted only when that slice actually reached `max_tokens`; otherwise the
budget doubles and the text is tokenized again. Growth is geometric, so the
pathological case costs about twice the single-pass work.

Three properties make this exact rather than approximate:

- The scan only stops on UTF-8 lead bytes, so the cut always lands on a
  codepoint boundary.
- WordPiece segmentation is word-local, so the token stream of a prefix cut at a
  word boundary is a prefix of the token stream of the whole document.
- Normalization carries no state across a word boundary: combining marks are
  dropped independently, CJK characters are wrapped in spaces individually, and
  control characters are deleted pointwise.

The boundary predicate is `se_prefix_boundary_len`, and it lives in
`se_tokenizer.c` next to the tokenizer so it can call the very same
`is_whitespace` / `is_punct` / `se_is_cjk` predicates rather than a parallel
copy of them. A position is a legal cut when the codepoint ending there is
whitespace or punctuation, or when either side of it is a CJK codepoint - CJK is
space-wrapped by `emit_cleaned`, so both of its edges are word boundaries.
Codepoints that the normaliser rewrites (lowercase map, NFD map, dropped
combining marks) are skipped: their class after normalisation is not necessarily
their class before it.

Restricting the cut to ASCII, as 0.1.0 did, was correct but degenerate: a CJK or
NBSP-separated document has no ASCII byte anywhere, so the budget grew to the
full length and the whole input was copied under the GVL after all. That is why
the predicate is Unicode-aware. Text with genuinely no legal cut inside the scan
window - one enormous word, a run of combining marks - still falls back to a
full copy, because there is nothing else that would be correct.

`test/prefix_chunking_test.rb` sweeps every printable ASCII separator, checks
CJK, NBSP and Unicode-punctuation documents against
`embed_token_ids(tokenize(text))`, and asserts that quadrupling the input does
not triple the cost.

With `max_tokens: false` there is nothing to truncate and the whole input is
copied.

The window bounds tokenizing, not the whole call. Ruby has to answer whether the
`String` is valid UTF-8, and `rb_enc_str_coderange` answers it for the whole
`String` — an O(bytes) scan when Ruby has not computed a coderange yet, running
before the prefix is even chosen. `validate_encoding: :prefix` skips that scan
and leans on the fact that the C side already validates what it reads:
`decode_one` rejects malformed UTF-8, and `se_prefix_boundary_len` returns 0
when it cannot decode, which degrades to a full copy rather than reading out of
bounds. So `:prefix` is memory-safe in every case; what it gives up is noticing
invalid bytes the tokenizer never reaches. A cached coderange is always
honoured, which keeps the cheap answer authoritative when Ruby already has one.

## Mapping the model file

POSIX uses `mmap` with `PROT_READ`/`MAP_PRIVATE`; Windows uses
`CreateFileMapping` plus `MapViewOfFile`. Both give the same three properties
the format was designed around: pages fault in lazily instead of being read up
front, the page cache is shared between forked or sibling processes, and the
runtime never owns a writable copy of the matrix. Windows previously read the
file into the heap, which cost every process a private copy of the model and a
full read before the first query.

Windows keeps the heap path as a fallback when mapping fails, since some network
filesystems refuse it, and `model->mapped` records which one was taken so
`se_model_close` unmaps or frees correctly. `verify` is separate from loading and
streams the file through SHA-256 in chunks, so checking an artifact never
materialises it either.

## Concurrency

Small calls run inline. Large `embed_batch` calls copy Ruby input into C-owned
memory and then run outside the GVL. This is enough for Puma's threaded model:
one request can compute embeddings while other Ruby threads keep running.

When a fiber scheduler is installed, large batches are handed to one Ruby
thread before the caller waits, so the reactor is not pinned by the C loop. This
costs one OS thread per call, so under high concurrency batching beats many
small `embed` calls. The runtime intentionally does not expose internal
`threads:` parallelism; offline indexing should split work at the
job/application layer.

Cancellation is cooperative: the tokenizer checks a flag every 1024 codepoints
(counted by iteration, not by byte offset, so the interval does not depend on
how wide the input's codepoints are), and the ASCII fast path checks on the same
cadence in bytes so a long ASCII run is no less interruptible. The pooling loop
checks every 256 rows, and the top-k scan every 1024 rows. `unblock_cancel` sets that flag when Ruby
interrupts a GVL-free region.

No global mutable state exists in C. The model is immutable after load and
scratch buffers are per call.

## Where this fits in a RAG pipeline

Dense retrieval with static embeddings is a weaker first stage than a
transformer. The cheap fix is fusion, not a bigger model: BM25 and dense
vectors fail in different ways — BM25 misses paraphrases, dense misses rare
identifiers and exact phrases — and Reciprocal Rank Fusion over both recovers
most of the difference at zero additional cost, since SQLite's FTS5 gives you
BM25 for free.

```
chunks -> FTS5 (BM25) ---\
                          >-- RRF --> top 20 -> optional cross-encoder rerank
chunks -> .semb vectors --/
```

## Open questions before v1.0

1. **Which potion models actually match `BERT_WORDPIECE_V1`.** The 32M family
   has a larger vocabulary than the bge-base tokenizer it was distilled from,
   which means tokens were added somewhere. If they live in `added_tokens`
   rather than `model.vocab`, HF matches them with a separate trie pass over
   raw text that this runtime does not implement. The converter refuses such
   models today; whether it has to is an audit question.
2. **Russian.** Distilling a multilingual teacher into a WordPiece vocabulary
   we control keeps the pure-C path, but skips the Tokenlearn pre-training
   that gives the published potion models their quality. That gap has to be
   measured before the multilingual story is real.
3. **int8.** The format has the slot; the runtime does not have the path.
   f32 stays the reference forever regardless; f16 is only a returned storage
   encoding.
