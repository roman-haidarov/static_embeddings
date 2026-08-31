# Limitations

`static_embeddings` is a runtime for one narrow thing: turning text into a
vector with a converted Model2Vec / potion model, and scanning a matrix of such
vectors. It is not a vector database, not a general embedding library, and not a
tokenizer toolkit.

This page is the answer to "does it do X". If X is here, the answer is no, and
the entry says whether that is a decision or just unfinished work.

## Models

- **Only `.semb` files from this repository's converter.** Loading a
  HuggingFace directory at runtime is not supported and will not be. The
  converter is the only thing that reads third-party files, and it runs offline.
- **Only the `BERT_WORDPIECE_V1` tokenizer profile.** SentencePiece, BPE, Unigram
  and byte-level tokenizers are rejected at conversion time rather than
  approximated.
- **No model training, distillation or fine-tuning.** Produce the model with
  upstream `model2vec`, then convert it.
- **No transformer inference.** Static embeddings have no attention and no
  context; a word contributes the same row wherever it appears. If you need
  contextual embeddings, this is the wrong tool.

## Runtime

- **No internal parallelism.** `threads:` is rejected rather than ignored. Split
  work at the job level and run more application workers.
- **No Ractor support.** Model, and everything holding native state, are not
  shareable; `Ractor.make_shareable` raises.
- **No GPU, no BLAS, no external SIMD library.** The kernels are plain C with
  compiler autovectorisation, plus a hand-written half-precision path.
- **No streaming input.** `embed` and `embed_batch` take Ruby `String`s that are
  already in memory.
- **`embed_batch` peaks at roughly twice the returned blob**, because the result
  is built in C memory and then copied into a Ruby `String`. Chunk large
  corpora.
- **Big-endian hosts are refused at load.** The header is decoded
  little-endian explicitly, but the mmapped structures and the float matrix are
  read in native order.

## Search

- **No index.** `dot_top_k` and `cosine_top_k` are exhaustive scans. There is no
  ANN, no IVF, no HNSW, no clustering.
- **No batched queries.** One query per scan; N queries stream the matrix N
  times.
- **No candidate filtering.** You cannot restrict a scan to a subset of rows.
- **No precomputed row norms.** `cosine_top_k` recomputes each row's norm on
  every query. Normalise your rows once and use `dot_top_k` instead.
- **No persistence, no ids, no updates.** A matrix is a frozen `String`; mapping
  rows to documents, appending, and deleting are the caller's problem. If you
  want that, use pgvector and keep this gem for producing the vectors.
- **No metric other than dot product and cosine.** No Euclidean, no Manhattan,
  no Hamming.

## Storage formats

- **`f32` and `f16` only.** No int8, no binary quantisation.
- **`f16` is a storage encoding, not a different model.** Top-k over an f16
  matrix can be faster or slower than f32 depending on the decode kernel; see
  `docs/PERFORMANCE.md`. `embed_batch(format: :f16)` still encodes the blob
  with the scalar converter.
- **`f16` rounding is half-up, not ties-to-even.** Blobs written by this gem can
  differ from NumPy or PyTorch by one ULP on exact halfway values.

## Platforms

- **Windows is built and smoke-tested by cross-compilation under wine, not on a
  real Windows runner.** The loader uses `CreateFileMapping`, with a
  read-into-heap fallback.
- **No JRuby or TruffleRuby.** The gem is a CRuby C extension.

## Verification

- **`load` does not check the SHA-256.** Structural validation always runs, but
  a bit flip inside the matrix is only caught by `verify: true` or
  `StaticEmbeddings.verify`.
- **Parity is per model and per runtime version.** A converted model is not
  trusted until it has a record in `docs/MODEL_AUDIT.md`, and that record
  expires when the tokenizer changes.

## Known open questions

These are not decisions, just work that has not been done or measured:

- The `SE_GVL_UNLOCK_THRESHOLD` value has a benchmark now
  (`rake benchmark:gvl_threshold`) but has not been retuned from it.
- Top-k keeps its candidates in a sorted array, which degrades for large `k`; a
  heap would be better somewhere above a few dozen.
- `madvise(MADV_RANDOM)` and huge pages for the embedding section are untested.
- The C allocation counters cover the runtime's own allocations only; Ruby heap,
  fragmentation and mmap residency are outside them.
