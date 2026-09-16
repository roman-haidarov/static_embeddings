# Model Audit

A converted model is trusted only after a recorded comparison against an
external upstream oracle. An audit record covers one source-model snapshot,
one `.semb` format/runtime contract, pinned upstream package versions, and the
specific corpus that was checked. A passing corpus is evidence for those rows;
it is **not** an exhaustive proof that every Unicode string is equivalent.

`StaticEmbeddings::Reference` remains useful for high-volume differential fuzz
of the C implementation, but it is an implementation twin and is not counted as
independent upstream evidence.

## potion-retrieval-32m

| runtime | external result | note |
|---|---|---|
| 0.1.1 | 31-row corpus passed | superseded |
| 0.1.2 | not re-run | superseded before release |
| 0.1.3 | 31-row corpus passed | superseded |
| 0.1.4 | 31-row corpus passed | historical result below; later review found uncovered boundary cases |
| 0.1.5 | **434-row corpus passed** | format v3; CI `potion_audit` record below |

### Historical 0.1.4 record

Source model:

- Hugging Face repository: `minishlab/potion-retrieval-32M`
- Hugging Face snapshot: `6fc8051fab2a1e0ee76689cf08c853792ac285e7`
- Oracle implementation: `model2vec.StaticModel.from_pretrained`
- Python package: `model2vec 0.9.0` (`tokenizers 0.23.1`, `numpy 2.5.2`)
- Oracle rows in this recorded run: `31`
- Oracle dimension: `512`
- Oracle max length: `512`
- Runtime at time of this record: `static_embeddings 0.1.4`

Converted `.semb`:

- Format version: `2`
- Header size: `320`
- Bytes: `135411608`
- SHA256: `79e087863d2bab825779fd7de3574e5625542ef6a5ecad33fe681ea16d4b3ab0`

Recorded result (2026-08-31):

```text
rows=31
id_rows_checked=31
min_cosine=0.9999999999989528
max_abs_all=2.980232238769531e-07
token_id_failures=[]
vector_failures=[]
parity OK
```

The last line means **31/31 rows in that historical oracle passed**. It must not
be read as a global tokenizer-equivalence claim. The corpus did not cover, for
example, the corrected UNK-before-truncation ordering, standard AddedVocabulary
literals, or the Rust/Python CJK Extension E boundary.

## 0.1.5 record

0.1.5 separates three contracts that the old oracle mixed together:

1. raw Hugging Face `tokenizers 0.23.1` ids, including `[UNK]`;
2. this runtime's usable-id embedding contract (`[UNK]` drop, then token cap);
3. `model2vec.StaticModel` vectors where its character pre-cut does not change
   the usable token sequence.

Rows changed solely by Model2Vec's `max_length * median_token_length` character
pre-cut are reported as intentional deviations rather than hidden inside a pass.
A passing corpus is evidence for those 434 rows, not an exhaustive proof of
every Unicode string.

Recorded from GitHub Actions (`potion_audit`, Python 3.12.14, pinned
`model2vec==0.9.0` / `tokenizers==0.23.1` / `numpy==2.5.2`):

Source model:

- Hugging Face repository: `minishlab/potion-retrieval-32M`
- Hugging Face snapshot: `6fc8051fab2a1e0ee76689cf08c853792ac285e7`
- Oracle implementation: `model2vec.StaticModel.from_pretrained`
- Oracle rows: `434` from `tools/parity_cases.py`
- Oracle dimension: `512`
- Oracle max length: `512`
- Runtime at time of this record: `static_embeddings 0.1.5`

Converted `.semb` (CI artifact; Unicode tables stamped from the Ubuntu Ruby that
converted it, so the SHA is not expected to match a macOS local convert of the
same snapshot):

- Format version: `3`
- Bytes: `135411800`
- SHA256: `747231b5afbcb3b16bf2b04538f81d3a96be88a798982214b7cf01eddbdcf4eb`
- `dim=512` `vocab=63091`

Recorded result:

```text
rows=434
vectors_checked=432
intentional_character_pretruncate_deviations=2
min_cosine=0.9999999999999989
max_abs_all=1.4901161193847656e-08
raw_token_id_failures=[]
usable_token_id_failures=[]
embed_invariant_failures=[]
vector_failures=[]
corpus parity OK (434/434); intentional Model2Vec character pre-truncation deviations are reported separately
```

The two intentional deviations are `long:sparse-whitespace` and
`long:unknown-prefix`. Vectors were not required to match Model2Vec on those
rows. The remaining 432 rows were inside `cosine >= 1 - 1e-6` and
`max_abs <= 1e-5`.

The preceding CI job `upstream_parity` ran the same 434-row corpus against the
synthetic `tiny-wordpiece` fixture: `434/434`, all four failure lists empty,
`vectors_checked=429`, `intentional_character_pretruncate_deviations=5`,
`min_cosine=0.9999999999999988`, `max_abs_all=5.960464477539063e-08`. That job
proves the fixture loads in `StaticModel.from_pretrained` and that the checker
contracts hold; it is not a potion audit.

## 1.5.6 Patch 1 proof

Architecture: Model2Vec / Sentence Transformers import functions → immutable canonical data →
existing WordPiece `.semb` v3 writer. C runtime unchanged.

### potion-retrieval-32M reconvert

Official snapshot `6fc8051fab2a1e0ee76689cf08c853792ac285e7` ships a Sentence
Transformers `modules.json` (`StaticEmbedding` + `Normalize`). Detection uses
`config.json` `model_type=model2vec` first, so this stays UNK_DROP / L2 / 512.

New file `potion-retrieval-32m-reconvert.semb` vs existing
`potion-retrieval-32m-v3.semb`: vocab/hash/embeddings/norm_tables/tries
byte-identical. 10-text corpus `max_abs=0`. Provenance JSON differs (new keys).

### static-retrieval-mrl-en-v1 vs SentenceTransformer.encode

Oracle: `tools/st_oracle.py` + `tools/check_st_parity.rb`,
`sentence-transformers 6.0.1`, `add_special_tokens=false`,
`normalize_embeddings=false`, 436 rows including `all-unk:private-use` and
`long:known-words-600`.

```text
1024: 436/436  min_cosine=0.999999999999999  max_abs_all=7.62939453125e-06
512:  436/436  min_cosine=0.9999999999999989 max_abs_all=7.62939453125e-06
```

512 compared as prefix of the 1024-d encode vectors. Ruby Reference matched C
separately. Diagnostic `StaticEmbedding.forward` matched encode on this model.

### UNK and max_tokens on the real models

- potion-retrieval-32M `🧬 🧬 🧬` → UNK ids dropped → zero vector; 600×`hello`
  → `truncated=true pooled=512`.
- static-retrieval-mrl-en-v1-1024 same emoji string → three UNK ids pooled
  (non-zero, mean of UNK rows); 600×`hello` → `truncated=false pooled=600`.
- Private-use codepoints are BertNormalizer `Co` and become empty, not UNK.

## 1.5.6 Type A conversions

These records prove the offline import layer maps extra WordPiece sources onto
the existing runtime. They are **not** a replacement for the
potion-retrieval-32m Python oracle above. `static-retrieval-mrl-en-v1` has the
separate `SentenceTransformer.encode` oracle recorded above; the multilingual
similarity artifact was checked against the Ruby `.semb` Reference twin and the
small Russian retrieval fixture, not claimed as a general upstream retrieval
benchmark.

All conversions used `static_embeddings 1.5.6` on arm64-darwin24.

### minishlab/potion-base-8M

- Snapshot: `bf8b056651a2c21b8d2565580b8569da283cab23`
- Layout: Model2Vec (`UNK_DROP`, L2, `max_tokens` 512)
- `.semb`: `potion-base-8m.semb` `dim=256` `vocab=29528` `33346200` bytes
  SHA256 `8abd8d1f26511959e14ca26ec4dcacad1d3db9b4b66b2e67683fb2d06074f490`
- Native C matched Ruby Reference on short English/OOV/empty texts (`max_abs < 1e-5`)
- `model2vec 0.9.0` / `tokenizers 0.23.1` oracle: 434/434, vectors_checked=432,
  intentional character-pretruncate deviations=2, min_cosine=0.9999999999999989,
  max_abs_all=2.98e-08

### minishlab/potion-science-32M

- Snapshot: `7366079845507de14a4330007cdfa01bb92bca52`
- Layout: Model2Vec (`UNK_DROP`, L2, `max_tokens` 512)
- `.semb`: `potion-science-32m.semb` `dim=256` `vocab=124428` `140353240` bytes
  SHA256 `e968f93b89d59d9c63c6a8152495283d1138144e61ed5c08d306c21c00b7cdb1`
- Native C matched Ruby Reference on English and Russian snippets (`max_abs < 1e-5`)
- `model2vec 0.9.0` / `tokenizers 0.23.1` oracle: 434/434, vectors_checked=432,
  intentional character-pretruncate deviations=2, min_cosine=0.9999999999999986,
  max_abs_all=2.98e-08

### sentence-transformers/static-retrieval-mrl-en-v1

- Snapshot: `f60985c706f192d45d218078e49e5a8b6f15283a`
- Layout: Sentence Transformers StaticEmbedding (`UNK_INCLUDE`, no L2, unlimited)
- `static-retrieval-mrl-en-v1-1024.semb` `dim=1024` `vocab=30522` `128186264` bytes
  SHA256 `f316bb07348503418ed0f6d897ddbf7b44422d2077ca4860518f31f328610946`
- `static-retrieval-mrl-en-v1-512.semb` `--dimensions 512` `65677208` bytes
  SHA256 `efbb43c14def793d0bbb42ef591a0951caed23a49cdd94dcf72312569d1b7cff`
- Prefix of a 1024-d vector matched the 512-d artifact on in-vocabulary English
  (valid because this source does not L2-normalize)
- Native C matched Ruby Reference on short texts

### sentence-transformers/static-similarity-mrl-multilingual-v1

- Snapshot: `b68f4122911bcffcd6e1f695f2d99cd6788972d8`
- Layout: Sentence Transformers StaticEmbedding (`UNK_INCLUDE`, no L2, unlimited)
- `static-similarity-mrl-multilingual-v1-512.semb` `--dimensions 512` `vocab=105879`
  `227983824` bytes SHA256 `8d9a63d20ee23468ae2fd2584ef3f0ec75e720e68410e416157aaf2b873e110e`
- `static-similarity-mrl-multilingual-v1-256.semb` `--dimensions 256` `119563728` bytes
  SHA256 `2e9e689bd3afd18ed50a77f0f8842b4b5a5be39bfad75e3fa24ad99b478de068`
- Native C matched Ruby Reference on Russian text (`max_abs 9.1e-7`)

This model is published as similarity, not retrieval. `tools/eval_retrieval.rb`
on `test/fixtures/russian_faq_eval.json` (10 queries / 10 docs, cosine@10,
arm64-darwin24, 1.5.6):

| model | dim | MRR | nDCG@10 |
|---|---:|---:|---:|
| static-similarity-mrl-multilingual-v1-256 | 256 | 0.875 | 0.906 |
| static-similarity-mrl-multilingual-v1-512 | 512 | 0.850 | 0.889 |
| static-retrieval-mrl-en-v1-512 | 512 | 0.792 | 0.842 |
| potion-retrieval-32m | 512 | 0.733 | 0.799 |
| potion-base-8m | 256 | 0.712 | 0.780 |

Hit@10 was 1.0 for every model on this tiny labeled set. That is a domain sanity
check, not a public retrieval benchmark, and it does not make the similarity
model a hard-coded Russian default.
