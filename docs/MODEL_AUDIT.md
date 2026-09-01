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
