# Model Audit

A converted model is trusted only after parity against upstream
`model2vec.StaticModel` is recorded here. An audit record covers one `.semb`
file **and** one runtime version: changing the tokenizer, normalizer, prefix
window, pooling, output normalization or the oracle's corpus invalidates the
runtime half of it even though the file bytes are untouched.

## potion-retrieval-32m

| runtime | parity | note |
|---|---|---|
| 0.1.1 | `parity OK`, recorded below | superseded |
| 0.1.2 | **not re-run** | superseded before release |
| 0.1.3 | `parity OK` | tokenizer / pooling contract; same numbers as 0.1.4 |
| 0.1.4 | `parity OK`, recorded below | current |

0.1.2 changed `is_control()`, which changes token ids for any input containing
`U+007F`. `StaticEmbeddings::Reference` cannot settle whether the new behaviour
matches HuggingFace, because it is an implementation twin of the C runtime
written in this repository. The 0.1.3 release candidate was checked against a
fresh upstream `model2vec.StaticModel` oracle that includes DEL, control
characters, Unicode, OOV, long-word and truncation rows. 0.1.4 re-ran that
oracle against the SIMD L2 runtime on the same snapshot and `.semb`.

Source model:

- Hugging Face repository: `minishlab/potion-retrieval-32M`
- Hugging Face snapshot: `6fc8051fab2a1e0ee76689cf08c853792ac285e7`
- Oracle implementation: `model2vec.StaticModel.from_pretrained`
- Python package: `model2vec 0.9.0` (`tokenizers 0.23.1`, `numpy 2.5.2`)
- Oracle file: `tmp/model2vec_oracle.json`
- Oracle rows in this recorded run: `31`
- Oracle dimension: `512`
- Oracle max length: `512`
- Runtime at time of this record: `static_embeddings 0.1.4`

Converted `.semb`:

- Path: `$HOME/.cache/static_embeddings/models/potion-retrieval-32m.semb`
- Format version: `2`
- Header size: `320`
- Bytes: `135411608`
- SHA256: `79e087863d2bab825779fd7de3574e5625542ef6a5ecad33fe681ea16d4b3ab0`

Parity command:

```bash
python tools/model2vec_oracle.py \
  ~/.cache/huggingface/hub/models--minishlab--potion-retrieval-32M/snapshots/6fc8051fab2a1e0ee76689cf08c853792ac285e7 \
  --out tmp/model2vec_oracle.json

bundle exec rake parity \
  MODEL="$HOME/.cache/static_embeddings/models/potion-retrieval-32m.semb" \
  ORACLE=tmp/model2vec_oracle.json
```

Parity result (0.1.4, 2026-08-31):

```text
rows=31
id_rows_checked=31
min_cosine=0.9999999999989528
max_abs_all=2.980232238769531e-07
token_id_failures=[]
vector_failures=[]
parity OK
```

The printed `min_cosine` and `max_abs_all` are identical to the 0.1.3 record.
Worst vector row is still idx=29 (`"word" + 400 spaces`, 80 800 bytes) at
`max_abs=2.9802322e-07`. SIMD pairwise double L2 did not move the oracle
agreement on this corpus.

Decision:

- Tokenization parity: pass
- Vector parity: pass
- Empty input behavior: pass
- Whitespace behavior: pass
- Unicode normalization behavior: pass
- Unknown-word behavior: pass
- Long input / truncation behavior: pass
- DEL / control-character behavior: pass

Accepted for runtime and benchmark use **under 0.1.4**. The `.semb` file is
unchanged and its SHA256 still matches. The runtime half of the record was
refreshed after the 0.1.4 L2 kernel change.
