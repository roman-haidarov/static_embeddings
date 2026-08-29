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
| 0.1.2 | **not re-run** | required before release |

0.1.2 changed `is_control()`, which changes token ids for any input containing
`U+007F`. `StaticEmbeddings::Reference` cannot settle whether the new behaviour
matches HuggingFace, because it is an implementation twin of the C runtime
written in this repository. Only the oracle can, and the oracle now carries DEL
and control-character rows it did not have when the result below was recorded —
so `rows=28` would itself be evidence of a stale run.

Source model:

- Hugging Face repository: `minishlab/potion-retrieval-32M`
- Oracle implementation: `model2vec.StaticModel.from_pretrained`
- Python package: `model2vec 0.9.0`
- Oracle file: `tmp/model2vec_oracle.json`
- Oracle rows in this recorded run: `28`
- Oracle dimension: `512`
- Oracle max length: `512`
- Runtime at time of this record: `static_embeddings 0.1.1`

Converted `.semb`:

- Path: `$HOME/.cache/static_embeddings/models/potion-retrieval-32m.semb`
- Format version: `2`
- Header size: `320`
- Bytes: `135411608`
- SHA256: `79e087863d2bab825779fd7de3574e5625542ef6a5ecad33fe681ea16d4b3ab0`

Parity command:

```bash
bundle exec rake parity \
  MODEL="$HOME/.cache/static_embeddings/models/potion-retrieval-32m.semb" \
  ORACLE=tmp/model2vec_oracle.json
```

Parity result:

```text
rows=28
id_rows_checked=28
min_cosine=0.9999999999989528
max_abs_all=2.980232238769531e-07
token_id_failures=[]
vector_failures=[]
parity OK
```

Decision:

- Tokenization parity: pass
- Vector parity: pass
- Empty input behavior: pass
- Whitespace behavior: pass
- Unicode normalization behavior: pass
- Unknown-word behavior: pass
- Long input / truncation behavior: pass

Accepted for runtime and benchmark use **under 0.1.1**. The `.semb` file is
unchanged and its SHA256 still matches; what expired is the runtime half of the
record.
