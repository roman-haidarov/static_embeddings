# Model Audit

This document records the trust and parity status of converted `.semb` models.
A converted production model is accepted only after parity against the upstream
Python `model2vec.StaticModel` implementation is recorded here.

## potion-retrieval-32m

Status: audited / accepted

Source model:

- Hugging Face repository: `minishlab/potion-retrieval-32M`
- Oracle implementation: `model2vec.StaticModel.from_pretrained`
- Python package: `model2vec 0.9.0`
- Oracle file: `tmp/model2vec_oracle.json`
- Oracle rows: `28`
- Oracle dimension: `512`
- Oracle max length: `512`
- Runtime checked with: `static_embeddings 0.1.1`
- Runtime source note: re-run parity after changing tokenizer, normalizer, prefix-window, pooling, or output-normalization code.

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

This converted model is accepted for runtime and benchmark use.
