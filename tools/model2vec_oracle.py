"""Generate an external compatibility oracle from pinned Model2Vec/HF code.

The runtime exposes two related contracts:

* Model#tokenize returns raw Hugging Face tokenizer ids (including [UNK]) and
  applies max_tokens to raw ids.
* embedding pools the ids selected by the model's recorded truncation policy:
  - raw_ids    (model2vec >= 0.10.0): cap raw ids, [UNK] included, then drop [UNK];
  - usable_ids (model2vec 0.9.0):     drop [UNK], then cap.

Model2Vec additionally applies a character pre-cut before tokenization. The gem
intentionally does not reproduce that heuristic. Rows where that heuristic
changes Model2Vec's pooled ids are recorded, but strict vector parity is not
claimed for those rows.
"""

import argparse
import json
from importlib.metadata import version

import numpy as np
from model2vec import StaticModel
from tokenizers import Tokenizer

from parity_cases import build_cases

SCHEMA_VERSION = 3
PINNED_MODEL2VEC = "0.10.0"
PINNED_TOKENIZERS = "0.23.3"


def _as_id_rows(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [[int(token_id) for token_id in row] for row in value]


def _version_tuple(text):
    return tuple(int(part) for part in text.split(".")[:3])


def _truncation_policy(model2vec_version):
    return "raw_ids" if _version_tuple(model2vec_version) >= (0, 10, 0) else "usable_ids"


def _model2vec_ids(model, texts, max_length, policy):
    """The ids StaticModel.encode pools, character pre-cut included."""
    if policy == "usable_ids":
        return _as_id_rows(model.tokenize(texts, max_length=max_length))
    cut = max_length * model.median_token_length
    model.max_length = max_length
    return _as_id_rows(model.tokenize([text[:cut] for text in texts]))


def _static_pooled(raw, unk_id, max_length, policy):
    drop = (lambda ids: ids) if unk_id is None else (lambda ids: [i for i in ids if i != int(unk_id)])
    return drop(raw[:max_length]) if policy == "raw_ids" else drop(raw)[:max_length]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("--out", default="tmp/model2vec_oracle.json")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args()

    model2vec_version = version("model2vec")
    tokenizers_version = version("tokenizers")
    if not args.allow_version_mismatch:
        if model2vec_version != PINNED_MODEL2VEC:
            raise SystemExit(f"model2vec {model2vec_version} != pinned {PINNED_MODEL2VEC}")
        if tokenizers_version != PINNED_TOKENIZERS:
            raise SystemExit(f"tokenizers {tokenizers_version} != pinned {PINNED_TOKENIZERS}")
    policy = _truncation_policy(model2vec_version)

    cases = build_cases()
    texts = [case["text"] for case in cases]
    model = StaticModel.from_pretrained(args.source_dir)

    raw_tokenizer = Tokenizer.from_str(model.tokenizer.to_str())
    raw_tokenizer.no_truncation()
    raw_encodings = raw_tokenizer.encode_batch(texts, add_special_tokens=False)
    embeddings = model.encode(texts, max_length=args.max_length)
    model2vec_ids = _model2vec_ids(model, texts, args.max_length, policy)
    unk_id = model.unk_token_id

    rows = []
    for case, encoding, m2v_ids, vector in zip(cases, raw_encodings, model2vec_ids, embeddings):
        raw = [int(token_id) for token_id in encoding.ids]
        pooled = _static_pooled(raw, unk_id, args.max_length, policy)
        rows.append({
            "label": case["label"],
            "text": case["text"],
            "hf_raw_token_ids": raw[: args.max_length],
            "hf_raw_untruncated_length": len(raw),
            "static_pooled_token_ids": pooled,
            "model2vec_token_ids": m2v_ids,
            "model2vec_vector": np.asarray(vector, dtype=np.float32).tolist(),
            "model2vec_character_pretruncate_changes_ids": m2v_ids != pooled,
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "reference": {
            "model2vec": model2vec_version,
            "tokenizers": tokenizers_version,
            "max_length": args.max_length,
            "truncation": policy,
            "unk_token_id": None if unk_id is None else int(unk_id),
        },
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    deviations = sum(row["model2vec_character_pretruncate_changes_ids"] for row in rows)
    dim = len(rows[0]["model2vec_vector"]) if rows else 0
    print(f"wrote {args.out} rows={len(rows)} dim={dim} character_pretruncate_deviations={deviations}")


if __name__ == "__main__":
    main()
