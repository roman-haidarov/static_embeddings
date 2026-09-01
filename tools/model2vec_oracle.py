"""Generate an external compatibility oracle from pinned Model2Vec/HF code.

The runtime intentionally exposes two related contracts:

* Model#tokenize returns raw Hugging Face tokenizer ids (including [UNK]) and
  applies max_tokens to raw ids.
* embedding filters [UNK] first and then applies max_tokens to usable ids.

Model2Vec additionally applies a character pre-cut before tokenization. The gem
intentionally does not reproduce that heuristic. Rows where that heuristic
changes Model2Vec's usable ids are recorded, but strict vector parity is not
claimed for those rows.
"""

import argparse
import json
from importlib.metadata import version

import numpy as np
from model2vec import StaticModel

from parity_cases import build_cases

SCHEMA_VERSION = 2
PINNED_MODEL2VEC = "0.9.0"
PINNED_TOKENIZERS = "0.23.1"


def _as_id_rows(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [[int(token_id) for token_id in row] for row in value]


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

    cases = build_cases()
    texts = [case["text"] for case in cases]
    model = StaticModel.from_pretrained(args.source_dir)

    raw_encodings = model.tokenizer.encode_batch(texts, add_special_tokens=False)
    model2vec_ids = _as_id_rows(model.tokenize(texts, max_length=args.max_length))
    embeddings = model.encode(texts, max_length=args.max_length)
    unk_id = model.unk_token_id

    rows = []
    for case, encoding, m2v_ids, vector in zip(cases, raw_encodings, model2vec_ids, embeddings):
        raw = [int(token_id) for token_id in encoding.ids]
        usable = [token_id for token_id in raw if unk_id is None or token_id != int(unk_id)]
        static_usable = usable[: args.max_length]
        rows.append({
            "label": case["label"],
            "text": case["text"],
            "hf_raw_token_ids": raw[: args.max_length],
            "hf_raw_untruncated_length": len(raw),
            "static_usable_token_ids": static_usable,
            "model2vec_token_ids": m2v_ids,
            "model2vec_vector": np.asarray(vector, dtype=np.float32).tolist(),
            "model2vec_character_pretruncate_changes_ids": m2v_ids != static_usable,
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "reference": {
            "model2vec": model2vec_version,
            "tokenizers": tokenizers_version,
            "max_length": args.max_length,
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
