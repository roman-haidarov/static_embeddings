"""Generate an upstream oracle from SentenceTransformer.encode.

Primary contract: SentenceTransformer.encode(..., normalize_embeddings=False).
StaticEmbedding.forward is recorded only as a diagnostic second vector.

Token ids are Hugging Face encode_batch(..., add_special_tokens=False).
[UNK] is kept. There is no default token cap.
"""

import argparse
import json
from importlib.metadata import version

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

from parity_cases import build_cases

SCHEMA_VERSION = 1


def _as_ids(encoding):
    return [int(token_id) for token_id in encoding.ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("--out", default="tmp/st_oracle.json")
    parser.add_argument("--dimensions", type=int, default=None)
    args = parser.parse_args()

    cases = build_cases()
    extra = [
        {"label": "all-unk:private-use", "text": "\uE000\uE001\uE002 \uF8FF"},
        {"label": "long:known-words-600", "text": " ".join(["hello"] * 600)},
    ]
    cases.extend(extra)
    texts = [case["text"] for case in cases]

    model = SentenceTransformer(args.source_dir, device="cpu")
    static = model[0]
    if not isinstance(static, StaticEmbedding):
        raise SystemExit(f"expected first module StaticEmbedding, got {type(static)}")

    encodings = static.tokenizer.encode_batch(texts, add_special_tokens=False)
    encoded = model.encode(
        texts,
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if args.dimensions is not None:
        encoded = encoded[:, : args.dimensions]

    import torch

    weight = static.embedding.weight
    diagnostic = []
    for encoding in encodings:
        ids = np.asarray(encoding.ids, dtype=np.int64)
        if ids.size == 0:
            diagnostic.append(np.zeros((int(weight.shape[1]),), dtype=np.float32))
            continue
        with torch.no_grad():
            vec = static.embedding(
                torch.from_numpy(ids),
                torch.zeros(1, dtype=torch.int64),
            ).cpu().numpy()[0]
        diagnostic.append(np.asarray(vec, dtype=np.float32))

    if args.dimensions is not None:
        diagnostic = [row[: args.dimensions] for row in diagnostic]

    rows = []
    for case, encoding, vector, diag in zip(cases, encodings, encoded, diagnostic):
        raw = _as_ids(encoding)
        rows.append({
            "label": case["label"],
            "text": case["text"],
            "hf_raw_token_ids": raw,
            "hf_raw_untruncated_length": len(raw),
            "st_encode_vector": np.asarray(vector, dtype=np.float32).tolist(),
            "static_embedding_vector": np.asarray(diag, dtype=np.float32).tolist(),
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "reference": {
            "oracle": "sentence_transformers.SentenceTransformer.encode",
            "sentence_transformers": version("sentence_transformers"),
            "add_special_tokens": False,
            "normalize_embeddings": False,
            "max_tokens": None,
            "dimensions": args.dimensions,
        },
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


if __name__ == "__main__":
    main()
