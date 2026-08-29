"""Dump reference token ids and vectors from Python model2vec.StaticModel.

The Ruby reference implementation in this repository is an implementation twin
of the C runtime, so it cannot prove compatibility with HuggingFace tokenizers
or model2vec. This script produces the only external oracle we accept.

It records token ids as well as vectors, because the correctness contract in
the README is stated on ids ("token ids must match the reference exactly") and
two different tokenizations can still land inside cosine tolerance.

Usage:
    python tools/model2vec_oracle.py ./potion-retrieval-32M --out tmp/oracle.json
"""

import argparse
import json
import numpy as np
from model2vec import StaticModel

BASE_TEXTS = [
    "",
    " ",
    "\t\n\r",
    "postgres pipeline mode in ruby",
    "ruby postgres pipelining and database queries",
    "electrolysis hair removal aftercare",
    "banana smoothie with milk",
    "Hello WORLD",
    "hello, world!",
    "hello\u007fworld",
    "hello\u007f world",
    "\u007f",
    "caf\u00e9",
    "CAF\u00c9",
    "cafe\u0301",
    "na\u00efve",
    "\u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442 \u043f\u0440\u043e \u043f\u043e\u0441\u0442\u0433\u0440\u0435\u0441",
    "\u041f\u0420\u0418\u0412\u0415\u0422, \u041c\u0418\u0420!",
    "\u4e2d\u6587\u6d4b\u8bd5",
    "\U0001f9ac\U0001f9ac\U0001f9ac",
    "zero\u200bwidth",
    "non\u00a0breaking\u2009space",
    "testing ##ing edge",
    "zzzzz unknownword qqqqq",
    "a" * 300,
]

SEED = (
    "postgres pipeline mode in ruby static embeddings local rag search "
    "vector database latency throughput "
)


def repeat_to_bytes(seed, target):
    out = []
    size = 0
    while size < target:
        out.append(seed)
        size += len(seed.encode("utf-8"))
    return "".join(out)


LARGE_TEXTS = [
    repeat_to_bytes(SEED, 8 * 1024),
    repeat_to_bytes(SEED, 96 * 1024),
    repeat_to_bytes(SEED, 512 * 1024),
    "a" * (96 * 1024),
    ("word" + " " * 400) * 200,
    "\u0441\u043b\u043e\u0432\u043e\u00a0" * 20000,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("--out", default="tmp/model2vec_oracle.json")
    parser.add_argument("--text", action="append", dest="texts")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()

    if args.texts:
        texts = args.texts
    else:
        texts = list(BASE_TEXTS)
        if not args.skip_large:
            texts += LARGE_TEXTS

    model = StaticModel.from_pretrained(args.source_dir)
    embeddings = model.encode(texts, max_length=args.max_length)
    encodings = model.tokenizer.encode_batch(texts, add_special_tokens=False)

    rows = []
    for text, vector, encoding in zip(texts, embeddings, encodings):
        ids = list(encoding.ids)
        rows.append(
            {
                "text": text,
                "token_ids": ids[: args.max_length],
                "token_ids_untruncated_length": len(ids),
                "vector": np.asarray(vector, dtype=np.float32).tolist(),
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"max_length": args.max_length, "rows": rows}, f, ensure_ascii=False)

    dim = len(rows[0]["vector"]) if rows else 0
    print("wrote %s rows=%d dim=%d max_length=%d" % (args.out, len(rows), dim, args.max_length))


if __name__ == "__main__":
    main()
