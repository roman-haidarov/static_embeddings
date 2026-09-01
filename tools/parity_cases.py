"""Deterministic boundary corpus for the pinned HF/model2vec compatibility contract."""


def _case(label, text):
    return {"label": label, "text": text}


def _contexts(label, char):
    return [
        _case(f"{label}:alone", char),
        _case(f"{label}:middle", f"hello{char}world"),
        _case(f"{label}:left", f"{char}hello"),
        _case(f"{label}:right", f"hello{char}"),
    ]


def build_cases():
    cases = []
    basics = [
        "", " ", "\t\n\r", "hello", "Hello WORLD", "hello, world!",
        "café", "CAFÉ", "cafe\u0301", "naïve", "привет мир", "中文测试",
        "🧬 emoji 🧬", "zero\u200bwidth", "non\u00a0breaking\u2009space",
        "testing ##ing edge", "zzzzz unknownword qqqqq", "a" * 300,
    ]
    cases.extend(_case(f"basic:{i:02d}", text) for i, text in enumerate(basics))

    # BERT clean_text/control boundaries. Note that tokenizers 0.23.1 calls
    # unicode_categories 0.1.1 `is_other()`. Despite a misleading comment in
    # bert.rs mentioning Cn, that crate actually returns true only for Cc/Cf/Co.
    controls = [
        0x0000, 0x0001, 0x0008, 0x000B, 0x000C, 0x001F, 0x007F,
        0x0080, 0x009F, 0x00AD, 0x061C, 0x180E,
        0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0xE000,
    ]
    for cp in controls:
        cases.extend(_contexts(f"control:U+{cp:04X}", chr(cp)))
    for cp in (0x0378, 0x0380):
        cases.extend(_contexts(f"unassigned-not-control:U+{cp:04X}", chr(cp)))

    # Rust tokenizers 0.23.1 BertNormalizer CJK ranges. Every edge gets -1/0/+1
    # so a copied Python transformers boundary cannot silently pass.
    cjk_ranges = [
        (0x4E00, 0x9FFF),
        (0x3400, 0x4DBF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0x2B920, 0x2CEAF),
        (0xF900, 0xFAFF),
        (0x2F800, 0x2FA1F),
    ]
    seen = set()
    for lo, hi in cjk_ranges:
        for cp in (lo - 1, lo, lo + 1, hi - 1, hi, hi + 1):
            if cp in seen or cp < 0 or cp > 0x10FFFF:
                continue
            seen.add(cp)
            cases.extend(_contexts(f"cjk-boundary:U+{cp:05X}", chr(cp)))

    # One representative from each punctuation subclass plus ASCII punctuation.
    punctuation = {
        "Pc": [0x005F, 0x203F],
        "Pd": [0x002D, 0x2014],
        "Ps": [0x0028, 0x2018],
        "Pe": [0x0029, 0x2019],
        "Pi": [0x00AB, 0x201C],
        "Pf": [0x00BB, 0x201D],
        "Po": [0x0021, 0x3002],
    }
    for category, points in punctuation.items():
        for cp in points:
            cases.extend(_contexts(f"punct:{category}:U+{cp:04X}", chr(cp)))

    # Whitespace and combining-mark behavior.
    for cp in [0x0020, 0x0009, 0x000A, 0x000D, 0x00A0, 0x1680, 0x2000, 0x2009, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000]:
        cases.extend(_contexts(f"space:U+{cp:04X}", chr(cp)))
    for text in ["e\u0301", "A\u030A", "и\u0301", "\u0301" * 16, "cafe\u0301 noir"]:
        cases.append(_case(f"combining:{len(cases)}", text))

    # The only AddedVocabulary subset accepted by the converter.
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for token in specials:
        cases.extend([
            _case(f"added:{token}:alone", token),
            _case(f"added:{token}:adjacent", f"hello{token}world"),
            _case(f"added:{token}:spaced", f"hello {token} world"),
            _case(f"added:{token}:double", token + token),
            _case(f"added:{token}:lower", token.lower()),
        ])
    cases.extend([
        _case("added:mixed", "[CLS]hello[MASK]world[SEP]"),
        _case("added:all", "[PAD][UNK][CLS][SEP][MASK]"),
    ])

    # Truncation/UNK matrix. Single-letter known words keep these texts below
    # Model2Vec's character pre-cut so they isolate UNK-before-truncate.
    for n in (511, 512, 513, 700):
        cases.append(_case(f"truncate:known:{n}", " ".join(["a"] * n)))
    for ratio, period in [(10, 10), (25, 4), (50, 2), (90, 10)]:
        words = []
        for i in range(900):
            if ratio == 90:
                unknown = (i % period) != 0
            else:
                unknown = (i % period) == 0
            words.append("🧬" if unknown else "a")
        cases.append(_case(f"truncate:unk-ratio-{ratio}", " ".join(words)))

    # Prefix/window behavior, including an intentional difference from
    # Model2Vec's max_length * median_token_length character pre-cut.
    cases.extend([
        _case("long:sparse-whitespace", "hello" + (" " * 100_000) + ("world " * 600)),
        _case("long:dense-ascii", "hello world ruby static embedding " * 800),
        _case("long:unknown-prefix", ("🧬 " * 700) + ("hello " * 600)),
        _case("long:combining", "\u0301" * 20_000),
    ])

    labels = [case["label"] for case in cases]
    if len(labels) != len(set(labels)):
        raise RuntimeError("duplicate parity-case label")
    return cases


if __name__ == "__main__":
    corpus = build_cases()
    print(f"rows={len(corpus)}")
