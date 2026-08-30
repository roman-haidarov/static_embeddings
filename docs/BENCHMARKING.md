# Benchmarking

Two different tools, for two different questions.

`benchmark/` answers "how fast is this, and is the difference I am looking at
real". `samples/` answers "where does the time go" by keeping a process alive
for a profiler to attach to. Do not use one for the other's job.

```bash
bundle exec rake benchmark                 # everything
bundle exec rake benchmark:core_paths
bundle exec rake benchmark:gvl_threshold

BENCH_MODEL="$HOME/.cache/static_embeddings/models/potion-retrieval-32m.semb" \
  bundle exec rake benchmark:core_paths
```

Without `BENCH_MODEL` the demo model is used, which has `dim=8` and a 176-token
vocabulary. It proves the harness runs; it says nothing about production speed.

## Reading the output

```text
case                                      ns/op      ops/sec    spread%   alloc/op
embed words=20                           8540.5       117089       3.0%       1.00
```

`spread_pct` is the distance between the fastest and slowest repeat. **A
difference smaller than `spread_pct` is not a result.** On a laptop or a shared
CI runner it is routinely 5-10%, which is wider than most changes worth making.
If you want to claim an improvement, either get the spread down or show the
change across several independent runs.

`alloc/op` is Ruby object allocations per operation. It moves for different
reasons than time does, and it is the more stable signal of the two.

Generated text inputs are primed with `valid_encoding?` before timing, so these
benchmarks measure the cached-coderange hot path. Fresh strings from disk or the
database can pay an extra whole-string UTF-8 scan; see `docs/PERFORMANCE.md`.

Iteration counts are calibrated automatically so every measurement runs for at
least `BENCH_MIN_SECONDS`. Knobs:

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_MODEL` | demo model | model to load |
| `BENCH_REPEATS` | 7 | repeats per case, median reported |
| `BENCH_MIN_SECONDS` | 0.25 | minimum duration of one repeat |
| `BENCH_WARMUP_SECONDS` | 0.05 | warmup before timing |
| `BENCH_ADAPTIVE` | 1 | set to 0 to use the literal iteration count |
| `BENCH_FORMAT` | table | set to `kv` for machine-readable lines |

## What the two benchmarks are for

`core_paths` covers tokenize, single embed, batch embed, pooling in isolation
and top-k in both formats. The `embed` and `embed rotating corpus` rows are
deliberately both present: the first repeats one string and keeps its matrix
rows in cache, the second rotates through 2000 texts and does not. The gap
between them is the cache effect, and quoting the first as throughput is how you
end up with a number nobody can reproduce.

`gvl_threshold` sweeps input sizes around `SE_GVL_UNLOCK_THRESHOLD`, the point
where a single `embed` stops running under the GVL and is handed to the batch
machinery. Look for the step, not the slope. On one x86-64 run the step cost
about 32% latency and four extra allocations, and it is also the point where a
second Ruby thread starts making progress at all. Both halves of that trade have
to be measured on your hardware before the constant is worth changing.

## Traps this repository has already fallen into

**Quoting noise.** An `embed_batch f16` change was reported as a 6% speedup; it
was inside the run-to-run spread and the next run showed the opposite sign. The
memory result from the same change was real and reproducible. Report the one you
can reproduce.

**Cache-resident working sets.** Embedding the same text in a loop keeps a tiny
slice of the matrix hot. Real corpora do not.

**Logical vs processed bytes.** With truncation active, a 3 MB document is
tokenized only until `max_tokens` is reached. Any MB/s figure computed from the
string length divides by bytes that were never read.

**GC state.** `GC.disable` does not stop an in-flight incremental cycle from
sweeping. The harness leaves GC on and reports allocations instead.

See `docs/PERFORMANCE.md` for the budget itself and where the time actually
goes.
