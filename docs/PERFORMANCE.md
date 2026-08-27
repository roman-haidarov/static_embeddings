# Performance budget

Absolute latency numbers are not comparable across models: `potion-base-2M`
and `potion-retrieval-32M` differ by roughly 8x in row width, so swapping the
model would read as a regression. The budget is therefore normalised.

Run `ruby tools/benchmark.rb [model.semb]`.

## Metrics tracked

| Metric | Unit | Why |
|---|---|---|
| tokenization | ns / input byte | independent of `dim`; catches normalizer regressions |
| pooling | ns / token / 100 dims | comparable across models |
| single embed | µs / call | the latency a query pays |
| batch throughput | texts/s, tokens/s | indexing capacity |
| RSS overhead | bytes beyond the mapped file | must stay O(scratch), not O(model) |
| cold vs warm | first query after a cold page cache | mmap means the first pass faults in the matrix |
| time to first query | ms from `load` to first vector | what a Puma `before_fork` or a serverless cold start pays |

`samples/cold_start.rb` measures the last row. It is single shot on purpose:
`load` walks the whole vocabulary hash table and both tries exactly once, and
`warmup!` only faults pages in once. Drop the page cache first
(`sudo purge` on macOS, `echo 3 > /proc/sys/vm/drop_caches` on Linux) or the
output will label itself a warm start.

Every other sample under `samples/` runs a hot loop and measures steady state.
`warmup_hot_path.rb` in particular re-touches pages that are already resident;
its `mapped_range_gb_per_sec` is address-range coverage, not memory bandwidth,
because warmup reads one byte per page.

## CI status

The CI benchmark job is informational. Treat regression gates as meaningful only when a checked-in baseline is compared on the same runner class. Absolute thresholds on shared CI hardware produce noise, not signal.

## Benchmark hygiene

Two failure modes have bitten this repository already and are worth stating:

- **Logical vs processed bytes.** With truncation active, a 3 MB document is
  tokenized only until `max_tokens` is reached. Any "MB/s of input" figure
  divides by bytes that were never read. Samples that still print one label it
  `logical_input_mb_per_sec` and set `truncation_active`.
- **Resident working sets.** Embedding the same text, or the same frozen id
  array, in a loop keeps a tiny slice of the matrix in cache and overstates
  throughput. `random_pooling_hot_path.rb` rotates through many id sets and
  prints `touched_matrix_mb` so the reader can check it exceeds the last level
  cache.

`GC.disable` is off by default in the sample harness: it does not stop an
already started incremental cycle from sweeping, so zeroed GC counters used to
appear next to profiles full of `gc_sweep`. Set `GC_DISABLE=1` deliberately and
read `gc_disabled=` before quoting `gc_delta`.

## Where the time actually goes

For static embeddings, tokenization dominates: pooling is a handful of row
reads and adds. That is why the tokenizer is the file to profile, and why
SIMD in the pooling loop is not where the wins are — the loop is memory-bound
on random row lookups, and the compiler already vectorises it.
