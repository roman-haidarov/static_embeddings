# Performance budget

Absolute latency is not comparable across models: `potion-base-2M` and
`potion-retrieval-32M` differ by roughly 8x in row width, so swapping the model
would read as a regression. The budget is therefore normalised.

```bash
ruby tools/benchmark.rb [model.semb]
```

## Metrics tracked

| Metric | Unit | Why |
|---|---|---|
| tokenization | ns / processed byte | independent of `dim`; catches normalizer regressions |
| pooling | ns / token / 100 dims | comparable across models |
| single embed | µs / call | the latency a query pays |
| batch throughput | texts/s, tokens/s | indexing capacity |
| RSS overhead | bytes beyond the mapped file | must stay O(scratch), not O(model) |
| time to first query | ms from `load` to first vector | what a Puma `before_fork` or a serverless cold start pays |
| C allocation stats | bytes and counts by native category | optional build, explains heap churn |

## Where the time actually goes

Tokenization dominates; pooling is a handful of row reads and adds. That is why
the tokenizer is the file to profile, and why SIMD in the pooling loop is not
where the wins are — that loop is memory-bound on random row lookups and the
compiler already vectorises it.

Out-of-vocabulary text is the main cliff. In-vocabulary words hit the
vocabulary hash directly; everything else falls through to trie-driven subword
splitting, which on a synthetic all-OOV corpus measured several times the
per-text cost. Non-English input on an English model pays this on top of the
`[UNK]` quality problem.

## `f16` is a storage trade-off

`format: :f16` halves the bytes a top-k scan streams and doubles the decode
work. Which one wins is a property of the machine, and this repository's own
sample runs disagree with each other — same corpus, same `k`, native decode
kernel in both cases:

```text
                     f32        f16
x86-64, f16c       3.00 ms    1.65 ms    f16 1.8x faster
M1 Pro, neon-fp16  2.29 ms    2.88 ms    f16 1.3x slower
```

Neither ratio transfers. Confirm `StaticEmbeddings.simd_backend`, then measure
on the hardware you will run on. On the lookup-table fallback `f16` is usually
slower than `f32`.

## C allocation counters

Compiling with `STATIC_EMBEDDINGS_ALLOC_STATS=1` wraps the runtime's own
allocations and records bytes and counts per category. Default builds do not
define the internal methods and do not pay for the counters.

```bash
STATIC_EMBEDDINGS_ALLOC_STATS=1 samples/run_all.sh
```

The harness forces `rake clobber compile` in this mode, because an existing
mkmf Makefile otherwise keeps the previous `DEFS`, and it fails the run if the
instrumented methods did not load. Samples then print
`c_alloc.<category>.<metric>` lines.

Two limits worth stating. The counters cover only allocations made through the
instrumented wrappers, so Ruby object heap, allocator fragmentation and mmap
residency are all outside them — use Instruments, heaptrack, Massif or `vmmap`
for those. And the instrumentation is not free: about 13% on `tokenize`, 3% on
`embed`, within noise on `embed_batch`. A run captured with it is not
comparable to one captured without it.

## Benchmark hygiene

Three failure modes have bitten this repository already.

**Logical vs processed bytes.** With truncation active a 3 MB document is
tokenized only until `max_tokens` is reached, so any "MB/s of input" figure
computed from the string length divides by bytes that were never read. Samples
print both `logical_input_mb_per_sec` and a measured
`processed_input_mb_per_sec`; quote the second. `tools/benchmark.rb` refuses to
print a ns/byte figure at all when truncation makes the two differ.

**Resident working sets.** Embedding the same text, or the same frozen id array,
in a loop keeps a tiny slice of the matrix in cache and overstates throughput.
`random_pooling_hot_path.rb` rotates through many id sets and prints
`touched_matrix_mb` so you can check it exceeds the last level cache.

**GC state.** `GC.disable` does not stop an already started incremental cycle
from sweeping, so zeroed GC counters used to appear next to profiles full of
`gc_sweep`. It is off by default in the harness; set `GC_DISABLE=1`
deliberately and read `gc_disabled=` before quoting `gc_delta`.

## Two costs that are not the tokenizer

`embed` cannot start until Ruby has confirmed the `String` is valid UTF-8, and
for a `String` whose coderange has not been computed that is an O(bytes) scan
which runs before the prefix window is chosen. On a 3 MB freshly read document
it dominated: 442 µs against 71 µs for the same string once Ruby had cached the
answer. Reuse the `String`, call `valid_encoding?` outside the hot path, or pass
`validate_encoding: :prefix` to bound validation the same way tokenizing is
bounded — 78 µs, at the cost of not inspecting bytes past the window.

`StaticEmbeddings.verify` hashes the artifact and is deliberately not on the
load path. It streams in 1 MiB chunks: on a 62 MB model, 249 ms and 30 MB of
peak RSS. Reading the file whole and duplicating it to zero the checksum field
cost 433 ms and 280 MB.

## Cold start

`samples/cold_start.rb` measures `load`, first embed and `warmup!` in one shot,
because each of those happens exactly once: `load` walks the vocabulary hash and
both tries, and `warmup!` faults pages in. Drop the page cache first
(`sudo purge` on macOS, `echo 3 > /proc/sys/vm/drop_caches` on Linux) or the
output labels itself a warm start.

Every other sample runs a hot loop and measures steady state. `warmup_hot_path.rb`
in particular re-touches resident pages; its `mapped_range_gb_per_sec` is
address-range coverage, not memory bandwidth, because warmup reads one byte per
page.

## CI

The benchmark job is informational. Absolute thresholds on shared runners
produce noise, not signal; a regression gate is only meaningful against a
checked-in baseline on the same runner class.
