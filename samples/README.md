# Samples

These scripts are native hot-path probe targets, not just benchmark reporters. They intentionally stay alive, print their PID, sleep before the hot loop, and print a one-line macOS `sample + filtercalltree` command to copy into a second console.

Compile first:

```bash
bundle exec rake compile
bundle exec rake demo_model
```

Run one probe target:

```bash
MODEL="$HOME/.cache/static_embeddings/models/potion-retrieval-32m.semb" \
  DURATION=25 bundle exec ruby samples/embed_batch_corpus_hot_path.rb
```

Then copy the printed command that starts with:

```bash
mkdir -p ".../samples/results"; OUT="..."; SAMPLE="..."; { sample <pid> ...
```

The captured call tree is written to `samples/results/*.txt` by default. Set `RESULT_DIR=...` to redirect it.

Useful probes:

```bash
bundle exec ruby samples/embed_single_large_hot_path.rb
MODE=ascii       bundle exec ruby samples/embed_batch_corpus_hot_path.rb
MODE=hashes      bundle exec ruby samples/embed_batch_corpus_hot_path.rb
MODE=base64      bundle exec ruby samples/embed_batch_corpus_hot_path.rb
MODE=identifiers bundle exec ruby samples/embed_batch_corpus_hot_path.rb
MODE=long_words  bundle exec ruby samples/tokenize_wordpiece_stress_hot_path.rb
bundle exec ruby samples/unicode_corner_hot_path.rb
ROWS=100000 bundle exec ruby samples/cosine_top_k_hot_path.rb
bundle exec ruby samples/cancellation_gvl_stress.rb
bundle exec ruby samples/warmup_hot_path.rb
```

`run_all.sh` can launch every probe as a separate Ruby process and, on macOS when `sample` and `filtercalltree` exist, capture each call tree automatically:

```bash
MODEL="$HOME/.cache/static_embeddings/models/potion-retrieval-32m.semb" \
  DURATION=15 samples/run_all.sh
```

For a quick syntax/smoke pass without long profiling:

```bash
QUICK=1 samples/run_all.sh
```


`random_pooling_hot_path.rb` bypasses tokenization with random token ids and isolates embedding-matrix row access, pooling, normalization and cache locality.

`cancellation_gvl_stress.rb` has two useful modes: default fairness mode and `FULL_SCAN=1 TIMEOUT=0.005`, which disables truncation so Timeout reaches the cancellation path.

## Reading the numbers

`cold_start.rb` is the only single-shot sample. It reports `load_ms`,
`warmup_ms`, `first_embed_ms` and `time_to_first_query_ms` — the figures a Puma
`before_fork` block or a serverless cold start actually pays. Drop the OS page
cache before running it (`sudo purge` on macOS,
`echo 3 > /proc/sys/vm/drop_caches` on Linux); the output prints
`cold_page_cache=` so you can tell whether you did.

`warmup_hot_path.rb` measures the opposite thing: re-touching pages that are
already resident. Its `mapped_range_gb_per_sec` is address-range coverage, not
memory bandwidth, because `warmup!` reads one byte per page.

`cosine_top_k_hot_path.rb` takes `METRIC=dot` (default) or `METRIC=cosine`.
They are different kernels: cosine also computes each row norm.

`random_pooling_hot_path.rb` rotates through `ID_SETS` distinct random id sets
so the touched rows do not stay in cache. Check `touched_matrix_mb` in the
output: if it is smaller than the last level cache, the result is cache
bandwidth rather than matrix locality.

GC is enabled by default in the harness. `GC.disable` does not stop an already
started incremental cycle from sweeping, so zeroed GC counters can appear next
to profiles full of `gc_sweep`. Set `GC_DISABLE=1` deliberately, and read
`gc_disabled=` in `gc_delta` before quoting it.

Metrics named `logical_input_mb_per_sec` divide by bytes that truncation may
never have read. When `truncation_active=true`, use `ops_per_sec`.
