# Samples

These scripts are native hot-path probe targets, not benchmark reporters. They
stay alive, print their PID, sleep before the hot loop, and print a one-line
macOS `sample + filtercalltree` command to paste into a second console.

```bash
bundle exec rake compile
bundle exec rake demo_model

MODEL="$HOME/.cache/static_embeddings/models/potion-retrieval-32m.semb" \
  DURATION=25 bundle exec ruby samples/embed_batch_corpus_hot_path.rb
```

Captured call trees go to `samples/results/`; set `RESULT_DIR=...` to redirect.

## Probes

```bash
bundle exec ruby samples/embed_single_large_hot_path.rb
MODE=ascii       bundle exec ruby samples/embed_batch_corpus_hot_path.rb
MODE=hashes      bundle exec ruby samples/embed_batch_corpus_hot_path.rb
MODE=base64      bundle exec ruby samples/embed_batch_corpus_hot_path.rb
MODE=identifiers bundle exec ruby samples/embed_batch_corpus_hot_path.rb
MODE=long_words  bundle exec ruby samples/tokenize_wordpiece_stress_hot_path.rb
bundle exec ruby samples/unicode_corner_hot_path.rb
ROWS=100000      bundle exec ruby samples/cosine_top_k_hot_path.rb
bundle exec ruby samples/cancellation_gvl_stress.rb
bundle exec ruby samples/warmup_hot_path.rb
```

`random_pooling_hot_path.rb` bypasses tokenization with random token ids to
isolate matrix row access, pooling, normalization and cache locality.

`cosine_top_k_hot_path.rb` takes `METRIC=dot` (default) or `METRIC=cosine`.
They are different kernels: cosine also computes each row norm.

`cancellation_gvl_stress.rb` has two modes. The default measures GVL fairness
and is *not* cancellation evidence. `FULL_SCAN=1 TIMEOUT=0.005` disables
truncation so `Timeout` actually reaches the cancellation path; that run must
show non-zero timeouts and a bounded `rss_delta_excluding_mapped_kb` or it
proves nothing.

## Running everything

```bash
MODEL="$HOME/.cache/static_embeddings/models/potion-retrieval-32m.semb" \
  DURATION=15 samples/run_all.sh
```

`run_all.sh` launches every probe as a separate process, runs `rake test` into
`00_test.log`, and on macOS captures each call tree automatically when `sample`
and `filtercalltree` exist. `QUICK=1` does a fast syntax and smoke pass;
`TEST=0` skips the suite. Output goes to a timestamped
`samples/results/run_<UTC>/` directory with `env.txt`, `summary.tsv`,
`replay.sh` and per-probe logs.

To also get C allocation counters, rebuild with the instrumented path:

```bash
STATIC_EMBEDDINGS_ALLOC_STATS=1 \
MODEL="$HOME/.cache/static_embeddings/models/potion-retrieval-32m.semb" \
  DURATION=15 samples/run_all.sh
```

See `docs/PERFORMANCE.md` for what those counters do and do not cover, and for
why an instrumented run is not comparable to a normal one.

## Reading the numbers

`failures=0` in `summary.tsv` means the probes completed, not that the suite
passed. Check `00_test.log` before saying tests are green.

`cold_start.rb` is the only single-shot sample: `load_ms`, `warmup_ms`,
`first_embed_ms`, `time_to_first_query_ms`. Drop the OS page cache first, or
`page_cache_state` will tell you it could not confirm a cold start.

`warmup_hot_path.rb` measures the opposite thing — re-touching resident pages.
Its `mapped_range_gb_per_sec` is address-range coverage, not bandwidth.

`random_pooling_hot_path.rb` rotates through `ID_SETS` distinct id sets so the
touched rows do not stay in cache. If `touched_matrix_mb` is smaller than the
last level cache, you measured cache bandwidth, not matrix locality.

Large-input samples print `logical_input_mb_per_sec` next to a measured
`processed_input_mb_per_sec`. Quote the second: with truncation active the
tokenizer stops well before the end of the string. The processed figure is
omitted when the corpus is degenerate enough that the measurement cannot be
made, for example an all-OOV text whose vector does not depend on prefix length.

GC is enabled by default. See `docs/PERFORMANCE.md` before quoting `gc_delta`.
