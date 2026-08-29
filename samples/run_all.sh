#!/usr/bin/env bash
set -u
set -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root" || exit 1

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
result_root="${RESULT_ROOT:-samples/results}"
out_dir="$result_root/run_$stamp"
mkdir -p "$out_dir"
ln -sfn "run_$stamp" "$result_root/latest" 2>/dev/null || true

if [[ -n "${RUBY_CMD:-}" ]]; then
  read -r -a ruby_cmd <<< "$RUBY_CMD"
else
  ruby_cmd=(bundle exec ruby)
fi

if [[ -n "${RAKE_CMD:-}" ]]; then
  read -r -a rake_cmd <<< "$RAKE_CMD"
else
  rake_cmd=(bundle exec rake)
fi

quick="${QUICK:-0}"
if [[ "$quick" == "1" ]]; then
  default_duration="${DURATION:-1}"
  default_sleep="${SLEEP_BEFORE_HOT_LOOP:-0.2}"
  default_texts="${TEXTS:-300}"
  default_bytes_per_text="${BYTES_PER_TEXT:-160}"
  default_input_bytes="${INPUT_BYTES:-150000}"
  default_rows="${ROWS:-5000}"
  default_units="${UNITS:-16}"
else
  default_duration="${DURATION:-25}"
  default_sleep="${SLEEP_BEFORE_HOT_LOOP:-7}"
  default_texts="${TEXTS:-5000}"
  default_bytes_per_text="${BYTES_PER_TEXT:-192}"
  default_input_bytes="${INPUT_BYTES:-3000000}"
  default_rows="${ROWS:-50000}"
  default_units="${UNITS:-40}"
fi

detect_model() {
  if [[ -n "${MODEL:-}" ]]; then
    printf '%s\n' "$MODEL"
    return 0
  fi

  local candidate
  for candidate in \
    "$HOME/.cache/static_embeddings/models/potion-retrieval-32m.semb" \
    "$repo_root/tmp/test-tiny.semb" \
    "$repo_root/lib/models/demo.semb"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

write_env() {
  {
    echo "run_id=run_$stamp"
    echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo_root=$repo_root"
    echo "result_dir=$out_dir"
    echo "quick=$quick"
    echo "ruby_cmd=${ruby_cmd[*]}"
    echo "rake_cmd=${rake_cmd[*]}"
    echo "MODEL=${MODEL:-}"
    echo "DURATION=$default_duration"
    echo "SLEEP_BEFORE_HOT_LOOP=$default_sleep"
    echo "TEXTS=$default_texts"
    echo "BYTES_PER_TEXT=$default_bytes_per_text"
    echo "INPUT_BYTES=$default_input_bytes"
    echo "ROWS=$default_rows"
    echo "UNITS=$default_units"
    echo "STATIC_EMBEDDINGS_ALLOC_STATS=${STATIC_EMBEDDINGS_ALLOC_STATS:-}"
    echo
    "${ruby_cmd[@]}" -v 2>/dev/null || true
    uname -a 2>/dev/null || true
    sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    nproc 2>/dev/null || true
  } > "$out_dir/env.txt"
}

append_summary_header() {
  printf 'case\tstatus\tseconds\truby_log\tcapture_log\n' > "$out_dir/summary.tsv"
}

run_step() {
  local name="$1"
  shift
  local log="$out_dir/${name}.log"
  local started ended elapsed status

  started="$(date +%s)"
  {
    echo "case=$name"
    echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "command=$*"
    echo
    "$@"
  } > "$log" 2>&1
  status=$?
  ended="$(date +%s)"
  elapsed=$((ended - started))
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$elapsed" "$log" "" >> "$out_dir/summary.tsv"
  return "$status"
}

sample_available() {
  command -v sample >/dev/null 2>&1 && command -v filtercalltree >/dev/null 2>&1
}

wait_for_line() {
  local file="$1"
  local pattern="$2"
  local timeout_seconds="$3"
  local i
  for ((i=0; i<timeout_seconds*10; i++)); do
    if grep -q "$pattern" "$file" 2>/dev/null; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

extract_value() {
  local file="$1"
  local key="$2"
  grep -m 1 "^${key}=" "$file" | sed "s/^${key}=//"
}

run_sample() {
  local name="$1"
  local file="$2"
  shift 2
  local ruby_log="$out_dir/${name}.ruby.log"
  local capture_log="$out_dir/${name}.sample.txt"
  local started ended elapsed status target_pid sample_seconds sample_file native_grep

  started="$(date +%s)"
  env RESULT_DIR="$out_dir" MODEL="$MODEL" "$@" "${ruby_cmd[@]}" "samples/$file" > "$ruby_log" 2>&1 &
  target_pid=$!

  if wait_for_line "$ruby_log" "^Copy this one-line capture command:" 60; then
    sample_seconds="$(extract_value "$ruby_log" sample_seconds)"
    sample_file="$(extract_value "$ruby_log" sample_file)"
    native_grep="$(grep -m 1 '^Copy this one-line capture command:' -A1 "$ruby_log" | tail -1 | sed -n 's/.*grep -E "\([^"]*\)".*/\1/p')"

    if sample_available; then
      {
        echo "case=$name"
        echo "target_pid=$target_pid"
        echo "sample_seconds=$sample_seconds"
        echo "sample_file=$sample_file"
        echo
        sample "$target_pid" "$sample_seconds" -f "$sample_file"
        echo
        echo "===== focused native symbols ====="
        filtercalltree "$sample_file" | grep -E "$native_grep" | head -320
        echo
        echo "===== filtercalltree head -320 ====="
        filtercalltree "$sample_file" | head -320
      } > "$capture_log" 2>&1
    else
      {
        echo "case=$name"
        echo "target_pid=$target_pid"
        echo "sample/filtercalltree not found; run the printed command manually on macOS."
        echo
        grep -m 1 '^mkdir -p ' "$ruby_log" || true
      } > "$capture_log" 2>&1
    fi
  else
    {
      echo "case=$name"
      echo "target_pid=$target_pid"
      echo "sample did not print capture command before timeout"
    } > "$capture_log" 2>&1
  fi

  wait "$target_pid"
  status=$?
  ended="$(date +%s)"
  elapsed=$((ended - started))
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$elapsed" "$ruby_log" "$capture_log" >> "$out_dir/summary.tsv"
  return "$status"
}

write_commands() {
  cat > "$out_dir/replay.sh" <<EOF
#!/usr/bin/env bash
set -u
set -o pipefail
cd "$repo_root" || exit 1
export MODEL="$MODEL"
export RESULT_DIR="$out_dir"
export DURATION="$default_duration"
export SLEEP_BEFORE_HOT_LOOP="$default_sleep"
export TEXTS="$default_texts"
export BYTES_PER_TEXT="$default_bytes_per_text"
export INPUT_BYTES="$default_input_bytes"
export ROWS="$default_rows"
export UNITS="$default_units"

${ruby_cmd[*]} samples/embed_single_large_hot_path.rb
FORMAT=f16 ${ruby_cmd[*]} samples/embed_single_large_hot_path.rb
MODE=ascii       ${ruby_cmd[*]} samples/embed_batch_corpus_hot_path.rb
MODE=ascii FORMAT=f16 ${ruby_cmd[*]} samples/embed_batch_corpus_hot_path.rb
MODE=hashes      ${ruby_cmd[*]} samples/embed_batch_corpus_hot_path.rb
MODE=base64      ${ruby_cmd[*]} samples/embed_batch_corpus_hot_path.rb
MODE=identifiers ${ruby_cmd[*]} samples/embed_batch_corpus_hot_path.rb
MODE=long_words  ${ruby_cmd[*]} samples/embed_batch_corpus_hot_path.rb
MODE=unicode     ${ruby_cmd[*]} samples/embed_batch_corpus_hot_path.rb
MODE=unicode FORMAT=f16 ${ruby_cmd[*]} samples/embed_batch_corpus_hot_path.rb
MODE=hashes      ${ruby_cmd[*]} samples/tokenize_wordpiece_stress_hot_path.rb
MODE=base64      ${ruby_cmd[*]} samples/tokenize_wordpiece_stress_hot_path.rb
MODE=identifiers ${ruby_cmd[*]} samples/tokenize_wordpiece_stress_hot_path.rb
MODE=long_words  ${ruby_cmd[*]} samples/tokenize_wordpiece_stress_hot_path.rb
${ruby_cmd[*]} samples/unicode_corner_hot_path.rb
FORMAT=f16 ${ruby_cmd[*]} samples/unicode_corner_hot_path.rb
MODE=ascii METRIC=dot    ${ruby_cmd[*]} samples/cosine_top_k_hot_path.rb
MODE=ascii METRIC=cosine ${ruby_cmd[*]} samples/cosine_top_k_hot_path.rb
MODE=ascii FORMAT=f16 METRIC=dot    ${ruby_cmd[*]} samples/cosine_top_k_hot_path.rb
MODE=ascii FORMAT=f16 METRIC=cosine ${ruby_cmd[*]} samples/cosine_top_k_hot_path.rb
${ruby_cmd[*]} samples/cancellation_gvl_stress.rb
FULL_SCAN=1 TIMEOUT=0.005 ${ruby_cmd[*]} samples/cancellation_gvl_stress.rb
${ruby_cmd[*]} samples/random_pooling_hot_path.rb
FORMAT=f16 ${ruby_cmd[*]} samples/random_pooling_hot_path.rb
${ruby_cmd[*]} samples/warmup_hot_path.rb
${ruby_cmd[*]} samples/cold_start.rb
${ruby_cmd[*]} samples/f16_quality_check.rb
EOF
  chmod +x "$out_dir/replay.sh"
}

failures=0
write_env
append_summary_header

if [[ "${COMPILE:-1}" != "0" ]]; then
  if [[ "${STATIC_EMBEDDINGS_ALLOC_STATS:-}" == "1" ]]; then
    run_step "00_compile" "${rake_cmd[@]}" clobber compile || failures=$((failures + 1))
    run_step "00_alloc_stats_check" "${ruby_cmd[@]}" -Ilib -e 'require "static_embeddings"; abort "allocation stats build was not loaded" unless StaticEmbeddings.respond_to?(:__alloc_stats__) && StaticEmbeddings.respond_to?(:__alloc_stats_reset__)' || failures=$((failures + 1))
    if [[ "$failures" -ne 0 ]]; then
      echo "allocation stats build failed; see $out_dir/00_compile.log and $out_dir/00_alloc_stats_check.log"
      exit 1
    fi
  else
    run_step "00_compile" "${rake_cmd[@]}" compile || failures=$((failures + 1))
  fi
fi

if [[ "${TEST:-1}" != "0" ]]; then
  run_step "00_test" "${rake_cmd[@]}" test || failures=$((failures + 1))
fi

if ! model_path="$(detect_model)"; then
  if [[ "${BUILD_MODEL:-1}" != "0" ]]; then
    run_step "00_demo_model" "${rake_cmd[@]}" demo_model || true
  fi
  model_path="$(detect_model || true)"
fi

if [[ -z "${model_path:-}" || ! -f "$model_path" ]]; then
  echo "model not found" | tee "$out_dir/model_error.txt"
  echo "Set MODEL=/path/to/model.semb or run bundle exec rake demo_model." | tee -a "$out_dir/model_error.txt"
  exit 2
fi

export MODEL="$model_path"
echo "resolved_model=$MODEL" >> "$out_dir/env.txt"
write_commands

echo "writing results to $out_dir"
echo "model=$MODEL"
echo

run_sample "01_embed_single_large" "embed_single_large_hot_path.rb" \
  DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" INPUT_BYTES="$default_input_bytes" || failures=$((failures + 1))

run_sample "01b_embed_single_large_f16" "embed_single_large_hot_path.rb" \
  FORMAT=f16 DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" INPUT_BYTES="$default_input_bytes" || failures=$((failures + 1))

run_sample "02_embed_batch_ascii" "embed_batch_corpus_hot_path.rb" \
  MODE=ascii DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" TEXTS="$default_texts" BYTES_PER_TEXT="$default_bytes_per_text" || failures=$((failures + 1))

run_sample "02b_embed_batch_ascii_f16" "embed_batch_corpus_hot_path.rb" \
  MODE=ascii FORMAT=f16 DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" TEXTS="$default_texts" BYTES_PER_TEXT="$default_bytes_per_text" || failures=$((failures + 1))

run_sample "03_embed_batch_hashes" "embed_batch_corpus_hot_path.rb" \
  MODE=hashes DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" TEXTS="$default_texts" BYTES_PER_TEXT="$default_bytes_per_text" || failures=$((failures + 1))

run_sample "04_embed_batch_base64" "embed_batch_corpus_hot_path.rb" \
  MODE=base64 DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" TEXTS="$default_texts" BYTES_PER_TEXT="$default_bytes_per_text" || failures=$((failures + 1))

run_sample "05_embed_batch_identifiers" "embed_batch_corpus_hot_path.rb" \
  MODE=identifiers DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" TEXTS="$default_texts" BYTES_PER_TEXT="$default_bytes_per_text" || failures=$((failures + 1))

run_sample "06_embed_batch_long_words" "embed_batch_corpus_hot_path.rb" \
  MODE=long_words DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" TEXTS="$default_texts" BYTES_PER_TEXT="$default_bytes_per_text" || failures=$((failures + 1))

run_sample "07_embed_batch_unicode" "embed_batch_corpus_hot_path.rb" \
  MODE=unicode DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" TEXTS="$default_texts" BYTES_PER_TEXT="$default_bytes_per_text" || failures=$((failures + 1))

run_sample "07b_embed_batch_unicode_f16" "embed_batch_corpus_hot_path.rb" \
  MODE=unicode FORMAT=f16 DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" TEXTS="$default_texts" BYTES_PER_TEXT="$default_bytes_per_text" || failures=$((failures + 1))

run_sample "08_tokenize_hashes" "tokenize_wordpiece_stress_hot_path.rb" \
  MODE=hashes DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" UNITS="$default_units" || failures=$((failures + 1))

run_sample "09_tokenize_base64" "tokenize_wordpiece_stress_hot_path.rb" \
  MODE=base64 DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" UNITS="$default_units" || failures=$((failures + 1))

run_sample "10_tokenize_identifiers" "tokenize_wordpiece_stress_hot_path.rb" \
  MODE=identifiers DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" UNITS="$default_units" || failures=$((failures + 1))

run_sample "11_tokenize_long_words" "tokenize_wordpiece_stress_hot_path.rb" \
  MODE=long_words DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" UNITS="$default_units" || failures=$((failures + 1))

run_sample "12_unicode_corner" "unicode_corner_hot_path.rb" \
  DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" INPUT_BYTES="${UNICODE_INPUT_BYTES:-16384}" || failures=$((failures + 1))

run_sample "12b_unicode_corner_f16" "unicode_corner_hot_path.rb" \
  FORMAT=f16 DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" INPUT_BYTES="${UNICODE_INPUT_BYTES:-16384}" || failures=$((failures + 1))

run_sample "13_dot_top_k_ascii" "cosine_top_k_hot_path.rb" \
  MODE=ascii METRIC=dot DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" ROWS="$default_rows" BYTES_PER_TEXT="$default_bytes_per_text" K="${K:-10}" || failures=$((failures + 1))

run_sample "13b_cosine_top_k_ascii" "cosine_top_k_hot_path.rb" \
  MODE=ascii METRIC=cosine DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" ROWS="$default_rows" BYTES_PER_TEXT="$default_bytes_per_text" K="${K:-10}" || failures=$((failures + 1))

run_sample "13c_dot_top_k_ascii_f16" "cosine_top_k_hot_path.rb" \
  MODE=ascii FORMAT=f16 METRIC=dot DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" ROWS="$default_rows" BYTES_PER_TEXT="$default_bytes_per_text" K="${K:-10}" || failures=$((failures + 1))

run_sample "13d_cosine_top_k_ascii_f16" "cosine_top_k_hot_path.rb" \
  MODE=ascii FORMAT=f16 METRIC=cosine DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" ROWS="$default_rows" BYTES_PER_TEXT="$default_bytes_per_text" K="${K:-10}" || failures=$((failures + 1))

run_sample "14_cancellation_gvl_fairness" "cancellation_gvl_stress.rb" \
  DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" TIMEOUT="${TIMEOUT:-0.05}" INPUT_BYTES="$default_input_bytes" TEXTS="$default_texts" || failures=$((failures + 1))

run_sample "15_cancellation_full_scan" "cancellation_gvl_stress.rb" \
  FULL_SCAN=1 DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" TIMEOUT="${FULL_SCAN_TIMEOUT:-0.005}" INPUT_BYTES="$default_input_bytes" TEXTS="$default_texts" || failures=$((failures + 1))

run_sample "16_random_pooling" "random_pooling_hot_path.rb" \
  DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" TOKENS="${TOKENS:-512}" ID_SETS="${ID_SETS:-256}" || failures=$((failures + 1))

run_sample "16b_random_pooling_f16" "random_pooling_hot_path.rb" \
  FORMAT=f16 DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" TOKENS="${TOKENS:-512}" ID_SETS="${ID_SETS:-256}" || failures=$((failures + 1))

run_sample "17_warmup" "warmup_hot_path.rb" \
  DURATION="$default_duration" SLEEP_BEFORE_HOT_LOOP="$default_sleep" || failures=$((failures + 1))

# Single-shot checks finish too quickly for the profiler harness, so run them directly.
run_step "18_cold_start" env MODEL="$MODEL" "${ruby_cmd[@]}" samples/cold_start.rb || failures=$((failures + 1))
run_step "19_f16_quality" env MODEL="$MODEL" ROWS="${F16_QUALITY_ROWS:-256}" BYTES_PER_TEXT="$default_bytes_per_text" "${ruby_cmd[@]}" samples/f16_quality_check.rb || failures=$((failures + 1))

{
  echo
  echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "failures=$failures"
} >> "$out_dir/env.txt"

echo
cat "$out_dir/summary.tsv"
echo
if [[ "$failures" -eq 0 ]]; then
  echo "OK: all sample probe targets completed"
else
  echo "FAILED: $failures sample probe target(s) failed"
fi
echo "results=$out_dir"
exit "$failures"
