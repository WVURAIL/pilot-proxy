#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${1:-"$project_dir/../estimator_transfer_2026-08-25"}
runtime=${PILOT_PROXY_PYTHON:-/home/djg/rail/venvs/ppci/bin/python}

if [[ -e "$result_dir" ]]; then
  echo "Result directory already exists: $result_dir" >&2
  exit 2
fi

mkdir -p "$result_dir"
cd "$project_dir"

common=(
  -m pilot_proxy.testbench.evaluate_snr
  --input-iq generated/atsc/atsc_8vsb_complex64.cfile
  --waveform-audit-json generated/atsc/atsc_waveform_audit.json
  --physical-channel 14
  --frame-size-samples 16384
  --num-input-streams 4
  --detector-backend cuda
  --synthesis-backend cuda
  --noise-source gnuradio
  --gnuradio-python /usr/bin/python3
  --pilot-below-data-db-from-audit
  --spectral-sense normal
  --no-reference-archive-phase
)

run_point() {
  local label=$1
  local snr_db=$2
  local trials=$3
  local seed=$4
  local output_dir=$5

  mkdir -p "$(dirname "$output_dir")"
  PYTHONNOUSERSITE=1 PYTHONPATH=src "$runtime" -u "${common[@]}" \
    --requested-data-shelf-snr-db "$snr_db" \
    --noise-trials "$trials" --seed "$seed" \
    --output-dir "$output_dir" \
    >"$result_dir/${label}.log" 2>&1
}

wait_for_jobs() {
  local status=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

point_pids=()
run_point snr_m45 -45 240 2026082705 "$result_dir/extreme_low/snr_m45" & point_pids+=("$!")
run_point snr_p3 3 60 2026082706 "$result_dir/high" & point_pids+=("$!")
run_point snr_p6 6 60 2026082707 "$result_dir/high/snr_p6" & point_pids+=("$!")
run_point snr_p9 9 60 2026082708 "$result_dir/high/snr_p9" & point_pids+=("$!")
run_point snr_p12 12 60 2026082709 "$result_dir/high/snr_p12" & point_pids+=("$!")
run_point snr_p15 15 60 2026082710 "$result_dir/high/snr_p15" & point_pids+=("$!")
run_point snr_p18 18 60 2026082711 "$result_dir/high/snr_p18" & point_pids+=("$!")
run_point snr_p21 21 60 2026082712 "$result_dir/high/snr_p21" & point_pids+=("$!")
wait_for_jobs "${point_pids[@]}"

PYTHONNOUSERSITE=1 PYTHONPATH=src "$runtime" -u "${common[@]}" \
  --snr-start-db -42 --snr-stop-db -30 --snr-step-db 3 \
  --noise-trials 240 --seed 20260825 \
  --output-dir "$result_dir/lower" \
  2>&1 | tee "$result_dir/lower.log"

PYTHONNOUSERSITE=1 PYTHONPATH=src "$runtime" -u "${common[@]}" \
  --snr-start-db -27 --snr-stop-db 0 --snr-step-db 3 \
  --noise-trials 60 --seed 20260826 \
  --output-dir "$result_dir/upper" \
  2>&1 | tee "$result_dir/upper.log"

additional_pids=()
run_point lower_m45_part1 -45 380 2026082800 "$result_dir/lower_additional/snr_m45/part_1" & additional_pids+=("$!")
run_point lower_m45_part2 -45 380 2026082801 "$result_dir/lower_additional/snr_m45/part_2" & additional_pids+=("$!")
run_point lower_m42_part1 -42 380 2026082802 "$result_dir/lower_additional/snr_m42/part_1" & additional_pids+=("$!")
run_point lower_m42_part2 -42 380 2026082803 "$result_dir/lower_additional/snr_m42/part_2" & additional_pids+=("$!")
run_point lower_m39_part1 -39 380 2026082804 "$result_dir/lower_additional/snr_m39/part_1" & additional_pids+=("$!")
run_point lower_m39_part2 -39 380 2026082805 "$result_dir/lower_additional/snr_m39/part_2" & additional_pids+=("$!")
run_point lower_m36_part1 -36 380 2026082806 "$result_dir/lower_additional/snr_m36/part_1" & additional_pids+=("$!")
run_point lower_m36_part2 -36 380 2026082807 "$result_dir/lower_additional/snr_m36/part_2" & additional_pids+=("$!")
run_point lower_m33_part1 -33 380 2026082808 "$result_dir/lower_additional/snr_m33/part_1" & additional_pids+=("$!")
run_point lower_m33_part2 -33 380 2026082809 "$result_dir/lower_additional/snr_m33/part_2" & additional_pids+=("$!")
run_point lower_m30_part1 -30 380 2026082810 "$result_dir/lower_additional/snr_m30/part_1" & additional_pids+=("$!")
run_point lower_m30_part2 -30 380 2026082811 "$result_dir/lower_additional/snr_m30/part_2" & additional_pids+=("$!")
wait_for_jobs "${additional_pids[@]}"

inputs=(
  "$result_dir/extreme_low/snr_m45"
  "$result_dir/lower"
)
for snr in m45 m42 m39 m36 m33 m30; do
  inputs+=(
    "$result_dir/lower_additional/snr_${snr}/part_1"
    "$result_dir/lower_additional/snr_${snr}/part_2"
  )
done
inputs+=(
  "$result_dir/upper"
  "$result_dir/high"
  "$result_dir/high/snr_p6"
  "$result_dir/high/snr_p9"
  "$result_dir/high/snr_p12"
  "$result_dir/high/snr_p15"
  "$result_dir/high/snr_p18"
  "$result_dir/high/snr_p21"
)

plot_args=()
for input_dir in "${inputs[@]}"; do
  plot_args+=(--input-csv "$input_dir/dtv_snr_summary.csv")
done
for input_dir in "${inputs[@]}"; do
  plot_args+=(--trial-csv "$input_dir/dtv_snr_eval.csv")
done

PYTHONNOUSERSITE=1 PYTHONPATH=src "$runtime" \
  -m pilot_proxy.testbench.plot_results \
  "${plot_args[@]}" \
  --conditioning-trial-csv "$result_dir/lower/dtv_snr_eval.csv" \
  --conditioning-json "$result_dir/estimator_transfer_conditioning.json" \
  --bootstrap-samples 10000 \
  --smooth-window 1 \
  --y-min-db -60 \
  --output-png "$result_dir/estimator_transfer.png" \
  --title "GNU Radio input, CPU/GPU estimator transfer"
