#!/usr/bin/env bash
# Local CHIME workflow: one positive-excess detector run with offset diagnostics.
set -euo pipefail

INPUT_DIR="${PILOT_PROXY_CHIME_INPUT_DIR:-$HOME/dataset/canfar_pilots_10s}"
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "CHIME input directory not found: $INPUT_DIR" >&2
  echo "Set PILOT_PROXY_CHIME_INPUT_DIR to your CHIME baseband HDF5 directory." >&2
  exit 1
fi
ROOT=generated/chime_real

ANALYSIS_RUN=${PILOT_PROXY_CHIME_OUTPUT_DIR:-${ROOT}/canfar_pilots_10s_positive_excess_full}
FREQUENCY_OFFSET_HALF_WIDTH_HZ=${PILOT_PROXY_FREQUENCY_OFFSET_HALF_WIDTH_HZ:-10000}
FREQUENCY_OFFSET_BACKEND=${PILOT_PROXY_FREQUENCY_OFFSET_BACKEND:-auto}
FREQUENCY_OFFSET_STREAM_BATCH_SIZE=${PILOT_PROXY_FREQUENCY_OFFSET_STREAM_BATCH_SIZE:-2048}
FRAMES_PER_CHUNK=${PILOT_PROXY_FRAMES_PER_CHUNK:-2}
MAX_FRAMES=${PILOT_PROXY_MAX_FRAMES:-}

if [[ "${PILOT_PROXY_CLEAN_RUNS:-0}" == "1" ]]; then
  rm -rf "$ANALYSIS_RUN"
fi

echo "Using CHIME input directory: $INPUT_DIR"
echo "Using frequency-offset search half-width: ${FREQUENCY_OFFSET_HALF_WIDTH_HZ} Hz"
echo "Using frequency-offset backend: ${FREQUENCY_OFFSET_BACKEND}"
echo "Using frequency-offset stream batch size: ${FREQUENCY_OFFSET_STREAM_BATCH_SIZE}"
echo "Using frames per chunk: ${FRAMES_PER_CHUNK}"
echo "Using CHIME output directory: ${ANALYSIS_RUN}"
if [[ -n "$MAX_FRAMES" ]]; then
  echo "Using max frames: ${MAX_FRAMES}"
fi

CHIME_RUN_EXTRA_ARGS=()
if [[ -n "$MAX_FRAMES" ]]; then
  CHIME_RUN_EXTRA_ARGS+=(--max-frames "$MAX_FRAMES")
fi

PYTHONPATH=src python -m pilot_proxy.cli chime-run \
  --input-dir "$INPUT_DIR" \
  --physical-channel-range 14:36 \
  --frames-per-chunk "$FRAMES_PER_CHUNK" \
  --frequency-offset-diagnostic \
  --frequency-offset-peak-search-half-width-hz "$FREQUENCY_OFFSET_HALF_WIDTH_HZ" \
  --frequency-offset-backend "$FREQUENCY_OFFSET_BACKEND" \
  --frequency-offset-stream-batch-size "$FREQUENCY_OFFSET_STREAM_BATCH_SIZE" \
  --output-dir "$ANALYSIS_RUN" \
  --plot \
  "${CHIME_RUN_EXTRA_ARGS[@]}"

PYTHONPATH=src python -m pilot_proxy.cli choose-detector-k \
  --frequency-offset "$ANALYSIS_RUN/frequency_offset_outputs.npz" \
  --candidate-k 128 256 \
  --candidate-reference-spacing-policy fixed_skipped_guard fixed_hz_reference_spacing \
  --min-peak-prominence-db 25 \
  --max-capture-loss-db 1.0 \
  --output "$ANALYSIS_RUN/tables/k_candidate_summary.csv"

PYTHONPATH=src python -m pilot_proxy.cli validate-products \
  --run-dir "$ANALYSIS_RUN" \
  --output-json "$ANALYSIS_RUN/product_validation.json"
