#!/usr/bin/env bash
set -euo pipefail

# Standalone CUDA + GNU Radio release sanity check. The defaults below are
# environment-specific (a WSL + miniconda CUDA setup) and will not match most
# machines -- override them as needed:
#   SM                    GPU compute capability (nvidia-smi --query-gpu=compute_cap --format=csv)
#   CUDA_PYTHON           interpreter with cupy + this package installed
#   GNURADIO_PYTHON       interpreter with GNU Radio (for the ATSC generator)
#   CUDA_LD_LIBRARY_PATH  path to the CUDA/driver libs (WSL: /usr/lib/wsl/lib)
# This assumes a CUDA-capable environment; it is NOT a CPU-only smoke test.
SM="${SM:-89}"
CUDA_PYTHON="${CUDA_PYTHON:-$HOME/miniconda3/envs/pilot-proxy/bin/python}"
GNURADIO_PYTHON="${GNURADIO_PYTHON:-/usr/bin/python3}"
CUDA_LD_LIBRARY_PATH="${CUDA_LD_LIBRARY_PATH:-/usr/lib/wsl/lib}"
unset LD_LIBRARY_PATH

make build-kernel SM="$SM"

PYTHONNOUSERSITE=1 PYTHONPATH=src "$GNURADIO_PYTHON" \
  -m pilot_proxy.testbench.generate_atsc_signal \
  --output-iq generated/atsc/atsc_8vsb_complex64.cfile \
  --num-iq-samples 600000

PYTHONPATH=src "$CUDA_PYTHON" \
  -m pilot_proxy.testbench.audit_atsc_signal \
  --input-iq generated/atsc/atsc_8vsb_complex64.cfile

PYTHONPATH=src "$CUDA_PYTHON" \
  -m pilot_proxy.testbench.quantize \
  --input-iq generated/atsc/atsc_8vsb_complex64.cfile \
  --physical-channel 14 \
  --frame-size-samples 16384 \
  --num-input-streams 1

LD_LIBRARY_PATH="$CUDA_LD_LIBRARY_PATH" PYTHONPATH=src "$CUDA_PYTHON" \
  -m pilot_proxy.testbench.evaluate_snr \
  --input-iq generated/atsc/atsc_8vsb_complex64.cfile \
  --output-dir generated/quickstart/dtv_snr_eval \
  --physical-channel 14 \
  --frame-size-samples 16384 \
  --num-input-streams 1 \
  --requested-snr-shelf-db -26 \
  --noise-trials 10 \
  --noise-source gnuradio \
  --gnuradio-python "$GNURADIO_PYTHON"

LD_LIBRARY_PATH="$CUDA_LD_LIBRARY_PATH" PYTHONPATH=src "$CUDA_PYTHON" \
  -m pilot_proxy.detect \
  --input-detector-matrix generated/detector_input/detector_matrix_i4.npy \
  --physical-channel 14
