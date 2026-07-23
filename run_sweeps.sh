#!/bin/bash
export CFILE=/mnt/c/Users/dylan/DataspellProjects/pilot-proxy/generated/atsc/atsc_8vsb_stationary_trimmed.cfile
source ~/pilot-proxy-datatrawl/bin/activate
JOBS=$(( $(nproc) - 2 ))
mkdir -p ~/pd_pts
for off in 0 -1000 1000 -1526 1526; do
  for ((snr=-60; snr<=-20; snr+=3)); do
    echo "$off $snr"
  done
done | xargs -P "$JOBS" -n 2 bash -c '
  pilot-proxy evaluate-snr \
    --input-iq "$CFILE" \
    --detector-backend cpu-reference \
    --noise-source python \
    --num-input-streams 4 \
    --frequency-offset-hz "$0" \
    --requested-snr-shelf-db "$1" \
    --noise-trials 1000 \
    --output-dir "$HOME/pd_pts/off_${0}_snr_${1}" \
    > "$HOME/pd_pts/off_${0}_snr_${1}.log" 2>&1'
echo "ALL POINTS DONE"
