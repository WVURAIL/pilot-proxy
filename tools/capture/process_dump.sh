#!/bin/bash
# Bring one reduced dump home and run everything on it: products, pilot files, both kernel banks, frame analysis,
# ladder placement. usage: process_dump.sh <event_id>   (needs the event reduced on the analysis host: 280 products)
# Site configuration: CHIME/FRB internal values, never committed; export them before running.
#   JUMP_HOST      ssh jump host into the CHIME network
#   ANALYSIS_HOST  ssh destination of the analysis host, <user>@<host>
#   USER_DATA_DIR  the operator's user-data directory on the analysis host (products are in pilot_reduce/)
# Reading CHIME baseband data requires CHIME/FRB authorization; see dumps/README.md.
set -u
: "${JUMP_HOST:?}" "${ANALYSIS_HOST:?}" "${USER_DATA_DIR:?}"
EV=$1
R=/home/djg/rail/output/channel-ruling-execution-2026-09-14/rebuild/author_actions/capture-runbook/reduce
PY=/home/djg/rail/venvs/archive-local/bin/python
W=$R/weights_ch33/chime_dtv_weights_k128_ch33measured.bin
mkdir -p /home/djg/rail/datasets/pilot_reduce_$EV
rsync -a -e "ssh -o ConnectTimeout=20 -J $JUMP_HOST" "$ANALYSIS_HOST:$USER_DATA_DIR/pilot_reduce/$EV/" /home/djg/rail/datasets/pilot_reduce_$EV/ 2>&1 | grep -v conda
n=$(ls /home/djg/rail/datasets/pilot_reduce_$EV/*.npz 2>/dev/null | wc -l); echo "$EV products $n"; [ "$n" -ge 280 ] || { echo "not fully reduced"; exit 1; }
$R/pull_pilots_ready.sh $EV 2>&1 | tail -1
L=/home/djg/rail/datasets/pilot_dump_$EV; CH=""
for f in $(ls $L | sed 's/.*_//; s/.h5//'); do ch=$(python3 -c "f=800-$f*0.390625; print(14+int((f-470)//6))"); [ "$ch" -le 36 ] && CH="$CH --physical-channel $ch"; done
cd /home/djg/rail/pilot-proxy
for tag in k230 k230_ch33measured; do O=$R/kernel_${EV}_$tag; rm -rf $O; extra=""; [ $tag = k230_ch33measured ] && extra="--weights-path $W"
  PYTHONPATH=src $PY -m pilot_proxy.cli chime-run --input-dir $L --output-dir $O $CH --frame-size-samples 16384 --frames-per-chunk 2 --lib-path /home/djg/rail/pilot-proxy/cuda/libfstatistic.so $extra 2>&1 | grep -iE "error" | head -2
  PYTHONPATH=src $PY -m pilot_proxy.cli validate-products --run-dir $O --output-json $O/product_validation.json 2>&1 | tail -1
done
cd $R/frame_analysis
$PY frame_residual.py /home/djg/rail/datasets/pilot_reduce_$EV $R/kernel_${EV}_k230 frame_residual_$EV.csv 2>&1 | grep -E "rows ->|Error"
$PY ladder_place.py $R/kernel_${EV}_k230 2>&1 | awk "/kernel_${EV}/{p=1} p" | head -20
sha256sum $R/kernel_${EV}_k230/chime_detector_outputs.npz $R/kernel_${EV}_k230_ch33measured/chime_detector_outputs.npz frame_residual_$EV.csv >> HASHES_frame_analysis.txt
B="/mnt/c/Users/dylan/West Virginia University/WVU RAIL - Documents/RFI Mitigation/Datasets/matched-capture-2026-09/science_dumps/$EV/reduce"; mkdir -p "$B" && cp -r $R/kernel_${EV}_k230 $R/kernel_${EV}_k230_ch33measured "$B/" && cp frame_residual_$EV.csv "$B/" && echo "$EV backed up"
