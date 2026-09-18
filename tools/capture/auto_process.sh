#!/bin/bash
# Pull and process every reduced event as it completes on frb-analysis (280 products), once each.
RB=/home/djg/rail/output/channel-ruling-execution-2026-09-14/rebuild/author_actions/capture-runbook
DONE=$RB/reduce/done_events.txt; LOG=$RB/reduce/auto_process.log
touch $DONE
for i in $(seq 1 300); do
  out=$(ssh -o ConnectTimeout=20 -J chimenet dgormley@frb-analysis 'for d in /data/user-data/dgormley/pilot_reduce/2026091*; do n=$(ls $d/*.npz 2>/dev/null | wc -l); [ "$n" -eq 280 ] && echo "$(basename $d)"; done' 2>/dev/null | grep -v conda)
  for ev in $out; do
    grep -q "^$ev$" $DONE && continue
    echo "$(date -u +%FT%TZ) processing $ev" >> $LOG
    (cd $RB && bash reduce/process_dump.sh $ev > reduce/process_$ev.log 2>&1); rc=$?
    echo "$(date -u +%FT%TZ) $ev exit $rc" >> $LOG
    [ $rc -eq 0 ] && echo "$ev" >> $DONE
  done
  sleep 120
done
