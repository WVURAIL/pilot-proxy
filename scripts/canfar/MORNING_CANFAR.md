# Morning runbook v2 -- updated 2026-09-01 ~14:20 UTC

OVERNIGHT NEWS: at ~11:04 UTC the CADC permissions service broke
("unexpected exception calling permissions service(s)") and ALL THREE scans
-- both CANFAR shards and the local run -- died cleanly within the same
minute (fatal fetch -> ordered stop -> durable flush -> exit). Nothing is
running anywhere; there is NOTHING TO KILL. The v1 kill steps are obsolete.

Progress banked before the outage (all durable):
  shard1: 506 = 5,108/8,311 units; 521 untouched
  shard2: 598 COMPLETE (1,534 done + 9 quarantined, 7 of them the poison
          rows); 614 = 4,648/8,940; 629 just started
  local : 506 partial; stays stopped (evidence only -- CANFAR replaces it)

## 1. In any CANFAR session terminal (notebook2 if alive, else a new one):
    bash /arc/home/dgormley/pp_switch/canfar_shard2_repair.sh
No process is running, so it goes straight to the ledger edit: backs up,
drops exactly the 7 staging-ENOENT rows, keeps all legitimate sub-frame
rows. The 7 units then re-enter scope on resume.

## 2. Wait for CADC recovery, then relaunch both shards
Check recovery any time (each gate refuses while the outage lasts, so it is
always safe to try):
    PP_RESUME_CONFIRM=YES bash /arc/home/dgormley/pp_switch/canfar_shard.sh 1 resume   # notebook1
    PP_RESUME_CONFIRM=YES bash /arc/home/dgormley/pp_switch/canfar_shard.sh 2 resume   # notebook2
Resume re-enters 506/521 and 614/629 (+ the 7 freed 598 units), keeps all
overnight work. The fixed launcher makes `canfar_shard.sh N stop` work from
now on. If the notebook sessions died with the platform trouble (skaha was
returning 500s), launch two fresh GPU sessions and run the SAME commands --
each new node needs `bash /arc/home/dgormley/pp_switch/canfar_probe_bootstrap.sh`
once first (env var PP_SKIP_PROBE=1 skips the throughput probe).

## 3. After ~15 min: verify
    bash /arc/home/dgormley/pp_switch/canfar_shard.sh 1 status
    bash /arc/home/dgormley/pp_switch/canfar_shard.sh 2 tripwire
Shard-2 tripwire expectation: unexpected 0.

## Notes
- Outage signature vNEW: minoc /capabilities answers 200 while cadcget fails
  with the permissions-service error -- a 200 no longer proves health; only
  the byte gate does (it is part of every launch/resume gate).
- /tmp staging dirs: leave them alone; resume reuses them safely.
- A recovery watcher is polling from the local machine; Claude will have
  current status when you return.
