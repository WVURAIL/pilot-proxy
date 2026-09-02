# CANFAR shard tooling for the 2026-09 archive run

These are the exact scripts that ran the CHIME pilot-tone archive scan on
CANFAR in September 2026, committed as provenance rather than as a general
deployment kit. Paths under `/arc/home/dgormley/`, the session hostnames, the
channel partition, and every digest they assert are specific to that run and
are meant to be read alongside the run ledger.

| file | role |
|---|---|
| `canfar_probe_bootstrap.sh` | Per-node setup: clone at the frozen revision, restore the runtime offline from the freeze bundle wheelhouse, build and preserve the node's kernel, fetch-throughput probe. Run once per session. |
| `canfar_smoke_844.sh` | Cross-arch qualification: rerun the local B2d rehearsal (channel 844, first 8 files) on the node and compare the product unit-by-unit against the local sm89 reference. Integer fields must be bit-identical. |
| `canfar_shard.sh` | Shard controller: `update`, `gate`, `launch`, `resume`, `run` (foreground, for headless sessions), `tripwire`, `status`, `stop`. Every launch/resume re-runs the full gate chain, ending in an md5-verified byte fetch over the route the scan will actually use. |
| `canfar_supervise.sh` | Keeps one shard running across archive outages and transient fetch failures by resuming from outside; distinguishes a refused gate (nothing ran) from a scan that ran and exited. |
| `canfar_shard2_repair.sh` | One-off ledger repair for the 2026-09-01 staging-removal incident; kept because the incident is in the ledger. |
| `MORNING_CANFAR.md` | Operator runbook as it stood on 2026-09-01. |

Things these scripts taught us, each recorded in the run ledger and, where it
is a code change, in `POST_RUN_DEFERRED.md`:

* a bash async job (`nohup cmd &`) inherits SIGINT ignored, so scans are
  started with `setsid -f` and stopped with INT then TERM;
* `/minoc/capabilities` stays HTTP 200 through outages -- only a real byte
  fetch, or `/raven/availability`, proves the archive is serving;
* a pinned replica must be throughput-probed before use (UVic measured
  0.18-1.7 MiB/s against raven's 15-160 when both were healthy);
* removing a staging directory under a live scan permanently quarantines the
  units in flight -- never remove staging until `stop` confirms exit;
* one dead download worker ends the scan, and not every transient is
  retried inside `fetch()`; the supervisor exists until that is fixed.

See `docs/CANFAR_RUNBOOK.md` for the general procedure and
`docs/CADC_OUTAGE_2026-09-01.md` for the outage report filed during the run.
