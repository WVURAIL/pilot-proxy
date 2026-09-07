# Deferred until after the archive run (repo is frozen at the tag)

Source and docs changes that would dirty the worktree, in rough priority order.

1. Remove the datatrail prefix workaround once CHIMEFRB fixes the service
   (_restore_collection in datatrail_client.py is a no-op after the fix; the
   upstream report draft is at
   UPSTREAM_DATATRAIL_PREFIX_REGRESSION.md in the archived inventory-rebuild
   evidence (WVU OneDrive, Datasets/pilot-tone-pipeline/archive/
   inventory_rebuild_2026-08/), venue
   CHIMEFRB/datatrail -- file it whenever, independent of the run).
2. Promote the outage-seam rehearsal into the test tree
   (evidence/40_freeze_replay/rehearsal-20260830/outage_rehearsal.py) so the
   production entry point's outage behavior is pinned in CI.
3. LOCAL_PROCESSING.md: repin the v5 bundle blocks to the rebuild (inventory
   path/sha, no exclusions files), fix the cert-renewal block to the
   two-command form (bare canfar login yields 10-day certs), document
   --stop-after-checkpoint (exists in the CLI, in no doc), and record that
   release-check deletes cuda/libfstatistic.so (pin the digest-named kernel).
4. RERUN_PARAMETER_REGISTER.md: cert expiry line is stale (says 2026-09-03).
5. Launch-script nits from the mutation test (G6): EXIT trap instead of RETURN
   in gate; a proper LAUNCH-BLOCK message for a missing venv; optionally assert
   the cert CN.
6. Port the DS001 data sheet / UG001 user guide TeX sources out of the datatrawl
   bundles and attach them to pilot-proxy releases (the article's Data
   Availability statement now promises this; the only sources are
   ~/rail/repo-archives/datatrawl-*.bundle and the local clone).
7. PAPER_PLAN.md / CANFAR_RUNBOOK.md prose mentions of datatrawl (no links;
   cosmetic).
8. Push the dissertation commit 1b154eb (datatrawl reference removal) --
   local-only until pushed.
9. Delete the datatrawl GitHub repos when ready (both already archived;
   bundles verified AND restore-tested 2026-08-30; also decide
   datatrawl-analyzer-template).
10. Consider wiring accept_products.py and run_status.sh into the repo as
    tools/ after the run, with tests.
11. v6 schema consideration: p_ref_sum_u64 is derivable (lower+upper);
    kept in v5 as the integrity identity the gates assert. Decide keep
    (self-checking product) vs drop (8 bytes/frame) for any v6.

12. **Rename `.datatrawl.lock` runtime lock-file suffix.** The archive scan's  [status: open]
    per-product lock files are named `.<product>.datatrawl.lock` (seen live in
    `_per_pilot/` during the 20260829 run). The suffix comes from the ported
    runtime source. Transient dotfiles, so no release artifact carries the
    name, but the source string must be renamed (e.g. `.pilotproxy.lock`)
    before publication, alongside any other in-source `datatrawl` strings.
    Cannot change during the frozen run.

13. **Reclassify staging-file ENOENT/truncation as transient, not quarantine.**  [status: done]
    Observed 2026-09-01 on CANFAR shard 2: staging dir removed during the
    post-SIGINT drain; the reader raised UnreadableUnitError on 7 vanished
    staged files and quarantined the units permanently (silent scope loss on
    every later resume -- caught only by the tripwire's unexpected-class
    check). probe/read failures whose cause is a missing or short STAGED copy
    (errno 2, or size < the inventory's size_bytes) should requeue the fetch
    instead of writing a quarantine row. Evidence: shard-2 quarantine backup
    `quarantine.jsonl.pre_repair_20260901` beside the live ledger.

14. **Real interrupt-resume rehearsal + SIGINT hygiene.** The B2d phase-1  [status: done (handler restored, join 30 s; the live interrupt rehearsal is still to run)]
    SIGINT was a no-op (bash async job -> SIGINT SIG_IGN inherited; scan ran
    to completion). Any claim citing B2d as interrupt evidence should cite
    the SIGKILL test / machine-crash recovery instead. Post-run: rerun a
    genuine interrupt rehearsal with the fixed foreground-fork launcher
    (setsid -f), and consider having the CLI restore SIGINT to default at
    startup when it finds it ignored, plus raising _WORKER_JOIN_SECONDS
    (2.0 s) so an interrupt exit is a drain rather than a scratch-retention
    abort.

15. **Make the storage service selectable (outage resilience).**  [status: done in b59b5c0 (PILOT_PROXY_STORAGE_SERVICE)]
    `_make_client` hardcodes the library default
    `ivo://cadc.nrc.ca/global/raven`. When global raven broke on 2026-09-01,
    every fetch failed even though the UVic minoc replica was healthy and
    served byte-identical objects. Add a `--storage-service` option (or
    `PILOT_PROXY_STORAGE_SERVICE` env var) threaded into
    StorageInventoryClient(resource_id=...), record the resolved value in
    scan_scope.json execution metadata so the data path is provenance-visible,
    and consider automatic failover across replicas on repeated locator 5xx.
    Verified bypass command and evidence: run ledger, "Verified bypass exists".

16. **Make staging removal safe by construction.** Twice now (06:04 and 23:10  [status: superseded: the CANFAR launcher scripts are retired]
    on 2026-09-01) a staging directory was removed under a live scan, and both
    times readable units were permanently quarantined as staging-ENOENT.
    Deferred item 13 fixes the misclassification; this adds the operational
    half: `canfar_shard.sh` should own staging removal behind a mode that
    refuses while any scan matching that run directory is alive, so no
    hand-typed `rm -rf` is ever the interface.

17. **Retry transient transport failures instead of aborting the run.**  [status: done]
    `cadc_transport.expected_errors()` returns (OSError, cadcutils
    HttpException, requests RequestException). A raw
    `urllib3.exceptions.ProtocolError` -- "Connection broken:
    IncompleteRead(N bytes read, M more expected)" -- matches none of them, so
    fetch() does not retry it, the download worker dies, and
    ActiveDownloadWorkersError ends the whole scan. Observed twice on
    2026-09-02 at ~01:17 UTC, killing both shards within the same minute after
    ~35 min of clean running. Add the urllib3 exception family (and
    http.client.IncompleteRead) to expected_errors so a truncated transfer is
    retried like any other transient. Until then, canfar_supervise.sh restarts
    the scan externally.

18. **Format exceptions defensively in the fetch retry loop.**  [status: done]
    `sources/cadc.py:466` builds its retry message with
    `f"{type(exc).__name__}: {exc}"`. cadcutils raises an HttpException whose
    `__str__` returns a non-string, so formatting it raises
    `TypeError: __str__ returned non-string (type HTTPError)`, which escapes
    the `except self._expected_errors` clause, kills the download worker, and
    ends the scan -- turning an error the retry loop exists to absorb into a
    fatal one. Observed on both shards on 2026-09-03. Use a defensive
    formatter (`repr(exc)` inside try/except, or
    `getattr(exc, "orig_exception", exc)`) so no exception's own
    representation can end a run. Pairs with item 17 (urllib3 ProtocolError
    not being retried at all).
