# artifacts/ — data behind the published survey pages

Snapshots of the JSON inlined into the two published artifact pages:

| file | page | built by |
|---|---|---|
| report_data.json | Pilot Proxy Trawl (pilot_proxy_trawl.html) | analysis/make_report_data.py |
| policy_data.json | DTV Masking Policy (dtv_masking_policy.html) | analysis/make_policy_data.py |

Snapshot state: the complete 23-channel per-pilot products of 2026-08-19.

Regenerate on a new products snapshot and render the pages:

    python3 analysis/make_report_data.py --products DIR --out out
    python3 analysis/make_policy_data.py --products DIR --out out
    python3 analysis/render_artifacts.py --data-dir out

Both make_* scripts need the released `baonoise` package
(bao-noise-tolerance). The policy methodology constants (since=2025-01,
the coherence bracket, the inclusive-keep and collection-ceased
overrides, the eta grid) are locked in make_policy_data.py; changing
them is a policy revision, not a rebuild.
