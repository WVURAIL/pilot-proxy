# Per-pilot product schema

```text
schema_name = "pilotproxy_per_pilot_product"
schema_revision = 3
schema_version = "pilotproxy_per_pilot_product_v3"
source_event_key_schema_version = "pilotproxy_namespaced_source_event_key_v1"
```

This is the only supported per-pilot product contract. PilotProxy had no public
product release before this revision, so development snapshots are not
accepted, converted, repaired, or resumed. Delete them and regenerate from
authoritative inputs with the current checkout.

Detailed arrays are listed in
[`PER_PILOT_PRODUCT_FIELDS.md`](PER_PILOT_PRODUCT_FIELDS.md) and
[`FINE_REDUCTION_PRODUCTS.md`](FINE_REDUCTION_PRODUCTS.md).

## Decision contract

`decision_contract_json` separates three roles:

1. `active_decision`: `coarse_normalized_positive_excess`, implemented as
   an exact host integer comparison and stored in `reject_mask`;
2. `fine_diagnostic`: `per_frame_robust_null_bulk_threshold`, diagnostic only;
3. `fine_candidate_decision`: `fine_order_statistic_cfar`, implemented and
   bit-tested in kernel core 2.3.0 but inactive while runtime calibration is
   `pending_campaign`.

A stored fine spectrum or diagnostic threshold exceedance therefore does not
imply that the fine candidate produced the rejection mask.

## Fine null-bulk diagnostic

`fine_null_bulk_exceedance_fraction` is the fraction of the same independent
null-bulk bins used to estimate a frame's diagnostic threshold that exceed that
threshold. It is an in-sample diagnostic, not an independently measured
false-alarm probability or false-alarm rate.

## Resume and combine boundary

Resume and combine require the exact schema identity, decision contract,
geometry, timing identity, weights, and detector provenance. A mismatch is an
error. Current runtime code contains no aliases, adapters, repair paths, or
fallback readers for pre-release products.

The source-event identity version is required even though the enclosing product
schema remains revision 1. It gates the change from historical basename-only
keys to keys that retain the complete archive or campaign namespace. Products
without this field cannot be combined or resumed because identical basenames in
different campaigns are not the same acquisition.
