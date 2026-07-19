# Survey-composition and FRB-stratification provenance (2026-07-18)

We generated this directory with `analysis/survey_composition.py`. The script
joins `inventory.jsonl`, `event_presence_keys.csv.gz`, and the per-frame dump.
These source inputs are not committed here, so the CSVs in this directory are
retained results rather than a self-contained regeneration bundle.

## Unit-to-event assignment

The detector products identify capture units by time, while the inventory
identifies events by observing date. The script searches the same day and
`+/-1` day, then uses frame count to choose among candidate events. On
singleton days, equality between unit frame count and the floored inventory
`n_frames` value was recorded as 97.4%. That value cannot be independently
recomputed without the uncommitted source inputs.

The script records exact matches, class-consistent ambiguous matches,
single-class day fallbacks, mixed-class cases, and orphans. Mixed and orphaned
units do not enter the `classified.FRB` stratum. In the retained
`survey_assignment_quality.csv`, the assigned fraction ranges from 0.7465 to
0.8312 by channel. Therefore, the class-stratified rates are conditional on
the assigned subset and remain provisional. The range was independently
recomputed from the retained CSV.

## Files

- `survey_assignment_quality.csv` gives the assigned and unresolved unit
  counts for every channel.
- `survey_composition_by_channel.csv` compares the sampled and archive event
  classes.
- `survey_quarterly_exposure.csv` gives quarterly unit exposure for the four
  episodic channels used in the stratified figure: 17, 32, 33, and 35.
- `survey_frb_stratum_rates.csv` gives quarterly high-tail rates for those
  four channels in the all-event and `classified.FRB` strata.
- `survey_quarterly_rates_all23.csv`, added on 2026-07-18, extends the same
  quarterly calculation to every calibrated channel. Despite the historical
  filename, it contains 21 channels: channels 24 and 30 are excluded because
  they do not have trusted empirical zero points.

A quarterly high-tail rate is the fraction of valid frames satisfying

```text
F > mu_hat + 12e-3 * mu0
```

The script requires at least 40 valid all-event frames for a quarter. It
reports an FRB-stratum rate only when that stratum also contains at least 40
valid frames.

## Recorded interpretation

The percentages in this section were independently read or recomputed from
the retained quarterly CSV. They remain conditional on the incomplete event
assignment described above and are not estimates for unsampled quarters.

The completeness scan was created to test the Sec. 6.3 and Sec. 8.1 secular
claims. Its retained values support a gradual channel-34 change: all-event
rates are approximately 3--8% from 2018 through 2020, approximately 2% in
2021, and mostly 0--2% afterward. The later record is not uniformly below 1%:
for example, the all-event rate is 2.17% in 2024 Q2 and 3.02% in 2026 Q2,
while the corresponding FRB rates are 1.46% and 0.35%.

Channel 27 contains one sampled loud interval at 37.98% in 2020 Q3
(36.92% in the FRB stratum). Its sampled 2025--2026 endpoint is quieter, but
the intervening years are not sampled. This is an isolated observed interval,
not a measured transition date.

The same CSV is the evidence source for the recorded tier-flatness checks,
including channels 20 and 22 and the first-adoption group. Those statements
must be reported with the quarterly exposure and assignment denominators; the
CSV does not establish behavior in unsampled quarters.
