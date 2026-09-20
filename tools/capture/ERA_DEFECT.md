# The era boundary is not a transmitter date on five channels

Found 2026-09-19 by reading the per-channel spectrograms against the archive's recorded era starts.

## What the dating run actually does

`sections/era/current_first_month` in the per-channel ledger is cut on three different grounds, and
`current_evidence` says which. Across the 23 channels:

| what cut the current era | channels | count |
|---|---|---|
| archive start, i.e. no change at all | 14, 16, 18, 21, 22, 23, 25, 28, 29, 30, 34 | 11 |
| spectral-state transition, a real transmitter event | 19, 20, 26, 27, 32, 33, 36 | 7 |
| **station change, only the carrier's peak drifting in frequency** | **15, 17, 24, 31, 35** | **5** |

The five station-change channels are exactly the ones whose era lines disagree with the spectrograms.
Their carriers are continuously present; what moved was the peak position, by a fraction of a bin per
month. Channel 35 has ELEVEN eras on that basis and channel 31 has eight, with nine recorded station
excursions of 3 to 10 bins. Channel 17 drifts at 0.46 bins per month and channel 15 at 0.29.

## Why it matters

Amendment 1 condition (a) says the era "begins at the last persistent level change in the dating
run". The dating run does not implement that: it also cuts on position. So for these five the
"current era" is an artificially short and recent window that marks no transmitter event.

Read off the spectrograms, the last change in transmitter STATE is:

| channel | recorded era start | what the carrier actually does | recorded evidence |
|---|---|---|---|
| 15 | 2025-05 | carrier appears about 2021.6 | drift |
| 17 | 2025-10 | carrier continuous from 2019 | drift |
| 24 | 2019-03 | carrier present in every covered month to 2026-04 | drift |
| 31 | 2025-04 | carrier continuous from 2019 | drift |
| 35 | 2025-11 | carrier appears about 2021.8 | drift |

## What it does NOT change

No verdict. Amendment 1 condition (a) requires the current era to carry NO pilot, and all five of
these are proxy-high with a pilot present, so they fail (a) wherever the boundary is drawn. The
excisions of 15, 17, 31 and 35 rest on the 2026 capture and not on the archive era at all, and
channel 24's conditional rests on `off_era_current = False` and `n_off_frames = 0`, which are
whole-archive quantities rather than era-scoped ones.

## What it does change

1. Era dates must not be quoted as transmitter dates for these five. They are dating a frequency
   drift.
2. The archive chain quantities are computed on the current era, so for the five the coherent count
   and chain gain come from an artificially short window. That reaches the archive's booked
   ground-filter shares, which the ruling reports in its robustness comparison but does not use.
3. Where the thesis needs a transmitter date it should cite the spectral-state transition, which is
   what the seven state-cut channels carry, or the carrier itself as measured here.

## The figure

`channel_spectrograms.png`, built by `build_spectrograms.py`, now colours the era line by
`current_evidence`: solid for a state change, dotted for drift, dashed for an archive start. The
monthly reduction is cached in `spectrogram_cache.npz`, so redrawing is seconds.
