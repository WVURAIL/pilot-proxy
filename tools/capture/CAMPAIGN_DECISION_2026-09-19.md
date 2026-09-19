# The cadence campaign: designed, verified, and not recommended

A replacement cadence campaign was designed after the first proposal was shown to produce no
measurable coherence time at all. The replacement is geometrically sound. It is still not worth
flying, and this records why, so the decision is not revisited from memory.

## What was wrong with the first proposal

Eight dumps as four same-day pairs. cadence_tau.py line 92 discards every pair separated by a
sidereal day or more, so all cross-day pairs vanished: four surviving pairs, zero above the 7200 s
plateau start, no plateau, no crossing target, no coherence time on any channel. It would have cost
four days and 5.7 TB and returned nothing.

## What was wrong with my own framing

I said filling the empty 1200 to 7178 s lag gap was the most valuable thing a campaign could do.
That is false. Inverting the ruling against each range's own ratio at unit gain, every decision
threshold in the target set lies between 24.7 and 620.4 s: channel 22's long range at 24.7 s,
channel 14 at 31.1, channel 18 at 96.1, channel 22's short at 98.3, channel 25 at 102.2, channel 26
at 126.7, channel 16 at 131.9, channel 19 at 163.2, channel 27 at 620.4. All are below 1200 s.
Filling the gap decides no channel. The blockers are the estimator's trim-spread refusals (14, 18,
22, 27, 28) and constant-plateau returns (19, 26), not missing lag coverage.

## The replacement design, and why it still fails the cost test

Twenty dumps of 0.215 s in four posted waves at T0, +2200, +20000 and +42000 s, 2.19 TB written,
1.2 TB peak resident. Simulated through a re-implementation of the estimator built from source it
gives 190 pairs, eleven populated classes below 7200 s, 112 plateau pairs and no pair double-counted
between the plateau and class 3600. The geometry works. Three adversarial checks nonetheless return
"works with corrections", and the corrections are decisive.

1. It replaces the existing measurement rather than adding to it. Every pair between a record epoch
   and a new one exceeds a sidereal day and is discarded at line 92. The record carries 150 frames
   across fourteen epochs, three of them 33-frame science dumps that supply every long-lag pair. The
   campaign would trade that for about 100 frames of 5-frame dumps.
2. The yield case was out by about a factor of two. It modelled the record as fourteen uniform
   4-frame epochs. Recomputed against the true depth, the marginal gains are roughly half those
   claimed.
3. The refusal ceiling is structural, not a noise limit. At zero measurement noise on the new
   geometry the six priced rows still refuse 35 to 41 per cent of the time, and depth buys about
   five points between 5-frame epochs and infinite depth. The refusals come from the 75/90/95 trim's
   truncation behaviour, which no geometry fixes.
4. The grid is biased against the excisions it seeks. Seven of its eleven classes sit off-centre,
   and at channel 16's own 131.9 s threshold it returns 0.899 of the true coherence time, a 10 per
   cent undershoot in the direction that hides an excision.
5. Practical cost beyond the bytes: the firing script hard-codes a fixed offset ladder and cannot
   post these waves; about 90 GiB must be reclaimed before the first post against 73 GiB free; and
   the transfer holds a shared 45 to 50 MB/s path for twelve to thirteen hours.

## Decision

Do not fly it. The expected return is a few points of probability per channel on about six channels,
in a direction that is mostly excise-or-still-undetermined, bought by discarding a deeper existing
dataset. Defend the thesis on what is banked.

## What this leaves, and it is not nothing

Six channels excised on the 2026 capture, channel 24 excised on the 2020 basis, twelve channels
carrying a published numeric bound, three dated against the archive's transmitter record, and no
channel keeping anywhere. The one genuinely open item is a post-upgrade dump that records freq_id
676 to 691, which would let channel 24 be ruled on this capture's own basis and remove both its
conditionals. It is correctly carried as pending and it is outside the graduation timeline.

Also checked and refuted: that channel 28 could return a keep. Its long range is unpriced, its
keep-all reading being at the control floor, so no keep can be formed there at all; a keep reached
through a calibrated policy while keep-all sits at the floor is the artifact amendment 11 item 2
removed for channel 18. No channel keeps, and no achievable campaign changes that.
