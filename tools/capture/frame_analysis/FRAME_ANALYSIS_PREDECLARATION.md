# Frame-level analysis of the matched-capture dumps: predeclaration

Written 2026-09-17 before any science dump has landed. Applies to D1, D2, D3 (3 s, 71 frames) and, as a dry run, to the
pilot (0.5 s, 11 frames). Inputs are the per-frequency products of reduce_dump.py (one file per coarse bin: stacked
visibilities per frame over the 7,155 redundant baseline classes, per-input powers, block fine spectra, pilot cutouts)
and the kernel-of-record outputs of chime-run on the same dump (per-frame mask and pilot excess per channel). Frames are
aligned: both start at the common FPGA origin of the dump, so frame k is the same 16384 samples everywhere.

## Quantities, in the board's units

The board's A is a power ratio per kept frame: retained DTV power over the thermal power of the bin (board_method.md,
tradeoffs.csv `evaluation_G1_allowance`), and lambda_q is the tolerance in the same ratio (channels.csv
`conditional_reference_amplitude`). The frame-level physical residual is built as the same ratio in the visibility domain.

Per coarse bin b and baseline class s (below), with V_k the stack mean of the raw products in frame k (sum over the
16384 samples, averaged over the products in the class) and P_k the mean live-input power in frame k:

1. Coherent power over a frame set K, noise-bias free: C_s,b(K) = ( |sum_K V_k|^2 - sum_K |V_k|^2 ) / (|K| (|K|-1)).
   Two frames of pure noise contribute nothing on average; a component with a fixed phase across frames survives.
2. Amplitude ratio: a_s,b(K) = sqrt(max(C, 0)) / (16384 * mean_K P_k). For a plane wave from one transmitter that
   reaches every input with the same power this is the DTV-to-thermal power ratio of the bin, the board's A.
3. Sky and instrument reference: r_s,b = the 10th percentile of a_s,b(all frames) over the bins within 40 MHz of b.
   The DTV excess is e_s,b(K) = a_s,b(K) - r_s,b, floored at zero. Stated limitation: the reference is a lower
   envelope over neighbouring bins, not a fit; on a band where every neighbour carries DTV it is an upper bound on the
   sky term and the excess is a lower bound. That is the conservative direction for an exclusion.
4. Per-input retained power (the covariance route's U_i): for input i, U_i,b(K) = (mean_K auto_i,b,k - ref_i) / ref_i
   with ref_i the same input's power in the reference bins of the same frames; reported as the distribution over inputs.
5. Frame-lag structure: on the projection alpha_k = Re( V_k conj(m) ) / |m| with m = sum_K V_k, the structure function
   D(l) = mean over pairs of (alpha_k - alpha_{k+l})^2 for l = 1 .. |K|-1, normalised by the variance of alpha. The
   fast-variance fraction is phi_fast = D(1) / (2 var), and the within-dump coherence gain is
   G_dump = 1 + 2 sum_{l} rho(l) with rho(l) = 1 - D(l) / (2 var), truncated where rho first crosses zero. G_dump is a
   lower bound on G when the plateau is not reached within the dump; the archive's 300 s bound (30 s on 31 and 35)
   caps it from above.

Frame sets: all frames; kept frames (kernel mask 0); rejected frames (mask 1). A channel with fewer than 5 kept frames
reports its kept-frame quantities as absent, not as zero.

Baseline classes s: same-polarisation classes with (EW cylinder step, NS feed step) in
{(0,1), (0,2), (0,4), (0,8), (1,0), (1,1), (2,0), (3,0)} for both polarisations, keyed as in the products
(cylinder = id // 512, pol block = (id // 256) % 2, position = id % 256; EW 22.0 m per cylinder, NS 0.3048 m per feed).

## What is reported per channel

- The pilot bin and the in-band median over its bins of e_s,b for each class, on all, kept and rejected frames, with
  the kept-frame value carried as the physical A candidate; its ratio to the kernel's pilot-inferred shelf on the same
  frames is the proxy-to-physical transfer of RULE_CHANGE_2026-09-16.md.
- phi_fast and G_dump per class at the pilot bin and the in-band median, and the lag at which rho first crosses zero.
- U_i statistics: median, 90th percentile, fraction of inputs above lambda_q.
- Frame count, kept count, and the kernel's median pilot excess.

## Outcome rule (unchanged from MATCHED_CAPTURE.md item 5)

A channel whose kept-frame excess lower bound times G_low exceeds lambda_q on two of three dumps is excluded on physical
evidence; G_low = 1 where tau_needed is below one frame, otherwise G_dump. A channel whose kept-frame excess upper bound
(the all-frames value, which includes rejected frames) times the archive's G bound sits under lambda_q passes the
tolerance and proceeds to the floor gate and covariance route. Everything else is reported, not ruled.

No threshold in this document is tuned after seeing the data; changes are logged as amendments.

## Amendment 1 (author, 2026-09-16 evening local, revised the same night before any science-dump product was opened): the keep-all policy

Every channel is judged by the most forgiving mask policy that passes tolerance. The top rung of that ladder is
keep-all: a ceiling threshold set 6 dB above the highest per-frame pilot excess of the channel's current era, so that
in normal operation no frame is masked (retention reported, expected 100 percent) while a transmitter that turns on,
moves or strengthens is caught on the day. There is no separate "clean" verdict; a channel with no transmitter of its
own simply has keep-all as its passing policy.

A channel passes at keep-all when (a) its current era is representative and carries no pilot: the era begins at the
last persistent level change in the dating run (transmitter_dating/changes.csv; no change means the whole archive)
and spans at least 90 days and 60 archived units, within which the per-unit pilot-to-noise median is below 0 dB and
no unit exceeds 6 dB; and (b) in the dumps the all-frames in-band excess over the sky reference is below lambda_q
with the residual white at frame level (phi_fast at or above 0.9) on two of three dumps. The measured excess and its
upper bound are reported on every row; "below tolerance" is a statement about the bound, never a claim of zero.

If (a) holds and (b) fails, something radiates in band without a pilot of its own and the channel goes down the
policy ladder like any other. Candidates for (a) on today's evidence: 34 (never a pilot, 2018 to 2026), and the
switched-off channels 19 (off since 2024-11-26), 26 (2023-04-09), 27 (by 2022-10-19), 32 (by 2023-02-10).

Table of record. The deliverable is one row per freq_id 477 to 844: channel; role (pilot bin or data bin; the pilot
bin is recorded on every channel whatever its verdict, so that changes can be seen); disposition and passing policy
without the delay-filter credit; the same with the deployed 200 ns cut credited; the measured excess and its bound;
lambda_q in each world; the rule and the evidence file that decided the row. Dispositions are keep (with its policy),
pilot, or excise (no policy passes with every credit granted); excised rows carry the subtraction bar, the dB a
competing method would have to recover on kept frames. Rows the data cannot decide carry the reason.

## Amendment 2 (2026-09-17, on reading the first product metadata, before any analysis): 33 frames per dump

CHIME's kotekan caps a baseband dump at 546,875 samples (1.4 s), so each science dump holds 33 full frames, not 71,
and the structure function reaches lag 32 (1.3 s), not 70. Every estimator above is unchanged; frame counts in the
outcome rule are read as 33. Additional dumps at the same sidereal slots add frames without changing the design.

## Amendment 3 (2026-09-17, 19:46 UTC, written while the ten cadence events were still on the CHIME buffer or in transfer and before any cadence product was opened): the coherence time from the cadence campaign, and how it enters the table of record

Purpose. About twenty channels are undetermined in the provisional table because they pass at the within-dump gain
(lags to 1.3 s) and fail at the archive's gain bound, so their verdict is decided by where the coherence time tau_c
falls between 1.4 s and the bound. The cadence campaign (ten 0.2 s dumps at 0, 15, 30, 60, 120, 240, 480, 720, 960
and 1200 s after 16:02:08 UTC on 2026-09-17) supplies the lags from 15 s to 20 min that the archive does not have,
and the pilot and science dumps supply the lags from 2 h to one sidereal day. The estimator mirrors the archive's
definition of tau_c (RFIsher residual.py, correlation_time) so that the measured value replaces the archive's bound
in the same formula, G = min(tau_c, cap) / T_frame with the cap one sidereal day.

Per-epoch level. For each channel, on the baseline class (0, 1) with both polarisations averaged, and for two bin
sets (the pilot bin; the in-band median over the channel's bins), the epoch's level is the noise-bias-free amplitude
ratio of item 2 over all frames of the dump, x_d = sqrt(max(C_d, 0)) - r_d, with r_d the epoch's own sky reference of
item 3 (10th percentile over the bins within 40 MHz) and no floor at zero. Its noise variance is
sigma_d^2 = var_k(alpha_k) / n_d from the per-frame projections of item 5. The per-frame normalisation divides each
frame's stack mean by that frame's own live-input power rather than by the dump mean; the difference is below one
percent on every dump so far and is stated here so that it is not tuned later.

Structure function. Over every pair of epochs (a, b) less than one sidereal day apart,
D_ab = 0.5 [ (x_a - x_b)^2 - sigma_a^2 - sigma_b^2 ], the archive's noise-corrected form. Pairs are grouped to the
nearest predeclared lag class in log lag (15, 30, 45, 60, 90, 120, 180, 240, 360, 480, 600, 720, 960, 1200 s, then
1, 5, 10 and 24 h); a class is populated with one or more pairs (the archive's 40-pair gate is a sparse-sampling
guard that the campaign's per-pair noise, reported beside every D, makes unnecessary). The plateau is the mean D over
pairs with lag above 7200 s (the archive's plateau_start_seconds), with its uncertainty from the scatter of those
pairs. tau_c is the lag at which the class-mean D first reaches (1 - 1/e) of the plateau, linearly interpolated
between adjacent populated classes, requiring at least three populated classes; these are the archive's
crossing_fraction and minimum_populated_lag_bins.

Status and the gain. Four outcomes, in the order tested:
- constant: the plateau is not positive at two standard errors. The level does not vary within the day at the
  campaign's precision, the residual is a fixed bias over the integration, and G takes the cap (2,054,312), exactly
  as the archive books a refused tau_c. This is the pessimistic end and is deliberate.
- measured: D crosses the target inside the populated cadence classes. G = tau_c / T_frame.
- bound: the plateau is positive but D stays below the target through the longest populated cadence class. tau_c is
  at least that lag and at most 7200 s; G is reported as the range, and only the lower end is used, so that the bound
  can excise but never keep.
- no cadence lags: fewer than three populated classes below 7200 s (a failed campaign); the row keeps its provisional
  reason.

Table of record. Two columns are added, the measured G (with tau_c and its status) and R at that G in each world, and
the disposition on every channel with a live pilot bin is then read at the measured G: keep if the loosest policy
passes at the measured G with the deployed cut (marked "also without credit" when it passes in world none); excise
if no policy passes at the measured G, or at its lower bound, with the deployed cut. Excised rows carry two
subtraction bars: the dB over tolerance at the measured G with the deployed cut, which is what a competing method
must recover from the residual it leaves, and the dB at G = 1, the least any method could face if it also
decorrelated the residual frame to frame. The archive-bound keep of the provisional table stays valid where it
already held (channels 15 and 29). Channels with no live pilot bin are unchanged by this amendment.

What is not done. The class-mean D is not fitted to a model; the crossing is read off as in the archive. The
between-epoch phasor coherence of cadence_lags.py (a statement about phase stationarity, used in section 15 of the
methods notes) is reported alongside but does not enter G, because the archive's tau_c is defined on the level.

## Amendment 4 (2026-09-17, 22:35 UTC, on the manuscript review of the provisional table, before the fourth science dump or any cadence product beyond event 1 was placed): the physical A is the larger of the two polarisations

The provisional table read the in-band excess on the (0, 1) class from polarisation 0 alone. The review found that
polarisation 1 reads higher on most channels (channel 15: 0.0094 to 0.013 against a clipped zero; channel 29: 0.0023 to
0.0031 against zero; channel 16: 0.013 to 0.018 against 0.0017 to 0.010), consistent with a horizontally polarised
transmitter coupling unequally to the two feeds. A is a bound on the retained DTV-to-thermal ratio, and a rescue must
survive the worse reading, so from this amendment the table's A is the larger of the two same-polarisation readings of
the (0, 1) class on the kept dumps, with both readings carried as columns; G_dump and phi_fast are taken from the same
polarisation as A. The keep-all bound, the tolerance ratios in both worlds and the dispositions are recomputed from it.
The cadence level of amendment 3 is unchanged (both polarisations averaged; it measures time structure, not a bound).
Nothing else in the estimators changes. The polarisation-0 table is retained as table_of_record_provisional_pol0.csv.

## Amendment 5 (2026-09-18, 01:35 UTC, written after a preview of eleven of the fourteen epochs, before the last three cadence events were placed): the cadence level follows amendment 4, and the archive's trim probes are mirrored

Two omissions of amendment 3 are corrected, and the circumstances are stated so that the reader can weigh them.

1. Polarisation. Amendment 3 averaged the two polarisations' complex stack means before taking the level. That is a
   fixed linear functional of the field and tracks the DTV power, but it can be small where the two polarisations'
   instrumental phases oppose, and it is not the quantity the table rules on since amendment 4. From this amendment the
   epoch level is the per-polarisation amplitude ratio of item 2 on the polarisation that sets the table's A (the larger
   of the two readings on the kept dumps), and the other polarisation's result is reported beside it.
2. Trim probes. The archive's correlation_time trims the upper tail of the level at the 90th percentile before the
   structure function, probes 75, 90 and 95, and refuses when tau_c moves by more than a factor two across the probes.
   Amendment 3 omitted this. It is mirrored here on the epoch levels: the primary estimate drops the epochs above the
   90th percentile of the level (one of fourteen), the probes drop those above the 75th and 95th, and a spread above two
   across the probes gives the status refused, which takes the cap as the archive does. A status of bound at every
   probe is consistent. The untrimmed result is reported beside the trimmed one.

Why now. The preview showed one epoch (C240, 16:06:08 UTC) on channel 27 with a level three times its neighbours,
decaying within the dump's 0.17 s, which alone drove a crossing at 101 s; the archive's trim exists for exactly this
tail. The ruling's outcome is insensitive to the choice: no channel with a live pilot bin can pass at a gain above about
240 (channel 29 needs tau_c below 10 s; every other channel less), and at the campaign's shortest lag of 15 s the
structure function is below 0.4 of the plateau on every channel whose level is measurable, so no channel's tau_c lies
below 15 s. The amendment changes which number is printed for tau_c, not which channels pass.

The implementation is cadence_tau.py with --level polmax --pol-table <table_of_record csv> --trim archive; the
amendment-3 form remains available as --level polavg --trim none and is run for the report.

## Amendment 6 (2026-09-18, 03:16 UTC, after the table of record with the measured gain was built and with the author's decision): the lowest-epoch bound for channels whose pilot bin is not in the dumps

Facts that prompted it. Six channels (19, 20, 24, 30, 31, 34) have no live node on their pilot bin in the September 2026
dumps, so the per-dump mask statistic that the calibrated policies read cannot be formed, and the provisional rule left
them undetermined. The archive's own pilot-bin history shows: the bins were recorded until 2026-04-16 (19, 24, 31),
2026-06-13 (20) and 2026-07-06 (34), so five of the six nodes were dropped within the months before the dumps (30's
stopped in 2023-09); the current era's nominal pilot is at the floor on 19 (lost 2024-11-26, from 14.9 to -19.5 dB),
on 20 (lost 2022-08-31, from 3.0 to -9.9 dB and declining) and on 34 (never above the floor, 2018 to 2026), while the
same channels show DTV on their live in-band bins at every epoch (19: 4e-3 to 1.6e-2; 20: 6e-3 to 2.3e-2), so the
transmitter is on and the nominal pilot is not where the detector looks, the channel 33 situation; 30 and 31 carry
strong nominal pilots in the archive (Q medians 154 and 25). The dating run's "transmitter off" on 19 is therefore a
loss of the nominal pilot, not of the transmitter, and is reworded.

The bound. Every masking policy keeps some subset of the epochs, so the residual it leaves cannot lie below the
residual of the lowest epoch. For a channel whose pilot bin is not in the dumps, A_low is the minimum over all fourteen
epochs (the four dumps and the ten cadence dumps, all frames) of the in-band median excess on the polarisation that
sets A (amendment 4), and the channel is excised when A_low, with the deployed cut, fails tolerance at the measured
coherence gain (or the lower end of its bound, or the cap on a refusal): no policy can pass, whatever the mask would
have done. The bars are reported at A_low: the dB over tolerance at that gain, and at G = 1. A channel whose A_low
passes at that gain stays undetermined (a mask might pass; it cannot be evaluated). A channel with no products (24)
stays undetermined. The same column is reported for every channel; for the seventeen with a pilot bin it can only
confirm the verdict already made, since their kept-set residuals are above their lowest epoch.

Why it is valid. A_low is a lower bound on the kept-frame residual of any policy because masking removes frames and
cannot lower the residual of the frames it keeps below the lowest epoch's; the cadence epochs of four frames are
included so that the minimum is taken over the finest sampling available, which makes the bound weaker, that is,
harder to excise on. The direction is the conservative one for an exclusion.

## Amendment 7 (2026-09-18, 03:49 UTC, after the audit checks of the table of record, on the author's decision): the ruling is read on the baselines that carry the BAO measurement

Facts that prompted it. The table read A and the coherence time on the (0, 1) class, the 0.3 m baseline, where the
two feeds see the transmitter almost identically and the residual is largest. The 21 cm BAO scales at these redshifts
(k of 0.03 to 0.3 per Mpc) lie at baselines of roughly 10 to 100 m; the shortest baselines probe smaller k and do not
carry that measurement. On the three east-west classes in the same products, (1, 0), (2, 0) and (3, 0) at 22, 44 and
66 m, the in-band residual is 3 to 22 dB lower than on the 0.3 m baseline, and its level decorrelates sooner where the
measurement has the precision to say (tens of seconds to forty minutes), consistent with the low between-epoch phase
coherence on those baselines (section 15 of the notes). The rule's own logic, that an exclusion must survive every
credit the data can give, requires the exclusion to be read there.

The basis. For every channel, A_bao is the median over the three east-west classes of the per-class in-band excess
(each class read on the larger of its two polarisations, amendment 4; the median over the kept dumps of the policy
under test). The coherence time tau_bao is the median of the measured east-west class values of the amendment-5
estimator run on each class; where a class is bound its lower end is used; where every east-west class is refused
(the level near the estimator's floor, the trim probes disagreeing), the shortest-baseline result is used as the upper
bound on persistence and the row says so. G_bao = min(tau_bao, cap) / T_frame. The lowest-epoch bound of amendment 6
is read on the same basis (the minimum over the fourteen epochs of the east-west median).

Dispositions on this basis, with the 3 dB allowance the record section already carries: excise when no policy passes
at G_bao with the deployed cut by more than 3 dB; keep when the loosest policy passes at G_bao with the deployed cut
(marked also without credit where world none passes too), stated as keep on baselines longer than about 10 m, since
the 0.3 m reading is over tolerance on every channel; undetermined when the best policy lies within the 3 dB allowance
(marginal), or when the east-west coherence is unmeasurable and the verdict flips between the fallback and the
measured gains. The 0.3 m reading is kept in every row as the worst case (the columns of the previous table), and the
subtraction bars are reported on the BAO basis and on the worst case.

What it changes. From the table with the measured gain: channel 27 keeps (9 dB under on the BAO basis), channel 29 is
marginal (about 2 dB over), channel 14 is undetermined (its east-west coherence is refused in every class and the
fallback gain decides the sign); the other nineteen measurable channels remain excised by 9 dB or more on the BAO
basis, before the concession checks, which are repeated on this basis. Every bar falls; the direction is the one a
reviewer will accept.

Tolerance wording. The tolerance against which the excisions hold is the least strict the forecast prices: the primary
systematic budget of one sigma (zeta = 1) in the deployed world, with the 200 ns delay cut credited at 11.4 dB. The
ground-filter (sidereal-mean) credit is not included; it is granted in the robustness check, where on the east-west
baselines the measured between-epoch coherence of 0.1 to 0.7 makes it small.

## Amendment 8 (2026-09-18, 04:04 UTC, after the adversarial audit of the ruling, finding S1): the control floor and the baseline-resolved ruling

Facts that prompted it. Channel 37 (608 to 614 MHz, no DTV allocation) is the one control in the dumps. Read through
the same estimator it returns a positive, persistent excess on every class and epoch: 2.1e-3 on the 0.3 m class,
3.8e-4 at 2.4 m, 3e-5 to 8e-5 on the classes from 9.8 to 78 m and 22 to 66 m (baseline_floor.csv). The tolerance on A
at the ruling gains (1e-6 to 3e-6 at G of 85,831; 1e-7 at the cap) lies 16 to 45 dB below that reading. A channel whose
excess is not measured above the control's cannot be shown to be over tolerance, or within it, by this capture. The
predeclaration's item 3 called the excess a lower bound; the control shows it has a floor.

The floor. For each class and polarisation the floor is the mean of the channel 37 bins' excess over the four science
dumps, with its scatter over those bins and dumps. A channel's excess on a class is measured when it exceeds the floor
by more than three scatters; the excess net of the floor, with the scatter as its uncertainty, is what enters R.

The ranges. Baseline classes are grouped by what they carry: the shortest (0.3 to 2.4 m; below the BAO scales; the
worst case, kept as such), the short BAO baselines (9.8 and 19.5 m north-south), and the long BAO baselines (39 and
78 m north-south; 22, 44 and 66 m east-west), the 21 cm BAO modes at these redshifts lying at 10 to 100 m. The
coherence time is measured per class with the amendment-5 estimator; each range takes the median of the measured
values of its classes, the lower end of a bound, and the 0.3 m result as the persistence upper bound where every
class in the range is refused (stated on the row).

Dispositions. A channel is excised on a range when its excess is measured above the floor on at least one class of the
range and, net of the floor, fails tolerance by more than the 3 dB allowance at the range's gain with the deployed
cut under every policy; it is undetermined at the floor on a range when its excess is not measured above the floor
there (the capture cannot decide it either way, and the row states by how many dB the floor itself exceeds the
tolerance at that gain); it keeps on a range only when its excess is measured and passes, which the floor makes
impossible at these gains, so the table reports no keep. The row's verdict is given per range, and the summary
verdict for the 21 cm BAO measurement is: excise where excised on the long BAO range; excise on the short BAO
baselines only, where excised there and at the floor beyond; undetermined at the floor where at the floor on both.

What lowers the floor. Calibrated gains before stacking (the redundant stacks are uncalibrated and add incoherently),
a control block outside the DTV allocation and the 600 MHz cellular band in the post-upgrade dump, and longer
captures; the rows say so.

Addendum to amendment 8 (04:20 UTC, on reading the north-south class coherence times, before the per-range table was
built): where every class of a range is refused by the level estimator, the persistence bound is used only where the
between-epoch phasor coherence of the range's classes over the six dump pairs (5 to 22 h; cadence_lags on the
reference-subtracted excess) exceeds 0.7 in median, which is a measurement of persistence by a different estimator;
otherwise the range's gain is unmeasured between the within-dump gain and the persistence bound and the channel is
undetermined on that range (its measured excess above the floor is reported as unpriced). The per-class coherence
times are in cadence_tau_ns{32,64,128,255}.csv and cadence_tau_ew{1,2,3}.csv; the phasor coherences in
lag_coherence_allclasses_D.csv (classes (0,1), (0,8), (1,0) only; the north-south long classes are computed for this
addendum with the same script).

Second addendum to amendment 8 (04:35 UTC, author's decision): every credit the survey applies enters the ruling at
its measured value. The deployed 200 ns delay cut is credited at 11.4 dB on A as before. The ground filter (sidereal-
mean subtraction) is credited per baseline class from the between-epoch phasor coherence rho of the reference-
subtracted excess over the six dump pairs (5 to 22 h; the quantity the subtraction acts on, under RFIsher's model in
which a residual constant within a day is removed whatever it does day to day): the surviving power fraction is
1 - rho^2 with rho the median over the range's classes clipped to [0, 0.95]. The ratio with both credits is the ruling
quantity in the deployed world; the ratio with no credit at all remains the world-none column for the rescue
direction; the bar on an excised row is reported with both credits, and the credit itself is printed.

## Amendment 9 (2026-09-18, 05:30 UTC, after the second adversarial audit of the committed ruling, RULING_AUDIT_ROUND2_2026-09-18.md, findings R1 to R4; author's standing instruction that the ruling must survive every objection): the ruling is confined to the forecast's baseline domain and to measured coherence times

1. The forecast's baseline domain (R3). The forecast of record is the RadioFisher CHIME layout with baselines from
   20 to 128 m and a synthetic baseline density with zero weight below 20 m. Its tolerance therefore prices no
   residual on the 0.3, 2.4, 9.8 or 19.5 m classes. The range at 9.8 and 19.5 m is renamed "below the forecast's
   baseline cut"; its readings are reported (the excess above the control floor, the tolerance applied there for
   information) but rule nothing. The ruling is made on the long range alone (39 and 78 m north-south; 22, 44 and
   66 m east-west), which lies inside the domain. Rebuilding the bank with the as-built density is future work.
2. The lowest-epoch bound (R1). An epoch on which no class of the range is measured above the floor is an epoch at
   the floor, and the bound is then at the floor and cannot excise; the implementation is corrected to the rule as
   written in amendment 6.
3. The persistence fallback (R2). The between-epoch phasor coherence measures the share the ground filter removes,
   not the coherence time of what it leaves, so it no longer supplies a gain. Where the level estimator returns no
   measured class on the long range, the channel's own measured coherence time on the 19.5 m class is used if there is
   one (a measurement on the same channel at the nearest priced-adjacent baseline; stated on the row); otherwise the
   range is unpriced and the channel undetermined.
4. Marginal readings (R4). A bar within the 3 dB allowance is marginal, as before; a bar whose probe range (the three
   trim probes of the measured class) crosses the allowance is reported with that range.

The count this leaves is the one both audits support: the channels excised for the 21 cm BAO measurement are those
measured above the control floor on the long range and over the least strict tolerance there with both credits;
every other channel is undetermined, with its measured excess below the cut reported for the collaboration.
