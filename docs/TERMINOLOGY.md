# Detector measurement terminology

PilotProxy stores a target-to-local-reference **power ratio**, not an exact
F-distributed statistic:

\[
R_\mathrm{coarse}
  = \frac{2P_\mathrm{target}}
         {P_\mathrm{ref,lower}+P_\mathrm{ref,upper}}.
\]

The exact integer squared norms of the packed target and reference weights set
the flat-noise null ratio

\[
R_\mathrm{null}
  = \frac{2\lVert w_\mathrm{target}\rVert^2}
         {\lVert w_\mathrm{ref,lower}\rVert^2
          +\lVert w_\mathrm{ref,upper}\rVert^2}.
\]

The normalized ratio and physical pilot excess are

\[
Q_\mathrm{coarse}=R_\mathrm{coarse}/R_\mathrm{null},
\qquad
\rho=Q_\mathrm{coarse}-1.
\]

`reject_mask` uses the exact integer form of `Q_coarse > 1`.
`normalized_coarse_power_ratio_db = 10 log10(Q_coarse)` is the plotted level,
so the null and the active positive-excess boundary are both exactly 0 dB.
The reported `pilot_excess_db` is `10 log10(rho)` where `rho > 0`, and
`estimated_data_shelf_snr_db` is derived from that same normalized excess.
The linear `coarse_power_ratio` remains available for exact reconstruction;
`raw_pilot_excess = R_coarse - 1` is diagnostic-only and is not a physical PNR.

The fine product stores

\[
R_\mathrm{fine}[b]
  = \frac{2S_\mathrm{target}[b]}
         {S_\mathrm{ref,lower}[b]+S_\mathrm{ref,upper}[b]}.
\]

Its null is calibrated empirically from the independent non-designated bins.
The name `fine_power_ratio` therefore states exactly what is stored without
claiming an exact parametric null distribution.

Under independent complex-Gaussian projections with equal reference scales,
the corresponding idealized ratio can be written as an F variate.  The shipped
int4 weights do not satisfy the equal-scale condition exactly (the lower and
upper reference norms can differ), and telescope streams can be correlated or
non-Gaussian.  "F-statistic" is therefore retained only where it names the
existing CUDA/C ABI (`FStat_*`, `libfstatistic.so`) or discusses the idealized
model.  A later ABI-only series can rename implementation symbols without
mixing that mechanical change with scientific product semantics.

Frozen evidence and manuscript provenance are not rewritten by this hard cut.
They document earlier development snapshots and are not valid current product
inputs; regenerate current products and figures rather than adding readers for
those retired fields.
