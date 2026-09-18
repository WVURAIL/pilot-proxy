# Cadence campaign: coherence time of the DTV residual level

Estimator: FRAME_ANALYSIS_PREDECLARATION.md amendments 3 and 5 (cadence_tau.py). D is the noise-corrected structure function of the per-epoch level on the 0.3 m same-polarisation baseline, in A^2 units; the plateau is the mean D at lags above 7200 s; tau_c is the lag at which D reaches (1 - 1/e) of the plateau. Status constant means the plateau is not positive at two standard errors and G takes the sidereal cap; bound means D stays below the target through the longest populated cadence class (the class marked 3600 s holds the pairs between D3 and the cadence dumps, at lags of 7178 to 7193 s), and only the lower end of the range is used; refused means the trim probes disagree by more than a factor two and G takes the cap, as the archive does. The ruling reads the in-band row, the same quantity as A. Amendment 5 (per-polarisation level on the polarisation that sets A; the archive's trim probes) is the primary form.


## Pilot bin: tau_c and G

| ch | pol | status | tau_c (s) | tau_c high (s) | G | plateau (A^2) | plateau s.e. | pairs | classes | epochs used | probes 75 / 90 / 95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | 1 | measured | 15.0 | 15.0 | 357.6 | 7.190e-06 | 2.240e-06 | 65 | 12 | 12 | 75: measured 15 | 90: measured 15 | 95: measured 15 |
| 15 | 1 | bound | 3600.0 | 7200.0 | 85830.7 | 2.386e-06 | 3.115e-07 | 66 | 13 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 16 | 1 | bound | 3600.0 | 7200.0 | 85830.7 | 4.376e-06 | 5.241e-07 | 66 | 14 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 17 | 1 | bound | 3600.0 | 7200.0 | 85830.7 | 8.625e-05 | 1.199e-05 | 66 | 11 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 18 | 1 | bound | 3600.0 | 7200.0 | 85830.7 | 4.005e-06 | 5.682e-07 | 65 | 13 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 19 | 1 | no epochs |  |  |  |  |  | 0 | 0 | 0 | 75: no epochs | 90: no epochs | 95: no epochs |
| 20 | 0 | no epochs |  |  |  |  |  | 0 | 0 | 0 | 75: no epochs | 90: no epochs | 95: no epochs |
| 21 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 8.245e-06 | 1.642e-06 | 66 | 14 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 22 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 8.248e-06 | 1.460e-06 | 65 | 11 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 23 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 9.057e-06 | 1.264e-06 | 66 | 12 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 25 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 5.440e-06 | 8.457e-07 | 66 | 13 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 26 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 2.431e-06 | 3.873e-07 | 66 | 13 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 27 | 0 | refused: trim spread 132.9 | 86164.1 | 86164.1 | 2054312.0 | 2.051e-06 | 3.743e-07 | 65 | 12 | 12 | 75: bound 3600 | 90: bound 3600 | 95: measured 27 |
| 28 | 1 | bound | 1200.0 | 7200.0 | 28610.2 | 1.500e-07 | 3.556e-08 | 65 | 13 | 12 | 75: bound 480 | 90: bound 1200 | 95: bound 1200 |
| 29 | 1 | refused: trim spread 3.1 | 86164.1 | 86164.1 | 2054312.0 | 9.530e-08 | 2.318e-08 | 65 | 13 | 12 | 75: measured 28 | 90: measured 37 | 95: measured 89 |
| 30 | 0 | no epochs |  |  |  |  |  | 0 | 0 | 0 | 75: no epochs | 90: no epochs | 95: no epochs |
| 31 | 0 | no epochs |  |  |  |  |  | 0 | 0 | 0 | 75: no epochs | 90: no epochs | 95: no epochs |
| 32 | 1 | measured | 52.7 | 52.7 | 1257.1 | 1.130e-07 | 2.119e-08 | 65 | 13 | 12 | 75: measured 28 | 90: measured 53 | 95: measured 55 |
| 33 | 0 | refused: trim spread 5.9 | 86164.1 | 86164.1 | 2054312.0 | 6.562e-06 | 1.060e-06 | 65 | 13 | 12 | 75: bound 3600 | 90: bound 3600 | 95: measured 606 |
| 34 | 1 | no epochs |  |  |  |  |  | 0 | 0 | 0 | 75: no epochs | 90: no epochs | 95: no epochs |
| 35 | 0 | measured | 2712.6 | 2712.6 | 64673.2 | 9.894e-06 | 1.388e-06 | 65 | 12 | 12 | 75: measured 3011 | 90: measured 2713 | 95: measured 2678 |
| 36 | 1 | refused: trim spread 7.0 | 86164.1 | 86164.1 | 2054312.0 | 1.331e-07 | 3.078e-08 | 65 | 14 | 12 | 75: measured 33 | 90: measured 198 | 95: measured 229 |
| 37 | 1 | refused: trim spread 3.8 | 86164.1 | 86164.1 | 2054312.0 | 4.365e-08 | 1.035e-08 | 66 | 14 | 12 | 75: measured 15 | 90: measured 56 | 95: measured 57 |

### D(lag) / plateau by lag class (pilot); parentheses give the subtracted noise term over the plateau, same units

| ch | 15 s | 30 s | 45 s | 60 s | 90 s | 120 s | 180 s | 240 s | 360 s | 480 s | 600 s | 720 s | 960 s | 1200 s | 3600 s | plateau |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | +2.75 (0.00) | +2.09 (0.00) |  |  | +1.09 (0.00) | +1.72 (0.00) |  | +0.28 (0.00) | +1.12 (0.00) | +2.71 (0.00) | +1.15 (0.00) | +3.32 (0.00) | +1.77 (0.00) | +3.38 (0.00) | +1.64 (0.00) | 7.190258017887308e-06 |
| 15 | +0.04 (0.00) | +0.03 (0.00) | +0.00 (0.00) | +0.08 (0.00) | +0.00 (0.00) | +0.04 (0.00) | +0.01 (0.00) | +0.08 (0.00) | +0.07 (0.01) | +0.07 (0.01) | +0.35 (0.00) | +0.33 (0.00) |  |  | +0.57 (0.00) | 2.3863727694882666e-06 |
| 16 | -0.00 (0.00) | +0.03 (0.00) | +0.06 (0.00) | +0.04 (0.00) | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | +0.02 (0.00) | +0.08 (0.00) | +0.07 (0.00) |  | +0.06 (0.00) | +0.01 (0.00) | +0.02 (0.00) | +0.36 (0.00) | 4.376419285751156e-06 |
| 17 | +0.01 (0.00) | +0.02 (0.00) | +0.00 (0.00) | +0.01 (0.00) |  |  | +0.02 (0.00) | +0.06 (0.00) |  | +0.09 (0.00) |  | +0.06 (0.00) | +0.09 (0.00) | +0.25 (0.00) | +0.01 (0.00) | 8.624890772025336e-05 |
| 18 | -0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | +0.01 (0.00) | +0.01 (0.00) | -0.00 (0.00) |  | +0.00 (0.00) | +0.03 (0.00) | +0.01 (0.00) |  | +0.04 (0.00) | +0.00 (0.00) | +0.01 (0.00) | +0.31 (0.00) | 4.004596807357186e-06 |
| 19 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 20 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 21 | +0.01 (0.00) | +0.02 (0.00) | +0.03 (0.00) | +0.05 (0.00) | +0.00 (0.00) | +0.01 (0.00) | +0.05 (0.00) | +0.03 (0.00) | +0.24 (0.00) | +0.21 (0.00) | +0.34 (0.00) | +0.32 (0.00) | +0.32 (0.00) |  | +0.08 (0.00) | 8.244774911467313e-06 |
| 22 |  | +0.05 (0.00) |  |  | +0.02 (0.00) | +0.31 (0.00) |  | +0.26 (0.00) | +0.04 (0.00) | +0.05 (0.00) | +0.18 (0.00) | +0.17 (0.00) | +0.08 (0.00) | +0.08 (0.00) | +0.08 (0.00) | 8.248123358461048e-06 |
| 23 |  | +0.00 (0.00) |  | +0.01 (0.00) | -0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | -0.00 (0.00) | +0.01 (0.00) | +0.01 (0.00) | +0.01 (0.00) | +0.02 (0.00) |  | +0.25 (0.00) | 9.057226710137217e-06 |
| 25 | -0.00 (0.00) | +0.00 (0.00) | +0.01 (0.00) | +0.01 (0.00) | +0.00 (0.00) | +0.00 (0.00) | +0.03 (0.00) | +0.02 (0.00) | +0.04 (0.00) | +0.08 (0.00) | +0.17 (0.00) | +0.23 (0.00) |  |  | +0.10 (0.00) | 5.439550125917853e-06 |
| 26 | -0.00 (0.00) | +0.01 (0.00) | +0.01 (0.00) | +0.01 (0.00) | +0.00 (0.00) | +0.02 (0.00) | +0.07 (0.00) | +0.01 (0.00) | +0.03 (0.00) | +0.01 (0.00) | +0.06 (0.01) | +0.05 (0.00) |  |  | +0.01 (0.00) | 2.4311961891101403e-06 |
| 27 | -0.00 (0.00) | -0.00 (0.00) |  |  | +0.01 (0.00) | +0.01 (0.00) |  | +0.30 (0.01) | +0.02 (0.00) | +0.24 (0.00) | +0.00 (0.00) | +0.04 (0.00) | +0.57 (0.01) | +0.04 (0.00) | +0.02 (0.00) | 2.05096955521628e-06 |
| 28 | -0.05 (0.05) | +0.03 (0.04) | +0.08 (0.09) | +0.02 (0.07) | -0.01 (0.03) | -0.02 (0.03) | +0.05 (0.06) | -0.01 (0.03) | -0.02 (0.02) | +0.03 (0.03) |  | +0.06 (0.03) | +0.14 (0.05) | +0.01 (0.05) |  | 1.4996942651463254e-07 |
| 29 | -0.01 (0.04) | +0.58 (0.08) | +0.68 (0.07) | +0.28 (0.06) | +0.76 (0.05) | +0.33 (0.02) | +0.45 (0.10) | -0.05 (0.09) |  | +0.23 (0.05) | +1.56 (0.02) | +0.61 (0.07) | +0.49 (0.12) | +0.66 (0.05) |  | 9.530073403925499e-08 |
| 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 31 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 32 | -0.02 (0.05) | +0.28 (0.07) | +0.43 (0.09) | +0.83 (0.08) | -0.03 (0.05) | +0.05 (0.03) | +0.17 (0.07) | +0.12 (0.05) |  | +0.03 (0.03) | -0.02 (0.04) | +0.09 (0.06) | +0.22 (0.08) | +0.09 (0.05) |  | 1.1300322395477685e-07 |
| 33 | +0.03 (0.00) | +0.14 (0.00) | +0.35 (0.00) | +0.48 (0.00) | +0.06 (0.00) | +0.00 (0.00) |  | +0.12 (0.00) | +0.26 (0.01) | +0.48 (0.00) |  | +0.22 (0.00) | +0.38 (0.00) | +0.11 (0.00) | +0.28 (0.00) | 6.561826303345931e-06 |
| 34 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 35 | +0.05 (0.00) |  | +0.02 (0.00) | +0.08 (0.00) |  | +0.02 (0.00) |  | +0.02 (0.00) | +0.22 (0.00) | +0.16 (0.00) | +0.07 (0.00) | +0.08 (0.00) | +0.11 (0.00) | +0.15 (0.00) | +0.91 (0.00) | 9.893964577681666e-06 |
| 36 | -0.00 (0.06) | +0.13 (0.10) | +0.59 (0.11) | +0.38 (0.16) | -0.09 (0.11) | +0.07 (0.11) | +0.34 (0.10) | +1.31 (0.07) | -0.09 (0.13) | +0.23 (0.09) | +1.05 (0.13) | +1.27 (0.09) | +0.71 (0.10) | +2.44 (0.11) |  | 1.3309238943100135e-07 |
| 37 | +0.58 (0.16) | +0.59 (0.24) | -0.12 (0.19) | +0.90 (0.28) | -0.14 (0.16) | +0.36 (0.20) | +0.75 (0.30) | +0.49 (0.32) | +0.40 (0.39) | +0.62 (0.40) | +0.22 (0.30) | +0.59 (0.30) | +0.13 (0.15) |  | +1.91 (0.16) | 4.364576311613175e-08 |

## In-band median: tau_c and G

| ch | pol | status | tau_c (s) | tau_c high (s) | G | plateau (A^2) | plateau s.e. | pairs | classes | epochs used | probes 75 / 90 / 95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | 1 | measured | 15.0 | 15.0 | 357.6 | 1.593e-06 | 3.167e-07 | 66 | 13 | 12 | 75: measured 15 | 90: measured 15 | 95: measured 15 |
| 15 | 1 | bound | 3600.0 | 7200.0 | 85830.7 | 3.371e-06 | 3.660e-07 | 66 | 15 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 16 | 1 | bound | 3600.0 | 7200.0 | 85830.7 | 4.431e-06 | 5.814e-07 | 66 | 14 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 17 | 1 | bound | 3600.0 | 7200.0 | 85830.7 | 6.487e-05 | 1.012e-05 | 65 | 11 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 18 | 1 | bound | 3600.0 | 7200.0 | 85830.7 | 2.552e-06 | 3.911e-07 | 65 | 13 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 19 | 1 | bound | 3600.0 | 7200.0 | 85830.7 | 6.581e-06 | 9.792e-07 | 66 | 14 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 20 | 0 | refused: trim spread 11.3 | 86164.1 | 86164.1 | 2054312.0 | 1.236e-05 | 2.424e-06 | 66 | 13 | 12 | 75: bound 3600 | 90: measured 320 | 95: measured 331 |
| 21 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 6.522e-06 | 1.142e-06 | 66 | 14 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 22 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 1.348e-05 | 2.062e-06 | 65 | 12 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 23 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 9.964e-06 | 1.364e-06 | 66 | 14 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 25 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 3.363e-06 | 5.292e-07 | 66 | 13 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 26 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 1.804e-06 | 3.013e-07 | 66 | 13 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 27 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 1.203e-06 | 2.238e-07 | 65 | 13 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 28 | 1 | refused: trim spread 3.4 | 86164.1 | 86164.1 | 2054312.0 | 6.313e-08 | 1.054e-08 | 65 | 14 | 12 | 75: bound 1200 | 90: measured 348 | 95: bound 1200 |
| 29 | 1 | bound | 1200.0 | 7200.0 | 28610.2 | 9.014e-08 | 1.968e-08 | 65 | 13 | 12 | 75: bound 960 | 90: bound 1200 | 95: bound 1200 |
| 30 | 0 | measured | 30.0 | 30.0 | 715.3 | 1.805e-06 | 3.348e-07 | 65 | 13 | 12 | 75: measured 30 | 90: measured 30 | 95: measured 23 |
| 31 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 1.673e-05 | 1.716e-06 | 66 | 14 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 32 | 1 | measured | 35.4 | 35.4 | 843.4 | 5.759e-08 | 1.490e-08 | 65 | 13 | 12 | 75: measured 23 | 90: measured 35 | 95: measured 37 |
| 33 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 1.082e-05 | 1.606e-06 | 65 | 12 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 34 | 1 | refused: trim spread 518.0 | 86164.1 | 86164.1 | 2054312.0 | 5.364e-08 | 1.357e-08 | 65 | 14 | 12 | 75: constant 86164 | 90: measured 166 | 95: measured 874 |
| 35 | 0 | bound | 3600.0 | 7200.0 | 85830.7 | 6.168e-06 | 6.977e-07 | 66 | 15 | 12 | 75: bound 3600 | 90: bound 3600 | 95: bound 3600 |
| 36 | 1 | measured | 60.6 | 60.6 | 1444.4 | 5.872e-08 | 1.553e-08 | 65 | 14 | 12 | 75: measured 59 | 90: measured 61 | 95: measured 73 |
| 37 | 1 | refused: trim spread 5.7 | 86164.1 | 86164.1 | 2054312.0 | 3.593e-08 | 8.768e-09 | 66 | 14 | 12 | 75: measured 312 | 90: measured 55 | 95: measured 56 |

### D(lag) / plateau by lag class (inband); parentheses give the subtracted noise term over the plateau, same units

| ch | 15 s | 30 s | 45 s | 60 s | 90 s | 120 s | 180 s | 240 s | 360 s | 480 s | 600 s | 720 s | 960 s | 1200 s | 3600 s | plateau |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | +0.85 (0.00) | +3.60 (0.00) | +1.84 (0.00) | +0.06 (0.00) | +2.94 (0.00) | +0.84 (0.01) | +0.78 (0.02) | +3.67 (0.02) | +1.30 (0.01) | +2.98 (0.01) |  | -0.00 (0.02) | +2.95 (0.00) |  | +1.00 (0.01) | 1.5932425468665666e-06 |
| 15 | +0.03 (0.00) | +0.04 (0.01) | +0.04 (0.00) | +0.00 (0.00) | +0.00 (0.01) | +0.02 (0.00) | +0.05 (0.00) | +0.02 (0.00) | +0.01 (0.00) | +0.02 (0.00) | +0.09 (0.00) | +0.09 (0.00) | +0.02 (0.00) | +0.09 (0.00) | +0.41 (0.00) | 3.370724678007053e-06 |
| 16 | +0.00 (0.00) | +0.02 (0.01) | +0.06 (0.01) | +0.02 (0.01) | +0.02 (0.00) | +0.02 (0.00) | -0.01 (0.01) | +0.03 (0.00) | +0.01 (0.00) | +0.04 (0.00) |  | +0.02 (0.00) | +0.01 (0.00) | +0.01 (0.00) | +0.39 (0.00) | 4.431185072807332e-06 |
| 17 | +0.00 (0.00) | +0.00 (0.00) | +0.01 (0.00) | -0.00 (0.00) |  |  | +0.01 (0.00) | +0.05 (0.00) |  | +0.00 (0.00) |  | +0.16 (0.00) | +0.08 (0.00) | +0.30 (0.00) | +0.41 (0.00) | 6.487169954064164e-05 |
| 18 | -0.00 (0.01) | -0.00 (0.00) | +0.00 (0.00) | +0.01 (0.00) | +0.01 (0.00) | -0.00 (0.00) |  | +0.01 (0.00) | +0.04 (0.00) | +0.01 (0.00) |  | +0.05 (0.00) | +0.00 (0.00) | +0.01 (0.00) | +0.36 (0.00) | 2.5516252725868816e-06 |
| 19 | -0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | -0.00 (0.00) | -0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | -0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | -0.00 (0.00) |  | +0.34 (0.00) | 6.581199473264054e-06 |
| 20 | +0.09 (0.00) | +0.16 (0.00) | +0.05 (0.00) | +0.04 (0.00) | +0.01 (0.00) | +0.10 (0.00) | +0.03 (0.00) | +0.17 (0.00) | +0.86 (0.00) | +0.56 (0.00) |  | +0.45 (0.00) | +0.93 (0.00) |  | +0.15 (0.00) | 1.2360715108970714e-05 |
| 21 | +0.01 (0.00) | +0.02 (0.00) | +0.07 (0.00) | +0.04 (0.00) | -0.00 (0.00) | +0.09 (0.00) | +0.13 (0.00) | +0.25 (0.00) | +0.05 (0.00) | +0.12 (0.00) | +0.06 (0.00) | +0.14 (0.00) | +0.04 (0.00) |  | +0.07 (0.00) | 6.522421325243848e-06 |
| 22 | +0.01 (0.00) | +0.01 (0.00) |  |  | +0.00 (0.00) | +0.00 (0.00) |  | +0.02 (0.00) | +0.03 (0.00) | +0.03 (0.00) | +0.17 (0.00) | +0.15 (0.00) | +0.13 (0.00) | +0.18 (0.00) | +0.19 (0.00) | 1.348358113537674e-05 |
| 23 | +0.00 (0.00) | +0.00 (0.00) | +0.01 (0.00) | +0.01 (0.00) | +0.00 (0.00) | -0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | -0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | +0.01 (0.00) |  | +0.28 (0.00) | 9.964220842973366e-06 |
| 25 | +0.00 (0.00) | +0.01 (0.00) | +0.02 (0.00) | +0.02 (0.00) | +0.00 (0.00) | -0.00 (0.00) | +0.01 (0.00) | +0.01 (0.00) | +0.05 (0.00) | +0.07 (0.00) | +0.14 (0.00) | +0.16 (0.00) |  |  | +0.03 (0.00) | 3.362575076512026e-06 |
| 26 | -0.00 (0.01) | +0.00 (0.01) | +0.01 (0.01) | +0.01 (0.01) | +0.00 (0.00) | +0.00 (0.00) | +0.03 (0.01) | -0.00 (0.01) | +0.01 (0.00) | +0.00 (0.01) | +0.00 (0.00) | +0.00 (0.01) |  |  | +0.00 (0.00) | 1.8040275012543446e-06 |
| 27 | -0.00 (0.01) | +0.06 (0.01) | +0.16 (0.00) | +0.17 (0.00) | +0.01 (0.01) | +0.02 (0.00) |  | +0.02 (0.00) | +0.01 (0.00) | +0.11 (0.00) | -0.00 (0.00) | +0.07 (0.00) |  | +0.03 (0.01) | +0.10 (0.00) | 1.2026074226462558e-06 |
| 28 | -0.13 (0.14) | -0.09 (0.15) | +0.02 (0.11) | -0.12 (0.24) | -0.06 (0.14) | -0.00 (0.23) | +0.04 (0.12) | -0.06 (0.13) | +0.71 (0.24) | +0.18 (0.15) | +0.34 (0.17) | +0.09 (0.12) | -0.08 (0.08) | -0.07 (0.13) |  | 6.312653545924108e-08 |
| 29 | -0.07 (0.07) | +0.16 (0.07) | +0.38 (0.09) | +0.41 (0.12) | -0.12 (0.13) | -0.14 (0.15) | +0.57 (0.14) | -0.08 (0.10) |  | -0.07 (0.12) | -0.09 (0.17) | +0.23 (0.11) | +0.25 (0.09) | +0.59 (0.07) |  | 9.013608768817604e-08 |
| 30 |  | +1.00 (0.01) |  | +0.28 (0.01) | +0.21 (0.01) | +0.43 (0.00) | +1.40 (0.01) | +0.29 (0.00) | +0.07 (0.00) | +0.45 (0.01) | +0.01 (0.00) | +0.35 (0.01) | +0.57 (0.01) | +0.60 (0.01) | +0.30 (0.00) | 1.8048796666082673e-06 |
| 31 | +0.01 (0.00) | +0.02 (0.00) | +0.01 (0.00) | +0.02 (0.00) | +0.08 (0.00) | +0.06 (0.00) | +0.04 (0.00) | +0.01 (0.00) |  | +0.06 (0.00) | +0.18 (0.00) | +0.04 (0.00) | +0.02 (0.00) | +0.01 (0.00) | +0.52 (0.00) | 1.672758424867183e-05 |
| 32 | +0.13 (0.10) | +0.58 (0.18) | +0.72 (0.18) | +2.04 (0.22) | -0.05 (0.11) | +0.03 (0.10) | +0.85 (0.18) | +0.00 (0.13) | +0.07 (0.11) | +0.13 (0.16) |  | -0.14 (0.19) | +0.07 (0.23) | +0.22 (0.19) |  | 5.758905942793948e-08 |
| 33 | +0.01 (0.00) |  | +0.00 (0.00) | +0.01 (0.00) |  | +0.01 (0.00) |  | +0.02 (0.00) | +0.01 (0.00) | +0.02 (0.00) | +0.04 (0.00) | +0.03 (0.00) | +0.00 (0.00) | +0.03 (0.00) | +0.33 (0.00) | 1.0818607363402328e-05 |
| 34 | -0.07 (0.17) | +0.07 (0.13) | +0.08 (0.27) | +0.13 (0.31) | -0.29 (0.37) | -0.24 (0.38) | +0.89 (0.16) | +0.37 (0.14) | -0.32 (0.35) | +0.59 (0.14) | +0.21 (0.40) | +0.87 (0.16) | +1.33 (0.31) | -0.08 (0.19) |  | 5.3636081412275804e-08 |
| 35 | +0.05 (0.00) | +0.07 (0.00) | +0.04 (0.00) | +0.17 (0.00) | -0.00 (0.00) | +0.02 (0.00) | +0.11 (0.00) | +0.02 (0.00) | +0.05 (0.00) | +0.08 (0.00) | +0.08 (0.00) | +0.11 (0.00) | -0.00 (0.00) | +0.04 (0.00) | +0.44 (0.00) | 6.167514617886943e-06 |
| 36 | -0.02 (0.12) | +0.12 (0.10) | +0.04 (0.09) | +0.60 (0.03) | +2.26 (0.11) | +1.27 (0.08) | +0.52 (0.09) | +1.24 (0.09) | +0.05 (0.04) | +1.51 (0.07) | +4.25 (0.08) | +1.79 (0.09) | +1.87 (0.06) | +2.74 (0.05) |  | 5.8722935038225517e-08 |
| 37 | -0.02 (0.37) | -0.01 (0.51) | +0.06 (0.09) | +0.95 (0.24) | +0.09 (0.41) | -0.11 (0.24) | +0.80 (0.17) | +0.18 (0.24) | +0.49 (0.11) | +1.30 (0.19) | -0.12 (0.14) | +0.10 (0.23) | +0.01 (0.21) |  | +1.43 (0.15) | 3.592644717286524e-08 |

## The amendment-3 form beside it (both polarisations averaged, no trim; not the ruling's input)

| ch | status | tau_c (s) | G | plateau (A^2) |
|---|---|---|---|---|
| 14 | constant | 86164.1 | 2054312.0 | -2.753e-10 |
| 15 | measured | 128.9 | 3072.7 | 1.906e-06 |
| 16 | measured | 153.3 | 3654.3 | 3.209e-06 |
| 17 | bound | 3600.0 | 85830.7 | 3.810e-05 |
| 18 | measured | 219.0 | 5222.4 | 2.780e-06 |
| 19 | measured | 3003.7 | 71614.0 | 5.422e-07 |
| 20 | measured | 3477.5 | 82909.0 | 1.090e-06 |
| 21 | measured | 177.2 | 4225.0 | 1.544e-06 |
| 22 | bound | 3600.0 | 85830.7 | 7.669e-06 |
| 23 | bound | 3600.0 | 85830.7 | 3.753e-06 |
| 25 | measured | 1114.9 | 26581.9 | 9.761e-07 |
| 26 | bound | 3600.0 | 85830.7 | 5.185e-07 |
| 27 | measured | 99.2 | 2365.0 | 1.334e-06 |
| 28 | measured | 171.7 | 4093.0 | 6.801e-08 |
| 29 | measured | 2058.7 | 49084.3 | 6.008e-08 |
| 30 | measured | 2477.6 | 59070.2 | 4.999e-06 |
| 31 | measured | 2980.1 | 71051.1 | 4.326e-06 |
| 32 | bound | 3600.0 | 85830.7 | 1.716e-07 |
| 33 | bound | 3600.0 | 85830.7 | 3.982e-06 |
| 34 | bound | 3600.0 | 85830.7 | 1.283e-07 |
| 35 | measured | 2971.5 | 70847.1 | 2.119e-06 |
| 36 | measured | 2120.7 | 50561.9 | 7.184e-07 |
| 37 | measured | 2130.9 | 50804.6 | 5.868e-07 |

## Phase coherence of the reference-subtracted excess (cadence_lags.py, class ns1_x, in-band), mean rho by lag class

| ch | 15 s | 30 s | 45 s | 60 s | 90 s | 120 s | 180 s | 240 s | 360 s | 480 s | 600 s | 720 s | 960 s | 1200 s | 3600 s | 18000 s | 36000 s | 86400 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | +0.90 | +0.74 | +0.75 | +0.95 | +0.95 | +0.97 | +0.98 | +0.94 | +0.94 | +0.92 | +0.93 | +0.91 | +0.89 | +0.82 | +0.95 | +0.83 | +0.88 | +0.93 |
| 15 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +1.00 | +0.96 | +0.92 | +0.99 |
| 16 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.98 | +0.96 | +1.00 |
| 17 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 |
| 18 | +1.00 | +0.99 | +0.98 | +0.98 | +1.00 | +1.00 | +0.98 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +0.99 | +0.99 | +0.98 | +1.00 |
| 19 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +0.99 | +1.00 |
| 20 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +0.99 | +1.00 |
| 21 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +0.99 | +1.00 |
| 22 | +1.00 | +1.00 | +0.99 | +1.00 | +1.00 | +1.00 | +0.99 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +0.99 | +1.00 |
| 23 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +0.98 | +1.00 |
| 25 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +0.99 | +0.99 | +1.00 | +0.98 | +0.99 | +0.98 |
| 26 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +0.99 | +1.00 | +0.98 | +0.99 | +0.98 |
| 27 | +1.00 | +1.00 | +0.99 | +0.99 | +1.00 | +1.00 | +0.99 | +1.00 | +1.00 | +0.99 | +0.99 | +0.99 | +1.00 | +1.00 | +1.00 | +0.99 | +0.99 | +0.99 |
| 28 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +0.99 | +0.98 | +0.98 | +0.99 | +0.99 | +0.99 | +0.98 | +0.95 | +0.99 |
| 29 | +1.00 | +1.00 | +0.99 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +0.98 | +0.98 | +0.99 | +0.99 | +0.99 | +1.00 | +0.99 | +0.93 | +0.94 | +0.98 |
| 30 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 |
| 31 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 |
| 32 | +1.00 | +0.99 | +0.98 | +0.98 | +1.00 | +0.99 | +0.92 | +0.97 | +0.98 | +0.96 | +0.99 | +0.96 | +0.98 | +0.98 | +0.99 | +0.96 | +0.92 | +0.98 |
| 33 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +1.00 | +0.99 | +1.00 | +0.99 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +1.00 | +0.99 |
| 34 | +0.99 | +1.00 | +0.98 | +0.99 | +0.99 | +0.99 | +0.99 | +0.97 | +0.91 | +0.94 | +0.93 | +0.95 | +0.96 | +0.95 | +0.95 | +0.94 | +0.94 | +0.96 |
| 35 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +1.00 | +0.99 | +1.00 | +0.99 | +0.99 |
| 36 | +0.96 | +0.92 | +0.75 | +0.85 | +0.93 | +0.85 | +0.37 | +0.42 | +0.96 | +0.69 | +0.61 | +0.65 | +0.71 | +0.79 | +0.70 | +0.81 | +0.75 | +0.60 |
| 37 | +0.99 | +0.99 | +0.98 | +0.99 | +1.00 | +0.99 | +0.99 | +0.97 | +0.99 | +0.98 | +0.97 | +0.98 | +0.98 | +0.98 | +0.97 | +0.96 | +0.95 | +0.98 |

## Table of record at the measured gain (one line per channel; the CSV has one row per freq_id)

| ch | disposition | policy or reason | tau_c (s) | status | G measured | R none G_meas | R deployed G_meas | R deployed G=1 |
|---|---|---|---|---|---|---|---|---|
| 37 | undetermined | no tolerance on the board | 86164 | refused: trim spread 5.7 | 2054312 |  |  |  |
| 36 | undetermined | at the control floor on the BAO baselines: long, not measured above the control floor (the floor itself is 23.5 dB over tolerance at G 54734); short, at floor (not measured above the control floor (gain unmeasured)); worst case on the 0.3 m baseline: at floor (not measured above the control floor (the floor itself is 5.7 dB over tolerance at G 1445)) | 61 | measured | 1444 | 57.989 | 23.345 | 0.016 |
| 35 | excise | excised on the long BAO baselines (39 and 78 m north-south, 22 to 66 m east-west): 17.0 dB over tolerance under every policy (least: cal_q0.9) with both credits at G 9129 (measured); -22.6 dB at G = 1; on the short BAO baselines: excise (15.9 dB over tolerance under every policy (least: cal_q0.9) with both credits at G 5379 (measured); -21.4 dB at G = 1); worst case on the 0.3 m baseline: excise (22.3 dB over tolerance under every policy (least: cal_q0.9) with both credits at G 85831 (bound); -27.0 dB at G = 1); subtraction bar 17.0 dB, -22.6 dB even at G=1 | 3600 | bound | 85831 | 6349.821 | 2556.303 | 0.030 |
| 34 | undetermined | at the control floor on the BAO baselines: long, not measured above the control floor (the floor itself is 13.3 dB over tolerance at G 4310); short, at floor (not measured above the control floor (gain unmeasured)); worst case on the 0.3 m baseline: unpriced (measured above the floor but the gain is unmeasured: -28.1 dB at G = 1, 21.2 dB at the persistence bound; no measured class on the range and no borrowed measurement (amendment 9 item 3)) | 86164 | refused: trim spread 518.0 | 2054312 | 356193.775 | 51019.672 | 0.025 |
| 33 | excise | excised on the long BAO baselines (39 and 78 m north-south, 22 to 66 m east-west): 25.8 dB over tolerance under every policy (least: keep_all) with both credits at G 51036 (measured); -21.3 dB at G = 1; on the short BAO baselines: unpriced (measured above the floor but the gain is unmeasured: -23.1 dB at G = 1, 26.2 dB at the persistence bound; no measured class on the range and no borrowed measurement (amendment 9 item 3)); worst case on the 0.3 m baseline: excise (25.4 dB over tolerance under every policy (least: keep_all) with both credits at G 85831 (bound); -24.0 dB at G = 1); subtraction bar 25.8 dB, -21.3 dB even at G=1 | 3600 | bound | 85831 | 27845.227 | 3988.431 | 0.046 |
| 32 | undetermined | at the control floor on the BAO baselines: long, not measured above the control floor (the floor itself is 23.8 dB over tolerance at G 48718); short, at floor (not measured above the control floor (the floor itself is 15.7 dB over tolerance at G 49732)); worst case on the 0.3 m baseline: marginal (2.7 dB over tolerance (cal_q0.5), within the 3 dB allowance, at G 844 (measured)) | 35 | measured | 843 | 195.047 | 27.938 | 0.033 |
| 31 | excise | excised on the long BAO baselines (39 and 78 m north-south, 22 to 66 m east-west): 26.4 dB over tolerance on the lowest epoch (20260917140230) with both credits at G 60108 (borrowed); no policy can pass; -21.4 dB at G = 1; on the short BAO baselines: excise (25.1 dB over tolerance on the lowest epoch (20260917040230) with both credits at G 60108 (measured); no policy can pass; -22.7 dB at G = 1); worst case on the 0.3 m baseline: excise (25.6 dB over tolerance on the lowest epoch (20260917040230) with both credits at G 85831 (bound); no policy can pass; -23.7 dB at G = 1); subtraction bar 26.4 dB, -21.4 dB even at G=1 | 3600 | bound | 85831 | 38251.288 | 5478.951 | 0.064 |
| 30 | excise | excised on the long BAO baselines (39 and 78 m north-south, 22 to 66 m east-west): 28.6 dB over tolerance on the lowest epoch (20260916162300) with both credits at G 16365 (measured); no policy can pass; -13.5 dB at G = 1; on the short BAO baselines: excise (18.7 dB over tolerance on the lowest epoch (20260917161008) with both credits at G 1004 (measured); no policy can pass; -11.4 dB at G = 1); worst case on the 0.3 m baseline: excise (16.2 dB over tolerance on the lowest epoch (20260917090230) with both credits at G 715 (measured); no policy can pass; -12.3 dB at G = 1); subtraction bar 28.6 dB, -13.5 dB even at G=1 | 30 | measured | 715 | 3124.378 | 447.523 | 0.626 |
| 29 | undetermined | at the control floor on the BAO baselines: long, not measured above the control floor (the floor itself is 5.3 dB over tolerance at G 6225); short, at floor (not measured above the control floor (gain unmeasured)); worst case on the 0.3 m baseline: at floor (not measured above the control floor (the floor itself is 12.6 dB over tolerance at G 28610)) | 1200 | bound | 28610 | 2988.123 | 118.145 | 0.004 |
| 28 | undetermined | measured above the floor but unpriced: long BAO baselines, at floor (keep-all is not measured above the control floor (the floor itself is 8.0 dB over tolerance at G 13461); the calibrated policies read 10.5 dB over on their kept dumps); short BAO baselines, unpriced (measured above the floor but the gain is unmeasured: -34.3 dB at G = 1, 15.1 dB at the persistence bound; no measured class on the range and no borrowed measurement (amendment 9 item 3)); worst case on the 0.3 m baseline: at floor (not measured above the control floor (gain unmeasured)) | 86164 | refused: trim spread 3.4 | 2054312 | 149349.646 | 5905.009 | 0.003 |
| 27 | undetermined | measured above the floor but unpriced: long BAO baselines, at floor (not measured above the control floor (the floor itself is -3.9 dB over tolerance at G 744)); short BAO baselines, unpriced (measured above the floor but the gain is unmeasured: -38.7 dB at G = 1, 10.7 dB at the persistence bound; no measured class on the range and no borrowed measurement (amendment 9 item 3)); worst case on the 0.3 m baseline: excise (15.1 dB over tolerance under every policy (least: cal_q0.9) with both credits at G 85831 (bound); -34.2 dB at G = 1) | 3600 | bound | 85831 | 10337.270 | 408.717 | 0.005 |
| 26 | undetermined | measured above the floor but unpriced: long BAO baselines, at floor (not measured above the control floor (the floor itself is 18.7 dB over tolerance at G 24049)); short BAO baselines, unpriced (measured above the floor but the gain is unmeasured: -31.8 dB at G = 1, 17.5 dB at the persistence bound; no measured class on the range and no borrowed measurement (amendment 9 item 3)); worst case on the 0.3 m baseline: excise (27.3 dB over tolerance under every policy (least: cal_q0.9) with both credits at G 85831 (bound); -22.0 dB at G = 1) | 3600 | bound | 85831 | 43446.068 | 6047.612 | 0.070 |
| 25 | undetermined | measured above the floor but unpriced: long BAO baselines, at floor (not measured above the control floor (the floor itself is 18.9 dB over tolerance at G 18134)); short BAO baselines, unpriced (measured above the floor but the gain is unmeasured: -30.9 dB at G = 1, 18.5 dB at the persistence bound; no measured class on the range and no borrowed measurement (amendment 9 item 3)); worst case on the 0.3 m baseline: excise (29.1 dB over tolerance under every policy (least: cal_q0.9) with both credits at G 85831 (bound); -20.2 dB at G = 1) | 3600 | bound | 85831 | 63318.878 | 8813.870 | 0.103 |
| 24 | undetermined | no products |  |  |  |  |  |  |
| 23 | undetermined | measured above the floor but unpriced: long BAO baselines, at floor (not measured above the control floor (gain unmeasured)); short BAO baselines, unpriced (measured above the floor but the gain is unmeasured: -34.1 dB at G = 1, 15.3 dB at the persistence bound; no measured class on the range and no borrowed measurement (amendment 9 item 3)); worst case on the 0.3 m baseline: excise (31.4 dB over tolerance under every policy (least: cal_q0.9) with both credits at G 85831 (bound); -17.9 dB at G = 1) | 3600 | bound | 85831 | 192841.559 | 14619.237 | 0.170 |
| 22 | undetermined | below the forecast's baseline cut: measured above the floor on the 9.8 and 19.5 m classes (16.2 dB over tolerance under every policy (least: keep_all) with both credits at G 49310 (measured); -30.7 dB at G = 1), which the forecast does not price (amendment 9); on the long BAO baselines: unpriced (measured above the floor but the gain is unmeasured: -24.7 dB at G = 1, 24.6 dB at the persistence bound; no measured class on the range and no borrowed measurement (amendment 9 item 3)); worst case on the 0.3 m baseline: excise (28.9 dB over tolerance under every policy (least: cal_q0.9) with both credits at G 85831 (bound); -20.4 dB at G = 1) | 3600 | bound | 85831 | 240304.657 | 9153.347 | 0.107 |
| 21 | undetermined | at the control floor on the BAO baselines: long, not measured above the control floor (the floor itself is 16.8 dB over tolerance at G 26727); short, at floor (not measured above the control floor (the floor itself is 8.7 dB over tolerance at G 6732)); worst case on the 0.3 m baseline: excise (28.8 dB over tolerance under every policy (least: cal_q0.9) with both credits at G 85831 (bound); -20.6 dB at G = 1) | 3600 | bound | 85831 | 224811.860 | 8563.217 | 0.100 |
| 20 | undetermined | at the control floor on the BAO baselines: long, not measured above the control floor (the floor itself is 17.6 dB over tolerance at G 20337); short, at floor (not measured above the control floor (the floor itself is 13.6 dB over tolerance at G 47009)); worst case on the 0.3 m baseline: unpriced (measured above the floor but the gain is unmeasured: -21.2 dB at G = 1, 28.1 dB at the persistence bound; no measured class on the range and no borrowed measurement (amendment 9 item 3)) | 86164 | refused: trim spread 11.3 | 2054312 | 4164703.748 | 166208.406 | 0.081 |
| 19 | undetermined | measured above the floor but unpriced: long BAO baselines, at floor (not measured above the control floor (the floor itself is 22.6 dB over tolerance at G 68064)); short BAO baselines, unpriced (measured above the floor but the gain is unmeasured: -32.9 dB at G = 1, 16.5 dB at the persistence bound; no measured class on the range and no borrowed measurement (amendment 9 item 3)); worst case on the 0.3 m baseline: excise (24.9 dB over tolerance on the lowest epoch (20260917090230) with both credits at G 85831 (bound); no policy can pass; -24.4 dB at G = 1) | 3600 | bound | 85831 | 95200.723 | 5026.673 | 0.059 |
| 18 | undetermined | at the control floor on the BAO baselines: long, not measured above the control floor (the floor itself is 14.6 dB over tolerance at G 11139); short, keep (keep_all passes (-4.7 dB) with both credits at G 358 (measured)); worst case on the 0.3 m baseline: excise (28.5 dB over tolerance under every policy (least: keep_all) with both credits at G 85831 (bound); -20.8 dB at G = 1) | 3600 | bound | 85831 | 153735.093 | 8117.334 | 0.095 |
| 17 | excise | excised on the long BAO baselines (39 and 78 m north-south, 22 to 66 m east-west): 16.5 dB over tolerance under every policy (least: cal_q0.5) with both credits at G 896 (measured); -13.1 dB at G = 1; on the short BAO baselines: excise (11.8 dB over tolerance under every policy (least: cal_q0.5) with both credits at G 358 (measured); -13.8 dB at G = 1); worst case on the 0.3 m baseline: excise (35.1 dB over tolerance under every policy (least: cal_q0.5) with both credits at G 85831 (bound); -14.2 dB at G = 1); subtraction bar 16.5 dB, -13.1 dB even at G=1 | 3600 | bound | 85831 | 328943.340 | 35277.459 | 0.411 |
| 16 | undetermined | below the forecast's baseline cut: measured above the floor on the 9.8 and 19.5 m classes (4.1 dB over tolerance under every policy (least: cal_q0.9) with both credits at G 4077 (measured); -32.0 dB at G = 1), which the forecast does not price (amendment 9); on the long BAO baselines: at floor (not measured above the control floor (the floor itself is 12.7 dB over tolerance at G 3378)); worst case on the 0.3 m baseline: excise (30.2 dB over tolerance under every policy (least: cal_q0.5) with both credits at G 85831 (bound); -19.1 dB at G = 1) | 3600 | bound | 85831 | 14758.823 | 12547.609 | 0.146 |
| 15 | undetermined | below the forecast's baseline cut: measured above the floor on the 9.8 and 19.5 m classes (19.8 dB over tolerance under every policy (least: cal_q0.9) with both credits at G 49999 (measured); -27.2 dB at G = 1), which the forecast does not price (amendment 9); on the long BAO baselines: at floor (keep-all is not measured above the control floor (the floor itself is 17.6 dB over tolerance at G 15484); the calibrated policies read 21.0 dB over on their kept dumps); worst case on the 0.3 m baseline: excise (28.7 dB over tolerance under every policy (least: cal_q0.9) with both credits at G 85831 (bound); -20.6 dB at G = 1) | 3600 | bound | 85831 | 10110.657 | 8595.846 | 0.100 |
| 14 | undetermined | measured above the floor but unpriced: long BAO baselines, at floor (not measured above the control floor (gain unmeasured)); short BAO baselines, unpriced (measured above the floor but the gain is unmeasured: -25.7 dB at G = 1, 23.6 dB at the persistence bound; no measured class on the range and no borrowed measurement (amendment 9 item 3)); worst case on the 0.3 m baseline: excise (5.7 dB over tolerance under every policy (least: cal_q0.5) with both credits at G 358 (measured); -19.8 dB at G = 1) | 15 | measured | 358 | 34.409 | 29.254 | 0.082 |

Rows by disposition: excise 72, pilot 23, undetermined 273 (of 368).
