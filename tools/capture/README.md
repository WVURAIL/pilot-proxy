# Frame-level reduction of CHIME baseband dumps

Tools used on the 2026-09 matched-capture dumps. They read one converted CHIME baseband file per frequency
(`baseband` uint8 [n_time, 2048], offset-binary 4+4 bit, columns in correlator input order with `index_map/input`
giving the chan_id) and work per 16384-sample frame, the archive detector's frame.

- `reduce_dump.py` one file: full 2048-input correlation per frame stacked into the redundant baseline classes,
  per-input powers, 16384-bin fine spectrum per 256-input block, per-input pilot cutouts. Single process; use a
  common `--grid-fpga` so frames align across frequencies.
- `band_shape.py` per coarse bin: mean power, frame variability, short-baseline coherence, pilot line and offset.
- `first_look.py` one row per file with line strength and per-input line SNR.
- `line_check.py` where the pilot line sits under both spectral senses; the CHIME fine FFT along time is inverted
  (bin +k is sky below the coarse centre).

Integer-MHz lines and the DC bin are instrumental spurs; exclude them from any line search.
