# Transmitter census provenance

Source: FCC LMS and ISED station data, compiled as
`TV_Stations_UHF_within500mi_DRAO.xlsx`. Retrieved <FILL: date>.
Query criterion: UHF ATSC television stations within 500 statute miles of
DRAO (49.3208 N, 119.6239 W). <FILL: any additional query parameters.>

Derivation (`census_from_xlsx.py`, deterministic; regenerate with
`python census_from_xlsx.py TV_Stations_UHF_within500mi_DRAO.xlsx census.csv`):

- Population = Type "on-air" only. Dropped: 23 off-air (no emission),
  9 analog (no ATSC pilot), 4 ATSC 3.0 (OFDM; no 8-VSB pilot tone).
- "Physical Ch(s)" parsing: every integer token in 14..36 becomes one row;
  multi-channel entries expand. One on-air row (CH5643-DT) carried no UHF
  token and is excluded. <FILL: confirm VHF-only or note reason.>
- Channel-sharing partners (same rf_channel at the same site) emit one
  physical carrier and are merged into single emitter rows (4 merges;
  494 -> 490 emitter-channel rows: 93 primary, 397 non-primary).
- detectability_db = modeled Field Strength (dBuV/m) where available
  (43 rows); association ranking falls back to distance otherwise.
- Distances converted miles -> km; bearings, CHIME channel indices, and
  the source sheet's Frequency Tolerance carried through. Note the
  regulatory asymmetry preserved from the source: every on-air
  full-power/Class A/Relay specifies +/-1 kHz; every Translator/LPTV is
  "None specified".
