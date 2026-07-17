# Transmitter census provenance

Source: FCC LMS and ISED station data, compiled as
`TV_Stations_UHF_within500mi_DRAO.xlsx`. Retrieved 2026-06-09.
Query criterion: UHF ATSC television stations within 500 statute miles of
DRAO (49.3208 N, 119.6239 W). Collection method: listing collection only
(all stations in radius; over-the-air filter; transmitter-class
identification for offset-tolerance context). No propagation model was
applied by this census.

Field strengths (detectability_db, 42 merged rows; 43 values before the
channel-sharing merge, one merge collapsing two): RabbitEars Signal Search
Map study, 2026-06-09 12:15 ET, shareable id 2738863, 120-statute-mile
search radius, receive height set to its maximum (99,999,999 ft AGL),
which effectively removes terrain blocking; values are therefore
optimistic/upper-bound estimates. Rows beyond 120 miles carry no field
strength and rank by distance. The RabbitEars result-list printout is
archived alongside this file.

Derivation (`census_from_xlsx.py`, deterministic; regenerate with
`python census_from_xlsx.py TV_Stations_UHF_within500mi_DRAO.xlsx census.csv`):

- Population = Type "on-air" only. Dropped: 23 off-air (no emission),
  9 analog (no ATSC pilot), 4 ATSC 3.0 (OFDM; no 8-VSB pilot tone).
- "Physical Ch(s)" parsing: every integer token in 14..36 becomes one row;
  multi-channel entries expand. One on-air row (CH5643-DT) carried no UHF
  token and is excluded: confirmed VHF-only. VHF allocations (channels
  2-13, 54-216 MHz) lie entirely below CHIME's 400-800 MHz band, so no
  VHF pilot can appear in-band; the census scope is UHF by construction.
- Channel-sharing partners (same rf_channel at the same site) emit one
  physical carrier and are merged into single emitter rows (4 merges;
  494 -> 490 emitter-channel rows: 93 primary, 397 non-primary).
- detectability_db = RabbitEars Signal Search field strength (dBuV/m)
  where available (42 merged rows within the 120-mile study radius; 43
  pre-merge values); association ranking falls back to distance
  otherwise.
- Distances converted miles -> km; bearings, CHIME channel indices, and
  the source sheet's Frequency Tolerance carried through. Note the
  regulatory asymmetry preserved from the source: every on-air
  full-power/Class A/Relay specifies +/-1 kHz; every Translator/LPTV is
  "None specified".
