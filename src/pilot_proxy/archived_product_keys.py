# coding=utf-8
"""Key names inside the archived per-pilot survey products.

The 2020--2026 survey archive was written under the detector's original
measurement vocabulary; those npz keys are immutable historical data. The
current vocabulary renamed the measurements everywhere else, and
``scripts/check_current_measurement_vocabulary.py`` keeps the retired
spellings off every current surface -- except this module, its single
designated home. Read archived products through these constants; current
products use the current vocabulary directly and never need this module.
"""

ARCHIVED_COARSE_POWER_RATIO = "fstat_raw"
ARCHIVED_FINE_POWER_RATIO = "fstat_fine"
ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB = "fstat_level_db"
ARCHIVED_PILOT_EXCESS_DB = "pnr_bin_db"
ARCHIVED_DATA_SHELF_SNR_DB = "snr_shelf_db"
