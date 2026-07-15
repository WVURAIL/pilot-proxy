"""Shared locations for the pilot-proxy paper analysis chain.

Every location can be overridden with an environment variable. Defaults
assume this directory lives at <repo>/analysis/ and that data dumps follow
the ~/paper/ layout (see analysis/README.md):

  PP_REPO      pilot-proxy repo root        (default: parent of analysis/)
  PP_OUT       output dir for figures/CSVs  (default: ~/paper/out)
  PP_DUMPS     dir holding the npz dumps    (default: ~/paper/dumps)
  PP_PERFRAME  per-frame detector dump      (default: $PP_DUMPS/perframe.npz)
  PP_POWER     per-frame baseband power     (default: $PP_DUMPS/power.npz)
  PP_SPECTRA   integrated spectra dump      (default: $PP_DUMPS/all_spectra.npz)
  PP_RESULTS   extracted results bundle dir (default: ~/paper/results_bundle)
  PP_SWEEP_CSV merged evaluate-snr summary  (default: $PP_DUMPS/dtv_snr_summary.csv)

Importing this module also puts <repo>/src on sys.path so that
`from pilot_proxy...` imports work without installing the package.
"""
import os
import sys
from pathlib import Path


def _p(env, default):
    return Path(os.environ.get(env, str(default))).expanduser()


REPO = _p("PP_REPO", Path(__file__).resolve().parents[1])
sys.path.insert(0, str(REPO / "src"))

OUT = _p("PP_OUT", "~/paper/out")
OUT.mkdir(parents=True, exist_ok=True)
DUMPS = _p("PP_DUMPS", "~/paper/dumps")
PERFRAME = _p("PP_PERFRAME", DUMPS / "perframe.npz")
POWER = _p("PP_POWER", DUMPS / "power.npz")
SPECTRA = _p("PP_SPECTRA", DUMPS / "all_spectra.npz")
RESULTS = _p("PP_RESULTS", "~/paper/results_bundle")
SWEEP_CSV = _p("PP_SWEEP_CSV", DUMPS / "dtv_snr_summary.csv")
