"""Data locations for all scripts.

Set the environment variables below, or create an untracked paths_local.py
that defines any of WORK, HEEDB_ROOT and MIMIC_ROOT.

  EKG_WORK    restricted work directory for patient-level derived files
  HEEDB_ROOT  Harvard-Emory ECG Database root (contains ECG/I0001, ECG/I0006)
  MIMIC_ROOT  directory containing mimic-iv-ecg/1.0, mimic-iv-ed-2.2, mimiciv/3.1
"""

import os
from pathlib import Path

WORK = Path(os.environ.get('EKG_WORK', 'work'))
HEEDB_ROOT = Path(os.environ.get('HEEDB_ROOT', 'data/heedb'))
MIMIC_ROOT = Path(os.environ.get('MIMIC_ROOT', 'data/mimic'))

try:
    from paths_local import *  # noqa: F401,F403  (machine-specific overrides)
except ImportError:
    pass

MIMIC_MACHINE = MIMIC_ROOT / 'mimic-iv-ecg/1.0/machine_measurements.csv'
