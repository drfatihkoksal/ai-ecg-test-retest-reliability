"""Torch-free WFDB reading and resampling shared by the reliability scripts."""

import numpy as np
import wfdb
from scipy.signal import resample_poly


LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
N_250 = 2500
N_LIMA = 4096


def read_ecg(path: str) -> tuple[np.ndarray | None, int | None, str]:
    """Return (samples x 12 mV array at native fs, fs, reason)."""
    try:
        x, hdr = wfdb.rdsamp(path)
    except Exception as exc:
        return None, None, f'read_error:{type(exc).__name__}'
    fs = int(hdr['fs'])
    channels = list(hdr['sig_name'])
    if fs not in (250, 500) or abs(len(x) / fs - 10) > 0.05:
        return None, fs, 'format_or_duration'
    if set(channels) != set(LEADS) or len(channels) != 12:
        return None, fs, 'lead_set'
    if any(unit != 'mV' for unit in hdr['units']):
        return None, fs, 'unit'
    x = x[:, [channels.index(lead) for lead in LEADS]].astype(np.float32)
    if not np.isfinite(x).all():
        return None, fs, 'nonfinite'
    if np.any(np.std(x, axis=0) < 0.005):
        return None, fs, 'flat_lead'
    if np.max(np.abs(x)) > 20:
        return None, fs, 'extreme_amplitude'
    x -= np.median(x, axis=0, keepdims=True)
    return x, fs, ''


def to_250(x: np.ndarray, fs: int) -> np.ndarray:
    y = resample_poly(x, 1, 2, axis=0) if fs == 500 else x
    return y[:N_250].T.astype(np.float32)


def to_lima(x: np.ndarray, fs: int) -> np.ndarray:
    y = resample_poly(x, 4, 5, axis=0) if fs == 500 else resample_poly(x, 8, 5, axis=0)
    y = y[:4000]
    pad = N_LIMA - len(y)
    out = np.zeros((12, N_LIMA), dtype=np.float32)
    out[:, pad // 2:pad // 2 + len(y)] = y.T
    return out
