"""POST-HOC: signal-based electrode reversal detection for reliability pairs.

For each consecutive ECG pair, median beats are compared after applying every
limb-electrode permutation (expressed exactly through Einthoven relations on
leads I and II) and every adjacent precordial transposition to ECG B. A pair
is flagged when some permutation fits ECG A much better than the identity.
A single-ECG rule (negative QRS in I with positive QRS in aVR) is also
computed. Thresholds are chosen against MIMIC machine statements and then
applied unchanged to HEEDB. See reliability_protocol.md change log.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from paths import WORK
from reliability_models import read_ecg, to_250


PRE, POST = 62, 112          # -250 ms .. +450 ms at 250 Hz
QRS = slice(PRE - 15, PRE + 25)  # -60 .. +100 ms
MAX_SHIFT = 5
# (I', II') as functions of (I, II) for each limb electrode permutation.
LIMB = {
    'ra_la': lambda i, ii: (-i, ii - i),
    'la_ll': lambda i, ii: (ii, i),
    'ra_ll': lambda i, ii: (i - ii, -ii),
    'cyclic_1': lambda i, ii: (ii - i, -i),
    'cyclic_2': lambda i, ii: (-ii, i - ii),
}
CHEST_SWAPS = [(k, k + 1) for k in range(5)]


def median_beat(x: np.ndarray) -> np.ndarray | None:
    """x: 12 x 2500 mV. Returns 12 x (PRE+POST) median beat or None."""
    energy = np.abs(np.diff(x, axis=1)).sum(axis=0)
    energy = np.convolve(energy, np.ones(8) / 8, mode='same')
    peaks, _ = find_peaks(energy, distance=75, height=0.4 * np.percentile(energy, 99))
    peaks = peaks[(peaks >= PRE) & (peaks + POST <= x.shape[1])]
    if len(peaks) < 3:
        return None
    beats = np.stack([x[:, p - PRE:p + POST] for p in peaks])
    beat = np.median(beats, axis=0)
    return beat - beat[:, :10].mean(axis=1, keepdims=True)


def fit_error(a: np.ndarray, b: np.ndarray) -> float:
    """Relative error of a vs best-scaled, best-shifted b (leads x time)."""
    best = np.inf
    for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
        bs = np.roll(b, s, axis=1)[:, MAX_SHIFT:-MAX_SHIFT]
        aa = a[:, MAX_SHIFT:-MAX_SHIFT]
        gain = float((aa * bs).sum() / max((bs * bs).sum(), 1e-9))
        err = np.linalg.norm(aa - gain * bs) / max(np.linalg.norm(aa), 1e-9)
        best = min(best, err)
    return float(best)


def single_rule(beat: np.ndarray) -> bool:
    qrs = beat[:, QRS].sum(axis=1)
    return bool(qrs[0] < 0 and qrs[3] > 0)   # lead I negative, aVR positive


def check_pair(job: tuple[str, str]) -> dict:
    out = {}
    beats = []
    for path in job:
        x, fs, reason = read_ecg(path)
        if x is None:
            return {'ok': False}
        beat = median_beat(to_250(x, fs))
        if beat is None:
            return {'ok': False}
        beats.append(beat)
    a, b = beats
    out['ok'] = True
    out['single_rule_a'] = single_rule(a)
    out['single_rule_b'] = single_rule(b)
    limb_a = a[[0, 1]]
    out['limb_identity'] = fit_error(limb_a, b[[0, 1]])
    for name, f in LIMB.items():
        out[f'limb_{name}'] = fit_error(limb_a, np.stack(f(b[0], b[1])))
    chest_a, chest_b = a[6:12], b[6:12]
    out['chest_identity'] = fit_error(chest_a, chest_b)
    swap_errors = []
    for i, j in CHEST_SWAPS:
        perm = list(range(6))
        perm[i], perm[j] = perm[j], perm[i]
        swap_errors.append(fit_error(chest_a, chest_b[perm]))
    out['chest_best_swap'] = float(min(swap_errors))
    out['chest_best_swap_index'] = int(np.argmin(swap_errors))
    return out


def run(work: Path, workers: int) -> None:
    pairs = pd.read_parquet(work / 'rel_pairs.parquet',
                            columns=['site', 'bin', 'patient_id', 'record_a', 'record_b',
                                     'path_a', 'path_b'])
    jobs = list(zip(pairs['path_a'], pairs['path_b']))
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(check_pair, jobs, chunksize=64))
    result = pd.concat([pairs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    limb_cols = [f'limb_{k}' for k in LIMB]
    result['limb_best'] = result[limb_cols].min(axis=1)
    result['limb_best_name'] = result[limb_cols].idxmin(axis=1).str.replace('limb_', '')
    result['limb_ratio'] = result['limb_best'] / result['limb_identity']
    result['chest_ratio'] = result['chest_best_swap'] / result['chest_identity']
    result.to_parquet(work / 'rel_reversal_check.parquet', index=False)
    print(json.dumps({'pairs': len(result), 'ok': int(result['ok'].sum())}))


LIMB_THRESHOLD = 0.8    # chosen on MIMIC machine statements (AUC 0.90)
CHEST_THRESHOLD = 0.7


def summarize(work: Path, out: Path) -> None:
    import duckdb
    from sklearn.metrics import roc_auc_score
    from reliability_analysis import (MIMIC_MACHINE, MINUTES_PER_YEAR, WORK as _,
                                      attach, load_scores)
    r = pd.read_parquet(work / 'rel_reversal_check.parquet')
    r = r[r['ok'] == True].copy()  # noqa: E712
    for c in ('limb_ratio', 'chest_ratio'):
        r[c] = r[c].replace([np.inf, -np.inf], np.nan).fillna(99.0)
    r['flag_limb'] = r['limb_ratio'] < LIMB_THRESHOLD
    r['flag_chest'] = r['chest_ratio'] < CHEST_THRESHOLD
    r['flag_any'] = r['flag_limb'] | r['flag_chest']
    cols = ','.join(f'report_{i}' for i in range(18))
    machine = duckdb.sql(f"""
        SELECT CAST(study_id AS VARCHAR) AS rid,
               regexp_matches(lower(concat_ws(' ', {cols})), 'lead reversal') AS rev
        FROM read_csv('{MIMIC_MACHINE}')""").df()
    m = (r[r['site'].isin(['mimic', 'mimic_ed'])]
         .merge(machine.rename(columns={'rid': 'record_a', 'rev': 'rev_a'}), on='record_a')
         .merge(machine.rename(columns={'rid': 'record_b', 'rev': 'rev_b'}), on='record_b'))
    xor = m['rev_a'] ^ m['rev_b']
    validation = {'pairs': int(len(m)), 'machine_one_ecg_reversal': int(xor.sum()),
                  'auc_limb_ratio': float(roc_auc_score(xor, -m['limb_ratio']))}
    for t in (0.5, 0.6, 0.7, 0.8):
        flag = m['limb_ratio'] < t
        validation[f'threshold_{t}'] = {'flag_rate': float(flag.mean()),
                                        'sensitivity': float(flag[xor].mean()),
                                        'ppv': float(xor[flag].mean())}

    scores, _ = load_scores(work)
    keep = ['site', 'bin', 'record_a', 'record_b', 'flag_limb', 'flag_chest', 'flag_any']
    p = pd.read_parquet(work / 'rel_pairs.parquet').merge(
        r[keep], on=['site', 'bin', 'record_a', 'record_b'], how='inner')
    p = attach(p, scores, ('a', 'b'))
    p = p[p['ok_a'].fillna(False) & p['ok_b'].fillna(False) & (p['md5_a'] != p['md5_b'])
          & p['age_a'].between(18, 89, inclusive='left')].copy()
    shift = p['gap_min'] / MINUTES_PER_YEAR
    p['dL'] = p['lima_age_b'] - p['lima_age_a'] - shift
    p['dH'] = p['heedb_age_b'] - p['heedb_age_a'] - shift
    p['dS'] = p['sex_logit_b'] - p['sex_logit_a']
    sw = lambda x: float(np.std(x, ddof=1) / np.sqrt(2)) if len(x) > 20 else None
    rows = []
    for (site, bin_name), g in p.groupby(['site', 'bin']):
        clean, flagged = g[~g['flag_any']], g[g['flag_any']]
        rows.append({'site': site, 'bin': bin_name, 'n': int(len(g)),
                     'flag_limb_pct': 100 * float(g['flag_limb'].mean()),
                     'flag_chest_pct': 100 * float(g['flag_chest'].mean()),
                     'lima_sw_all': sw(g['dL']), 'lima_sw_clean': sw(clean['dL']),
                     'lima_sw_flagged': sw(flagged['dL']),
                     'heedb_sw_all': sw(g['dH']), 'heedb_sw_clean': sw(clean['dH']),
                     'sex_sw_all': sw(g['dS']), 'sex_sw_clean': sw(clean['dS']),
                     'sex_flip_all': float(((g['sex_logit_a'] > 0) != (g['sex_logit_b'] > 0)).mean()),
                     'sex_flip_clean': float(((clean['sex_logit_a'] > 0) != (clean['sex_logit_b'] > 0)).mean())})
    result = {'thresholds': {'limb_ratio': LIMB_THRESHOLD, 'chest_ratio': CHEST_THRESHOLD},
              'validation_vs_mimic_machine': validation, 'by_site_bin': rows}
    out.write_text(json.dumps(result, indent=1))
    print(pd.DataFrame(rows).round(3).to_string())
    print(json.dumps(validation, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'summarize'))
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--workers', type=int, default=28)
    args = parser.parse_args()
    if args.action == 'run':
        run(args.work, args.workers)
    else:
        summarize(args.work, Path(__file__).with_name('reliability_reversal_check_2026-10-06.json'))


if __name__ == '__main__':
    main()
