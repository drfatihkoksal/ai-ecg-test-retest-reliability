"""POST-HOC: HEEDB 12SL (v24) reversal and data-quality statements.

(1) Validates the signal-based pairwise reversal detector against 12SL
    reversal statements in HEEDB, and (2) recomputes within-patient SDs after
    excluding pairs with any 12SL reversal or poor-quality statement and/or a
    detector flag. See reliability_protocol.md change log.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from paths import HEEDB_ROOT
from reliability_analysis import MINUTES_PER_YEAR, WORK, attach, load_scores
from reliability_reversal_check import CHEST_THRESHOLD, LIMB_THRESHOLD


HEEDB = HEEDB_ROOT / 'ECG'
REVERSAL_CODES = (1672, 1595, 1679)
QUALITY_CODES = (1302, 1500, 1501, 1502, 1504, 1673)
OUT = Path(__file__).with_name('reliability_heedb_12sl_check_2026-10-06.json')


def statements() -> pd.DataFrame:
    frames = []
    for site in ('I0001', 'I0006'):
        path = HEEDB / site / '12SL_diagnoses' / 'diagnoses_v24.csv'
        frames.append(duckdb.sql(f"""
            WITH d AS (
                SELECT regexp_extract(trim(FileName), '([^/]+?)(\\.hea)?\\s*$', 1) AS stem,
                       list_transform(string_split(codes, ','), x -> TRY_CAST(trim(x) AS INTEGER)) AS c
                FROM read_csv('{path}', header=true, all_varchar=true, quote='"',
                              strict_mode=false, parallel=false)
            )
            SELECT stem,
                   list_has_any(c, {list(REVERSAL_CODES)}) AS reversal_12sl,
                   list_has_any(c, {list(QUALITY_CODES)}) AS quality_12sl
            FROM d
        """).df())
    return pd.concat(frames).drop_duplicates('stem')


def main() -> None:
    s = statements()
    det = pd.read_parquet(WORK / 'rel_reversal_check.parquet')
    det = det[(det['ok'] == True) & det['site'].isin(['i0001', 'i0006'])].copy()  # noqa: E712
    for c in ('limb_ratio', 'chest_ratio'):
        det[c] = det[c].replace([np.inf, -np.inf], np.nan).fillna(99.0)
    det['flag_detector'] = (det['limb_ratio'] < LIMB_THRESHOLD) | (det['chest_ratio'] < CHEST_THRESHOLD)

    pairs = pd.read_parquet(WORK / 'rel_pairs.parquet')
    pairs = pairs[pairs['site'].isin(['i0001', 'i0006'])]
    for side in ('a', 'b'):
        pairs[f'stem_{side}'] = pairs[f'path_{side}'].str.extract(r'([^/]+)$')[0]
        pairs = pairs.merge(s.rename(columns={'stem': f'stem_{side}',
                                              'reversal_12sl': f'rev_{side}',
                                              'quality_12sl': f'qual_{side}'}),
                            on=f'stem_{side}', how='left')
    pairs = pairs.merge(det[['site', 'bin', 'record_a', 'record_b', 'limb_ratio',
                             'flag_detector']],
                        on=['site', 'bin', 'record_a', 'record_b'], how='inner')
    coverage = float(pairs[['rev_a', 'rev_b']].notna().all(axis=1).mean())
    for c in ('rev_a', 'rev_b', 'qual_a', 'qual_b'):
        pairs[c] = pairs[c].fillna(False).astype(bool)

    xor = pairs['rev_a'] ^ pairs['rev_b']
    validation = {'pairs': int(len(pairs)), 'statement_coverage': coverage,
                  'one_ecg_12sl_reversal': int(xor.sum()),
                  'auc_limb_ratio': float(roc_auc_score(xor, -pairs['limb_ratio']))}
    for t in (0.5, 0.6, 0.7, 0.8):
        flag = pairs['limb_ratio'] < t
        validation[f'threshold_{t}'] = {'flag_rate': float(flag.mean()),
                                        'sensitivity': float(flag[xor].mean()),
                                        'ppv': float(xor[flag].mean())}

    scores, _ = load_scores(WORK)
    p = attach(pairs, scores, ('a', 'b'))
    p = p[p['ok_a'].fillna(False) & p['ok_b'].fillna(False) & (p['md5_a'] != p['md5_b'])
          & p['age_a'].between(18, 89, inclusive='left')].copy()
    shift = p['gap_min'] / MINUTES_PER_YEAR
    p['dL'] = p['lima_age_b'] - p['lima_age_a'] - shift
    p['dH'] = p['heedb_age_b'] - p['heedb_age_a'] - shift
    p['dS'] = p['sex_logit_b'] - p['sex_logit_a']
    p['flag_12sl_rev'] = p['rev_a'] | p['rev_b']
    p['flag_12sl_qual'] = p['qual_a'] | p['qual_b']
    p['flag_all'] = p['flag_12sl_rev'] | p['flag_12sl_qual'] | p['flag_detector']
    sw = lambda x: float(np.std(x, ddof=1) / np.sqrt(2)) if len(x) > 20 else None
    rows = []
    for (site, bin_name), g in p.groupby(['site', 'bin']):
        clean = g[~g['flag_all']]
        rows.append({
            'site': site, 'bin': bin_name, 'n': int(len(g)),
            'first_ecg_12sl_rev_pct': 100 * float(g['rev_a'].mean()),
            'second_ecg_12sl_rev_pct': 100 * float(g['rev_b'].mean()),
            'first_ecg_12sl_quality_pct': 100 * float(g['qual_a'].mean()),
            'second_ecg_12sl_quality_pct': 100 * float(g['qual_b'].mean()),
            'any_flag_pct': 100 * float(g['flag_all'].mean()),
            'lima_sw_all': sw(g['dL']), 'lima_sw_clean': sw(clean['dL']),
            'lima_sw_12sl_rev_only': sw(g.loc[g['flag_12sl_rev'], 'dL']),
            'heedb_sw_all': sw(g['dH']), 'heedb_sw_clean': sw(clean['dH']),
            'sex_sw_all': sw(g['dS']), 'sex_sw_clean': sw(clean['dS']),
            'sex_flip_all': float(((g['sex_logit_a'] > 0) != (g['sex_logit_b'] > 0)).mean()),
            'sex_flip_clean': float(((clean['sex_logit_a'] > 0) != (clean['sex_logit_b'] > 0)).mean()),
        })
    OUT.write_text(json.dumps({'validation_vs_12sl': validation, 'by_site_bin': rows}, indent=1))
    print(json.dumps(validation, indent=1))
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == '__main__':
    main()
