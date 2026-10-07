"""Test-retest agreement of the published Ribeiro et al. (2020) abnormality model.

For each class, compares the AI output between two ECGs of the same patient
and benchmarks it against the conventional machine statement (GE 12SL v24 in
HEEDB; machine report text in MIMIC) on the same pairs. Conduction
abnormalities (1st-degree AV block, RBBB, LBBB) are the primary classes
because they rarely change within hours. Aggregate output only.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from reliability_analysis import (MIMIC_MACHINE, N_BOOT, RNG, WORK, attach, binary_agreement,
                                  continuous_agreement, load_scores)
from reliability_heedb_12sl_check import HEEDB


CLASSES = ['avb1', 'rbbb', 'lbbb', 'sb', 'af', 'st']
PRIMARY_CLASSES = ['avb1', 'rbbb', 'lbbb']
CODES_12SL = {'avb1': [101], 'rbbb': [440, 442], 'lbbb': [460], 'sb': [21, 24],
              'af': [161], 'st': [23]}
MIMIC_REGEX = {
    'avb1': r'(1st|first) degree a-?v block',
    'rbbb': r'(?<!incomplete )right bundle branch block',
    'lbbb': r'(?<!incomplete )left bundle branch block',
    'sb': r'sinus bradycardia',
    'af': r'atrial fibrillation',
    'st': r'sinus tachycardia',
}
BINS = ('1_lt5m', '2_5_60m', '3_1_6h', '4_6_24h', '8_gt1y')
OUT = Path(__file__).with_name('reliability_diagnostic_results_2026-10-06.json')


def heedb_statements() -> pd.DataFrame:
    frames = []
    for site in ('I0001', 'I0006'):
        path = HEEDB / site / '12SL_diagnoses' / 'diagnoses_v24.csv'
        frames.append(duckdb.sql(f"""
            SELECT regexp_extract(trim(FileName), '([^/]+?)(\\.hea)?\\s*$', 1) AS key,
                   list_transform(string_split(codes, ','), x -> TRY_CAST(trim(x) AS INTEGER)) AS c
            FROM read_csv('{path}', header=true, all_varchar=true, quote='"',
                          strict_mode=false, parallel=false)""").df())
    codes = pd.concat(frames).drop_duplicates('key')
    for name, values in CODES_12SL.items():
        codes[f'm_{name}'] = codes['c'].apply(
            lambda c: any(v in c for v in values) if isinstance(c, (list, np.ndarray)) else np.nan)
    return codes.drop(columns='c')


def mimic_statements() -> pd.DataFrame:
    cols = ','.join(f'report_{i}' for i in range(18))
    frame = duckdb.sql(f"""
        SELECT CAST(study_id AS VARCHAR) AS key, lower(concat_ws(' | ', {cols})) AS t
        FROM read_csv('{MIMIC_MACHINE}')""").df()
    for name, pattern in MIMIC_REGEX.items():
        frame[f'm_{name}'] = frame['t'].str.contains(pattern, regex=True)
    return frame.drop(columns='t')


def flip_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: boolean calls at ECG A and ECG B."""
    out = binary_agreement(a, b)
    pos = a.astype(bool)
    out['n_positive_a'] = int(pos.sum())
    out['lost_on_b_given_positive_a'] = float((~b[pos]).mean()) if pos.sum() else None
    if pos.sum() >= 20:
        idx = RNG.integers(0, pos.sum(), size=(N_BOOT, pos.sum()))
        lost = (~b[pos])[idx].mean(axis=1)
        out['lost_ci'] = [float(np.percentile(lost, 2.5)), float(np.percentile(lost, 97.5))]
    return out


def main() -> None:
    scores, _ = load_scores(WORK)
    ribeiro = pd.concat(pd.read_parquet(f)
                        for f in sorted((WORK / 'rel_scores_ribeiro').glob('ribeiro_*.parquet')))
    scores = scores[['waveform_path', 'ok', 'md5']].merge(ribeiro, on='waveform_path', how='left')
    pairs = pd.read_parquet(WORK / 'rel_pairs.parquet')
    pairs = pairs[pairs['bin'].isin(BINS)]
    pairs = attach(pairs, scores, ('a', 'b'))
    pairs = pairs[pairs['ok_a'].fillna(False) & pairs['ok_b'].fillna(False)
                  & (pairs['md5_a'] != pairs['md5_b'])].copy()

    heedb = heedb_statements()
    mimic = mimic_statements()
    is_heedb = pairs['site'].isin(['i0001', 'i0006'])
    for s in 'ab':
        pairs[f'key_{s}'] = np.where(is_heedb, pairs[f'path_{s}'].str.extract(r'([^/]+)$')[0],
                                     pairs[f'record_{s}'])
        stmt = pd.concat([heedb, mimic]).drop_duplicates('key')
        pairs = pairs.merge(stmt.rename(columns={'key': f'key_{s}', **{
            f'm_{c}': f'm_{c}_{s}' for c in CLASSES}}), on=f'key_{s}', how='left')

    results = {'note': 'AI call = published-model probability > 0.5; machine = 12SL v24 '
                       '(HEEDB) or MIMIC machine report text', 'sites': {}}
    for (site, bin_name), g in pairs.groupby(['site', 'bin']):
        entry = {'n': int(len(g))}
        for c in CLASSES:
            pa, pb = g[f'ribeiro_{c}_a'].to_numpy(), g[f'ribeiro_{c}_b'].to_numpy()
            ok = np.isfinite(pa) & np.isfinite(pb)
            ma, mb = g[f'm_{c}_a'], g[f'm_{c}_b']
            has_m = (ma.notna() & mb.notna()).to_numpy() & ok
            logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            ai_a, ai_b = pa[has_m] > 0.5, pb[has_m] > 0.5
            m_a, m_b = ma.to_numpy()[has_m].astype(bool), mb.to_numpy()[has_m].astype(bool)
            concordant = m_a == m_b
            item = {
                'n_with_statements': int(has_m.sum()),
                'logit': continuous_agreement(logit(pa[ok]), logit(pb[ok])),
                'ai': flip_stats(ai_a, ai_b),
                'machine': flip_stats(m_a, m_b),
                'ai_flip_when_machine_concordant': float((ai_a != ai_b)[concordant].mean()),
                'ai_flip_when_machine_concordant_positive':
                    float((ai_a != ai_b)[concordant & m_a].mean()) if (concordant & m_a).sum() else None,
            }
            entry[c] = item
        results['sites'].setdefault(site, {})[bin_name] = entry

    OUT.write_text(json.dumps(results, indent=1, default=float))
    rows = []
    for site, bins in results['sites'].items():
        for b, e in bins.items():
            for c in CLASSES:
                i = e[c]
                rows.append({'site': site, 'bin': b, 'class': c, 'n': i['n_with_statements'],
                             'prev_ai': i['ai']['prevalence_a'], 'flip_ai': i['ai']['discordance'],
                             'lost_ai': i['ai']['lost_on_b_given_positive_a'],
                             'prev_m': i['machine']['prevalence_a'],
                             'flip_m': i['machine']['discordance'],
                             'lost_m': i['machine']['lost_on_b_given_positive_a'],
                             'kappa_ai': i['ai']['kappa'], 'kappa_m': i['machine']['kappa'],
                             'ai_flip_m_conc': i['ai_flip_when_machine_concordant'],
                             'logit_icc': i['logit']['icc_a1']})
    table = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(table[table['bin'].isin(['1_lt5m', '3_1_6h'])].round(3).to_string())


if __name__ == '__main__':
    main()
