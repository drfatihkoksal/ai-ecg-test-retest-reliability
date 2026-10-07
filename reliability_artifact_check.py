"""POST-HOC sensitivity: artifact and lead-reversal flags (see reliability_protocol.md change log).

Recomputes within-patient SDs after excluding (a) pairs in which either ECG
is in the top 10% of its cohort for high-frequency noise or baseline wander,
and (b) in MIMIC, pairs with a machine statement of lead reversal, unsuitable
recording, repeat request, or external noise.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from reliability_analysis import MIMIC_MACHINE, MINUTES_PER_YEAR, WORK, attach, load_scores


OUT = Path(__file__).with_name('reliability_artifact_check_2026-10-06.json')
BINS = ('1_lt5m', '2_5_60m', '3_1_6h', '8_gt1y')


def machine_flags() -> pd.DataFrame:
    cols = ','.join(f'report_{i}' for i in range(18))
    return duckdb.sql(f"""
        WITH m AS (
            SELECT CAST(study_id AS VARCHAR) AS record_id,
                   lower(concat_ws(' | ', {cols})) AS t
            FROM read_csv('{MIMIC_MACHINE}'))
        SELECT record_id,
               regexp_matches(t, 'lead reversal') AS reversal,
               regexp_matches(t, 'unsuitable for analysis|please repeat|external noise') AS unsuitable
        FROM m""").df()


def sw(x: pd.Series) -> float:
    return x.std() / np.sqrt(2)


def main() -> None:
    scores, _ = load_scores(WORK)
    p = attach(pd.read_parquet(WORK / 'rel_pairs.parquet'), scores, ('a', 'b'))
    p = p[p['ok_a'].fillna(False) & p['ok_b'].fillna(False) & (p['md5_a'] != p['md5_b'])
          & p['age_a'].between(18, 89, inclusive='left')].copy()
    shift = p['gap_min'] / MINUTES_PER_YEAR
    p['dL'] = p['lima_age_b'] - p['lima_age_a'] - shift
    p['dH'] = p['heedb_age_b'] - p['heedb_age_a'] - shift
    p['dS'] = p['sex_logit_b'] - p['sex_logit_a']

    flags = machine_flags()
    for s in 'ab':
        p = p.merge(flags.rename(columns={'record_id': f'record_{s}', 'reversal': f'rev_{s}',
                                          'unsuitable': f'uns_{s}'}), on=f'record_{s}', how='left')

    # Top 10% of each cohort's ECGs for high-frequency noise or baseline wander.
    noisy = pd.Series(False, index=p.index)
    for site, g in p.groupby('site'):
        hf = np.quantile(np.r_[g['hf_noise_a'], g['hf_noise_b']], 0.9)
        bw = np.quantile(np.r_[g['baseline_wander_a'], g['baseline_wander_b']], 0.9)
        noisy[g.index] = ((g['hf_noise_a'] > hf) | (g['hf_noise_b'] > hf)
                          | (g['baseline_wander_a'] > bw) | (g['baseline_wander_b'] > bw))
    p['noisy'] = noisy

    rows = []
    for (site, b), g in p.groupby(['site', 'bin']):
        if b not in BINS:
            continue
        clean = g[~g['noisy']]
        row = {'site': site, 'bin': b, 'n': len(g), 'swL': sw(g['dL']), 'swH': sw(g['dH']),
               'swS': sw(g['dS']), 'noisy%': 100 * g['noisy'].mean(),
               'swL_clean': sw(clean['dL']), 'swH_clean': sw(clean['dH']), 'swS_clean': sw(clean['dS'])}
        if site.startswith('mimic'):
            fa = g['rev_a'].fillna(False).astype(bool) | g['uns_a'].fillna(False).astype(bool)
            fb = g['rev_b'].fillna(False).astype(bool) | g['uns_b'].fillna(False).astype(bool)
            reversal = g['rev_a'].fillna(False).astype(bool) | g['rev_b'].fillna(False).astype(bool)
            unflagged, both, flagged = g[~(fa | fb)], g[~(fa | fb) & ~g['noisy']], g[fa | fb]
            row.update({'flagA%': 100 * fa.mean(), 'flagB%': 100 * fb.mean(), 'rev%': 100 * reversal.mean(),
                        'swL_noflag': sw(unflagged['dL']), 'swH_noflag': sw(unflagged['dH']),
                        'swS_noflag': sw(unflagged['dS']),
                        'swL_both': sw(both['dL']), 'swH_both': sw(both['dH']), 'n_both': len(both),
                        'swL_flagged': sw(flagged['dL']) if len(flagged) > 20 else np.nan,
                        'n_flagged': len(flagged)})
        rows.append(row)
    out = pd.DataFrame(rows).round(3)
    pd.set_option('display.width', 250)
    print(out.to_string())
    out.to_json(OUT, orient='records', indent=1)


if __name__ == '__main__':
    main()
