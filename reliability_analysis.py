"""Test-retest reliability of AI-ECG age and sex outputs (reliability_protocol.md).

Reads patient-level scores from the restricted work directory and writes only
aggregate results.
"""

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from paths import MIMIC_MACHINE, WORK


SEEDS = (1, 2, 3, 4, 5)
AGE_MODELS = ('lima_age', 'heedb_age', 'bracke_age')
LIMB = ('I', 'II')
CHEST = ('V1', 'V2', 'V3', 'V4', 'V5', 'V6')
PRIMARY_BIN = '3_1_6h'
MINUTES_PER_YEAR = 525960.0
AGE_GAP_THRESHOLD = 8.0
DUPLICATE_MV = 0.01
N_BOOT = 2000
RNG = np.random.default_rng(20261005)


# ------------------------------------------------------------------ loading

def load_scores(work: Path) -> tuple[pd.DataFrame, dict]:
    frames, fingerprints = [], {}
    for path in sorted((work / 'rel_scores').glob('scores_*.parquet')):
        frame = pd.read_parquet(path)
        shard = path.stem.split('_')[1]
        fp = np.load(work / 'rel_scores' / f'fingerprint_{shard}.npy', mmap_mode='r')
        frame['fp_shard'] = shard
        frame['fp_row'] = np.arange(len(frame))
        fingerprints[shard] = fp
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    bracke = sorted((work / 'rel_scores_bracke').glob('bracke_*.parquet'))
    if bracke:
        scores = scores.merge(pd.concat(pd.read_parquet(b) for b in bracke),
                              on='waveform_path', how='left', validate='one_to_one')
    else:
        scores['bracke_age'] = np.nan
    scores['heedb_age'] = scores[[f'age_seed{s}' for s in SEEDS]].mean(axis=1)
    scores['sex_logit'] = scores[[f'sex_seed{s}' for s in SEEDS]].mean(axis=1)
    scores['age_seed_sd'] = scores[[f'age_seed{s}' for s in SEEDS]].std(axis=1)
    scores['sex_seed_sd'] = scores[[f'sex_seed{s}' for s in SEEDS]].std(axis=1)
    return scores, fingerprints


def attach(pairs: pd.DataFrame, scores: pd.DataFrame, suffixes: tuple[str, ...]) -> pd.DataFrame:
    keep = [c for c in scores.columns if c != 'waveform_path']
    for s in suffixes:
        renamed = scores.rename(columns={c: f'{c}_{s}' for c in keep})
        pairs = pairs.merge(renamed, left_on=f'path_{s}', right_on='waveform_path',
                            how='left', validate='many_to_one').drop(columns='waveform_path')
    return pairs


def duplicate_flags(pairs: pd.DataFrame, fingerprints: dict) -> np.ndarray:
    dup = (pairs['md5_a'] == pairs['md5_b']).to_numpy()
    for i in np.flatnonzero(~dup & pairs['ok_a'].fillna(False).to_numpy()
                            & pairs['ok_b'].fillna(False).to_numpy()):
        row = pairs.iloc[i]
        fa = fingerprints[row['fp_shard_a']][int(row['fp_row_a'])].astype(np.float32)
        fb = fingerprints[row['fp_shard_b']][int(row['fp_row_b'])].astype(np.float32)
        dup[i] = float(np.max(np.abs(fa - fb))) < DUPLICATE_MV
    return dup


def mimic_machine(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    report = ', '.join(f'report_{i}' for i in range(18))
    return con.execute(f"""
        WITH m AS (
            SELECT CAST(study_id AS VARCHAR) AS record_id, cart_id, filtering,
                   bandwidth, qrs_end - qrs_onset AS qrs_ms,
                   lower(concat_ws(' ', {report})) AS txt
            FROM read_csv(?)
        )
        SELECT record_id, cart_id, filtering, bandwidth, qrs_ms,
               regexp_matches(txt, 'paced|pacemaker') AS paced,
               CASE WHEN regexp_matches(txt, 'atrial fibrillation|atrial flutter') THEN 'af'
                    WHEN regexp_matches(txt, 'sinus') THEN 'sinus' ELSE 'other' END AS rhythm
        FROM m
    """, [str(MIMIC_MACHINE)]).df()


# ------------------------------------------------------------------ metrics

def icc_a1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two-way random, absolute agreement, single measurement; vectorised on axis -1."""
    n = a.shape[-1]
    grand = (a.mean(-1) + b.mean(-1)) / 2
    m = (a + b) / 2
    msr = 2 * ((m - grand[..., None]) ** 2).sum(-1) / (n - 1)
    msc = n * ((a.mean(-1) - grand) ** 2 + (b.mean(-1) - grand) ** 2)
    sst = ((a - grand[..., None]) ** 2 + (b - grand[..., None]) ** 2).sum(-1)
    mse = (sst - msr * (n - 1) - msc) / (n - 1)
    return (msr - mse) / (msr + mse + 2 * (msc - mse) / n)


def kappa(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    po = (x == y).mean(-1)
    px, py = x.mean(-1), y.mean(-1)
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe)


def boot_ci(func, *arrays) -> tuple[float, float, float]:
    point = float(func(*arrays))
    n = len(arrays[0])
    stats = []
    for start in range(0, N_BOOT, 200):
        idx = RNG.integers(0, n, size=(min(200, N_BOOT - start), n))
        stats.append(func(*(a[idx] for a in arrays)))
    lo, hi = np.percentile(np.concatenate(stats), [2.5, 97.5])
    return point, float(lo), float(hi)


def continuous_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    d = b - a
    sw = lambda x, y: np.std(y - x, axis=-1, ddof=1) / np.sqrt(2)
    point_sw, lo_sw, hi_sw = boot_ci(sw, a, b)
    icc, lo_icc, hi_icc = boot_ci(icc_a1, a, b)
    return {
        'n': int(len(a)), 'bias': float(d.mean()),
        'sd_diff': float(d.std(ddof=1)),
        'loa': [float(d.mean() - 1.96 * d.std(ddof=1)),
                float(d.mean() + 1.96 * d.std(ddof=1))],
        'sw': point_sw, 'sw_ci': [lo_sw, hi_sw],
        'rc': 2.77 * point_sw, 'rc_ci': [2.77 * lo_sw, 2.77 * hi_sw],
        'median_abs_diff': float(np.median(np.abs(d))),
        'icc_a1': icc, 'icc_ci': [lo_icc, hi_icc],
    }


def binary_agreement(x: np.ndarray, y: np.ndarray) -> dict:
    disc = lambda u, v: (u != v).mean(-1)
    p, lo, hi = boot_ci(disc, x.astype(float), y.astype(float))
    k, klo, khi = boot_ci(kappa, x.astype(float), y.astype(float))
    return {'prevalence_a': float(x.mean()), 'discordance': p,
            'discordance_ci': [lo, hi], 'kappa': k, 'kappa_ci': [klo, khi]}


def age_block(p: pd.DataFrame, score: str) -> dict:
    p = p[np.isfinite(p[f'{score}_a']) & np.isfinite(p[f'{score}_b'])]
    a = p[f'{score}_a'].to_numpy()
    b = p[f'{score}_b'].to_numpy()
    chrono_a = p['age_a'].to_numpy()
    chrono_b = chrono_a + p['gap_min'].to_numpy() / MINUTES_PER_YEAR
    gap_a, gap_b = a - chrono_a, b - chrono_b
    return {
        'predicted_age': continuous_agreement(a, b),
        'age_gap': continuous_agreement(gap_a, gap_b),
        'age_gap_gt8': binary_agreement(gap_a > AGE_GAP_THRESHOLD,
                                        gap_b > AGE_GAP_THRESHOLD),
        'mae_a': float(np.mean(np.abs(gap_a))),
    }


def sex_block(p: pd.DataFrame, score: str) -> dict:
    a = p[f'{score}_a'].to_numpy()
    b = p[f'{score}_b'].to_numpy()
    pa, pb = 1 / (1 + np.exp(-a)), 1 / (1 + np.exp(-b))
    out = {'logit': continuous_agreement(a, b),
           'class_flip': binary_agreement(a > 0, b > 0),
           'abs_dprob_gt_0.2': float(np.mean(np.abs(pb - pa) > 0.2))}
    male = p['male'].to_numpy()
    for label, mask in (('female_label', male == 0), ('male_label', male == 1)):
        if mask.sum() >= 50:
            out[f'logit_{label}'] = continuous_agreement(a[mask], b[mask])
    return out


# --------------------------------------------------------------- covariates

def add_pair_covariates(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p['d_hr'] = (p['heart_rate_b'] - p['heart_rate_a']).abs()
    p['rr_cv_max'] = p[['rr_cv_a', 'rr_cv_b']].max(axis=1)
    p['log_hf_noise_max'] = np.log(p[['hf_noise_a', 'hf_noise_b']].max(axis=1) + 1e-6)
    p['log_wander_max'] = np.log(p[['baseline_wander_a', 'baseline_wander_b']].max(axis=1) + 1e-6)
    for name, leads in (('limb', LIMB), ('chest', CHEST)):
        p[f'd_log_rms_{name}'] = np.mean(
            [np.abs(np.log(p[f'rms_{l}_b'] / p[f'rms_{l}_a'])) for l in leads], axis=0)
    p['log_gap_min'] = np.log(p['gap_min'])
    return p


def determinants(p: pd.DataFrame, outcome: np.ndarray, extra: list[str]) -> dict:
    base = ['d_hr', 'rr_cv_max', 'log_hf_noise_max', 'log_wander_max',
            'd_log_rms_limb', 'd_log_rms_chest', 'log_gap_min', 'age_a', 'male']
    cols = base + extra
    x = p[cols].astype(float)
    mask = x.notna().all(axis=1).to_numpy() & np.isfinite(outcome)
    x = x[mask]
    binary = [c for c in cols if set(np.unique(x[c])) <= {0.0, 1.0}]
    z = x.copy()
    for c in cols:
        if c not in binary:
            z[c] = (x[c] - x[c].mean()) / x[c].std()
    fit = sm.OLS(outcome[mask], sm.add_constant(z)).fit(cov_type='HC3')
    ci = fit.conf_int()
    return {'n': int(mask.sum()), 'r_squared': float(fit.rsquared),
            'outcome_mean': float(outcome[mask].mean()),
            'coefficients': {c: {'beta': float(fit.params[c]),
                                 'ci': [float(ci.loc[c, 0]), float(ci.loc[c, 1])],
                                 'p': float(fit.pvalues[c]),
                                 'unit': 'per unit (binary)' if c in binary else 'per SD'}
                             for c in cols}}


# --------------------------------------------------------------------- main

def eligible_age(p: pd.DataFrame) -> pd.Series:
    return p['age_a'].between(18, 89, inclusive='left')


def eligible_sex(p: pd.DataFrame) -> pd.Series:
    return p['male'].notna()


def sex_target(series: pd.Series) -> pd.Series:
    s = series.fillna('').str.strip().str.upper()
    return s.map({'MALE': 1.0, 'M': 1.0, 'FEMALE': 0.0, 'F': 0.0})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).with_name('reliability_results_2026-10-05.json'))
    args = parser.parse_args()
    con = duckdb.connect()
    scores, fingerprints = load_scores(args.work)
    pairs = pd.read_parquet(args.work / 'rel_pairs.parquet')
    pairs = attach(pairs, scores, ('a', 'b'))
    sa, sb = sex_target(pairs['sex_a']), sex_target(pairs['sex_b'])
    pairs['male'] = sa.where(sa == sb)
    pairs['duplicate'] = duplicate_flags(pairs, fingerprints)
    pairs['both_ok'] = pairs['ok_a'].fillna(False) & pairs['ok_b'].fillna(False)

    machine = mimic_machine(con)
    mimic_mask = pairs['site'].isin(['mimic', 'mimic_ed'])
    for s in ('a', 'b'):
        m = machine.rename(columns={c: f'{c}_{s}' for c in machine.columns if c != 'record_id'})
        pairs = pairs.merge(m, left_on=f'record_{s}', right_on='record_id', how='left') \
                     .drop(columns='record_id')
    pairs['device_change'] = np.where(mimic_mask, (pairs['cart_id_a'] != pairs['cart_id_b']).astype(float), np.nan)
    pairs['filter_change'] = np.where(mimic_mask, ((pairs['filtering_a'] != pairs['filtering_b'])
                                                   | (pairs['bandwidth_a'] != pairs['bandwidth_b'])).astype(float), np.nan)
    pairs['rhythm_change'] = np.where(mimic_mask, (pairs['rhythm_a'] != pairs['rhythm_b']).astype(float), np.nan)
    pairs['paced_either'] = np.where(mimic_mask, (pairs['paced_a'].fillna(False) | pairs['paced_b'].fillna(False)).astype(float), np.nan)
    pairs['d_qrs_ms'] = (pairs['qrs_ms_b'] - pairs['qrs_ms_a']).abs()
    pairs = add_pair_covariates(pairs)

    flow = (pairs.groupby(['site', 'bin'])
            .agg(pairs=('patient_id', 'size'), both_ok=('both_ok', 'sum'),
                 duplicates=('duplicate', 'sum')).reset_index())
    analysed = pairs[pairs['both_ok'] & ~pairs['duplicate']].copy()

    results = {'flow': flow.to_dict(orient='records'), 'sites': {}}
    for (site, bin_name), p in analysed.groupby(['site', 'bin']):
        entry = {}
        pa = p[eligible_age(p)]
        if len(pa) >= 100:
            entry['n_age'] = int(len(pa))
            for score in AGE_MODELS:
                if np.isfinite(pa[f'{score}_a']).sum() >= 100:
                    entry[score] = age_block(pa, score)
            entry['heedb_age_seed1'] = age_block(pa, 'age_seed1')
        ps = p[eligible_sex(p)]
        if len(ps) >= 100:
            entry['n_sex'] = int(len(ps))
            entry['sex_logit'] = sex_block(ps, 'sex_logit')
            entry['sex_seed1'] = sex_block(ps, 'sex_seed1')
            entry['sex_limb_only'] = sex_block(ps, 'sex_limb_only')
            entry['sex_chest_only'] = sex_block(ps, 'sex_chest_only')
        # Same-ECG spread across training seeds versus between-ECG spread.
        entry['seed_sd_age_median'] = float(np.median(np.r_[p['age_seed_sd_a'], p['age_seed_sd_b']]))
        entry['seed_sd_sex_median'] = float(np.median(np.r_[p['sex_seed_sd_a'], p['sex_seed_sd_b']]))
        entry['median_gap_min'] = float(p['gap_min'].median())
        results['sites'].setdefault(site, {})[bin_name] = entry

    # Share of long-term within-person variance already present over hours.
    results['short_vs_long'] = {}
    for site in ('i0001', 'i0006', 'mimic'):
        s = results['sites'].get(site, {})
        if PRIMARY_BIN in s and '8_gt1y' in s:
            results['short_vs_long'][site] = {
                score: (s[PRIMARY_BIN][score]['age_gap']['sw'] ** 2 /
                        s['8_gt1y'][score]['age_gap']['sw'] ** 2)
                for score in AGE_MODELS if score in s[PRIMARY_BIN]}

    # Determinants in the primary interval.
    results['determinants'] = {}
    for site in ('i0001', 'i0006', 'mimic', 'mimic_ed'):
        p = analysed[(analysed['site'] == site) & (analysed['bin'] == PRIMARY_BIN)]
        p = p[eligible_age(p) & eligible_sex(p)]
        extra = (['device_change', 'filter_change', 'rhythm_change', 'paced_either', 'd_qrs_ms']
                 if site.startswith('mimic') else [])
        chrono_shift = p['gap_min'].to_numpy() / MINUTES_PER_YEAR
        results['determinants'][site] = {
            'lima_abs_age_gap_change': determinants(
                p, np.abs(p['lima_age_b'] - p['lima_age_a'] - chrono_shift).to_numpy(), extra),
            'heedb_abs_age_gap_change': determinants(
                p, np.abs(p['heedb_age_b'] - p['heedb_age_a'] - chrono_shift).to_numpy(), extra),
            'bracke_abs_age_gap_change': determinants(
                p, np.abs(p['bracke_age_b'] - p['bracke_age_a'] - chrono_shift).to_numpy(), extra),
            'abs_sex_logit_change': determinants(
                p, np.abs(p['sex_logit_b'] - p['sex_logit_a']).to_numpy(), extra),
        }
        # Sensitivity: regular rhythm only.
        reg = p[p['rr_cv_max'] <= 0.15]
        if len(reg) >= 100 and eligible_age(reg).all():
            results['sites'][site][PRIMARY_BIN]['regular_rhythm_only'] = {
                **{score: age_block(reg, score) for score in AGE_MODELS},
                'sex_logit': sex_block(reg, 'sex_logit')}
        if site.startswith('mimic'):
            same = p[(p['device_change'] == 0) & (p['filter_change'] == 0)]
            if len(same) >= 100:
                results['sites'][site][PRIMARY_BIN]['same_device_filter'] = {
                    'n': int(len(same)),
                    **{score: age_block(same, score) for score in AGE_MODELS},
                    'sex_logit': sex_block(same, 'sex_logit')}

    # POST-HOC (added after first results; see protocol change log).
    results['post_hoc'] = {}
    sw = lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(2))
    for site in ('i0001', 'i0006', 'mimic', 'mimic_ed'):
        site_pairs = analysed[analysed['site'] == site]
        ref = site_pairs[(site_pairs['bin'] == '1_lt5m') & eligible_age(site_pairs)]
        p = site_pairs[(site_pairs['bin'] == PRIMARY_BIN) & eligible_age(site_pairs)].copy()
        shift = p['gap_min'] / MINUTES_PER_YEAR
        p['dL'] = p['lima_age_b'] - p['lima_age_a'] - shift
        p['dH'] = p['heedb_age_b'] - p['heedb_age_a'] - shift
        p['dB'] = p['bracke_age_b'] - p['bracke_age_a'] - shift
        p['chest_q'] = pd.qcut(p['d_log_rms_chest'], 5, labels=False)
        entry = {'by_chest_amplitude_change_quintile': [
            {'quintile': int(q), 'n': int(len(g)),
             'median_d_log_rms_chest': float(g['d_log_rms_chest'].median()),
             'lima_sw': sw(g['dL']), 'heedb_sw': sw(g['dH']), 'bracke_sw': sw(g['dB'])}
            for q, g in p.groupby('chest_q')]}
        if len(ref):
            entry['median_signal_change_by_bin'] = {
                b: {'d_log_rms_chest': float(g['d_log_rms_chest'].median()),
                    'd_log_rms_limb': float(g['d_log_rms_limb'].median()),
                    'd_hr': float(g['d_hr'].median())}
                for b, g in site_pairs.groupby('bin')}
            similar = p[(p['d_log_rms_chest'] <= ref['d_log_rms_chest'].median())
                        & (p['d_log_rms_limb'] <= ref['d_log_rms_limb'].median())
                        & (p['d_hr'] <= ref['d_hr'].median())]
            entry['signal_similar_to_lt5m'] = {'n': int(len(similar)),
                                               'lima_sw': sw(similar['dL']),
                                               'heedb_sw': sw(similar['dH']),
                                               'bracke_sw': sw(similar['dB'])}
        # Age-bias-corrected gap (Barthels et al. 2025): residual of the gap on
        # chronological age, fitted per site and model on first ECGs of all bins.
        aged = site_pairs[eligible_age(site_pairs)]
        for score in AGE_MODELS:
            fit = aged[np.isfinite(aged[f'{score}_a'])]
            if len(fit) < 100:
                continue
            slope, intercept = np.polyfit(fit['age_a'], fit[f'{score}_a'] - fit['age_a'], 1)
            corr_a = p[f'{score}_a'] - p['age_a'] - (intercept + slope * p['age_a'])
            chrono_b = p['age_a'] + shift
            corr_b = p[f'{score}_b'] - chrono_b - (intercept + slope * chrono_b)
            both = (np.isfinite(corr_a) & np.isfinite(corr_b)).to_numpy()
            corr_a, corr_b = corr_a.to_numpy()[both], corr_b.to_numpy()[both]
            entry[f'{score}_bias_corrected'] = {
                'slope': float(slope), 'intercept': float(intercept),
                'gap': continuous_agreement(corr_a, corr_b),
                'gap_gt8': binary_agreement(corr_a > AGE_GAP_THRESHOLD,
                                            corr_b > AGE_GAP_THRESHOLD)}
        adults = site_pairs[(site_pairs['bin'] == PRIMARY_BIN) & eligible_sex(site_pairs)
                            & (site_pairs['age_a'] >= 18)]
        entry['sex_adults_only'] = sex_block(adults, 'sex_logit')
        results['post_hoc'][site] = entry

    # Triples: does averaging two ECGs reduce disagreement with a third?
    triples = pd.read_parquet(args.work / 'rel_triples.parquet')
    triples = attach(triples, scores, ('a', 'b', 'c'))
    ok = triples[[f'ok_{s}' for s in 'abc']].fillna(False).all(axis=1)
    distinct = ((triples['md5_a'] != triples['md5_b']) & (triples['md5_b'] != triples['md5_c'])
                & (triples['md5_a'] != triples['md5_c']))
    triples = triples[ok & distinct & triples['age_a'].between(18, 89, inclusive='left')]
    results['triples'] = {}
    for site, t in triples.groupby('site'):
        entry = {'n': int(len(t))}
        for score in (*AGE_MODELS, 'sex_logit'):
            t_ok = t[np.isfinite(t[[f'{score}_{s}' for s in 'abc']]).all(axis=1)]
            single = t_ok[f'{score}_c'] - t_ok[f'{score}_a']
            mean2 = t_ok[f'{score}_c'] - (t_ok[f'{score}_a'] + t_ok[f'{score}_b']) / 2
            rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
            entry[score] = {'n': int(len(t_ok)),
                            'rms_single': rms(single), 'rms_mean_of_two': rms(mean2),
                            'ratio': rms(mean2) / rms(single),
                            'ratio_expected_exchangeable': float(np.sqrt(0.75))}
        results['triples'][site] = entry

    args.out.write_text(json.dumps(results, indent=1, default=float))
    flow.to_csv(args.work / 'rel_flow.csv', index=False)
    print(json.dumps({'written': str(args.out), 'analysed_pairs': int(len(analysed))}))


if __name__ == '__main__':
    main()
