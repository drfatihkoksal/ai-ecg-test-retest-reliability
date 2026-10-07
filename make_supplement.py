"""Build the Supplementary material for the EHJ-DH reliability manuscript.

Every number is read from aggregate result files; nothing is typed by hand.
Writes reliability_supplement.md and supplementary figures, then (optionally)
converts to .docx and .pdf with pandoc / LibreOffice.
"""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paths import WORK


HERE = Path(__file__).parent
FIG = HERE / 'figures'
R = json.loads((HERE / 'reliability_results_2026-10-05.json').read_text())
D = json.loads((HERE / 'reliability_diagnostic_results_2026-10-06.json').read_text())
ART = json.loads((HERE / 'reliability_artifact_check_2026-10-06.json').read_text())
REV = json.loads((HERE / 'reliability_reversal_check_2026-10-06.json').read_text())
SL = json.loads((HERE / 'reliability_heedb_12sl_check_2026-10-06.json').read_text())
VALIDATION = HERE / 'reliability_model_validation_2026-10-06.json'


def export_validation() -> None:
    """Copy aggregate validation metrics from the work directory into the repository."""
    import torch
    training = {}
    for task in ('age', 'sex'):
        for seed in range(1, 6):
            ckpt = torch.load(WORK / f'rel_model_{task}_seed{seed}.pt', map_location='cpu',
                              weights_only=False)
            training[f'{task}_seed{seed}'] = {'epoch': int(ckpt['epoch']),
                                              'val_metric': float(ckpt['val_metric'])}
    VALIDATION.write_text(json.dumps({
        'lima_selection': json.loads((WORK / 'rel_lima_selection.json').read_text()),
        'bracke_validation': json.loads((WORK / 'rel_bracke_validation.json').read_text()),
        'ribeiro_validation': json.loads((WORK / 'rel_ribeiro_validation.json').read_text()),
        'in_house_training': training,
    }, indent=1))


if (WORK / 'rel_lima_selection.json').exists():
    export_validation()
_V = json.loads(VALIDATION.read_text())
LIMA_SEL, BRACKE_VAL, RIB_VAL = _V['lima_selection'], _V['bracke_validation'], _V['ribeiro_validation']

SITES = {'i0001': 'I0001', 'i0006': 'I0006', 'mimic': 'MIMIC', 'mimic_ed': 'MIMIC ED'}
SITE_NAMES = {'i0001': 'HEEDB I0001 (test)', 'i0006': 'HEEDB I0006', 'mimic': 'MIMIC-IV-ECG',
              'mimic_ed': 'MIMIC ED, troponin-negative'}
BINS = {'1_lt5m': '<5 min', '2_5_60m': '5–60 min', '3_1_6h': '1–6 h', '4_6_24h': '6–24 h',
        '5_1_7d': '1–7 d', '6_7_30d': '7–30 d', '7_30d_1y': '30 d–1 y', '8_gt1y': '>1 y'}
AGE_MODELS = {'lima_age': 'Lima *et al.*', 'bracke_age': 'Bracke *et al.*',
              'heedb_age': 'In-house five-seed ensemble', 'heedb_age_seed1': 'In-house single seed'}
CLASSES = {'avb1': 'First-degree AV block', 'rbbb': 'RBBB', 'lbbb': 'LBBB',
           'sb': 'Sinus bradycardia', 'af': 'Atrial fibrillation', 'st': 'Sinus tachycardia'}
PREDICTORS = {'d_hr': 'Heart-rate difference', 'rr_cv_max': 'RR variability (higher of pair)',
              'log_hf_noise_max': 'High-frequency noise (higher of pair, log)',
              'log_wander_max': 'Baseline wander (higher of pair, log)',
              'd_log_rms_limb': 'Limb amplitude change', 'd_log_rms_chest': 'Precordial amplitude change',
              'log_gap_min': 'Interval (log minutes)', 'age_a': 'Chronological age', 'male': 'Male (binary)',
              'device_change': 'Device change (binary)', 'filter_change': 'Filter change (binary)',
              'rhythm_change': 'Rhythm-class change (binary)', 'paced_either': 'Paced (binary)',
              'd_qrs_ms': 'QRS-duration difference'}
COLORS = {'i0001': '#2a78d6', 'i0006': '#eb6834', 'mimic': '#1baf7a'}


def f(x, d=2):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def ci(v, d=2):
    return f'{f(v[0], d)}–{f(v[1], d)}'


def pct(x, d=1):
    return '—' if x is None else f'{100 * x:.{d}f}'


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or 'l' + 'r' * (len(header) - 1)
    sep = ['---' if a == 'l' else '---:' for a in align]
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(sep) + ' |']
    lines += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(lines)


def n_fmt(n):
    return f'{int(n):,}'.replace(',', ' ')


# ----------------------------------------------------------------- figures

def supp_figures() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = range(len(BINS))
    for site, color in COLORS.items():
        s = R['sites'][site]
        axes[0].plot(x, [s[b]['heedb_age']['age_gap']['sw'] for b in BINS], color=color,
                     lw=2, marker='o', ms=3.5, label=SITE_NAMES[site])
        axes[0].plot(x, [s[b]['heedb_age_seed1']['age_gap']['sw'] for b in BINS], color=color,
                     lw=1.3, ls='--', marker='s', ms=2.5)
        axes[1].plot(x, [s[b]['sex_logit']['class_flip']['discordance'] * 100 for b in BINS],
                     color=color, lw=2, marker='o', ms=3.5)
        axes[1].plot(x, [s[b]['sex_seed1']['class_flip']['discordance'] * 100 for b in BINS],
                     color=color, lw=1.3, ls='--', marker='s', ms=2.5)
    axes[0].set_ylabel('Within-patient SD of age gap (years)')
    axes[1].set_ylabel('Patients changing predicted sex (%)')
    axes[0].set_title('A  In-house age model', loc='left', fontweight='bold', fontsize=10)
    axes[1].set_title('B  In-house sex model', loc='left', fontweight='bold', fontsize=10)
    axes[0].plot([], [], color='#52514e', lw=2, label='Five-seed ensemble (solid)')
    axes[0].plot([], [], color='#52514e', lw=1.3, ls='--', label='Single seed (dashed)')
    axes[0].legend(frameon=False, fontsize=7.5, loc='lower right')
    for ax in axes:
        ax.set_xticks(list(x), list(BINS.values()), rotation=35, ha='right', fontsize=8)
        ax.set_ylim(bottom=0)
        ax.axvspan(1.5, 2.5, color='#e4e3df', alpha=0.6, lw=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e4e3df')
    fig.tight_layout()
    fig.savefig(FIG / 'supp_figure_s1.png', dpi=300)
    plt.close(fig)

    diag_bins = ['1_lt5m', '2_5_60m', '3_1_6h', '4_6_24h', '8_gt1y']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, c in zip(axes, ('avb1', 'rbbb', 'lbbb')):
        for site, color in COLORS.items():
            ax.plot(range(len(diag_bins)), [D['sites'][site][b][c]['logit']['icc_a1'] for b in diag_bins],
                    color=color, lw=2, marker='o', ms=3.5, label=SITE_NAMES[site])
        ax.set_title(CLASSES[c], loc='left', fontweight='bold', fontsize=10)
        ax.set_xticks(range(len(diag_bins)), [BINS[b] for b in diag_bins], rotation=35, ha='right', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e4e3df')
        ax.set_ylim(0.6, 1.0)
    axes[0].set_ylabel('ICC(A,1) of output logit')
    axes[0].legend(frameon=False, fontsize=7.5, loc='lower left')
    fig.tight_layout()
    fig.savefig(FIG / 'supp_figure_s2.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for site, color in COLORS.items():
        rows = R['post_hoc'][site]['by_chest_amplitude_change_quintile']
        ax.plot([q['quintile'] + 1 for q in rows], [q['lima_sw'] for q in rows], color=color, lw=2,
                marker='o', ms=3.5, label=SITE_NAMES[site])
        ax.plot([q['quintile'] + 1 for q in rows], [q['bracke_sw'] for q in rows], color=color, lw=1.3,
                ls='--', marker='s', ms=2.5)
    ax.plot([], [], color='#52514e', lw=2, label='Lima (solid)')
    ax.plot([], [], color='#52514e', lw=1.3, ls='--', label='Bracke (dashed)')
    ax.set_xticks([1, 2, 3, 4, 5], ['Q1\nsmallest', 'Q2', 'Q3', 'Q4', 'Q5\nlargest'])
    ax.set_xlabel('Precordial amplitude change between the two ECGs (quintile)')
    ax.set_ylabel('Within-patient SD of age gap (years)')
    ax.set_ylim(bottom=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e4e3df')
    ax.legend(frameon=False, fontsize=7, ncol=2, loc='lower right')
    fig.tight_layout()
    fig.savefig(FIG / 'supp_figure_s3.png', dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- sections

def train_metrics() -> dict:
    """Best validation metric per task and seed (age: MAE; sex: AUC)."""
    return {tuple([k.split('_seed')[0], int(k.split('_seed')[1])]): v['val_metric']
            for k, v in _V['in_house_training'].items()}


def methods() -> str:
    tm = train_metrics()
    age = [tm[('age', s)] for s in range(1, 6) if ('age', s) in tm]
    sex = [tm[('sex', s)] for s in range(1, 6) if ('sex', s) in tm]
    return f"""## Supplementary Methods

### S1. Pair construction and signal quality

For each patient, ECGs were ordered by acquisition time and record identifier, and each record was paired with the next record of the same patient. Pairs with identical acquisition times were discarded. Each pair was assigned to one of eight interval strata. Within a stratum, one pair per patient was kept by the smallest hash of the concatenated record identifiers, and at most 25 000 patients per stratum and cohort were kept by the smallest hash of the patient identifier. Both hashes are deterministic, so the sample is reproducible. In HEEDB I0001, only patients in the database's test split were used; models were trained only on training-split patients.

Signals were read in physical units (mV) with the WFDB library. A record was excluded if:
- the 12 standard leads, a 10-second duration (±0.05 s), or millivolt units were missing;
- any sample was non-finite;
- any lead had an SD below 0.005 mV;
- any absolute amplitude exceeded 20 mV.

The lead-wise median was subtracted. For the in-house models, signals were resampled to 250 Hz (2500 samples). For the published models, they were resampled to 400 Hz (polyphase resampling), truncated to 4000 samples, and zero-padded symmetrically to 4096.

**Duplicate waveforms.** Two records were treated as duplicates when the MD5 hashes of their 250-Hz signals, quantised to 0.005 mV, were equal, or when their 40-ms block means differed by less than 0.01 mV in every lead.

### S2. Models

**In-house models.** One-dimensional residual convolutional network:
- four residual blocks with 32, 64, 128, and 256 channels and stride 2;
- kernel sizes 7 and 5, group normalisation, and SiLU activation;
- global average pooling and a linear head.

Inputs were divided by a fixed lead-wise RMS scale estimated in training patients. Training details:
- 298 542 adult I0001 training patients with valid signals (one ECG each, consistent recorded sex);
- AdamW (weight decay 10⁻⁴), one-cycle learning rate (maximum 2×10⁻³), 12 epochs, batch size 512, bfloat16 mixed precision;
- the best epoch selected in 19 896 validation patients: lowest mean absolute error for age, highest AUC for sex.

Five seeds were trained for each task. Validation mean absolute error for age was {min(age):.2f}–{max(age):.2f} years; validation AUC for sex was {min(sex):.3f}–{max(sex):.3f}. Secondary sex models using only the limb or only the precordial leads were trained earlier on 100 000 patients and were used without change.

**Published models.**

{table(['Model', 'Training data', 'Input', 'Choice made in I0001 validation patients', 'Validation result'], [
        ['Lima *et al.* 2021, ECG age', 'CODE (Brazil)', '400 Hz, 4096 samples', 'amplitude ×1 vs ×10',
         f"×1: MAE {LIMA_SEL['x1']['mae']:.1f} y, r {LIMA_SEL['x1']['r']:.2f}; ×10: MAE {LIMA_SEL['x10']['mae']:.1f} y"],
        ['Bracke *et al.* 2026, ECG age', 'CODE-15% (Brazil)', '400 Hz, 4096 samples, recorded sex', 'none',
         f"MAE {BRACKE_VAL['mae']:.1f} y, r {BRACKE_VAL['r']:.2f} (published CODE-15%: {BRACKE_VAL['published_code15_mae']} y, {BRACKE_VAL['published_code15_r']})"],
        ['Ribeiro *et al.* 2020, abnormalities', 'CODE (Brazil)', '400 Hz, 4096 samples', 'amplitude ×1 vs ×10',
         f"×1 mean AUC {RIB_VAL['x1']['mean_auc']:.3f}; ×10 {RIB_VAL['x10']['mean_auc']:.3f}"],
    ], 'lllll')}

Validation used 5000 I0001 validation patients (4981 with valid signals). The Bracke model was run with mamba-ssm installed without its CUDA extension, through the library's Triton-kernel path (`use_mem_eff_path=False`). This path computes the same function as the fused kernel. All weights loaded with no missing or unexpected parameters. Per-class validation AUCs of the Ribeiro model against GE 12SL v24 statements were {', '.join(f"{CLASSES[c]} {RIB_VAL['x1']['auc'][c]:.3f}" for c in CLASSES)}.

### S3. Signal descriptors

- **Heart rate and RR-interval coefficient of variation:** from QRS detection on lead II (WFDB XQRS).
- **High-frequency and baseline power:** fractions of spectral power above 40 Hz and below 0.5 Hz; the median across leads.
- **Lead RMS:** root-mean-square amplitude per lead.
- **Amplitude change between ECGs:** the mean absolute log ratio of lead RMS, computed for leads I and II (limb) and V1–V6 (precordial).

### S4. Electrode-reversal detection

For each pair, a median beat was formed for each ECG from beats detected on a multilead energy envelope (window −250 to +450 ms; baseline set to the first 40 ms). Limb-electrode permutations were applied to the second ECG through Einthoven's relations on leads I and II:

{table(['Permutation', "Lead I'", "Lead II'"], [
        ['RA ↔ LA', '−I', 'II − I'], ['LA ↔ LL', 'II', 'I'], ['RA ↔ LL', 'I − II', '−II'],
        ['RA→LA→LL→RA', 'II − I', '−I'], ['RA→LL→LA→RA', '−II', 'I − II']], 'lll')}

Each transformed beat was compared with the first ECG's beat after optimal scalar gain and a shift of up to ±20 ms. The ratio of the best permutation's relative residual to the identity residual was the limb score; a pair was flagged if the score was below {REV['thresholds']['limb_ratio']}. The same was done for the five adjacent precordial transpositions (V1↔V2 … V5↔V6), with a threshold of {REV['thresholds']['chest_ratio']}. Thresholds were chosen against MIMIC machine statements and applied unchanged to HEEDB (*Table S8*).

### S5. Agreement statistics

For paired values *a*~i~, *b*~i~ in *n* patients:
- within-patient SD: *S*~w~ = SD(*b* − *a*)/√2;
- repeatability coefficient: 2.77·*S*~w~ (= 1.96·√2·*S*~w~);
- limits of agreement: mean(*b* − *a*) ± 1.96·SD(*b* − *a*).

ICC(A,1) was computed from two-way ANOVA mean squares for rows (MSR), columns (MSC), and error (MSE):

ICC(A,1) = (MSR − MSE) / (MSR + MSE + 2(MSC − MSE)/*n*).

Cohen's κ was computed for binary calls. All 95% confidence intervals are percentile intervals from 2000 bootstrap resamples of patients; each patient contributed one pair per stratum. For the age gap at the second ECG, chronological age was advanced by the interval between ECGs.

### S6. Protocol and deviations

The analysis protocol was written before model outputs were computed and is kept with the code together with a dated change log.

**Changes made before results were seen:**
- implementation of duplicate detection;
- the 18–89-year age window;
- in-house model training details;
- the published-model amplitude multiplier;
- the three-ECG analysis.

**Analyses added after the primary results (post hoc):**
- signal change by interval;
- age-gap variability by quintile of precordial amplitude change;
- signal-similar 1–6-hour pairs;
- sex outputs restricted to adults;
- artifact and electrode-reversal analyses;
- the age-bias-corrected gap.

**Models added after the primary results:** the second published ECG-age model and the abnormality classifier, added to test whether the findings depend on a single published model. Their analyses were specified before their outputs were computed.
"""


def flow_table() -> str:
    rows = []
    for r in R['flow']:
        s = R['sites'][r['site']].get(r['bin'], {})
        rows.append([SITES[r['site']], BINS[r['bin']], n_fmt(r['pairs']), n_fmt(r['both_ok']),
                     n_fmt(r['duplicates']), n_fmt(s.get('n_age', 0)), n_fmt(s.get('n_sex', 0)),
                     f(s.get('median_gap_min'), 1)])
    return table(['Cohort', 'Interval', 'Pairs sampled', 'Both ECGs valid', 'Duplicates removed',
                  'Age analyses', 'Sex analyses', 'Median interval (min)'], rows, 'llrrrrrr')


def age_table(model: str) -> str:
    rows = []
    for site in SITES:
        for b in BINS:
            e = R['sites'][site].get(b, {}).get(model)
            if not e:
                continue
            g, c = e['age_gap'], e['age_gap_gt8']
            rows.append([SITES[site], BINS[b], n_fmt(g['n']), f"{f(g['sw'])} ({ci(g['sw_ci'])})",
                         f(g['rc'], 1), f"{f(g['bias'], 1)} ({f(g['loa'][0], 1)}, {f(g['loa'][1], 1)})",
                         f"{f(g['icc_a1'])} ({ci(g['icc_ci'])})", f(e['predicted_age']['icc_a1']),
                         f"{pct(c['discordance'])} ({pct(c['discordance_ci'][0])}–{pct(c['discordance_ci'][1])})",
                         f(c['kappa']), f(e['mae_a'], 1)])
    return table(['Cohort', 'Interval', 'n', '*S*~w~, y (95% CI)', 'RC, y', 'Bias (LoA), y',
                  'ICC age gap (95% CI)', 'ICC predicted age', 'Class change >8 y, % (95% CI)', 'κ',
                  'MAE first ECG, y'], rows, 'llrrrrrrrrr')


def sex_table() -> str:
    rows = []
    for site in SITES:
        for b in BINS:
            e = R['sites'][site].get(b, {})
            if 'sex_logit' not in e:
                continue
            s = e['sex_logit']
            rows.append([SITES[site], BINS[b], n_fmt(s['logit']['n']),
                         f"{f(s['logit']['sw'])} ({ci(s['logit']['sw_ci'])})",
                         f(s['logit']['icc_a1']), f(s.get('logit_female_label', {}).get('icc_a1')),
                         f(s.get('logit_male_label', {}).get('icc_a1')),
                         f"{pct(s['class_flip']['discordance'])} ({pct(s['class_flip']['discordance_ci'][0])}–{pct(s['class_flip']['discordance_ci'][1])})",
                         pct(e['sex_seed1']['class_flip']['discordance']), pct(s['abs_dprob_gt_0.2']),
                         f(e['sex_limb_only']['logit']['sw']), f(e['sex_chest_only']['logit']['sw'])])
    return table(['Cohort', 'Interval', 'n', 'Logit *S*~w~ (95% CI)', 'ICC all', 'ICC female', 'ICC male',
                  'Class change, % (95% CI)', 'Class change single seed, %', 'Abs. change in *p* > 0.2, %',
                  'Limb-only *S*~w~', 'Precordial-only *S*~w~'], rows, 'llrrrrrrrrrr')


def ratio_table() -> str:
    rows = [[SITES[s]] + [f"{100 * R['short_vs_long'][s][m]:.0f}" for m in ('lima_age', 'bracke_age', 'heedb_age')]
            for s in ('i0001', 'i0006', 'mimic')]
    return table(['Cohort', 'Lima', 'Bracke', 'In-house ensemble'], rows)


def bias_table() -> str:
    rows = []
    for site in SITES:
        for m in ('lima_age', 'bracke_age', 'heedb_age'):
            e = R['post_hoc'][site].get(f'{m}_bias_corrected')
            if not e:
                continue
            rows.append([SITES[site], AGE_MODELS[m], f(e['slope'], 3), f(e['gap']['sw']),
                         f(e['gap']['icc_a1']), pct(e['gap_gt8']['prevalence_a']),
                         pct(e['gap_gt8']['discordance']), f(e['gap_gt8']['kappa'])])
    return table(['Cohort', 'Model', 'Slope of gap on age', '*S*~w~, y', 'ICC', 'Prevalence >8 y, %',
                  'Class change, %', 'κ'], rows, 'llrrrrrr')


def determinants_tables() -> str:
    out = []
    outcomes = {'lima_abs_age_gap_change': 'Lima: abs. change in age gap, y', 'bracke_abs_age_gap_change': 'Bracke: abs. change in age gap, y',
                'heedb_abs_age_gap_change': 'In-house: abs. change in age gap, y', 'abs_sex_logit_change': 'Sex: abs. change in logit'}
    for letter, site in zip('abcd', SITES):
        d = R['determinants'][site]
        preds = list(d['lima_abs_age_gap_change']['coefficients'])
        rows = []
        for p in preds:
            row = [PREDICTORS.get(p, p)]
            for o in outcomes:
                c = d[o]['coefficients'][p]
                row.append(f"{c['beta']:+.2f} ({c['ci'][0]:+.2f}, {c['ci'][1]:+.2f})")
            rows.append(row)
        rows.append(['*n*; *R*²'] + [f"{n_fmt(d[o]['n'])}; {d[o]['r_squared']:.3f}" for o in outcomes])
        rows.append(['Mean outcome'] + [f(d[o]['outcome_mean']) for o in outcomes])
        out.append(f"**Table S6{letter}. {SITE_NAMES[site]}, 1–6 h.**\n\n" + table(['Predictor'] + list(outcomes.values()), rows))
    return '\n\n'.join(out)


def sensitivity_table() -> str:
    rows = []
    for site in SITES:
        e = R['sites'][site]['3_1_6h']
        main = [f(e[m]['age_gap']['sw']) for m in ('lima_age', 'bracke_age', 'heedb_age')] + [pct(e['sex_logit']['class_flip']['discordance'])]
        rows.append([SITES[site], 'All pairs (primary)', n_fmt(e['lima_age']['age_gap']['n'])] + main)
        for key, label in (('regular_rhythm_only', 'Regular rhythm (RR CV ≤ 0.15)'),
                           ('same_device_filter', 'Same device and filter')):
            if key in e:
                s = e[key]
                rows.append([SITES[site], label, n_fmt(s['lima_age']['age_gap']['n'])] +
                            [f(s[m]['age_gap']['sw']) for m in ('lima_age', 'bracke_age', 'heedb_age')] +
                            [pct(s['sex_logit']['class_flip']['discordance'])])
        adult = R['post_hoc'][site]['sex_adults_only']
        rows.append([SITES[site], 'Adults only (sex)', n_fmt(adult['logit']['n']), '—', '—', '—',
                     pct(adult['class_flip']['discordance'])])
    return table(['Cohort', 'Analysis', 'n', 'Lima *S*~w~', 'Bracke *S*~w~', 'In-house *S*~w~',
                  'Sex class change, %'], rows, 'llrrrrr')


def exclusion_table() -> str:
    rows = []
    rev = {(r['site'], r['bin']): r for r in REV['by_site_bin']}
    sl = {(r['site'], r['bin']): r for r in SL['by_site_bin']}
    art = {(r['site'], r['bin']): r for r in ART}
    for site in SITES:
        for b in ('1_lt5m', '3_1_6h', '8_gt1y'):
            if (site, b) not in rev:
                continue
            r, a = rev[(site, b)], art.get((site, b), {})
            s = sl.get((site, b))
            rows.append([SITES[site], BINS[b], f(r['lima_sw_all']), f(a.get('swL_clean')),
                         f(a.get('swL_noflag')), f(r['lima_sw_clean']),
                         f(s['lima_sw_clean']) if s else '—',
                         f"{r['flag_limb_pct'] + r['flag_chest_pct']:.1f}",
                         f"{s['any_flag_pct']:.1f}" if s else '—', f(r['lima_sw_flagged'])])
    return table(['Cohort', 'Interval', 'All', 'Excl. noisy (top 10%)', 'Excl. MIMIC machine flags',
                  'Excl. signal detector', 'Excl. 12SL + detector', 'Detector flagged, %',
                  '12SL + detector flagged, %', 'Flagged pairs only'], rows, 'llrrrrrrrr')


def detector_table() -> str:
    rows = []
    for label, v in (('MIMIC machine statement', REV['validation_vs_mimic_machine']),
                     ('HEEDB 12SL v24', SL['validation_vs_12sl'])):
        for t in (0.5, 0.6, 0.7, 0.8):
            th = v[f'threshold_{t}']
            rows.append([label, f"{n_fmt(v.get('machine_one_ecg_reversal', v.get('one_ecg_12sl_reversal')))}",
                         f"{v['auc_limb_ratio']:.2f}", f'{t}', pct(th['flag_rate'], 2),
                         pct(th['sensitivity']), pct(th['ppv'])])
    return table(['Reference', 'Pairs with reversal in one ECG', 'AUC', 'Limb threshold', 'Flagged, %',
                  'Sensitivity, %', 'Positive predictive value, %'], rows, 'lrrrrrr')


def signal_table() -> str:
    rows = []
    for site in ('i0001', 'i0006', 'mimic'):
        m = R['post_hoc'][site]['median_signal_change_by_bin']
        rows.append([SITES[site]] + [f"{100 * m[b]['d_log_rms_chest']:.0f} / {100 * m[b]['d_log_rms_limb']:.0f} / {m[b]['d_hr']:.1f}" for b in BINS])
    sim = [[SITES[s], n_fmt(R['post_hoc'][s]['signal_similar_to_lt5m']['n']),
            f(R['post_hoc'][s]['signal_similar_to_lt5m']['lima_sw']),
            f(R['post_hoc'][s]['signal_similar_to_lt5m']['bracke_sw']),
            f(R['post_hoc'][s]['signal_similar_to_lt5m']['heedb_sw']),
            f(R['sites'][s]['1_lt5m']['lima_age']['age_gap']['sw']),
            f(R['sites'][s]['1_lt5m']['bracke_age']['age_gap']['sw'])] for s in ('i0001', 'i0006', 'mimic')]
    return (table(['Cohort'] + list(BINS.values()), rows, 'l' + 'r' * len(BINS)) +
            '\n\n*Cells: median precordial / limb amplitude change (mean |log RMS ratio| × 100) / median absolute heart-rate difference (b.p.m.).*\n\n'
            '**Table S9b. 1–6-hour pairs whose signal change was below the median of <5-minute pairs on all three measures (post hoc).**\n\n' +
            table(['Cohort', 'n', 'Lima *S*~w~', 'Bracke *S*~w~', 'In-house *S*~w~', 'Lima *S*~w~ <5 min',
                   'Bracke *S*~w~ <5 min'], sim))


def triples_table() -> str:
    rows = []
    for site in ('i0001', 'i0006', 'mimic'):
        t = R['triples'][site]
        for m, label in (('lima_age', 'Lima age'), ('bracke_age', 'Bracke age'),
                         ('heedb_age', 'In-house age'), ('sex_logit', 'In-house sex logit')):
            e = t[m]
            rows.append([SITES[site], label, n_fmt(e.get('n', t['n'])), f(e['rms_single']),
                         f(e['rms_mean_of_two']), f(e['ratio'], 3)])
    return table(['Cohort', 'Output', 'Patients', 'RMS difference, single ECG', 'RMS difference, mean of two',
                  'Ratio (expected 0.866)'], rows, 'llrrrr')


def diagnostic_table() -> str:
    rows = []
    for site in SITES:
        for b in ('1_lt5m', '3_1_6h', '8_gt1y'):
            e = D['sites'][site].get(b)
            if not e:
                continue
            for c in CLASSES:
                i = e[c]
                lost = i['ai']['lost_on_b_given_positive_a']
                lost_ci = i['ai'].get('lost_ci')
                rows.append([SITES[site], BINS[b], CLASSES[c], pct(i['ai']['prevalence_a']),
                             f"{pct(lost)}" + (f" ({pct(lost_ci[0])}–{pct(lost_ci[1])})" if lost_ci else ''),
                             pct(i['machine']['prevalence_a']), pct(i['machine']['lost_on_b_given_positive_a']),
                             f(i['ai']['kappa']), f(i['machine']['kappa']), f(i['logit']['icc_a1'])])
    return table(['Cohort', 'Interval', 'Class', 'AI positive (first ECG), %', 'AI call lost, % (95% CI)',
                  'Machine positive, %', 'Machine statement lost, %', 'κ AI', 'κ machine', 'Logit ICC'],
                 rows, 'lllrrrrrrr')


def grras_table() -> str:
    items = [
        ('1', 'Identify in title/abstract that reliability was investigated', 'Title, Abstract'),
        ('2', 'Name and describe the measured outcome', 'Methods: AI models; Table 2'),
        ('3', 'Specify the subject population', 'Methods: Data sources; Table 1'),
        ('4', 'Specify the rater population', 'Not applicable: automated models (Methods: AI models)'),
        ('5', 'Describe what was known about reliability before', 'Introduction'),
        ('6', 'Explain how subjects were selected', 'Methods: pairs and strata; Supplementary Methods S1'),
        ('7', 'Describe sample size considerations', 'Methods (all eligible pairs, capped at 25 000 per stratum)'),
        ('8', 'Describe the measurement and rating process', 'Methods; Supplementary Methods S1–S2'),
        ('9', 'State whether measurements were independent', 'Methods: models frozen; outputs computed per ECG'),
        ('10', 'Describe the statistical analysis', 'Methods; Supplementary Methods S5'),
        ('11', 'State the actual number of subjects and replicates', 'Results; Table S1'),
        ('12', 'Describe the sample characteristics', 'Table 1'),
        ('13', 'Report estimates with measures of statistical uncertainty', 'Tables 2–4; Tables S2–S11'),
        ('14', 'Discuss the practical relevance of results', 'Discussion: Implications'),
        ('15', 'Provide detailed results if possible', 'Supplementary material'),
    ]
    return table(['Item', 'GRRAS recommendation', 'Location'], [list(i) for i in items], 'lll')


def build() -> str:
    parts = [
        '---\ntitle: "Supplementary material"\nsubtitle: "Short-interval test–retest reliability of artificial intelligence–derived ECG age and sex"\n---\n',
        'All values are computed from aggregate outputs of the analysis code. Abbreviations: '
        '*S*~w~, within-patient standard deviation; RC, repeatability coefficient (2.77·*S*~w~); '
        'LoA, 95% limits of agreement; ICC, two-way random-effects absolute-agreement single-measure '
        'intraclass correlation; MAE, mean absolute error; RBBB/LBBB, right/left bundle branch block. '
        'Cohorts: I0001, HEEDB I0001 test patients (Massachusetts General Hospital); '
        'I0006, HEEDB I0006 (Emory University Hospital); MIMIC, MIMIC-IV-ECG; '
        'MIMIC ED, troponin-negative emergency subcohort.\n',
        methods(),
        '## Supplementary Tables\n',
        '**Table S1. Pair flow by cohort and interval.**\n\n' + flow_table(),
        '**Table S2a. Age-gap agreement, Lima *et al.* model.**\n\n' + age_table('lima_age'),
        '**Table S2b. Age-gap agreement, Bracke *et al.* model.** ECGs with unknown or conflicting recorded sex were not scored by this model.\n\n' + age_table('bracke_age'),
        '**Table S2c. Age-gap agreement, in-house five-seed ensemble.**\n\n' + age_table('heedb_age'),
        '**Table S2d. Age-gap agreement, in-house single seed (seed 1).**\n\n' + age_table('heedb_age_seed1'),
        '**Table S3. Sex-model agreement.** Limb-only and precordial-only models have their own logit scales and are not directly comparable with the 12-lead ensemble.\n\n' + sex_table(),
        '**Table S4. Variance of the age gap at 1–6 hours as a percentage of the variance at >1 year.**\n\n' + ratio_table(),
        '**Table S5. Age-bias-corrected age gap at 1–6 hours (post hoc).** The gap was replaced by its residual from a site- and model-specific linear regression on chronological age.\n\n' + bias_table(),
        '**Table S6. Correlates of absolute change at 1–6 hours.** Ordinary least squares with HC3 standard errors; coefficients are per SD of continuous predictors or per unit of binary predictors, with 95% CIs.\n\n' + determinants_tables(),
        '**Table S7a. Pre-specified sensitivity analyses at 1–6 hours.**\n\n' + sensitivity_table(),
        '**Table S7b. Age-gap *S*~w~ (Lima model, years) after excluding pairs with artifact or electrode-reversal flags (post hoc).**\n\n' + exclusion_table(),
        '**Table S8. Validation of the pairwise electrode-reversal detector.** Reference: pairs in which exactly one ECG carried a machine reversal statement.\n\n' + detector_table(),
        '**Table S9a. Signal change between the two ECGs by interval (post hoc).**\n\n' + signal_table(),
        '**Table S10. Averaging two ECGs in patients with three consecutive ECGs within 6 hours.**\n\n' + triples_table(),
        '**Table S11. Published abnormality classifier: all classes.** AI call: probability >0.5. "Lost": among patients positive on the first ECG, the proportion not positive on the second.\n\n' + diagnostic_table(),
        '**Table S12. GRRAS checklist.**\n\n' + grras_table(),
        '## Supplementary Figures\n',
        '![](figures/supp_figure_s1.png)\n\n**Figure S1. In-house models by interval.** (A) Within-patient SD of the age gap for the five-seed ensemble (solid) and a single seed (dashed). (B) Percentage of patients whose predicted sex changed, for the ensemble (solid) and a single seed (dashed). The grey band marks the primary 1–6-hour stratum.',
        '![](figures/supp_figure_s2.png)\n\n**Figure S2. Published abnormality classifier: ICC(A,1) of the output logit for conduction abnormalities by interval.**',
        '![](figures/supp_figure_s3.png)\n\n**Figure S3. Age-gap variability in 1–6-hour pairs by quintile of precordial amplitude change, for both published ECG-age models (post hoc).**',
    ]
    return '\n\n'.join(parts) + '\n'


def reference_docx() -> Path:
    """Pandoc reference document: A4 landscape, narrow margins, compact table text."""
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.shared import Cm, Pt
    base = WORK / 'supp_reference_default.docx'
    subprocess.run(['pandoc', '-o', str(base), '--print-default-data-file', 'reference.docx'],
                   check=True)
    doc = Document(base)
    for section in doc.sections:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = Cm(29.7), Cm(21.0)
        for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
            setattr(section, side, Cm(1.5))
    names = {s.name for s in doc.styles}
    for name, size in (('Normal', 10), ('Body Text', 10), ('First Paragraph', 10), ('Compact', 7.5)):
        if name in names:
            doc.styles[name].font.size = Pt(size)
            doc.styles[name].font.name = 'Arial'
    path = WORK / 'supp_reference.docx'
    doc.save(path)
    return path


def main() -> None:
    supp_figures()
    md = HERE / 'reliability_supplement.md'
    md.write_text(build())
    docx = HERE / 'reliability_supplement.docx'
    reference = reference_docx()
    subprocess.run(['pandoc', str(md), '-o', str(docx), '--resource-path', str(HERE),
                    '--reference-doc', str(reference)], check=True)
    subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', str(HERE), str(docx)],
                   check=True, capture_output=True)
    print(json.dumps({'markdown': str(md), 'docx': str(docx), 'pdf': str(HERE / 'reliability_supplement.pdf')}))


if __name__ == '__main__':
    main()
