"""Graphical abstract for EHJ - Digital Health (12.5 cm high x 18.0 cm wide, 600 dpi).

Uses aggregate results only.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).parent
RESULTS = json.loads((HERE / 'reliability_results_2026-10-05.json').read_text())
DIAG = json.loads((HERE / 'reliability_diagnostic_results_2026-10-06.json').read_text())
BINS = ['1_lt5m', '2_5_60m', '3_1_6h', '4_6_24h', '5_1_7d', '6_7_30d', '7_30d_1y', '8_gt1y']
LABELS = ['<5 min', '5–60 min', '1–6 h', '6–24 h', '1–7 d', '7–30 d', '30 d–1 y', '>1 y']
SITES = ('i0001', 'i0006', 'mimic')
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
PANEL = '#f3f2ee'
BLUE, ORANGE = '#2a78d6', '#eb6834'
CM = 1 / 2.54

rcParams['font.family'] = 'Liberation Sans'
rcParams['font.size'] = 9


def site_range(model: str, metric: str = 'age_gap') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.array([[RESULTS['sites'][s][b][model][metric]['sw'] for b in BINS] for s in SITES])
    return values.mean(axis=0), values.min(axis=0), values.max(axis=0)


def box(fig, x, y, w, h, color=PANEL):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.012',
                                      transform=fig.transFigure, facecolor=color,
                                      edgecolor='none', zorder=0))


def heading(fig, x, y, text):
    fig.text(x, y, text, fontsize=10.5, fontweight='bold', color=INK, va='top')


def main() -> None:
    fig = plt.figure(figsize=(18.0 * CM, 12.5 * CM), facecolor=SURFACE)

    # Title band
    fig.text(0.02, 0.965, 'How reproducible are AI-ECG outputs between ECGs recorded hours apart?',
             fontsize=12.5, fontweight='bold', color=INK, va='top')
    fig.text(0.02, 0.905, 'Age and recorded sex cannot change within hours, so any '
             'difference between two ECGs is variability',
             fontsize=9, color=MUTED, va='top')

    # Left: design
    box(fig, 0.02, 0.17, 0.27, 0.68)
    heading(fig, 0.035, 0.83, 'Design')
    design = [
        ('3 cohorts', 'HEEDB (MGH, Emory),\nMIMIC-IV-ECG'),
        ('532 891 pairs', 'consecutive ECGs,\n253 995 patients'),
        ('8 intervals', '<5 min to >1 year'),
        ('Models', '2 published ECG-age\n1 published diagnostic\nin-house age & sex'),
    ]
    y = 0.765
    for big, small in design:
        fig.text(0.035, y, big, fontsize=10, fontweight='bold', color=BLUE, va='top')
        fig.text(0.035, y - 0.045, small, fontsize=8.5, color=INK, va='top', linespacing=1.15)
        y -= 0.142

    # Middle: curve
    box(fig, 0.31, 0.17, 0.38, 0.68)
    heading(fig, 0.325, 0.83, 'ECG-age gap: within-patient SD')
    fig.text(0.325, 0.785, 'Line: mean of 3 cohorts; band: range',
             fontsize=8, color=MUTED, va='top')
    ax = fig.add_axes([0.375, 0.29, 0.30, 0.47])
    ax.set_zorder(2)
    x = np.arange(len(BINS))
    for model, color, label in (('lima_age', BLUE, 'Lima et al.'), ('bracke_age', ORANGE, 'Bracke et al.')):
        mean, lo, hi = site_range(model)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2, marker='o', markersize=3.5, label=label)
    ax.axvspan(1.5, 2.5, color=GRID, alpha=0.8, linewidth=0, zorder=0)
    ax.set_xticks(x[::1], LABELS, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 8.5)
    ax.set_ylabel('years', fontsize=8.5, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.set_facecolor(PANEL)
    ax.grid(axis='y', color='#dcdbd6', linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ratio = [RESULTS['short_vs_long'][s][m] for s in SITES for m in ('lima_age', 'bracke_age')]
    ax.annotate(f'{int(round(100 * min(ratio)))}–{int(round(100 * max(ratio)))}% of >1-year\n'
                'variance already\npresent within hours',
                xy=(2, site_range('lima_age')[0][2]), xytext=(3.3, 2.0), fontsize=8, color=INK,
                arrowprops={'arrowstyle': '-|>', 'color': MUTED, 'lw': 0.8})

    # Right: key numbers at 1-6 h
    box(fig, 0.71, 0.17, 0.27, 0.68)
    heading(fig, 0.725, 0.83, 'Over 1–6 hours')
    lima = [RESULTS['sites'][s]['3_1_6h']['lima_age'] for s in SITES]
    bracke = [RESULTS['sites'][s]['3_1_6h']['bracke_age'] for s in SITES]
    rc = [m['age_gap']['rc'] for m in lima + bracke]
    disc = [100 * m['age_gap_gt8']['discordance'] for m in lima + bracke]
    flip = [100 * RESULTS['sites'][s]['3_1_6h']['sex_logit']['class_flip']['discordance'] for s in SITES]
    lost = [100 * DIAG['sites'][s]['3_1_6h'][c]['ai']['lost_on_b_given_positive_a']
            for s in SITES for c in ('rbbb', 'lbbb')]
    facts = [
        (f'±{min(rc):.0f}–{max(rc):.0f} y', 'smallest detectable change\nin ECG age (published models)'),
        (f'{min(disc):.0f}–{max(disc):.0f}%', 'change "ECG age >8 y older"\ncategory'),
        (f'{min(flip):.0f}–{max(flip):.0f}%', 'change predicted sex'),
        (f'{min(lost):.0f}–{max(lost):.0f}%', 'of bundle branch block calls\nlost (machine reads no better)'),
    ]
    y = 0.765
    for big, small in facts:
        fig.text(0.725, y, big, fontsize=13, fontweight='bold', color=ORANGE, va='top')
        fig.text(0.725, y - 0.055, small, fontsize=8.5, color=INK, va='top', linespacing=1.15)
        y -= 0.055 + 0.04 * (small.count('\n') + 1) + 0.035

    # Bottom take-home band
    box(fig, 0.02, 0.025, 0.96, 0.12, color='#e6eef9')
    fig.text(0.035, 0.085, 'Take-home', fontsize=10, fontweight='bold', color=BLUE, va='center')
    fig.text(0.165, 0.085,
             'A single AI-ECG estimate is not a precise individual measurement. Report repeatability '
             'with accuracy,\naverage repeated ECGs, and use model ensembles before interpreting '
             'serial change in ECG age.',
             fontsize=9, color=INK, va='center', linespacing=1.2)

    out = HERE / 'figures'
    fig.savefig(out / 'graphical_abstract.png', dpi=600, facecolor=SURFACE)
    fig.savefig(out / 'graphical_abstract.tiff', dpi=600, facecolor=SURFACE,
                pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(out / 'graphical_abstract.pdf', facecolor=SURFACE)


if __name__ == '__main__':
    main()
