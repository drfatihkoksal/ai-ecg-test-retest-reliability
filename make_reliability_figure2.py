"""Figure 2: Bland–Altman density for the published ECG-age model (1–6 h, I0001)
and the effect of averaging two ECGs (aggregate triples results).

Patient-level differences are binned into a hexagonal density; no individual
points are drawn.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reliability_analysis import (MINUTES_PER_YEAR, PRIMARY_BIN, WORK, attach,
                                  eligible_age, load_scores)


HERE = Path(__file__).parent
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SITES = [('i0001', 'HEEDB I0001', '#2a78d6'), ('i0006', 'HEEDB I0006', '#eb6834'),
         ('mimic', 'MIMIC-IV-ECG', '#1baf7a')]
SCORES = [('lima_age', 'Published\nECG-age'), ('heedb_age', 'In-house\nECG-age'),
          ('sex_logit', 'In-house\nsex logit')]


def style(ax, title):
    ax.set_title(title, loc='left', fontsize=10, color=INK, fontweight='semibold')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    scores, _ = load_scores(WORK)
    pairs = pd.read_parquet(WORK / 'rel_pairs.parquet')
    pairs = pairs[(pairs['site'] == 'i0001') & (pairs['bin'] == PRIMARY_BIN)]
    p = attach(pairs, scores, ('a', 'b'))
    p = p[p['ok_a'].fillna(False) & p['ok_b'].fillna(False) & (p['md5_a'] != p['md5_b'])]
    p = p[eligible_age(p)]
    gap_a = p['lima_age_a'] - p['age_a']
    gap_b = p['lima_age_b'] - (p['age_a'] + p['gap_min'] / MINUTES_PER_YEAR)
    mean, diff = (gap_a + gap_b) / 2, gap_b - gap_a
    bias, sd = diff.mean(), diff.std(ddof=1)

    results = json.loads((HERE / 'reliability_results_2026-10-05.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    hb = ax.hexbin(mean, diff, gridsize=60, mincnt=5, cmap='Blues', linewidths=0,
                   extent=(-40, 50, -45, 45))
    for y, style_ in ((bias, '-'), (bias - 1.96 * sd, '--'), (bias + 1.96 * sd, '--')):
        ax.axhline(y, color=INK, linewidth=1, linestyle=style_)
    ax.text(49, bias + 1.96 * sd + 1.5, f'+1.96 SD: {bias + 1.96 * sd:.1f}', ha='right',
            fontsize=8, color=INK)
    ax.text(49, bias - 1.96 * sd - 4, f'−1.96 SD: {bias - 1.96 * sd:.1f}', ha='right',
            fontsize=8, color=INK)
    ax.text(49, bias + 1.5, f'bias: {bias:.1f}', ha='right', fontsize=8, color=INK)
    ax.set_xlabel('Mean age gap of the two ECGs (years)', fontsize=9, color=MUTED)
    ax.set_ylabel('Difference in age gap, second − first (years)', fontsize=9, color=MUTED)
    cb = fig.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label('Patients per cell (cells with <5 hidden)', fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    style(ax, f'A  Published ECG-age model, HEEDB I0001, 1–6 h (n = {len(p):,})')

    ax = axes[1]
    width = 0.25
    for i, (site, label, color) in enumerate(SITES):
        t = results['triples'][site]
        ratios = [t[s]['ratio'] for s, _ in SCORES]
        ax.bar(np.arange(3) + (i - 1) * width, ratios, width=width * 0.92, color=color,
               label=f"{label} (n = {t['n']:,})")
    ax.axhline(np.sqrt(0.75), color=INK, linewidth=1, linestyle='--')
    ax.text(2.45, np.sqrt(0.75) + 0.01, 'expected for independent errors (0.87)',
            ha='right', fontsize=8, color=INK)
    ax.set_xticks(range(3), [s for _, s in SCORES])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('RMS difference from third ECG:\nmean of two ÷ single ECG', fontsize=9, color=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    style(ax, 'B  Averaging two ECGs (three ECGs within 6 h)')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(HERE / 'figures' / f'reliability_figure2.{ext}', dpi=300, facecolor='#fcfcfb')


if __name__ == '__main__':
    main()
