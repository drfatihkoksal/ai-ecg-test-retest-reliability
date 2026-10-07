"""Figure for the AI-ECG test-retest reliability study (aggregate results only)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).parent
RESULTS = HERE / 'reliability_results_2026-10-05.json'
BINS = ['1_lt5m', '2_5_60m', '3_1_6h', '4_6_24h', '5_1_7d', '6_7_30d', '7_30d_1y', '8_gt1y']
BIN_LABELS = ['<5 min', '5–60 min', '1–6 h', '6–24 h', '1–7 d', '7–30 d', '30 d–1 y', '>1 y']
SITES = [('i0001', 'HEEDB I0001', '#2a78d6'), ('i0006', 'HEEDB I0006', '#eb6834'),
         ('mimic', 'MIMIC-IV-ECG', '#1baf7a')]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def style(ax, title):
    ax.set_title(title, loc='left', fontsize=10, color=INK, fontweight='semibold')
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def interval_panel(ax, r, getter, ylabel, title, ed_value=None):
    x = range(len(BINS))
    for key, label, color in SITES:
        y = [getter(r['sites'][key][b]) for b in BINS]
        ax.plot(x, y, color=color, linewidth=2, marker='o', markersize=4, label=label)
    if ed_value is not None:
        ax.plot([2], [ed_value], marker='D', markersize=6, color='#1baf7a',
                markerfacecolor='white', markeredgewidth=2, linestyle='none',
                label='MIMIC ED, troponin-negative')
    ax.axvspan(1.5, 2.5, color=GRID, alpha=0.5, linewidth=0)
    ax.set_xticks(list(x), BIN_LABELS, rotation=35, ha='right')
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_ylim(bottom=0)
    style(ax, title)


def main() -> None:
    r = json.loads(RESULTS.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    ed = r['sites']['mimic_ed']['3_1_6h']
    interval_panel(axes[0, 0], r, lambda e: e['lima_age']['age_gap']['sw'],
                   'Within-patient SD of ECG-age gap (years)',
                   'A  Published ECG-age model (Lima et al.)',
                   ed['lima_age']['age_gap']['sw'])
    interval_panel(axes[0, 1], r, lambda e: e['sex_logit']['logit']['sw'],
                   'Within-patient SD of sex logit',
                   'B  HEEDB-trained sex model (5-seed ensemble)',
                   ed['sex_logit']['logit']['sw'])
    ax = axes[0, 0]
    for key, label, color in SITES:
        ax.plot(range(len(BINS)), [r['sites'][key][b]['bracke_age']['age_gap']['sw'] for b in BINS],
                color=color, linewidth=1.5, linestyle='--', marker='s', markersize=3)
    ax.plot([2], [ed['bracke_age']['age_gap']['sw']], marker='s', markersize=5, color='#1baf7a',
            markerfacecolor='white', markeredgewidth=1.5, linestyle='none')
    ax.plot([], [], color=MUTED, linewidth=2, label='Lima et al. 2021 (solid)')
    ax.plot([], [], color=MUTED, linewidth=1.5, linestyle='--', label='Bracke et al. 2026 (dashed)')
    ax.set_title('A  Published ECG-age models', loc='left', fontsize=10, color=INK,
                 fontweight='semibold')
    ax.legend(frameon=False, fontsize=7.5, loc='lower right', ncol=2)

    ax = axes[1, 0]
    for key, label, color in SITES:
        rows = r['post_hoc'][key]['by_chest_amplitude_change_quintile']
        ax.plot([q['quintile'] + 1 for q in rows], [q['lima_sw'] for q in rows],
                color=color, linewidth=2, marker='o', markersize=4, label=label)
    ax.set_xticks([1, 2, 3, 4, 5], ['Q1\nsmallest', 'Q2', 'Q3', 'Q4', 'Q5\nlargest'])
    ax.set_xlabel('Precordial amplitude change between the two ECGs (quintile)',
                  fontsize=9, color=MUTED)
    ax.set_ylabel('Within-patient SD of ECG-age gap (years)', fontsize=9, color=MUTED)
    ax.set_ylim(bottom=0)
    style(ax, 'C  1–6 h pairs: spread rises with signal change')

    ax = axes[1, 1]
    x = range(len(BINS))
    for key, label, color in SITES:
        med = r['post_hoc'][key]['median_signal_change_by_bin']
        ax.plot(x, [100 * med[b]['d_log_rms_chest'] for b in BINS], color=color,
                linewidth=2, marker='o', markersize=4, label=label)
    ax.axvspan(1.5, 2.5, color=GRID, alpha=0.5, linewidth=0)
    ax.set_xticks(list(x), BIN_LABELS, rotation=35, ha='right')
    ax.set_ylabel('Median precordial amplitude change\n(mean |log RMS ratio| × 100)',
                  fontsize=9, color=MUTED)
    ax.set_ylim(bottom=0)
    style(ax, 'D  Signal change by interval between ECGs')

    for a in (axes[0, 0], axes[0, 1], axes[1, 1]):
        a.set_xlabel('Interval between consecutive ECGs', fontsize=9, color=MUTED)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(HERE / 'figures' / f'reliability_results.{ext}', dpi=200,
                    facecolor='#fcfcfb')


if __name__ == '__main__':
    main()
