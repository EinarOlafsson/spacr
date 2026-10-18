"""Recruitment and invasion plots for the deck, in viridis, every bar with its SD and its wells."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

O = '/mnt/firecuda2/Claude/toxoplasma_projects/presentation/images/outputs/'
BG, FG, MUTED = '#0E1217', '#E8EDF2', '#9AA6B2'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG, 'text.color': FG,
                     'axes.labelcolor': FG, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': MUTED,
                     'font.size': 15, 'axes.titlesize': 17, 'axes.spines.top': False, 'axes.spines.right': False})
VIR = matplotlib.colormaps['viridis']
rng = np.random.default_rng(0)


def bars(ax, groups, colors, xs, labels):
    for x, vals, c in zip(xs, groups, colors):
        m, sd = np.mean(vals), np.std(vals, ddof=1)
        ax.bar(x, m, width=0.7, color=c, alpha=0.9, zorder=1)
        ax.errorbar(x, m, yerr=sd, color=FG, capsize=6, lw=1.8, zorder=3)
        ax.scatter(x + rng.uniform(-0.22, 0.22, len(vals)), vals, s=22, color=FG, edgecolor=BG, lw=0.5, zorder=4, alpha=0.85)
    ax.set_xticks(xs); ax.set_xticklabels(labels)


# ---- recruitment: real well-level values from the preprint (Fig. 3B source data)
r = pd.read_csv('/home/olafsson/Downloads/source/source_data/Figure_3/Panel_B/Figure_3B_well_level_values.csv')
cond_col = {'NC': VIR(0.85), 'PC': VIR(0.25)}
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={'width_ratios': [2.3, 1]})
plates = sorted(r['plate'].unique())
xs, groups, cols, labels = [], [], [], []
for i, p in enumerate(plates):
    for j, c in enumerate(['NC', 'PC']):
        xs.append(i * 2.6 + j * 0.95); groups.append(r[(r.plate == p) & (r.condition == c)]['recruitment_mean'].values)
        cols.append(cond_col[c]); labels.append(c)
bars(a, groups, cols, np.array(xs), labels)
for i, p in enumerate(plates):
    a.text(i * 2.6 + 0.475, -0.16, p.replace('_', ' '), ha='center', transform=a.get_xaxis_transform(), color=FG, fontsize=14)
a.set_ylim(0.95, 1.42); a.set_ylabel('GFP-TSG101 recruitment\n(vacuole / cytoplasm)')
a.set_title('Every plate, every well (mean ± SD)', loc='left', pad=14)
rep = r.groupby(['plate', 'condition'])['recruitment_mean'].mean().unstack()
bars(b, [rep['NC'].values, rep['PC'].values], [cond_col['NC'], cond_col['PC']], np.array([0, 1]), ['NC\n(SAG1 gRNA)', 'PC\n(GRA14 gRNA)'])
b.set_ylim(0.95, 1.42); b.set_title('Plate means (n = 4)', loc='left', pad=14)
b.text(0.5, 1.36, 'p = 8.4 × 10⁻⁵', ha='center', color=FG, fontsize=14)
b.plot([0, 0, 1, 1], [1.33, 1.345, 1.345, 1.33], color=FG, lw=1.2)
fig.tight_layout(); fig.savefig(O + 'recruitment_viridis.png', dpi=150); plt.close(fig)

# ---- invasion: spaCR's analyze_invasion on the verification set
w = pd.read_csv('/mnt/wd4tb/spacr_testdata/assays/verify/invasion/results/analyze_invasion/well_invasion.csv')
conds = ['HeLa_vehicle', 'HeLa_inhibitor']
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={'width_ratios': [1.1, 1], 'wspace': 0.55})
inv = [w[w.condition == c]['invasion_efficiency'].values for c in conds]
att = [1 - v for v in inv]
x = np.arange(len(conds))
ci, ca = VIR(0.3), VIR(0.85)
a.bar(x, [np.mean(v) for v in inv], 0.6, color=ci, label='invaded (inside)')
a.bar(x, [np.mean(v) for v in att], 0.6, bottom=[np.mean(v) for v in inv], color=ca, label='attached (outside)')
a.errorbar(x, [np.mean(v) for v in inv], yerr=[np.std(v, ddof=1) for v in inv], color=FG, capsize=6, lw=1.8, ls='none')
for xi, v in zip(x, inv):
    a.scatter(xi + rng.uniform(-0.18, 0.18, len(v)), v, s=26, color=FG, edgecolor=BG, lw=0.5, zorder=4)
a.set_xticks(x); a.set_xticklabels(['vehicle', 'inhibitor']); a.set_ylim(0, 1.0); a.set_ylabel('fraction of parasites')
a.set_title('Invaded vs attached', loc='left', pad=14)
a.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=13)
bars(b, inv, [ci, ci], x, ['vehicle', 'inhibitor'])
b.set_ylim(0, 1.0); b.set_ylabel('invasion efficiency'); b.set_title('Per well (mean ± SD)', loc='left', pad=14)
fig.tight_layout(); fig.savefig(O + 'invasion_viridis.png', dpi=150); plt.close(fig)
print('ok')

# ---- replication: SYNTHETIC data (maintainer's request, 2026-09-21). Control follows 1:2:4:8:16:>16 =
# 30:20:20:15:10:5 %; a second condition with slower replication shows what a comparison looks like.
# Each condition: 6 wells, Dirichlet noise around the target fractions, then mean +- SD per bin.
labels = ['1', '2', '4', '8', '16', '>16']
target = {'control': [0.30, 0.20, 0.20, 0.15, 0.10, 0.05], 'slower replication': [0.45, 0.25, 0.15, 0.08, 0.05, 0.02]}
wells = {}
for c, t in target.items():
    w = rng.dirichlet(np.array(t) * 120, size=6)
    wells[c] = np.clip(w - w.mean(axis=0) + np.array(t), 0, 1)
segc = [VIR(v) for v in (0.1, 0.3, 0.48, 0.65, 0.82, 0.97)]
fig, a = plt.subplots(figsize=(12, 6.4))
x = np.arange(len(wells))
bottom = np.zeros(len(wells))
for i, lab in enumerate(labels):
    m = np.array([w[:, i].mean() for w in wells.values()]); sd = np.array([w[:, i].std(ddof=1) for w in wells.values()])
    a.bar(x, m, 0.5, bottom=bottom, color=segc[i], edgecolor=BG, lw=1, label=lab)
    a.errorbar(x, bottom + m, yerr=sd, color=FG if i < 4 else BG, capsize=5, lw=1.4, ls='none')
    for xi, mi, bi in zip(x, m, bottom):
        a.text(xi + 0.27, bi + mi / 2, f'{mi:.0%}', va='center', fontsize=12, color=MUTED)
    bottom += m
a.set_xticks(x); a.set_xticklabels(list(wells)); a.set_ylim(0, 1.05); a.set_xlim(-0.5, 1.75)
a.set_ylabel('fraction of vacuoles')
a.set_title('Parasites per vacuole (mean ± SD, 6 wells each)', loc='left', pad=14)
h, l = a.get_legend_handles_labels()
lg = a.legend(h[::-1], l[::-1], title='parasites / vacuole', frameon=False, fontsize=13, title_fontsize=13, loc='upper left', bbox_to_anchor=(1.0, 1.0), labelcolor=FG)
lg.get_title().set_color(FG)
fig.tight_layout(); fig.savefig(O + 'replication_synthetic.png', dpi=150); plt.close(fig)
print('replication ok')
