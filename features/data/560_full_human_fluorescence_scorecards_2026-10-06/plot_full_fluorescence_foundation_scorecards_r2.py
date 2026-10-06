from pathlib import Path
import hashlib
import json
import os
import sys

sys.meta_path = [finder for finder in sys.meta_path if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(Path.cwd()))
import spacr
assert Path(spacr.__file__).resolve().parent == Path('spacr').resolve()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from spacr.plot import save_figure
from spacr.figures.style import _apply_user_style

root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/560-human-fluorescence-primary-r1/full-fluorescence-foundation-CPU-scorecards-r2')
receipt = json.loads((root / 'complete-scorecards.json').read_text())
assert receipt['passed'] and len(receipt['comparisons']) == 12
models = ['resnet18', 'resnet50', 'vit_small_patch14_dinov2.lvd142m', 'openphenom', 'chada_vit', 'subcell']
names = ['ResNet-18', 'ResNet-50', 'DINOv2-S/14', 'OpenPhenom', 'ChAda-ViT', 'SubCell']
cohorts = [('HEp2-expert-DAPI', 'Epithelial DAPI · 10,000 crops · 7 classes'), ('CycleNET-expert-native', 'Yeast native markers · 10,499 crops · 9 classes')]
fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.7))
for column, (cohort, title) in enumerate(cohorts):
    cards = {card['backbone']: card for card in receipt['comparisons'] if card['cohort'] == cohort}
    assert set(cards) == set(models)
    for row, metric, label in [(0, 'map', 'Mean average precision: full self-excluded ranking'), (1, 'accuracy', 'Logistic regression: five stratified crop folds')]:
        axis = axes[row, column]
        values = [cards[model]['normal_retrieval']['map'] if row == 0 else cards[model]['normal_classifier']['accuracy'] for model in models]
        chance = cards[models[0]]['analytical_random_ranking_expected_map'] if row == 0 else cards[models[0]]['normal_classifier']['chance']
        positions = np.arange(len(models))
        axis.barh(positions, values, color=['#466d9a'] * 3 + ['#379281'] * 3, height=0.65)
        axis.axvline(chance, color='#666666', linestyle='--', linewidth=1.1, label=f'Chance {chance:.3f}')
        axis.set_yticks(positions, names)
        axis.invert_yaxis()
        axis.set_xlim(0, 1.10)
        axis.set_xticks([0, .25, .5, .75, 1])
        axis.set_xlabel(label, fontsize=9)
        if row == 0:
            axis.set_title(title, fontsize=12)
        for position, value in zip(positions, values):
            axis.text(value + .015, position, f'{value:.3f}', va='center', fontsize=9)
        axis.legend(loc='lower right', fontsize=8, frameon=False)
        axis.spines[['top', 'right']].set_visible(False)
fig.suptitle('Frozen spaCR encoders on original expert-labelled fluorescence crops', fontsize=15)
fig.text(.04, .06, 'SubCell epithelial input: DAPI + zero protein (DNA-only ablation). Yeast input: nuclear/bud-neck reference + GFP.\nAll eligible crops and original QC classes are included. Crop diagnostics; no patient, plate or gene holdout.', fontsize=9, va='bottom')
fig.subplots_adjust(left=.115, right=.98, top=.91, bottom=.17, hspace=.32, wspace=.37)
_apply_user_style(fig, kind='bar', force=True)
pdf = save_figure(fig, root / 'full-human-fluorescence-foundation-comparison.pdf', fmt='pdf', dpi=300, close=False)
png = save_figure(fig, root / 'full-human-fluorescence-foundation-comparison.png', fmt='png', dpi=160, close=True)
files = {str(path): {'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'bytes': Path(path).stat().st_size} for path in (pdf, png)}
(root / 'normal-scorecard-figure-artifacts.json').write_text(json.dumps({'normal_spacr_plot_and_user_style_used': True, 'actual_twelve_scorecards': True, 'files': files, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
print('PASS actual full-cohort normal spaCR figure exports', pdf, png, flush=True)
