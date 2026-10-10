"""Head-to-head on the 2026-10-09 retraining holdouts: new vs published vs stock.

Same test images, same eval options and metric as retrain_20261009/pipeline.py
(F1 at IoU 0.5 pooled over images; normalize=True; min_size 15 PV / 0 plaque;
max_size_fraction 1.0). PV test images that round 7 trained or validated on are
tagged, and F1 is also reported without them.
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np, tifffile, torch
from cellpose import models, metrics
from huggingface_hub import hf_hub_download

RUN = Path('/media/carruthers/mnt3/codex/scratch/retrain_20261009')
OUT = Path(__file__).parent
HOME = Path.home() / '.cellpose/models'

def read(p):
    p = Path(p)
    if p.suffix.lower() in ('.tif', '.tiff'): return tifffile.imread(p)
    import imageio.v2 as io; return io.imread(p)

def hf(uri):
    repo = uri.split('huggingface.co/')[1].split('/resolve/')[0]
    path = uri.split('/resolve/main/')[1].split('?')[0]
    return hf_hub_download(repo, path)

MODELS = {
 'pv': {
  'new_r8_best': RUN/'pv/models/pv_best',
  'toxoplasma_pv_v4': 'https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7/resolve/main/weights/cpsam_v2_toxo_r7',
  'toxoplasma_pv_v3': 'https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r6/resolve/main/weights/cpsam_v2_toxo_r6',
  'stock_cpsam_v2': HOME/'cpsam_v2'},
 'plaque': {
  'new_r6_best': RUN/'plaque/models/plaque_best',
  'toxoplasma_plaque_v3': 'https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc/resolve/main/weights/cpsam_plaque_r5_geldoc',
  'toxoplasma_plaque_v2': 'https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5/resolve/main/weights/cpsam_plaque_r5',
  'stock_cpsam': HOME/'cpsam', 'stock_cpsam_v2': HOME/'cpsam_v2'},
}
r7 = {}
with open('/nas_mnt/carruthers/sync/projects/toxoplasma_pv_model/metrics/final_r7/split.csv') as f:
    for row in csv.DictReader(f): r7[row['stem']] = row['set']

def r7_role(stem):
    for s, role in r7.items():
        if stem == s or stem.endswith('__' + s) or stem.endswith(s): return role
    return None

def f1(rows, keep=lambda r: True):
    t = {k: sum(r[k] for r in rows if keep(r)) for k in ('tp', 'fp', 'fn')}
    n = sum(1 for r in rows if keep(r))
    p = t['tp'] / max(1, t['tp'] + t['fp']); rc = t['tp'] / max(1, t['tp'] + t['fn'])
    return dict(n_images=n, f1=2*t['tp']/max(1, 2*t['tp']+t['fp']+t['fn']), precision=p, recall=rc, **t)

kind = sys.argv[1]
rows = [r for r in json.loads((RUN/kind/'prepared.json').read_text())['records'] if r['split'] == 'test']
opts = dict(normalize=True, min_size=0 if kind == 'plaque' else 15, max_size_fraction=1.0)
gts = {r['stem']: read(r['mask']) for r in rows}
imgs = {r['stem']: read(r['image']) for r in rows}
report = dict(kind=kind, n_test=len(rows), eval_options=opts, models={})
for name, src in MODELS[kind].items():
    path = hf(src) if isinstance(src, str) else str(src)
    t0 = time.time()
    model = models.CellposeModel(gpu=True, pretrained_model=path)
    per = []
    for r in rows:
        pred = model.eval(imgs[r['stem']], **opts)[0]
        ap, tp, fp, fn = metrics.average_precision([gts[r['stem']]], [pred], threshold=[.5])
        per.append(dict(stem=r['stem'], source=r['source'], group=r['group'],
                        r7=r7_role(r['stem']) if kind == 'pv' else None,
                        tp=int(tp[0, 0]), fp=int(fp[0, 0]), fn=int(fn[0, 0])))
    res = dict(weights=path, seconds=round(time.time()-t0, 1), overall=f1(per),
               by_source={s: f1(per, lambda r, s=s: r['source'] == s) for s in sorted({r['source'] for r in per})},
               peak_gpu_mib=round(torch.cuda.max_memory_allocated()/2**20))
    if kind == 'pv':
        res['excluding_r7_train_or_validation'] = f1(per, lambda r: r['r7'] not in ('train', 'validation'))
    res['per_image'] = per
    report['models'][name] = res
    print(name, json.dumps({k: v for k, v in res.items() if k not in ('per_image', 'by_source')}), flush=True)
    del model; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
if kind == 'pv':
    report['r7_overlap_counts'] = {k: sum(1 for r in rows if r7_role(r['stem']) == k) for k in ('train', 'validation', 'test', None)}
(OUT/f'{kind}_compare.json').write_text(json.dumps(report, indent=1, default=str))
print('DONE', kind)
