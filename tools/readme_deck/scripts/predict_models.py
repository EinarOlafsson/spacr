"""CPU inference: one held-out test field per model, for the slide deck.
Writes <out>/<model>.npz with image, ground truth, prediction and scores."""
import csv, os, sys, time
import numpy as np, tifffile, torch
torch.set_num_threads(12)
from cellpose import models, metrics

OUT = '/mnt/firecuda2/Claude/toxoplasma_projects/presentation/model_predictions'
C = '/mnt/wd4tb/af3/projects/cross_channel_models/data'
PV = '/mnt/wd4tb/af3/projects/toxoplasma_pv_model'
W = os.path.expanduser('~/.spacr/models')

RUNS = '/mnt/wd4tb/af3/projects/cross_channel_models/runs'


# Fields chosen by hand for the slides (maintainer feedback 2026-09-21): PV v3 on a dense field where it
# clearly beats stock; nuclei-from-cellmask on a parasite-free field so no vacuole is mistaken for a nucleus.
OVERRIDE = {'r6_perimage.csv': 'plate7_F18_2', 'nuclei_from_cellmask': 'CSA_screen__screen_20250125_171958__plate2_G03_14_1'}


def typical(path, keep):
    for k, v in OVERRIDE.items():
        if k in path:
            print('override', path, v, flush=True)
            return v
    """Name of a field whose recorded F1 sits at the median: a representative example, not a best case."""
    rows = [r for r in csv.DictReader(open(path)) if keep(r)]
    for r in rows:
        tp, fp, fn = (float(r[k]) for k in ('tp', 'fp', 'fn'))
        r['_f1'] = 2 * tp / max(1, 2 * tp + fp + fn)
    med = float(np.median([r['_f1'] for r in rows]))
    r = min(rows, key=lambda r: abs(r['_f1'] - med))
    print('picked', os.path.basename(path), r.get('name') or r.get('stem'), f"recorded F1 {r['_f1']:.3f}, median {med:.3f}", flush=True)
    return r.get('name') or r.get('stem')


def cc(ds, run, maskdir):
    n = typical(f'{RUNS}/{run}/best_perimage.csv', lambda r: r['host'] == 'HeLa' and 30 <= int(r['n_true']) <= 150)
    return f'{C}/{ds}/images/{n}.tif', f'{C}/{ds}/{maskdir}/{n}.tif'


jobs = {
    'toxoplasma_pv_v3': (f'{PV}/metrics/final_r6/models/cpsam_v2_toxo_r6',
                         *(lambda n: (f'{PV}/data/_seg_training_data/images/{n}.tif', f'{PV}/data/_seg_training_data/masks/{n}.tif'))(
                             typical(f'{PV}/metrics/final_r6/qc/r6_perimage.csv', lambda r: r['set'] == 'test' and int(r['n_true']) >= 20))),
    'toxoplasma_from_cellmask_v1': (f'{W}/toxoplasma_from_cellmask_pv', *cc('toxoplasma_from_cellmask', 'toxoplasma_from_cellmask_pv', 'masks_pv')),
    'nuclei_from_cellmask_v1': (f'{W}/nuclei_from_cellmask_best', *cc('nuclei_from_cellmask', 'nuclei_from_cellmask', 'masks')),
    'cell_from_hoechst_v1': (f'{W}/cell_from_hoechst_best', *cc('cell_from_hoechst', 'cell_from_hoechst', 'masks')),
    'toxoplasma_from_hoechst_v1': (f'{W}/toxoplasma_from_hoechst_pv', *cc('toxoplasma_from_hoechst', 'toxoplasma_from_hoechst_pv', 'masks_pv')),
}

only = sys.argv[1:]
for key, (w, img, gt) in jobs.items():
    if only and key not in only: continue
    if os.path.exists(f'{OUT}/{key}.npz'): continue
    t = time.time()
    im = tifffile.imread(img).astype(np.float32); g = tifffile.imread(gt)
    crop = int(os.environ.get('CROP', '0'))
    if crop and g.shape[0] > crop:
        best, by, bx = -1, 0, 0
        for y in range(0, g.shape[0] - crop + 1, 128):
            for x in range(0, g.shape[1] - crop + 1, 128):
                n = len(np.unique(g[y:y + crop, x:x + crop]))
                if n > best: best, by, bx = n, y, x
        im, g = im[by:by + crop, bx:bx + crop], g[by:by + crop, bx:bx + crop]
    m = models.CellposeModel(gpu=False, pretrained_model=w)
    pred = m.eval(im, batch_size=4)[0]
    ap, tp, fp, fn = metrics.average_precision([g], [pred], threshold=[0.5])
    tp, fp, fn = int(tp[0][0]), int(fp[0][0]), int(fn[0][0])
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    np.savez_compressed(f'{OUT}/{key}.npz', image=im, gt=g, pred=pred, f1=f1, tp=tp, fp=fp, fn=fn,
                        image_path=img, gt_path=gt, weights=w)
    print(key, os.path.basename(img), 'gt', len(np.unique(g)) - 1, 'pred', len(np.unique(pred)) - 1,
          f'F1@0.5={f1:.3f}', f'{time.time()-t:.0f}s', flush=True)
