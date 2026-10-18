"""Stock Cellpose-SAM (cpsam_v2, before any training on our images) on the same test crops as predict_models.py."""
import os, sys, time
import numpy as np, torch
torch.set_num_threads(12)
from cellpose import models, metrics

OUT = '/mnt/firecuda2/Claude/toxoplasma_projects/presentation/model_predictions'
model = models.CellposeModel(gpu=False, pretrained_model=os.path.expanduser('~/.cellpose/models/cpsam_v2'))
for key in sys.argv[1:]:
    dst = f'{OUT}/{key}_stock.npz'
    if os.path.exists(dst): continue
    t = time.time()
    d = np.load(f'{OUT}/{key}.npz')
    pred = model.eval(d['image'].astype(np.float32), batch_size=4)[0]
    ap, tp, fp, fn = metrics.average_precision([d['gt']], [pred], threshold=[0.5])
    tp, fp, fn = int(tp[0][0]), int(fp[0][0]), int(fn[0][0])
    np.savez_compressed(dst, pred=pred, f1=2 * tp / max(1, 2 * tp + fp + fn), tp=tp, fp=fp, fn=fn)
    print(key, 'stock pred', len(np.unique(pred)) - 1, f'F1@0.5={2 * tp / max(1, 2 * tp + fp + fn):.3f}', f'{time.time()-t:.0f}s', flush=True)
