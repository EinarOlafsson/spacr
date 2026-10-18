"""Slide 10: nuclei, cells and lipid droplets of one tutorial field, segmented again.

Nuclei and cells: Cellpose-SAM (cpsam_v2, spaCR's default) on the whole
field, CPU. Lipid droplets: spaCR's own spot segmentation
(spacr.object._segment_spots: white top-hat, threshold, watershed split).
"""
import sys
import time

import numpy as np
import tifffile
import torch

torch.set_num_threads(16)
D = '/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/spacr/tutorials/orig/W0107F0002T0001Z000C%d.tif'
OUT = '/mnt/firecuda2/Claude/scratch/deck589/seg/'
Y0, X0, S, PAD = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), 96
what = sys.argv[4:] or ['nuclei', 'cells', 'droplets']

nuc = tifffile.imread(D % 1).astype(np.float32)
cyt = tifffile.imread(D % 2).astype(np.float32)
lip = tifffile.imread(D % 3).astype(np.float32)
ya, xa = max(0, Y0 - PAD), max(0, X0 - PAD)
yb, xb = min(nuc.shape[0], Y0 + S + PAD), min(nuc.shape[1], X0 + S + PAD)
sl = (slice(ya, yb), slice(xa, xb))
np.savez_compressed(OUT + 'raw.npz', nuc=nuc[sl], cyt=cyt[sl], lip=lip[sl],
                    box=np.array([ya, xa, yb, xb, Y0, X0, S]))

if 'nuclei' in what or 'cells' in what:
    from cellpose import models
    model = models.CellposeModel(gpu=False, pretrained_model='cpsam_v2', use_bfloat16=False)
if 'nuclei' in what:
    t = time.time()
    m, _f, _s = model.eval(nuc[sl], diameter=None, flow_threshold=0.4,
                           cellprob_threshold=-1.0, min_size=60,
                           normalize={'percentile': [1, 99.8]})
    np.save(OUT + 'nuclei.npy', m.astype(np.int32))
    print('nuclei', int(m.max()), f'{time.time() - t:.0f}s', flush=True)
if 'cells' in what:
    t = time.time()
    img = np.stack([cyt[sl], nuc[sl], np.zeros_like(nuc[sl])], axis=-1)
    m, _f, _s = model.eval(img, channel_axis=-1, diameter=None,
                           flow_threshold=0.4, cellprob_threshold=-0.5,
                           min_size=400, normalize={'percentile': [1, 99.8]})
    np.save(OUT + 'cells.npy', m.astype(np.int32))
    print('cells', int(m.max()), f'{time.time() - t:.0f}s', flush=True)
if 'droplets' in what:
    from spacr.object import _segment_spots
    t = time.time()
    settings = {'organelle_tophat_radius': 4, 'organelle_watershed_spots': True,
                'organelle_min_area': 3, 'organelle_adaptive_block_size': 31,
                'organelle_adaptive_offset': 0}
    img = lip[sl]
    lo, hi = np.percentile(img, [1, 99.9])
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    m = _segment_spots(img, 'otsu', settings)
    np.save(OUT + 'droplets.npy', m.astype(np.int32))
    print('droplets', int(m.max()), f'{time.time() - t:.0f}s', flush=True)
