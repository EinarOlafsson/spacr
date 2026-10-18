"""Cellpose-SAM (stock cpsam_v2, as the Make Masks magnifier uses) on one 768 px Hoechst crop, cached for the GIF."""
import os, numpy as np, tifffile, torch
torch.set_num_threads(12)
from cellpose import models
SRC = '/mnt/wd4tb/af3/projects/cross_channel_models/data/cell_from_hoechst/images/MTOC_Screen__mtocScreen_20250530_111302__plate1_M07_9_1.tif'
OUT = '/mnt/firecuda2/Claude/toxoplasma_projects/presentation/images/magnifier_gif/seg.npz'
im = tifffile.imread(SRC).astype(np.float32)
y, x = 600, 600
crop = im[y:y + 768, x:x + 768]
m = models.CellposeModel(gpu=False, pretrained_model=os.path.expanduser('~/.cellpose/models/cpsam_v2'))
masks = m.eval(crop, batch_size=4)[0]
np.savez_compressed(OUT, image=crop, masks=masks, src=SRC, y=y, x=x)
print('objects', len(np.unique(masks)) - 1)
