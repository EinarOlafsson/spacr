"""Slide 10 (item 590, 2026-09-28): nuclei again, and the Toxoplasma panel. CPU.

nuclei_th: Cellpose-SAM (cpsam_v2) on the Hoechst crop after a white top-hat
  (disk 35 px) removes the cytoplasmic haze; flow 0.6, cellprob -1.5,
  percentile 1-99.95. Chosen over flow 0.4/cellprob -0.5 (missed dividing
  nuclei), flow 0.8/cellprob -2.5 (spilled into cytoplasm), StarDist
  2D_versatile_fluo and Cellpose 3 nuclei (false objects on the dim cytoplasm
  or cut nuclei). The result nuclei_th_v4.npy is copied to nuclei.npy.
pathogen: Cellpose-SAM on channel C4 (Toxoplasma), automatic diameter, the
  densest 450 px parasite region of the field (rows 1398-1848, cols 698-1148)
  padded by 48 px:  python segment_590.py pathogen 4 1350 650 1896 1196 0
"""
import numpy as np, time, sys, torch
from skimage.morphology import white_tophat, disk
torch.set_num_threads(int(sys.argv[2]))
what=sys.argv[1]
r=np.load('/mnt/wd4tb/scratch/590/seg/raw.npz')
from cellpose import models
model=models.CellposeModel(gpu=False, pretrained_model='cpsam_v2', use_bfloat16=False)
t=time.time()
if what=='nuclei_th':
    img=r['nuc'].astype(np.float32)
    img=white_tophat(img, disk(35))
    np.save('/mnt/wd4tb/scratch/590/seg/nuc_tophat.npy', img)
    m,_,_=model.eval(img, diameter=None, flow_threshold=0.6, cellprob_threshold=-1.5, min_size=150, normalize={'percentile':[1,99.95]})
elif what=='pathogen':
    y0,x0,y1,x1=[int(v) for v in sys.argv[3:7]]
    import tifffile
    D='/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/spacr/tutorials/orig/W0107F0002T0001Z000C4.tif'
    img=tifffile.imread(D).astype(np.float32)[y0:y1,x0:x1]
    np.save('/mnt/wd4tb/scratch/590/seg/pathogen_raw.npy', img)
    np.save('/mnt/wd4tb/scratch/590/seg/pathogen_box.npy', np.array([y0,x0,y1,x1]))
    m,_,_=model.eval(img, diameter=(float(sys.argv[7]) or None), flow_threshold=0.4, cellprob_threshold=-1.0, min_size=20, normalize={'percentile':[1,99.9]})
tag=what+('_d'+sys.argv[7] if what=='pathogen' else '')
np.save(f'/mnt/wd4tb/scratch/590/seg/{tag}_v4.npy', m.astype(np.int32)); print(what, int(m.max()), f'{time.time()-t:.0f}s', flush=True)
