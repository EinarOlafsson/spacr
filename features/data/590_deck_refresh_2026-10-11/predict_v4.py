import numpy as np, torch, time
torch.set_num_threads(12)
from cellpose import models, metrics
S='/mnt/firecuda2/Claude/toxoplasma_projects/presentation/model_predictions/'
d=np.load(S+'toxoplasma_pv_v3.npz')
im=d['image'].astype(np.float32); g=d['gt']
w='/mnt/wd4tb/scratch/deck-20261011/weights/weights/cpsam_v2_toxo_r7'
t=time.time()
pred=models.CellposeModel(gpu=False, pretrained_model=w).eval(im, batch_size=4)[0]
ap,tp,fp,fn=metrics.average_precision([g],[pred],threshold=[0.5]); tp,fp,fn=int(tp[0][0]),int(fp[0][0]),int(fn[0][0])
f1=2*tp/max(1,2*tp+fp+fn)
np.savez_compressed('model_predictions/toxoplasma_pv_v4.npz', image=im, gt=g, pred=pred, f1=f1, tp=tp, fp=fp, fn=fn, image_path=d['image_path'], gt_path=d['gt_path'], weights=w)
import shutil; shutil.copy(S+'toxoplasma_pv_v3_stock.npz','model_predictions/toxoplasma_pv_v4_stock.npz')
print('v4 F1', round(f1,3), tp, fp, fn, 'v3 F1', float(d['f1']), f'{time.time()-t:.0f}s')
