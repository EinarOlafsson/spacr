import faulthandler
faulthandler.enable()
import os
import numpy as np
print('actual NumPy BLAS configuration', flush=True)
np.show_config()
print('full-size scalar-neutral matrix diagnostic, thread settings', {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}, flush=True)
matrix = np.ones((17074, 384), dtype=np.float64)
print('allocated input bytes', matrix.nbytes, flush=True)
scores = matrix @ matrix.T
assert scores.shape == (17074, 17074) and scores.min() == scores.max() == 384
print('PASS actual full-size float64 dense product', scores.shape, scores.nbytes, flush=True)
