"""Capped reproduction of retained borrowed source views after a validation error."""
import resource
import ast
import os
import gc
import sys
from pathlib import Path

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
repo = Path(os.environ['SPACR_PROOF_REPO'])
sys.path.insert(0, str(repo))
import numpy as np
import spacr.object as O
from spacr.zstack import TStackSpec

root = Path(os.environ.get('SPACR_PROOF_DIR', Path(__file__).resolve().parent))
path = root / 'retained-error.npz'
np.savez_compressed(path, data=np.arange(24, dtype=np.float32).reshape(3, 4, 2),
                    filenames=np.array(['a.npy', 'b.npy', 'c.npy']))
reader = O._mask_archive_arrays
if os.environ.get('SPACR_TRACE_BEFORE'):
    source = (root / 'explicit_close_object.py').read_text()
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == '_mask_archive_arrays')
    namespace = dict(O.__dict__)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<frozen-explicit-close-reader>', 'exec'), namespace)
    reader = namespace[function.name]
error = None
with reader(path, native=True) as (stack, filenames, owner):
    try:
        O._require_t_axis(stack, TStackSpec(t_axis=0, z_axis=1, z_mode='volumetric'), str(path))
    except Exception as observed:
        error = observed
    del stack, owner
assert error is not None
traceback = error.__traceback__
while traceback.tb_frame.f_code.co_name != '_require_t_axis':
    traceback = traceback.tb_next
borrowed = traceback.tb_frame.f_locals['stack']
print('Reading retained validation pixels after context exit; core dumps disabled', flush=True)
print(float(borrowed.sum()), flush=True)
error = traceback = borrowed = None
gc.collect()
assert not list(root.glob('.spacr-native-mask-*'))
path.unlink()
