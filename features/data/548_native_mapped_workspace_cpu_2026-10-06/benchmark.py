import hashlib
import importlib.util
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import tifffile

from spacr import convert, io
from tests.test_watch_folder_and_analyse import MASK

variant=os.environ['VARIANT']
if variant=='baseline':
    source=Path('/mnt/wd4tb/scratch/f548-stream-proto-20261006/baseline_io.py')
    spec=importlib.util.spec_from_file_location('spacr._f548_baseline_io',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
else:
    source=Path(io.__file__)
    module=io

def status():
    rows={}
    for row in Path('/proc/self/status').read_text().splitlines():
        if ':' in row:
            key,value=row.split(':',1)
            if key in ('VmRSS','RssAnon','RssFile','VmHWM'):
                rows[key]=int(value.strip().split()[0])
    return rows

with tempfile.TemporaryDirectory(dir='/mnt/wd4tb/scratch/f548-stream-proto-20261006') as folder:
    root=Path(folder)
    raw=root/'raw'/'A01'
    raw.mkdir(parents=True)
    rng=np.random.default_rng(26545)
    for channel in range(1,5):
        frames=rng.integers(0,4096,size=(2,2,1280,1280),dtype=np.uint16)
        frames[::2,:,::13,::7]=0
        tifffile.imwrite(raw/f'field01_C{channel}.tif',frames,
                         metadata={'axes':'TZYX'},photometric='minisblack')
        del frames
    converted=root/'converted'
    assert convert.convert_folder(dict(src=str(root/'raw'),dst=str(converted),
                                       z_handling='keep',preview_rows=0)).is_complete
    settings=dict(MASK,src=str(converted),z_stack=True,t_stack=True,
                  t_axis_order='TZYX',z_axis=None,z_segmentation_mode='volumetric',
                  anisotropy=2,frame_interval_s=60.0,save_original_images=False,
                  batch_size=2,nucleus_channel=3,cell_channel=1,
                  pathogen_channel=None,lower_percentile=17)
    stop=threading.Event()
    samples=[]
    def observe():
        while not stop.is_set():
            samples.append(status())
            stop.wait(.01)
    observer=threading.Thread(target=observe)
    observer.start()
    started=time.perf_counter()
    try:
        result,_=module.preprocess_img_data(settings)
    finally:
        stop.set()
        observer.join()
    elapsed=time.perf_counter()-started
    archive=converted/'masks/plate1_A01_1_norm_timelapse.npz'
    with np.load(archive,allow_pickle=False) as packed:
        shape=list(packed['data'].shape)
        dtype=str(packed['data'].dtype)
    rec={'variant':variant,'source':str(source),
         'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
         'archive_bytes':archive.stat().st_size,
         'shape':shape,'dtype':dtype,'elapsed_s':elapsed,
         'max_rss_kib':max(s['VmRSS'] for s in samples),
         'max_anon_kib':max(s['RssAnon'] for s in samples),
         'max_file_kib':max(s['RssFile'] for s in samples),
         'last_status':status(),
         'selected_roles':[result['cellpose_nucleus_channel'],
                           result['cellpose_cell_channel']]}
    print(json.dumps(rec,indent=2),flush=True)
