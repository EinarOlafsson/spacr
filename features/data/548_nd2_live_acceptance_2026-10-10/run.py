"""F548 acceptance: real NAS ND2 SIM Z-stacks through the live watch path, real Cellpose on GPU.

Reads the NAS only. Each ND2 is written into the watch folder in chunks, the way a
microscope grows a file, while the watcher runs. Records pickup latency (watcher start
minus the moment the file stopped growing), total time, peak RAM and GPU memory.
"""
import json, os, shutil, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, '/media/carruthers/mnt3/claude/spacr-citation')
import psutil
from spacr import core
from spacr.settings import set_default_settings_preprocess_generate_masks
OUT = Path(__file__).parent
W = OUT / 'watched'
if W.exists(): shutil.rmtree(W)
W.mkdir()
SOURCES = [Path('/nas_mnt/carruthers/anu/Otherdata/hargk_SIM/nd024.nd2'),
           Path('/nas_mnt/carruthers/anu/Otherdata/hargk_SIM/nd025.nd2')]
settings = set_default_settings_preprocess_generate_masks({})
settings.update(dict(src=str(W), z_stack=True, timelapse=False, z_segmentation_mode='volumetric', z_axis=0, voxel_size_z_um=0.2, voxel_size_xy_um=0.065, channels=[0, 1],
                     nucleus_channel=0, cell_channel=1, pathogen_channel=None,
                     organelle_channel=None, watch_pipeline='mask', watch_poll_seconds=1,
                     watch_settle_seconds=3, watch_idle_minutes=1.5, plot=False,
                     verbose=False, test_mode=False, n_jobs=1, randomize=False))
completed = {}
def writer():
    time.sleep(5)  # the watch starts on an empty folder
    for src in SOURCES:
        data = src.read_bytes(); dst = W / src.name; step = len(data) // 11
        with open(dst, 'wb') as fh:
            for i in range(0, len(data), step):
                fh.write(data[i:i+step]); fh.flush(); os.fsync(fh.fileno()); time.sleep(2)
        completed[src.stem] = time.time()
        time.sleep(10)
peak = {'rss': 0, 'gpu_mib': 0}
stop = threading.Event()
def sampler():
    proc = psutil.Process()
    while not stop.is_set():
        rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
        peak['rss'] = max(peak['rss'], rss)
        try:
            used = [int(x) for x in subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True).stdout.split(',')[1::2]]
            mine = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
            for line in mine.splitlines():
                pid, mem = [x.strip() for x in line.split(',')]
                if int(pid) == os.getpid(): peak['gpu_mib'] = max(peak['gpu_mib'], int(mem))
        except Exception: pass
        time.sleep(0.5)
threading.Thread(target=writer, daemon=True).start(); threading.Thread(target=sampler, daemon=True).start()
t0 = time.time()
result = core._watch_folder_and_analyse(settings)
total = time.time() - t0
stop.set()
ledger = json.loads(Path(result['ledger']).read_text())
import torch
report = dict(sources=[str(s) for s in SOURCES], settings={k: settings[k] for k in ('z_stack', 'channels', 'nucleus_channel', 'cell_channel', 'watch_poll_seconds', 'watch_settle_seconds', 'watch_idle_minutes')},
              result={k: result[k] for k in ('done', 'failed', 'incomplete')}, total_seconds=round(total, 1),
              peak_rss_gib=round(peak['rss'] / 2**30, 2), peak_gpu_process_mib=peak['gpu_mib'],
              torch_max_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
              file_complete_epoch=completed, ledger=ledger)
(OUT / 'receipt.json').write_text(json.dumps(report, indent=1, default=str))
print(json.dumps({k: v for k, v in report.items() if k != 'ledger'}, indent=1, default=str))
