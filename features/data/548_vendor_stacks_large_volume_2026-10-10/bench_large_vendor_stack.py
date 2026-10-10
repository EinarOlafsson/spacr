"""Measure watcher latency and memory on a live 2048x2048x200 vendor OME-TIFF."""
import json, os, subprocess, sys, threading, time, resource

import numpy as np
import psutil

from spacr import core
from tests.test_watch_folder_and_analyse import MASK

pytest_plugins = ['tests.test_cov_object_masks_sam']
HERE = os.path.dirname(os.path.abspath(__file__))
PLANES = int(os.environ.get('BENCH_PLANES', '200'))
SIZE = int(os.environ.get('BENCH_SIZE', '2048'))
RATE = float(os.environ.get('BENCH_RATE', '10'))
TIMES = int(os.environ.get('BENCH_TIMES', '1'))


class Sampler:
    def __init__(self):
        self.proc = psutil.Process()
        self.peak = 0
        self.stop = False
        self.main = threading.get_ident()
        self.global_peak = 0
        self.timeline = []
        self.peak_stack = None
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        import traceback
        last = 0
        while not self.stop:
            rss = self.proc.memory_info().rss
            self.peak = max(self.peak, rss)
            now = time.time()
            if rss > self.global_peak * 1.02 + 2**28 or now - last > 1:
                frame = sys._current_frames().get(self.main)
                stack = [f'{os.path.basename(f.filename)}:{f.lineno}:{f.name}'
                         for f in traceback.extract_stack(frame)[-12:]] if frame else []
                self.timeline.append({'t': now, 'rss_mib': round(rss / 2**20), 'stack': stack[-6:]})
                last = now
            if rss > self.global_peak:
                self.global_peak = rss
                frame = sys._current_frames().get(self.main)
                if frame is not None:
                    self.peak_stack = [f'{os.path.basename(f.filename)}:{f.lineno}:{f.name}'
                                       for f in traceback.extract_stack(frame)[-14:]]
            time.sleep(0.02)


def test_large_vendor_volume(tmp_path, fake_model, monkeypatch):
    import spacr.object as spacr_object

    def volume_eval(self, x, **kwargs):
        volume = x[0] if isinstance(x, list) else x
        calls.append(list(volume.shape))
        labels = np.zeros(volume.shape[:3], np.uint16)
        labels[:, SIZE // 4:SIZE // 2, SIZE // 4:SIZE // 2] = 1
        return labels, None, None

    monkeypatch.setattr(spacr_object.cp_models.CellposeModel, 'eval', volume_eval)
    calls = []
    sampler = Sampler()
    phases = []

    def timed(module, name):
        original = getattr(module, name)

        def wrapper(*args, **kwargs):
            sampler.peak = 0
            before = psutil.Process().memory_info().rss
            start = time.time()
            try:
                return original(*args, **kwargs)
            finally:
                phases.append({'phase': name, 'start': start,
                               'seconds': round(time.time() - start, 3),
                               'rss_before_mib': round(before / 2**20, 1),
                               'peak_rss_mib': round(max(sampler.peak, before,
                                   psutil.Process().memory_info().rss) / 2**20, 1)})
        monkeypatch.setattr(module, name, wrapper)

    for name in ('_watch_copy_snapshot', '_watch_vendor_stage', '_watch_stage_volumes',
                 'preprocess_generate_masks', '_watch_collection_artifacts',
                 '_watch_collect'):
        timed(core, name)
    checks = []
    original_unreadable = core._watch_vendor_unreadable

    def unreadable(path, mode):
        start = time.time()
        reason = original_unreadable(path, mode)
        checks.append({'at': start, 'seconds': round(time.time() - start, 3),
                       'reason': reason})
        return reason
    monkeypatch.setattr(core, '_watch_vendor_unreadable', unreadable)

    watched = os.environ['BENCH_DIR']
    os.makedirs(watched, exist_ok=True)
    path = os.path.join(watched, 'embryo01.ome.tif')
    writer = subprocess.Popen([sys.executable, os.path.join(HERE, 'writer.py'), path,
                               str(PLANES), str(SIZE), str(RATE), str(TIMES)],
                              stdout=subprocess.PIPE, text=True)
    settings = dict(MASK, src=watched, channels=[0], nucleus_channel=0, cell_channel=None,
                    z_stack=True, z_segmentation_mode='volumetric', z_axis=0,
                    anisotropy=2, save_original_images=False,
                    watch_folder=True, watch_settle_seconds=2.0,
                    watch_poll_seconds=1.0, watch_idle_minutes=0.25)
    if TIMES > 1:
        settings.update(t_stack=True, t_axis_order='TZYX', z_axis=None,
                        frame_interval_s=60.0, batch_size=2)
    start = time.time()
    result = core._watch_folder_and_analyse(settings)
    total = time.time() - start
    timing = json.loads(writer.communicate()[0].strip().splitlines()[-1])
    sampler.stop = True
    ledger = json.load(open(os.path.join(watched, 'spacr_watch/watch_ledger.json')))
    entry = ledger['fields']['embryo01']
    premature = [c for c in checks if c['at'] < timing['writer_closed'] and c['reason'] is None]
    report = {
        'shape': [TIMES, PLANES // TIMES, SIZE, SIZE], 'dtype': 'uint16',
        'model_inputs': calls,
        'file_bytes': os.path.getsize(path), 'writer_rate_planes_per_s': RATE,
        'result': {k: v for k, v in result.items() if k != 'ledger'},
        'writer_seconds': round(timing['writer_closed'] - timing['writer_start'], 3),
        'completion_checks': checks, 'premature_ready': len(premature),
        'start_after_writer_closed_s': round(entry['started'] - timing['writer_closed'], 3),
        'done_after_writer_closed_s': round(entry['finished'] - timing['writer_closed'], 3),
        'ledger_waited_s': entry['waited'], 'ledger_seconds': entry['seconds'],
        'phases': [{**p, 'start': round(p['start'] - timing['writer_closed'], 3)} for p in phases],
        'process_maxrss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'watch_total_s': round(total, 3),
        'peak_stack': sampler.peak_stack,
        'timeline': [{**e, 't': round(e['t'] - timing['writer_closed'], 2)} for e in sampler.timeline],
    }
    out = os.environ.get('BENCH_OUT', os.path.join(HERE, 'report.json'))
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps(report, indent=1))
    assert result['done'] == ['embryo01'] and not premature
