import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/mnt/wd4tb/scratch/theme-refinement-20261006/final')
REVIEW = ROOT / 'review'
REVIEW.mkdir(parents=True, exist_ok=True)
REPO = Path('/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006')
sys.path.insert(0, str(REPO))
from PySide6.QtGui import QColor, QImage, QPainter
import spacr.qt.widgets

hashes = {}
for relative in ('spacr/qt/widgets/ambient.py', 'spacr/qt/theme.py',
                 'spacr/qt/night_themes.py', 'spacr/qt/preferences.py'):
    path = REPO / relative
    snapshot = ROOT / path.name
    shutil.copyfile(path, snapshot)
    hashes[relative] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('spacr.qt.widgets._final_theme_review', ROOT / 'ambient.py')
ambient = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ambient
spec.loader.exec_module(ambient)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
families = ('point_atlas', 'tissue_facets', 'chromatin_ribbon', 'genetic_advection',
            'impulse_lens', 'fungal_growth')
records = []


def composite(engine, width, height):
    started = time.perf_counter()
    shaded = engine.shade(width, height)
    shaded_done = time.perf_counter()
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor('#101418'))
    painter = QPainter(image)
    engine.blit(painter, shaded, width, height)
    painter.end()
    finished = time.perf_counter()
    return image, (shaded_done - started) * 1000, (finished - started) * 1000


for family in families:
    directory = REVIEW / family
    directory.mkdir(exist_ok=True)
    for width, height in ((1920, 1080), (3840, 2160)):
        for resolution in (1.0, 2.0):
            engine = ambient.make_engine('data_art_' + family, 'spacr', '#101418', seed=42,
                                         resolution=resolution, blur=0, speed=1, size=1, density=1)
            engine.set_max_pixels(width * height)
            shade_times, total_times, pixel_hashes = [], [], []
            for index in range(25):
                engine.set_time((29.0 if family == 'fungal_growth' else 8.0) + index / 24)
                if family == 'impulse_lens':
                    engine.set_pointer((0.48 + 0.10 * math.sin(index * 0.16),
                                        0.50 + 0.08 * math.cos(index * 0.12)))
                    if index == 4:
                        engine._add_impulse(engine.pointer)
                image, shade_ms, total_ms = composite(engine, width, height)
                shade_times.append(shade_ms)
                total_times.append(total_ms)
                pixel_hashes.append(hashlib.sha256(image.bits().tobytes()).hexdigest())
            if resolution == 2:
                image.save(str(directory / f'still-{width}x{height}.png'))
            warm_shade, warm_total = shade_times[1:], total_times[1:]
            p95 = lambda values: sorted(values)[math.ceil(len(values) * 0.95) - 1]
            record = {'family': family, 'display': [width, height], 'resolution': resolution,
                      'buffer': list(engine.buffer_size(width, height)), 'max_pixels': engine.max_pixels,
                      'cold_end_to_end_ms': total_times[0],
                      'shade_copy_median_ms': statistics.median(warm_shade),
                      'shade_copy_p95_ms': p95(warm_shade),
                      'end_to_end_median_ms': statistics.median(warm_total),
                      'end_to_end_p95_ms': p95(warm_total), 'warm_samples': len(warm_total),
                      '24fps_budget_pass_sampled': p95(warm_total) <= 1000 / 24,
                      'unique_pixel_frames': len(set(pixel_hashes)), 'pixel_hashes': pixel_hashes}
            records.append(record)
            receipt = {'source_commit': commit, 'source_sha256': hashes,
                       'imported_renderer': str(ambient.__file__), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                       'qt_platform': os.environ.get('QT_QPA_PLATFORM'), 'cpu_count': os.cpu_count(),
                       'load_average': list(os.getloadavg()),
                       'timing_scope': 'real engine shade + owned image copy + native-size destination fill and actual blit; file encoding excluded',
                       'records': records}
            (ROOT / 'render_perf.json').write_text(json.dumps(receipt, indent=2))
            print(json.dumps({key: value for key, value in record.items() if key != 'pixel_hashes'}), flush=True)
    engine = ambient.make_engine('data_art_' + family, 'spacr', '#101418', seed=42,
                                 resolution=2, blur=0, speed=1, size=1, density=1)
    engine.set_max_pixels(1920 * 1080)
    frames = 144 if family == 'fungal_growth' else 72
    for index in range(frames):
        stamp = (27.0 if family == 'fungal_growth' else 8.0) + index / 24.0
        engine.set_time(stamp)
        if family == 'impulse_lens':
            engine.set_pointer((0.38 + 0.17 * index / (frames - 1),
                                0.53 + 0.07 * math.sin(index * 0.09)))
            if index == 20:
                engine._add_impulse(engine.pointer)
        image, _, _ = composite(engine, 1920, 1080)
        image.save(str(directory / f'frame-{index:03d}.png'))
    print(json.dumps({'family': family, 'offline_24fps_frames': frames}), flush=True)

for relative, digest in hashes.items():
    assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == digest, relative
print(json.dumps({'source_stayed_stable': True, 'review': str(REVIEW)}), flush=True)
