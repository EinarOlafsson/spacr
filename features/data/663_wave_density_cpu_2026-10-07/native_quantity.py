"""Actual native wave population, density response and unchanged maximum proof."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys

import numpy as np
from PySide6.QtWidgets import QApplication

repo = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo))
from spacr.qt.widgets import ambient

assert Path(ambient.__file__).resolve() == repo / 'spacr/qt/widgets/ambient.py'
source_hash = hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('spacr.qt.widgets._atlas_before', root / 'before.py')
before = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = before
spec.loader.exec_module(before)
ambient._begin_ambient_startup()
before._begin_ambient_startup()
app = QApplication([])
width, height = 3840, 2160


def engine(module, palette, density, detail, background='#101418', radius=.5):
    result = module.make_engine('data_art_point_atlas', palette, background, seed=42,
                                density=density, resolution=detail, blur=0)
    result.set_max_pixels(width * height)
    result.set_time(8)
    result.set_gravity_radius(radius)
    result.set_pointer((.5, .5))
    return result


def words(image):
    return np.frombuffer(image.constBits(), np.uint32)


records, maximum_pairs = [], []
for palette in ('spacr', 'random'):
    density_frames, density_counts = [], []
    for density in (1, 2, 3):
        detail_frames = []
        for detail in (1, 2):
            current = engine(ambient, palette, density, detail)
            frame = current.shade(width, height)
            points = next(v for k, v in current._material_cache.items()
                          if isinstance(k, tuple) and k[0] == 'point_atlas')
            assert current.effective_density() == density
            assert np.all(words(frame) >> 24 == 255)
            assert frame.size().width() == width and frame.size().height() == height
            record = {'palette': palette, 'density': density, 'detail': detail,
                      'samples': len(points[0]), 'coordinate_bytes': sum(v.nbytes for v in points),
                      'frame_sha256': hashlib.sha256(frame.constBits()).hexdigest()}
            records.append(record)
            detail_frames.append(words(frame).copy())
            if detail == 2:
                density_counts.append(record['samples'])
                density_frames.append(detail_frames[-1])
                if palette == 'spacr':
                    frame.save(str(root / f'wave-density-{density}-native.png'))
            retained = bytes(frame.constBits())
            current.advance(.1)
            current.shade(width, height)
            assert bytes(frame.constBits()) == retained
            del current, frame, points, retained
        assert np.array_equal(*detail_frames)
    assert all(a < b for a, b in zip(density_counts, density_counts[1:]))
    for index in range(2):
        changed = int(np.count_nonzero(density_frames[index] != density_frames[index + 1]))
        assert changed > 1000
        records.append({'palette': palette, 'from_density': index + 1,
                        'to_density': index + 2, 'changed_native_pixels': changed})
    del density_frames, detail_frames
    for background in ('#101418', '#f6f7f9'):
        for radius in (0, .5):
            current = engine(ambient, palette, 3, 2, background, radius)
            reference = engine(before, palette, 3, 2, background, radius)
            image = current.shade(width, height)
            expected = reference.shade(width, height)
            assert np.array_equal(words(image), words(expected))
            maximum_pairs.append({'palette': palette, 'background': background,
                                  'radius': radius, 'changed_pixels': 0,
                                  'frame_sha256': hashlib.sha256(image.constBits()).hexdigest()})
            del current, reference, image, expected
assert hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest() == source_hash
receipt = {'source_sha256': source_hash,
           'before_sha256': hashlib.sha256((root / 'before.py').read_bytes()).hexdigest(),
           'source_parent': '57544560ec', 'native': [width, height],
           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'scope': '12 native density/detail frames plus8 exact maximum-density old/new pairs; no FPS acceptance',
           'records': records, 'maximum_density_pairs': maximum_pairs,
           'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'intentional_default_population_change': True,
           'maximum_density_original_pixels_retained': True,
           'hard_24fps_acceptance': False, 'human_aesthetic_acceptance': False}
(root / 'native_quantity.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
