import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from spacr.qt.widgets import ambient as after

source = Path('/mnt/wd4tb/scratch/theme-refinement-20261006/after_clock/ambient.py')
spec = importlib.util.spec_from_file_location('spacr.qt.widgets._point_before', source)
before = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = before
spec.loader.exec_module(before)
rows = []
for family in ('point_atlas', 'genetic_advection', 'impulse_lens'):
    for background in ('#101418', '#f6f7f9'):
        for seed in (7, 42):
            for detail in (1., 2.):
                for stamp in (0., 3.2):
                    engines = [module.make_engine('data_art_' + family, 'spacr', background,
                               seed=seed, resolution=detail, blur=0) for module in (before, after)]
                    for engine in engines:
                        engine.set_max_pixels(3840 * 2160)
                        if family == 'impulse_lens':
                            for index in range(24):
                                engine.set_time(index * .04)
                                engine._add_impulse((index / 24, .3 + .4 * ((index % 7) / 7)),
                                                    strength=.24 if index % 4 else 1.)
                        engine.set_time(stamp)
                        engine.set_pointer((.22, .68))
                    expected, actual = [engine.shade(3840, 2160) for engine in engines]
                    assert actual == expected, (family, background, seed, detail, stamp)
                    rows.append({'family': family, 'background': background, 'seed': seed,
                                 'detail': detail, 'clock': stamp,
                                 'size': [actual.width(), actual.height()], 'byte_identical': True})
print(json.dumps({'frames': len(rows), 'all_byte_identical': True}), flush=True)
record = {'before_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'after_sha256': hashlib.sha256(Path(after.__file__).read_bytes()).hexdigest(),
          'actual_import': after.__file__, 'records': rows}
Path('/mnt/wd4tb/scratch/theme-refinement-20261006/point_frame_parity.json').write_text(
    json.dumps(record, indent=2) + '\n')
