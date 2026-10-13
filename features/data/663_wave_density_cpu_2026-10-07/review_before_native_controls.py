"""Read-only final integrated native population and frame ownership review."""
import hashlib
import json
from pathlib import Path
import resource
import sys

import numpy as np
from PySide6.QtWidgets import QApplication

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
from spacr.qt.widgets import ambient

assert Path(ambient.__file__).resolve() == repo / 'spacr/qt/widgets/ambient.py'
paths = ['spacr/qt/widgets/ambient.py', 'spacr/qt/dialogs.py',
         'spacr/qt/widgets/glass.py', 'spacr/qt/widgets/availability_panel.py']
hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in paths}
ambient._begin_ambient_startup()
app = QApplication([])
width, height = 3840, 2160
records = []
for family in ('point_atlas', 'tissue_facets'):
    for density in (1, 3):
        same = []
        for detail in (1, 2):
            engine = ambient.make_engine('data_art_' + family, 'random', '#101418',
                                         seed=42, density=density, resolution=detail, blur=0)
            engine.set_max_pixels(width * height)
            assert engine.effective_density() == density
            assert engine.buffer_size(width, height) == (width, height)
            engine.set_time(2)
            engine.set_gravity_radius(.5)
            engine.set_pointer((.5, .5))
            first = engine.shade(width, height)
            before = bytes(first.constBits())
            engine.advance(.1)
            frame = engine.shade(width, height)
            assert bytes(first.constBits()) == before
            assert not np.array_equal(np.frombuffer(before, np.uint32),
                                      np.frombuffer(frame.constBits(), np.uint32))
            assert np.all(np.frombuffer(frame.constBits(), np.uint32) >> 24 == 255)
            material = next(v for k, v in engine._material_cache.items() if k[0] == family)
            record = {'family': family, 'density': density, 'detail': detail,
                      'effective_density': engine.effective_density(),
                      'buffer': engine.buffer_size(width, height),
                      'cache_keys': len(engine._material_cache),
                      'cache_names': [str(k) for k in engine._material_cache],
                      'frame_sha256': hashlib.sha256(frame.constBits()).hexdigest()}
            if family == 'point_atlas':
                record['samples'] = len(material[0])
                record['coordinate_bytes'] = sum(v.nbytes for v in material)
                assert len(engine._material_cache) == 3
            else:
                rotation = next(v for k, v in engine._material_cache.items() if k[0] == 'tissue_rotation')
                record['facets'] = len(material)
                record['angle_count'] = len(rotation[1])
                record['spinning_facets'] = sum(v != 0 for v in rotation[1])
                record['tile_bytes'] = sum(v[5].sizeInBytes() for v in material)
                assert len(rotation[1]) == len(material)
                assert len(engine._material_cache) == 2
            same.append(record)
            records.append(record)
            if density == 3 and detail == 2:
                frame.save(str(Path(__file__).parent / (family + '-random-max-native.png')))
            engine.set_resolution(.5)
            assert not engine._material_cache
            del engine, first, frame, before, material
        assert same[0]['frame_sha256'] == same[1]['frame_sha256']
        assert same[0].get('facets', same[0].get('samples')) == same[1].get('facets', same[1].get('samples'))
assert hashes == {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in paths}
receipt = {'source_root': str(repo), 'source_sha256': hashes,
           'scope': '8 actual native4K Random engines; population/cache/retained frames, not worker FPS',
           'records': records, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'hard_24fps_acceptance': False}
(Path(__file__).parent / 'native_controls.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
