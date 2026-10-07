import ast
import hashlib
import json
import subprocess
import textwrap
import types
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication
from spacr.qt.widgets import ambient

old_commit = 'ae7269a08c136ac35d488d9f657f82cbfd33c49a'
old_source = subprocess.check_output(
    ['git', 'show', f'{old_commit}:spacr/qt/widgets/ambient.py'], text=True)
module = ast.parse(old_source)
cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'AuroraEngine')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_paint_field')
namespace = dict(ambient.__dict__)
exec(textwrap.dedent(''.join(old_source.splitlines(keepends=True)[method.lineno - 1:method.end_lineno])), namespace)
old_method = namespace['_paint_field']
QApplication.instance() or QApplication([])
rows = []
for width, height, density, size, resolution in (
        (1280, 720, 1, 1, 1), (3840, 2160, 1, 1, 1), (3840, 2160, 3, 3, 2)):
    for palette, background in (('spacr', '#101418'), ('borealis', '#101418'), ('mono', '#f3f2ee')):
        before = ambient.make_engine('aurora', palette, background, seed=7,
                                     density=density, size=size, resolution=resolution)
        after = ambient.make_engine('aurora', palette, background, seed=7,
                                    density=density, size=size, resolution=resolution)
        before._paint_field = types.MethodType(old_method, before)
        before.set_max_pixels(width * height)
        after.set_max_pixels(width * height)
        for clock in (0.0, 3.33, 9.5, 60.2):
            before.set_time(clock)
            after.set_time(clock)
            left_image = before.shade(width, height)
            right_image = after.shade(width, height)
            assert left_image.size() == right_image.size()
            left = np.frombuffer(left_image.bits(), dtype=np.uint8)
            right = np.frombuffer(right_image.bits(), dtype=np.uint8)
            changed = np.any(left.reshape(-1, 4) != right.reshape(-1, 4), axis=1)
            rows.append({'size': [width, height], 'density': density,
                         'element_size': size, 'resolution': resolution,
                         'palette': palette, 'background': background,
                         'clock': clock, 'changed_pixels': int(changed.sum()),
                         'max_byte_diff': int(np.max(np.abs(left.astype(np.int16) - right.astype(np.int16))))})
print(json.dumps({'source': ambient.__file__,
                  'source_sha256': hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),
                  'old_method_git_sha': old_commit,
                  'rows': rows}, indent=2))
