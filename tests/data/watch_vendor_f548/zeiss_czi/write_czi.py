"""Write the synthetic multi-scene CZI fixtures of the F548 vendor watch tests.

Run with an environment that has pylibCZIrw:
    python write_czi.py OUTPUT_DIR
Each plane is ``czi_plane(scene, t, z, c)`` of tests/test_watch_vendor_formats_f548.py.
"""
import os
import sys

import numpy as np
from pylibCZIrw import czi as pyczi


def czi_plane(scene, t, z, c, size=32):
    plane = np.full((size, size), 40, np.uint16)
    plane[5:17, 5:17] = 500 + 37 * scene + 11 * t + 7 * z + 3 * c
    plane[18:29, 18:29] = 800 + 37 * scene + 11 * t + 7 * z + 3 * c
    return plane


def write(path, scenes, times, planes, channels):
    with pyczi.create_czi(path, exist_ok=True) as writer:
        for scene in range(scenes):
            for t in range(times):
                for z in range(planes):
                    for c in range(channels):
                        writer.write(data=czi_plane(scene, t, z, c),
                                     plane={'T': t, 'Z': z, 'C': c}, scene=scene,
                                     location=(scene * 50, 0))


out = sys.argv[1]
write(os.path.join(out, 'two_scenes_tzc.czi'), 2, 2, 3, 2)
write(os.path.join(out, 'two_scenes_tc.czi'), 2, 3, 1, 2)
print('ok')
