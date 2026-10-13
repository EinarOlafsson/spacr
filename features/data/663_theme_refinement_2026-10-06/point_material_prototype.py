import json
import time
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage
from spacr.qt.widgets import ambient

assert str(Path(ambient.__file__).resolve()).startswith('/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006/')


def prototype(self, width, height, x, y, light, spread=False):
    px = np.asarray(x, dtype=np.int32)
    py = np.asarray(y, dtype=np.int32)
    values = np.clip(np.asarray(light, dtype=np.float32) * self.alpha_scale(), 0., 1.)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py, values = px[inside], py[inside], values[inside]
    intensities = np.rint(values * 255).astype(np.uint8)
    levels = np.arange(256, dtype=np.float32) / 255.
    palette = self.paint_colors
    lookup = np.full(256, np.uint32(0xFF000000), dtype=np.uint32)
    for channel, shift in (('red', 16), ('green', 8), ('blue', 0)):
        primary = getattr(palette[0], channel)()
        accent = getattr(palette[min(1, len(palette) - 1)], channel)()
        ink = .78 * primary + .22 * accent
        value = ink * levels if self.dark else 255. - (255. - ink) * levels
        lookup |= np.asarray(value, dtype=np.uint32) << shift
    image = QImage(width, height, QImage.Format_RGB32)
    flat = np.frombuffer(image.bits(), dtype=np.uint32, count=width * height)
    flat.fill(lookup[0])
    combine = np.maximum.at if self.dark else np.minimum.at
    combine(flat, py * width + px, lookup[intensities])
    if spread:
        for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            xx, yy = px + dx, py + dy
            valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            weight = .24 if dx and dy else .68
            levels_at = np.rint(intensities[valid] * weight).astype(np.uint8)
            combine(flat, yy[valid] * width + xx[valid], lookup[levels_at])
    return image


rows = []
for width, height in ((1920,1080),(3840,2160)):
    for dark in (True,False):
        engine = ambient.make_engine('data_art_impulse_lens','spacr', '#101418' if dark else '#f6f7f9',seed=42)
        engine.set_max_pixels(width*height)
        rng=np.random.default_rng(42)
        count=130000 if width==3840 else 35000
        x=rng.integers(-2,width+2,count,dtype=np.int32)
        y=rng.integers(-2,height+2,count,dtype=np.int32)
        x[::6]=x[1::6][:len(x[::6])]
        light=rng.uniform(-.1,1.2,count).astype(np.float32)
        for spread in (False,True):
            expected=engine._point_material(width,height,x,y,light,spread)
            actual=prototype(engine,width,height,x,y,light,spread)
            assert actual==expected,(width,dark,spread)
            samples={}
            for name,fn in (('before',engine._point_material),('prototype',lambda *args:prototype(engine,*args))):
                times=[]
                for _ in range(16):
                    began=time.perf_counter()
                    image=fn(width,height,x,y,light,spread)
                    times.append((time.perf_counter()-began)*1000)
                samples[name]={'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95))}
            row={'size':[width,height],'dark':dark,'spread':spread,'samples':samples}
            rows.append(row)
            print(json.dumps(row),flush=True)
Path('/mnt/wd4tb/scratch/theme-refinement-20261006/point_material_prototype.json').write_text(json.dumps(rows,indent=2)+'\n')
