import hashlib
import json
import resource
import statistics
import time
from pathlib import Path
from spacr.qt.widgets.ambient import make_engine
import spacr.qt.widgets.ambient as ambient

root = Path('/mnt/wd4tb/spacr-worktrees/codex-theme-growth-advection-20261007')
source = root / 'spacr/qt/widgets/ambient.py'
assert Path(ambient.__file__).resolve() == source.resolve()
out = Path('/mnt/wd4tb/scratch/theme-growth-advection-20261007')
results = []
for density, size in ((1, 1), (3, 2.5)):
    for detail in (1, 2):
        engine = make_engine('data_art_fungal_growth', 'spacr', '#080a12', seed=7)
        engine.set_max_pixels(3840*2160)
        engine.set_resolution(detail)
        engine.set_density(density)
        engine.set_size(size)
        durations = []
        image = None
        for index in range(3):
            engine.set_time(57 + index*.25)
            started = time.perf_counter()
            image = engine.shade(3840, 2160)
            durations.append((time.perf_counter()-started)*1000)
        path = out / f'mycelium-final-native4k-d{density}-s{size}-detail{detail}.png'
        assert image.save(str(path),'PNG')
        results.append({'density':density,'size':size,'detail':detail,'screen':[3840,2160],'buffer':engine.buffer_size(3840,2160),'shade_ms':durations,'median_ms':statistics.median(durations),'frame':str(path),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
receipt={'source_commit':'604269d2487ab2513989736415047465b9432ff9','ambient_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_path':str(source),'renderer':'actual _FungalGrowthEngine.shade, 3 sequential frames per setting','qt_platform':'offscreen','records':results}
path=out/'mycelium-final-native4k.json'
path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
