import hashlib
import json
from pathlib import Path
from spacr.qt.widgets.ambient import make_engine
import spacr.qt.widgets.ambient as ambient

root = Path('/mnt/wd4tb/spacr-worktrees/codex-theme-growth-advection-20261007')
source = root / 'spacr/qt/widgets/ambient.py'
assert Path(ambient.__file__).resolve() == source.resolve()
out = Path('/mnt/wd4tb/scratch/theme-growth-advection-20261007/mycelium-final-frames')
out.mkdir(exist_ok=True)
engine = make_engine('data_art_fungal_growth', 'spacr', '#080a12', seed=7)
for frame in range(61):
    engine.set_time(float(frame * 2))
    image = engine.shade(960, 540)
    assert image.save(str(out / f'frame-{frame:03d}.png'), 'PNG')
(out.parent/'mycelium-final-timelapse.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_commit':'604269d2487ab2513989736415047465b9432ff9','renderer':'actual _FungalGrowthEngine.shade','viewport':[960,540],'frames':61,'animation_seconds':[0,120],'sampling_step_animation_seconds':2,'playback_fps':12,'palette':'spacr','background':'#080a12','seed':7,'density':1,'size':1},indent=2)+'\n')
