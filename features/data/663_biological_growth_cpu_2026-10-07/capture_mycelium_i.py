from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from spacr.qt.widgets.ambient import make_engine

out = Path('/mnt/wd4tb/scratch/theme-growth-advection-20261007')
engine = make_engine('data_art_fungal_growth', 'spacr', '#080a12', seed=7)
engine.set_density(1.0)
engine.set_size(1.0)
stamps = (7, 15, 27, 37, 45, 57, 67, 75, 87, 97, 105, 117)
frames = []
for stamp in stamps:
    engine.set_time(float(stamp))
    image = engine.shade(960, 540)
    path = out / f'mycelium-i-{stamp}.png'
    assert image.save(str(path), 'PNG')
    frame = Image.open(path).convert('RGB')
    frame.thumbnail((480, 270))
    panel = Image.new('RGB', (480, 290), '#080a12')
    panel.paste(frame, (0, 20))
    ImageDraw.Draw(panel).text((8, 4), f't={stamp}s', fill='white')
    frames.append(panel)
contact = Image.new('RGB', (4*480, 3*290), '#080a12')
for i, frame in enumerate(frames):
    contact.paste(frame, ((i % 4)*480, (i//4)*290))
contact.save(out/'mycelium-i-contact.png')
print(out/'mycelium-i-contact.png')
