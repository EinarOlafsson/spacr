"""The deck clips microscopy tiles to the filled canonical logo silhouette."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def test_cover_preparation_keeps_interior_and_removes_attached_and_detached_exterior(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    grid = 8
    occupied = np.zeros((grid, grid), dtype=bool)
    occupied[2:6, 2:6] = True
    occupied[0, 0] = occupied[7, 7] = True
    occupied[3, 1] = True
    Image.new('RGB', (160, 160), (80, 120, 160)).save(source / 'mosaic.png')
    (source / 'photomosaic.json').write_text(json.dumps({
        'grid': grid, 'image': 'mosaic.png',
        'occupied': np.packbits(occupied).tobytes().hex()}))
    icons = tmp_path / 'icons'
    icons.mkdir()
    silhouette = Image.new('RGBA', (160, 160))
    ImageDraw.Draw(silhouette).rectangle((40, 40, 119, 119), outline='white', width=4)
    silhouette.save(icons / 'logo_spacr.png')
    deck = tmp_path / 'deck' 
    (deck / 'scripts').mkdir(parents=True)
    (deck / 'images/icons').mkdir(parents=True)
    script = deck / 'scripts/prep_assets.py'
    shutil.copyfile(ROOT / 'tools/readme_deck/scripts/prep_assets.py', script)
    environment = dict(os.environ, PYTHONPATH=str(ROOT), QT_QPA_PLATFORM='offscreen')
    subprocess.run([sys.executable, str(script), str(icons), '--mosaic-only',
                    '--mosaic-source', str(source)], check=True, env=environment)
    pixels = np.asarray(Image.open(deck / 'images/icons/spacr_logo_color.png'))
    alpha = pixels[..., 3]
    assert not alpha[:40].any() and not alpha[120:].any()
    assert not alpha[:, :40].any() and not alpha[:, 120:].any()
    assert alpha[40:120, 40:120].all()
    assert pixels[80, 80, :3].any()
