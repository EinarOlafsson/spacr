"""Icons -> white-on-transparent 512px with strokes thick enough to read; image size index for build.js.

Item 589 (2026-09-26): the maintainer found the icons' white lines too thin to
see. Each icon's alpha is grown by a 13-px max filter per 1024 source pixels
(about 0.3 px either side of every line at the tile's size on a 3200-px
slide) before it is cut down, and it is kept at 512 px rather than 256.
Grown alpha exposes the colour under formerly transparent pixels, which is
black in the source files, so the colour is set to white everywhere.

Paths are relative to this script's folder, so a copy of the presentation
folder builds from its own images.

    python3 scripts/prep_assets.py [path/to/spacr/resources/icons/]
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_fill_holes

Image.MAX_IMAGE_PIXELS = None
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("icons", nargs="?", type=Path,
                    default=ROOT / "spacr/resources/icons")
parser.add_argument("--mosaic-only", action="store_true")
parser.add_argument("--mosaic-source", type=Path,
                    default=ROOT / "spacr/resources/splash")
args = parser.parse_args()
I = str(args.icons) + '/'
sys.path.insert(0, str(args.mosaic_source.resolve().parents[2]))
from PySide6.QtCore import QSize
from PySide6.QtGui import QImageReader
from spacr.qt.widgets.loading_screen import _black_to_alpha

meta = json.loads((args.mosaic_source / 'photomosaic.json').read_text())
grid = int(meta['grid'])
tile = 20
side = grid * tile
occupied = np.unpackbits(np.frombuffer(bytes.fromhex(meta['occupied']),
                                      dtype=np.uint8))[:grid * grid]
occupied = occupied.astype(bool).reshape(grid, grid)
reader = QImageReader(str(args.mosaic_source / meta['image']))
reader.setScaledSize(QSize(side, side))
image = reader.read()
if image.isNull():
    raise ValueError('The packaged startup mosaic could not be decoded')
image = _black_to_alpha(image)
pixels = np.frombuffer(image.bits(), dtype=np.uint32).reshape(
    side, image.bytesPerLine() // 4)[:, :side]
pixels.reshape(grid, tile, grid, tile)[:] *= occupied[:, None, :, None]
silhouette = Image.open(args.icons / 'logo_spacr.png').convert('RGBA')
silhouette = silhouette.resize((side, side), Image.Resampling.LANCZOS)
inside = binary_fill_holes(np.asarray(silhouette)[..., 3] > 128)
pixels[~inside] = 0
if not image.save(P + 'images/icons/spacr_logo_color.png'):
    raise OSError('The clipped deck mosaic could not be saved')
if args.mosaic_only:
    raise SystemExit(0)
names = ('mask measure annotate classify map_barcodes regression foreign embeddings run_compare experiment_design '
         'power dose_response qc_dashboard make_masks align umap gate_editor graph_builder analyze_plaques recruitment '
         'invasion replication model_zoo timelapse motility report batch pipeline_graph barcode_qc image_scatter').split()
for n in names:
    im = Image.open(I + n + '.png').convert('RGBA')
    a = np.array(im)
    alpha = a[..., :3].max(axis=2) if a[..., 3].min() == 255 else a[..., 3]
    k = max(3, int(round(13 * max(im.size) / 1024)) | 1)
    alpha = Image.fromarray(alpha).filter(ImageFilter.MaxFilter(k))
    white = Image.new('L', im.size, 255)
    icon = Image.merge('RGBA', (white, white, white, alpha))
    icon.thumbnail((512, 512), Image.LANCZOS)
    icon.save(P + 'images/icons/' + n + '.png', optimize=True)
im = Image.open(I + 'logo_spacr_readme.png')
im.save(P + 'images/icons/logo.png')
sizes = {}
for f in glob.glob(P + 'images/**/*.*', recursive=True):
    if f.lower().endswith(('.png', '.jpg', '.gif')):
        with Image.open(f) as im:
            sizes[os.path.relpath(f, P)] = im.size
json.dump(sizes, open(P + 'scripts/sizes.json', 'w'), indent=0)
print(len(sizes))
