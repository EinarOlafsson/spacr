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
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
I = (sys.argv[1] if len(sys.argv) > 1 else
     '/mnt/firecuda2/Claude/repo/spacr/spacr/resources/icons/')
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
