"""Draw slide 10's four panels from the raw channels and the masks.

Cells and lipid droplets come from the 900 px crop of tutorial field
W0107F0002T0001Z000 (segment_field.py, 2026-09-26); nuclei were segmented
again for item 590 (segment_590.py) and the pathogen panel is a
450 px close-up of the Toxoplasma channel (C4) segmented with Cellpose-SAM
(segment_590.py). Usage: python render_panels.py [nuclei cells
pathogen droplets] -- NUCLEI=<file> picks the nuclei mask.
"""
import colorsys
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

OUT = os.environ.get('SEG', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seg_590') + '/')
IMG = os.environ.get('IMGOUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'images', 'mask_new') + '/')
NUCLEI = os.environ.get('NUCLEI', 'nuclei.npy')
ZOOM = (200, 250)

raw = np.load(OUT + 'raw.npz')
ya, xa, yb, xb, Y0, X0, S = raw['box']
crop = (slice(Y0 - ya, Y0 - ya + S), slice(X0 - xa, X0 - xa + S))


def grey(a, lo_p=0.5, hi_p=99.7, gamma=0.8):
    lo, hi = np.percentile(a, [lo_p, hi_p])
    return np.clip((a - lo) / (hi - lo), 0, 1) ** gamma


def up(a, k, order=1):
    return ndi.zoom(a, k, order=order)


def palette(labels, seed=3):
    """Bright, saturated colours; the dark blue-violet hues are left out so
    every outline reads on a grey image at slide size."""
    rng = np.random.default_rng(seed)
    n = int(labels.max()) + 1
    hues = rng.random(n) * 0.78
    hues = np.where(hues > 0.55, hues + 0.22, hues) % 1.0
    cols = np.array([colorsys.hsv_to_rgb(h, 0.7, 1.0) for h in hues])
    cols[0] = 0
    return cols


def outlined_multi(base, labels, k, width):
    rgb = np.repeat(base[..., None], 3, axis=2)
    lab = up(labels, k, order=0)
    b = find_boundaries(lab, mode='inner')
    if width > 1:
        b = ndi.binary_dilation(b, iterations=width - 1) & (lab > 0)
    rgb[b] = palette(labels)[lab][b]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


if __name__ == '__main__':
    os.makedirs(IMG, exist_ok=True)
    which = sys.argv[1:] or ['nuclei', 'cells', 'pathogen', 'droplets']
    for name in which:
        if name == 'pathogen':
            img = np.load(OUT + 'pathogen_raw.npy').astype(np.float32)
            lab = np.load(OUT + 'pathogen_d0.npy')
            p = 48
            img, lab = img[p:-p, p:-p], lab[p:-p, p:-p]
            out = outlined_multi(up(grey(img, 0.5, 99.8, 0.6), 4), lab, 4, 7)
        elif name == 'droplets':
            z0, z1 = ZOOM
            lab = np.load(OUT + 'droplets.npy')[crop][z0:z0 + 450, z1:z1 + 450]
            img = raw['lip'][crop].astype(np.float32)[z0:z0 + 450, z1:z1 + 450]
            out = outlined_multi(up(grey(img, 0.5, 99.7, 0.6), 4), lab, 4, 3)
        else:
            f, chan = (NUCLEI, 'nuc') if name == 'nuclei' else ('cells.npy', 'cyt')
            lab = np.load(OUT + f)[crop]
            img = raw[chan][crop].astype(np.float32)
            out = outlined_multi(up(grey(img, 0.5, 99.8, 0.85), 2), lab, 2, 6)
        Image.fromarray(out).save(IMG + f'mask_{name}.png', optimize=True)
        print(name, out.shape, int(len(np.unique(lab)) - 1), 'objects in view')
