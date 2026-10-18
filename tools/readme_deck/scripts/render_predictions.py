"""Three-panel figure per model: input channel | prediction (magenta) | curated truth (cyan).
Reads model_predictions/<key>.npz, writes images/predictions/<key>.png."""
import glob, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk

P = '/mnt/firecuda2/Claude/toxoplasma_projects/presentation/'
os.makedirs(P + 'images/predictions', exist_ok=True)
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34)
WIN = 900


def norm(im):
    im = im.astype(np.float32)
    if im.ndim == 3: im = im.mean(axis=-1) if im.shape[-1] <= 4 else im.mean(axis=0)
    lo, hi = np.percentile(im, (1, 99.7))
    return np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)


def window(gt, WIN):
    h, w = gt.shape
    best, by, bx = -1, 0, 0
    for y in range(0, max(1, h - WIN + 1), 150):
        for x in range(0, max(1, w - WIN + 1), 150):
            n = len(np.unique(gt[y:y + WIN, x:x + WIN]))
            if n > best: best, by, bx = n, y, x
    return by, bx


def overlay(g, labels, rgb):
    out = np.stack([g * 0.85] * 3, -1)
    b = binary_dilation(find_boundaries(labels, mode='inner'), disk(1))
    out[b] = rgb
    return out


for f in sorted(g for g in glob.glob(P + 'model_predictions/*.npz') if not g.endswith('_stock.npz')):
    key = os.path.basename(f)[:-4]
    d = np.load(f)
    g, gt, pr = norm(d['image']), d['gt'], d['pred']
    WIN = min(900, *gt.shape)
    y, x = window(gt, WIN)
    sl = (slice(y, y + WIN), slice(x, x + WIN))
    panels = [(np.stack([g[sl]] * 3, -1), 'input channel')]
    sp = P + f'model_predictions/{key}_stock.npz'
    if os.path.exists(sp):
        panels.append((overlay(g[sl], np.load(sp)['pred'][sl], (1.0, 0.72, 0.3)), 'stock Cellpose-SAM (untrained)'))
    panels += [(overlay(g[sl], pr[sl], (1.0, 0.2, 0.75)), 'our fine-tuned model'), (overlay(g[sl], gt[sl], (0.2, 0.9, 0.85)), 'curated truth')]
    gap = 24
    cols = 2 if len(panels) == 4 else 3
    rows = len(panels) // cols
    canvas = Image.new('RGB', (cols * WIN + (cols - 1) * gap, rows * WIN + (rows - 1) * gap), (14, 18, 23))
    for i, (p, lab) in enumerate(panels):
        tile = Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(tile)
        dr.rectangle((0, 0, 20 + 21 * len(lab), 56), fill=(14, 18, 23))
        dr.text((12, 10), lab, fill=(232, 237, 242), font=FONT)
        canvas.paste(tile, ((i % cols) * (WIN + gap), (i // cols) * (WIN + gap)))
    canvas.save(P + f'images/predictions/{key}.png')
    print(key, f"F1@0.5={float(d['f1']):.3f}", 'tp', int(d['tp']), 'fp', int(d['fp']), 'fn', int(d['fn']))
