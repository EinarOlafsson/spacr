"""Paper panels for the deck, with the panel letters taken out cleanly (item 589).

crop_paper2.py painted a rectangle over each letter, sized in 1100-px preview
coordinates (one unit is 9 source pixels) and filled with the median of the
pixels around it. The rectangles were too coarse: some left part of a glyph
showing (slide 2's "A"), and where the ring touched a dark image the fill
came out as a black square on white (Classify: image model).

Here the rough region only says WHERE a letter is. Inside it the glyph is
found as ink -- pixels unlike the local background -- split into connected
pieces, and only the pieces that lie wholly inside the (slightly grown)
region and are small are removed: those pixels and a 2-px antialiasing halo
are set to the background colour. Nothing that reaches outside the region
(an axis, an image, a line) is touched, so no box edge is ever drawn.

Sources are the full-resolution figure PNGs (9922 px wide) or, for Fig. 3,
the PDF rendered at 1200 dpi. Crops are kept up to 4000 px wide.
"""
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None
D = '/home/olafsson/Downloads/source/'
W = '/mnt/firecuda2/Claude/scratch/deck589/'
SRC = {'F1': D + 'figures/Figure_1_spaCR_and_spaCRPower_workflows.png',
       'F2': D + 'figures/Figure_2_spaCR_screen_experimental_workflow.png',
       'F3': W + 'paper/fig3hi.png',
       'F4': D + 'figures/Figure_4_XGBoost_feature_attribution_and_MaxViT_saliency_2.png',
       'F5': D + 'figures/Figure_5_EAF1_TSG101_recruitment_and_ESCRT_association_2.png',
       'S7': D + 'supplementary_figures/Figure_S7_gene_level_OLS_diagnostics_and_permutation_2.png'}
MAXW = 4000


def background(pixels):
    """The commonest colour among ``pixels`` (N x 3), to 8 levels a channel."""
    q = (pixels // 8).astype(np.int64)
    key = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    values, counts = np.unique(key, return_counts=True)
    top = values[np.argmax(counts)]
    return np.clip(pixels[key == top].astype(np.float64).mean(axis=0), 0, 255)


def erase_glyphs(arr, box, grow=0.35, halo=2, max_frac=0.6):
    """Remove the ink pieces that sit inside ``box`` (x0, y0, x1, y1)."""
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    gx, gy = round((x1 - x0) * grow), round((y1 - y0) * grow)
    ex0, ey0 = max(0, x0 - gx), max(0, y0 - gy)
    ex1, ey1 = min(w, x1 + gx), min(h, y1 + gy)
    wx0, wy0 = max(0, ex0 - 3 * gx - 4), max(0, ey0 - 3 * gy - 4)
    wx1, wy1 = min(w, ex1 + 3 * gx + 4), min(h, ey1 + 3 * gy + 4)
    win = arr[wy0:wy1, wx0:wx1].astype(np.float32)
    inner = arr[ey0:ey1, ex0:ex1].reshape(-1, 3).astype(np.float32)
    bg = background(inner)
    ink = np.abs(win - bg).max(axis=2) > 40
    pieces, n = ndi.label(ink, structure=np.ones((3, 3)))
    if n == 0:
        return 0, bg
    slices = ndi.find_objects(pieces)
    area = (ex1 - ex0) * (ey1 - ey0)
    chosen = np.zeros(n + 1, bool)
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        py0, py1 = sl[0].start + wy0, sl[0].stop + wy0
        px0, px1 = sl[1].start + wx0, sl[1].stop + wx0
        inside = px0 >= ex0 and py0 >= ey0 and px1 <= ex1 and py1 <= ey1
        touches = px1 > x0 and px0 < x1 and py1 > y0 and py0 < y1
        small = (px1 - px0) * (py1 - py0) <= max_frac * area
        chosen[i] = inside and touches and small
    remove = chosen[pieces]
    if not remove.any():
        return 0, bg
    keep = ink & ~remove
    remove = ndi.binary_dilation(remove, iterations=halo) & ~keep
    region = arr[wy0:wy1, wx0:wx1]
    region[remove] = np.clip(np.round(bg), 0, 255).astype(arr.dtype)
    return int(chosen.sum()), bg


def erase_edge_strays(arr, side, width):
    """Remove ink that a neighbouring panel left along one edge of the crop.

    Only pieces that touch that edge and lie wholly within ``width`` pixels
    of it are removed; anything reaching further in belongs to the panel.
    """
    h, w = arr.shape[:2]
    band = {'left': (slice(None), slice(0, width)),
            'right': (slice(None), slice(w - width, w)),
            'top': (slice(0, width), slice(None)),
            'bottom': (slice(h - width, h), slice(None))}[side]
    view = arr[band]
    bg = background(view.reshape(-1, 3).astype(np.float32))
    ink = np.abs(view.astype(np.float32) - bg).max(axis=2) > 40
    full = np.zeros((h, w), bool)
    full[band] = ink
    reach = np.zeros((h, w), bool)
    grown = np.abs(arr.astype(np.float32) - bg).max(axis=2) > 40
    pieces, n = ndi.label(grown, structure=np.ones((3, 3)))
    edge_ids = {'left': pieces[:, 0], 'right': pieces[:, -1],
                'top': pieces[0, :], 'bottom': pieces[-1, :]}[side]
    removed = 0
    for i in np.unique(edge_ids[edge_ids > 0]):
        sl = ndi.find_objects(pieces == i)[0]
        inside = {'left': sl[1].stop <= width, 'right': sl[1].start >= w - width,
                  'top': sl[0].stop <= width, 'bottom': sl[0].start >= h - width}[side]
        if inside:
            reach |= pieces == i
            removed += 1
    reach = ndi.binary_dilation(reach, iterations=2) & ~(grown & ~reach)
    arr[reach] = np.clip(np.round(bg), 0, 255).astype(arr.dtype)
    return removed


def fill_empty_tile(arr, box, sample, gutter):
    """Give a grid tile that only held the panel letter its neighbours' background.

    Once the letter is gone its tile is plain black among tiles whose
    background is a colour (the saliency grid's dark purple), and reads as a
    black box. The tile's black pixels at least ``gutter`` pixels inside the
    grid's black outline take the median colour of ``sample`` (a patch of a
    neighbouring tile's background), so the grid lines keep their width.
    """
    x0, y0, x1, y1 = box
    sx0, sy0, sx1, sy1 = sample
    colour = np.median(arr[sy0:sy1, sx0:sx1].reshape(-1, 3), axis=0)
    black = arr.mean(axis=2) < 20
    inside = ndi.distance_transform_edt(black) >= gutter
    region = np.zeros_like(black)
    region[y0:y1, x0:x1] = True
    arr[region & inside & black] = np.round(colour).astype(arr.dtype)
    return tuple(int(v) for v in colour)


def run(jobs, out_dir, review_dir):
    cache = {}
    for name, j in jobs.items():
        src = j['src']
        if src not in cache:
            cache[src] = np.asarray(Image.open(SRC[src]).convert('RGB'))
        im = cache[src]
        f = im.shape[1] / 1100
        x0, y0, x1, y1 = j['box']
        crop = im[round(y0 * f):round(y1 * f), round(x0 * f):round(x1 * f)].copy()
        before = crop.copy()
        report = []
        for cov in j.get('cover', []):
            a, b, c, d = cov[:4]
            box = (max(0, round((a - x0) * f)), max(0, round((b - y0) * f)),
                   min(crop.shape[1], round((c - x0) * f)),
                   min(crop.shape[0], round((d - y0) * f)))
            n, bg = erase_glyphs(crop, box, **j.get('erase', {}))
            report.append((box, n, tuple(int(v) for v in bg)))
        if 'fill_tile' in j:
            hh, ww = crop.shape[:2]
            to_px = lambda b: (round(b[0] * ww), round(b[1] * hh), round(b[2] * ww), round(b[3] * hh))
            t = j['fill_tile']
            colour = fill_empty_tile(crop, to_px(t['box']), to_px(t['sample']),
                                     round(t['gutter'] * ww))
            print(f'  {name}: empty tile filled with {colour}')
        for side, width in j.get('edge_strays', {}).items():
            n = erase_edge_strays(crop, side, round(width * f))
            print(f'  {name}: {n} stray piece(s) removed along the {side} edge')
        out = Image.fromarray(crop)
        if out.width > MAXW:
            out = out.resize((MAXW, round(out.height * MAXW / out.width)), Image.LANCZOS)
        out.save(out_dir + name + '.png', optimize=True)
        tiles = []
        for box, n, bg in report:
            bx0, by0, bx1, by1 = box
            pad = max(bx1 - bx0, by1 - by0)
            sl = (slice(max(0, by0 - pad), by1 + pad), slice(max(0, bx0 - pad), bx1 + pad))
            pair = np.concatenate([before[sl], np.full((before[sl].shape[0], 6, 3), 255, np.uint8) * 0 + [255, 0, 0], crop[sl]], axis=1).astype(np.uint8)
            t = Image.fromarray(pair)
            t = t.resize((600, max(1, round(t.height * 600 / t.width))))
            tiles.append(t)
            print(f'  {name}: region {box} -> {n} piece(s) removed, background {bg}')
        if tiles:
            sheet = Image.new('RGB', (600, sum(t.height + 8 for t in tiles)), (60, 60, 60))
            yy = 0
            for t in tiles:
                sheet.paste(t, (0, yy))
                yy += t.height + 8
            sheet.save(review_dir + 'rev_' + name + '.png')
        print(name, out.size)


if __name__ == '__main__':
    import os
    jobs = json.load(open(sys.argv[1]))
    only = sys.argv[2:]
    if only:
        jobs = {k: v for k, v in jobs.items() if k in only}
    rev = W + 'paper/review/'
    os.makedirs(rev, exist_ok=True)
    run(jobs, W + 'images/manuscript/', rev)
