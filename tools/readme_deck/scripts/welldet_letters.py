"""Slide "Plaque Well Detector v2": take the panel letters out of the published
figure the detector is shown on (item 589), and the cut-off line of the
page's title above it.

Each letter is found as the bright (white letters) or dark (black letters)
pixels in a small box, kept only where they form a piece that lies inside
the box, grown by 2 px for the antialiasing, and inpainted from the image
around it (OpenCV Telea), so the photograph continues underneath.
"""
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SRC = sys.argv[1]
DST = sys.argv[2]
LETTERS = {
    'A': ((18, 22, 50, 54), 'white'), 'B': ((18, 336, 48, 366), 'white'),
    'C': ((634, 24, 660, 54), 'white'), 'D': ((634, 258, 660, 290), 'white'),
    'E': ((10, 596, 40, 628), 'black'), 'F': ((230, 596, 258, 628), 'black'),
    'G': ((450, 596, 478, 628), 'black'), 'H': ((670, 596, 698, 628), 'black'),
    'I': ((8, 818, 36, 858), 'black'),
}

im = np.asarray(Image.open(SRC).convert('RGB')).copy()
lum = im.mean(axis=2)
mask = np.zeros(lum.shape, np.uint8)
for name, ((x0, y0, x1, y1), colour) in LETTERS.items():
    sub = lum[y0:y1, x0:x1]
    ink = sub > 200 if colour == 'white' else sub < 60
    pieces, n = ndi.label(ink, structure=np.ones((3, 3)))
    keep = np.zeros_like(ink)
    for i, sl in enumerate(ndi.find_objects(pieces), start=1):
        on_edge = (sl[0].start == 0 or sl[1].start == 0
                   or sl[0].stop == ink.shape[0] or sl[1].stop == ink.shape[1])
        if not on_edge and (pieces[sl] == i).sum() >= 8:
            keep |= pieces == i
    grown = ndi.binary_dilation(keep, iterations=2)
    mask[y0:y1, x0:x1] |= grown.astype(np.uint8)
    print(name, int(keep.sum()), 'glyph pixels')
out = cv2.inpaint(im[..., ::-1].copy(), mask * 255, 5, cv2.INPAINT_TELEA)[..., ::-1]
top = 22
Image.fromarray(np.ascontiguousarray(out[top:])).save(DST, optimize=True)
review = np.concatenate([im[top:], np.full((im.shape[0] - top, 8, 3), 255, np.uint8), out[top:]], axis=1)
Image.fromarray(review).save(DST.replace('.png', '_review.png'))
