#!/usr/bin/env python3
"""Replace local/maintainer path text in tutorial scene frames (item 447).

The published tutorial masters are composed from per-scene still captures
(``production/<lesson>/visual.json``). Some older captures show real paths of
the maintainer's computer in folder fields, console lines, terminals and file
dialogs. This tool finds that text with the OCR the frame sweep uses
(``rapidocr_onnxruntime``; native-size full frame, half-overlapping 1080p
tiles, and the sweep's own brightness-lifted tiles) and repaints ONLY the local
part of each path:

* ``/mnt/<disk>/.../refresh_2026-09-09/replication_runs/X`` ->
  ``/data/example/replication_runs/X``
* ``/home/<account>/...`` -> ``/home/user/...``; a bare account label -> ``user``
* a wrapped console row that continues such a path loses the private folders

Only the replaced root is drawn: its font, size, colour and renderer gamma are
fitted to the original glyphs (the app's Open Sans weights first, then common
system and terminal fonts), and the fit must correlate at least MIN_FIT. The
trailing file/folder names and any text after them keep their captured pixels
and move to follow the shorter root. The old text is cleared by inpainting its
text band from the surrounding background; field borders, scrollbars and all
other pixels are untouched, and a difference mask is checked to stay inside the
recorded boxes. Each frame is then re-read as the sweep reads it and around
every painted box, and anything newly found is redacted the same way.

Sub-commands (run all Python through tools/run_capped.sh):

  detect  --lesson ID --stage STAGE --out DIR      OCR every unique scene image
  apply   --lesson ID --stage STAGE --out DIR      write redacted copies, repoint visual.json
  record  --lesson ID --stage STAGE --out DIR --sweep JSON   add media hashes/checks to the receipt
  verify  --image PNG ...                          re-OCR; exit 1 if a path remains

The redacted copies are new files under <stage>/captures/path_redacted_447; the
original captures are not modified. Receipts never contain the original path
text, only kinds and boxes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from sample_tutorial_frames import LOCAL_ACCOUNT, PUBLIC_URL, path_hits  # noqa: E402

NEUTRAL_ROOT = '/data/example'
NEUTRAL_HOME = '/home/user'

# Broader than the sweep's patterns: also catches fragments of a wrapped or
# clipped path (".../Claude/toxo", "plasma_projects/...").
EXTRA_TOKENS = re.compile(
    r'/claude/|claude/toxo|toxoplasmaprojects|oplasmaprojects|plasmaprojects|firecuda|wd4tb|nasmnt|/mnt/|codex/|'
    r'workflowauthoring|userwalkthrough|scratchdocs|refresh2026|/users/|[a-z]:\\')

FONT_DIRS = [REPO / 'spacr/qt/resources/fonts', REPO / 'spacr/resources/font/open_sans/static',
             Path('/usr/share/fonts/truetype/dejavu'), Path('/usr/share/fonts/truetype/liberation'),
             Path('/usr/share/fonts/truetype/noto'), Path('/usr/share/fonts/truetype/cantarell'),
             Path('/usr/share/fonts/opentype/cantarell'), Path('/usr/share/fonts/truetype/ubuntu')]
FONT_NAMES = ['OpenSans-Regular.ttf', 'OpenSans-Light.ttf', 'OpenSans-SemiBold.ttf', 'OpenSans-Bold.ttf',
              'NotoSans-Regular.ttf', 'NotoSans-Medium.ttf', 'NotoSans-Bold.ttf', 'Cantarell-Regular.otf',
              'Cantarell-VF.otf', 'Ubuntu-R.ttf', 'DejaVuSans.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSansMono.ttf',
              'LiberationSans-Regular.ttf', 'LiberationMono-Regular.ttf', 'NotoSansMono-Regular.ttf',
              'UbuntuMono-R.ttf', 'UbuntuSansMono[wght].ttf', 'UbuntuSans[wdth,wght].ttf', 'Ubuntu-M.ttf']


# ---------------------------------------------------------------- text rules

def squeeze(text):
    return re.sub(r'\s+', '', text).lower()


def local_account(text):
    """True when a bare maintainer account name (not the public author/URL) is shown."""
    return bool(LOCAL_ACCOUNT.search(PUBLIC_URL.sub('', squeeze(text))))


def path_kinds(text):
    """Kinds of local path named by an OCR line (empty when neutral)."""
    kinds = {k for k, _ in path_hits([text]) if k != 'generic_home'}
    loose = squeeze(text).replace('_', '').replace('-', '')
    if EXTRA_TOKENS.search(loose):
        kinds.add('path_fragment')
    if 'maintainer_account' in kinds and not local_account(text):
        kinds.discard('maintainer_account')
    return sorted(kinds)


# A path-like token: runs of path characters that contain a slash or a
# maintainer token. Spaces end a token (file dialogs and console lines separate
# the path from surrounding words with spaces or quotes).
TOKEN = re.compile(r"(?:[A-Za-z]:\\)?[^\s'\"`=,;:()\[\]{}<>]+")
STAGE_DIR = re.compile(r'refresh[_\-]?2026[-_]?\d\d[-_]?\d\d[^/]*/?', re.I)
PREFIXES = [
    # longest first: anything up to a known stage/work root collapses to the neutral root
    re.compile(r'^.*?user[-_]?walkthrough[-_]?stage/?', re.I),
    re.compile(r'^.*?workflow[-_]?authoring[-_]?\d*/?', re.I),
    re.compile(r'^.*?scratch[-_]?docs/[^/]*/?', re.I),
    re.compile(r'^.*?refresh[_\-]?2026[-_]?\d\d[-_]?\d\d[^/]*/?', re.I),
    re.compile(r'^.*?toxoplasma[_\-]?projects/tutorials/[^/]*/?', re.I),
    re.compile(r'^.*?toxoplasma[_\-]?projects/?', re.I),
    re.compile(r'^.*?/claude/?', re.I),
    re.compile(r'^.*?/codex/(repo/)?', re.I),
    re.compile(r'^.*?/mnt/[^/]*/?', re.I),
    re.compile(r'^.*?/nas_mnt/[^/]*/?', re.I),
]
ROOT_ANCHOR = re.compile(r'/(home|mnt|nas_mnt|Users|tmp|data|media|srv|opt)/', re.I)
HOME = re.compile(r'^(.*?)/home/([^/\s]+)', re.I)


def is_path_token(token, context=None):
    """True for a path token naming a local path; context (the text before it) exempts the author's name."""
    if '/' in token or '\\' in token:
        return bool(path_kinds(token))
    return 'olafsson' in token.lower() and local_account((context or '')[-24:] + token)


PRIVATE_COMPONENT = re.compile(
    r'^(claude|toxoplasma_projects|tutorials|codex|firecuda\d*|refresh_2026[-_0-9a-z]*|'
    r'workflow-authoring[-_0-9]*|user-walkthrough-stage|scratch-docs)$', re.I)
PRIVATE_NAMES = ('claude', 'toxoplasma_projects', 'tutorials', 'codex', 'firecuda2', 'refresh_2026-09-09',
                 'workflow-authoring-20260922', 'user-walkthrough-stage', 'scratch-docs')


def extend_private(token, cut):
    """Extend a cut over following private folder names, and over a clipped start of one."""
    while cut < len(token) and token[cut] == '/':
        following = token[cut + 1:].split('/', 1)[0]
        last = '/' not in token[cut + 1:]
        clipped = last and len(following) >= 3 and any(
            name.startswith(following.lower().rstrip('.…')) for name in PRIVATE_NAMES)
        if not following or not (PRIVATE_COMPONENT.match(following) or clipped):
            break
        cut += 1 + len(following)
    return cut


def neutral_cut(token):
    """How much of a path token to replace, and with what.

    Returns (cut, replacement_prefix, kind): token[:cut] (the local root or the
    account part) is replaced by replacement_prefix; token[cut:] (the trailing
    file and folder names) is kept exactly as captured. (None, None, None) when
    the token names no local path. The kind never contains the original text.
    """
    first = token[1:].split('/', 1)[0] if token.startswith('/') else ''
    if first and PRIVATE_COMPONENT.match(first) and not ROOT_ANCHOR.match(token):
        # a wrapped console row continuing a path whose root was already replaced
        return extend_private(token, 1 + len(first)), '', 'wrapped_local_root->(removed)'
    home = HOME.search(token)
    if home and not re.search(r'/mnt/|toxoplasma|firecuda|refresh', token, re.I):
        return home.end(), home.group(1) + NEUTRAL_HOME, 'account_home->/home/user'
    for pattern in PREFIXES:
        match = pattern.match(token)
        if match:
            cut = match.end() - (1 if match.group(0).endswith('/') else 0)
            return extend_private(token, cut), NEUTRAL_ROOT, 'local_root->/data/example'
    if local_account(token):
        where = re.search(r'olafsson', token, re.I)
        if where and where.start() == 0:
            return where.end(), 'user', 'account_name->user'
    return None, None, None


def neutral_path(token):
    """Neutral replacement for a whole path token: (replacement, kind)."""
    cut, prefix, kind = neutral_cut(token)
    if cut is None:
        return None, None
    return prefix + token[cut:], kind


def path_spans(text):
    """(start, end, token) of every local path token in an OCR line string."""
    spans = []
    for match in TOKEN.finditer(text):
        token, start = match.group(0), match.start()
        # OCR can drop the space before a path ("saved to/home/..."): start at the root.
        anchor = ROOT_ANCHOR.search(token)
        if anchor and anchor.start() > 0 and ('/' not in token[:anchor.start()]
                                              or not token[:anchor.start()].strip('/')):
            start += anchor.start()
            token = token[anchor.start():]
        if is_path_token(token, text[:start]):
            account = re.search(r'olafsson', token, re.I)
            if neutral_cut(token)[0] is None and account and account.start() > 0:
                # the account inside a name ("/tmp/pytest-of-<account>/..."): replace just it
                start += account.start()
                token = token[account.start():]
            spans.append((start, match.end(), token))
    return spans


# ---------------------------------------------------------------------- OCR

_ENGINE = None


def engine(threads=3):
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        # max_side_len above 4K keeps full-frame detection at native size.
        _ENGINE = RapidOCR(intra_op_num_threads=threads, inter_op_num_threads=1, max_side_len=4000)
    return _ENGINE


def _rows(result, dx=0, dy=0, words=False):
    rows = []
    for row in result or []:
        quad = np.asarray(row[0], dtype=float) + [dx, dy]
        item = {'text': row[1], 'score': float(row[2]),
                'box': [int(quad[:, 0].min()), int(quad[:, 1].min()), int(quad[:, 0].max()), int(quad[:, 1].max())]}
        if words and len(row) > 4 and row[3]:
            chars, glyphs = row[3], row[4]
            item['chars'] = [[int(min(p[0] for p in c)) + dx, int(min(p[1] for p in c)) + dy,
                              int(max(p[0] for p in c)) + dx, int(max(p[1] for p in c)) + dy] for c in chars]
            item['glyphs'] = list(glyphs)
        rows.append(item)
    return rows


def ocr_lines(pixels, tiles=True):
    """OCR rows of a frame: native-size full frame plus half-overlapping 1080p tiles."""
    ocr = engine()
    rows = _rows(ocr(pixels)[0])
    if tiles:
        height, width = pixels.shape[:2]
        th, tw = min(1080, height), min(1920, width)
        for top in sorted({*range(0, max(1, height - th + 1), th // 2), max(0, height - th)}):
            for left in sorted({*range(0, max(1, width - tw + 1), tw // 2), max(0, width - tw)}):
                tile = np.ascontiguousarray(pixels[top:top + th, left:left + tw])
                rows.extend(_rows(ocr(tile)[0], left, top))
    return rows


def ocr_crop(pixels, box, pad=(80, 12), scale=1.0):
    """Re-read one region with per-character boxes (optionally upscaled for the detector)."""
    import cv2
    height, width = pixels.shape[:2]
    x0, y0 = max(0, box[0] - pad[0]), max(0, box[1] - pad[1])
    x1, y1 = min(width, box[2] + pad[0]), min(height, box[3] + pad[1])
    crop = np.ascontiguousarray(pixels[y0:y1, x0:x1])
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    rows = _rows(engine()(crop, return_word_box=True)[0], 0, 0, words=True)
    for row in rows:
        row['box'] = [int(round(v / scale)) + (x0 if i % 2 == 0 else y0) for i, v in enumerate(row['box'])]
        if 'chars' in row:
            row['chars'] = [[int(round(v / scale)) + (x0 if i % 2 == 0 else y0) for i, v in enumerate(c)]
                            for c in row['chars']]
    return rows


def rec_line(pixels, box, pad=(3, 3)):
    """Recognition only (no detector) of one known text line, with character boxes."""
    height, width = pixels.shape[:2]
    x0, y0 = max(0, box[0] - pad[0]), max(0, box[1] - pad[1])
    x1, y1 = min(width, box[2] + pad[0]), min(height, box[3] + pad[1])
    crop = np.ascontiguousarray(pixels[y0:y1, x0:x1])
    ocr = engine()
    rec, _ = ocr.text_rec([crop], True)
    quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    text, score, quads, glyphs = ocr.cal_rec_boxes([crop], [quad], rec)[0][:4]
    if not text:
        return []
    item = {'text': text, 'score': float(score), 'box': [x0, y0, x1, y1]}
    if quads and len(quads) == len(glyphs):
        item['chars'] = [[int(min(p[0] for p in q)), int(min(p[1] for p in q)),
                          int(max(p[0] for p in q)), int(max(p[1] for p in q))] for q in quads]
        item['glyphs'] = list(glyphs)
    return [item]


def merge_boxes(boxes):
    """Union boxes that sit on the same text row and touch or overlap."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        for box in boxes:
            for other in out:
                vertical = min(box[3], other[3]) - max(box[1], other[1])
                height = min(box[3] - box[1], other[3] - other[1])
                gap = max(box[0], other[0]) - min(box[2], other[2])
                if vertical > 0.5 * height and gap < 25:
                    other[:] = [min(box[0], other[0]), min(box[1], other[1]),
                                max(box[2], other[2]), max(box[3], other[3])]
                    changed = True
                    break
            else:
                out.append(box)
        boxes = out
    return boxes


def read_region(pixels, box):
    """Best per-character re-read of a region: the longest path-naming line overlapping it.

    The detector is sensitive to the crop, so several paddings are tried.
    """
    best = []
    for pad, scale in (((10, 10), 1.0), ((20, 30), 1.0), ((80, 12), 1.0), ((40, 20), 1.0), ((4, 16), 1.0),
                       ((20, 20), 1.5), ((20, 20), 2.0), ((60, 30), 0.75)):
        for line in ocr_crop(pixels, box, pad, scale):
            if not path_kinds(line['text']) or 'chars' not in line:
                continue
            lb = line['box']
            if lb[3] - lb[1] < 0.6 * (box[3] - box[1]):
                continue  # a sliver of the neighbouring line cut by the crop
            overlap = min(lb[2], box[2]) - max(lb[0], box[0])
            vertical = min(lb[3], box[3]) - max(lb[1], box[1])
            if overlap <= 0 or vertical < 0.5 * (lb[3] - lb[1]):
                continue
            best.append(line)
        if best and max(len(b['text']) for b in best) >= 0.9 * max(1, (box[2] - box[0]) / 10.5):
            break
    if not best:
        for line in rec_line(pixels, box):
            if path_kinds(line['text']) and 'chars' in line:
                best.append(line)
    # One line per row: keep the longest reading.
    best.sort(key=lambda line: -len(line['text']))
    chosen = []
    for line in best:
        if all(min(line['box'][2], c['box'][2]) - max(line['box'][0], c['box'][0]) <= 0
               or min(line['box'][3], c['box'][3]) - max(line['box'][1], c['box'][1]) <= 0 for c in chosen):
            chosen.append(line)
    return chosen


def detect(pixels):
    """Regions of an image whose OCR text names a local path, re-read with character boxes."""
    rows = [r for r in ocr_lines(pixels) if path_kinds(r['text'])]
    regions = []
    for box in merge_boxes([r['box'] for r in rows]):
        inside = [r for r in rows if r['box'][0] >= box[0] - 1 and r['box'][2] <= box[2] + 1
                  and r['box'][1] >= box[1] - 1 and r['box'][3] <= box[3] + 1]
        regions.append({'box': box, 'rows': inside, 'lines': read_region(pixels, box)})
    return regions


# ------------------------------------------------------------------ painter

_FONTS = None


def font_files():
    global _FONTS
    if _FONTS is None:
        _FONTS = []
        for name in FONT_NAMES:
            for folder in FONT_DIRS:
                if (folder / name).exists():
                    _FONTS.append(str(folder / name))
                    break
    return _FONTS


def glyph_mask(patch, polarity=None, level=0.3):
    """Stroke mask of text in an RGB patch and its polarity (+1 light text, -1 dark text)."""
    import cv2
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    if polarity is None:
        median = float(np.median(gray))
        polarity = 1 if np.percentile(gray, 99) - median >= median - np.percentile(gray, 1) else -1
    hat = top if polarity > 0 else black
    peak = float(np.percentile(hat, 99.5))
    mask = hat > max(12.0 * level / 0.3, level * peak)
    return mask, hat, polarity


def background(patch, mask, grow_by=2):
    """Patch with the masked glyph pixels filled from the surrounding pixels."""
    import cv2
    grow = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=grow_by)
    return cv2.inpaint(np.ascontiguousarray(patch), grow, 4, cv2.INPAINT_TELEA), grow.astype(bool)


def text_color(patch, bg, mask):
    """Colour of fully covered glyph pixels: the glyph pixels farthest from the background."""
    distance = np.abs(patch.astype(np.float32) - bg.astype(np.float32)).sum(-1)
    inside = distance[mask]
    full = mask & (distance >= np.percentile(inside, 97))
    return np.median(patch[full].reshape(-1, 3), axis=0)


def alpha_of(patch, bg, color):
    diff = patch.astype(np.float32) - bg.astype(np.float32)
    axis = np.asarray(color, np.float32) - bg.astype(np.float32)
    norm = (axis ** 2).sum(-1)
    return np.clip((diff * axis).sum(-1) / np.maximum(norm, 1.0), 0, 1)


def render_mask(text, font_path, size):
    """Anti-aliased coverage of text, its baseline row and left origin column."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, size)
    ascent, descent = font.getmetrics()
    width = int(np.ceil(font.getlength(text))) + 8
    canvas = Image.new('L', (width, ascent + descent + 8), 0)
    ImageDraw.Draw(canvas).text((4, 4 + ascent), text, font=font, fill=255, anchor='ls')
    return np.asarray(canvas, np.float32) / 255.0, 4 + ascent, 4


def fit_font(alpha, text, width_hint, fonts=None, sizes=None):
    """(score, font, size, x, baseline) whose rendering of text best matches alpha.

    x and baseline are in alpha's coordinates (the pen origin of the text).
    The app's own font (Open Sans) is preferred when it scores within 0.03 of
    the best candidate.
    """
    import cv2
    from PIL import ImageFont
    pad = 24
    blur = (0, 0)
    padded = cv2.GaussianBlur(np.pad(alpha, pad).astype(np.float32), blur, 0.8)
    results = []
    for path in fonts or font_files():
        if sizes is None:
            probe = ImageFont.truetype(path, 40).getlength(text)
            if probe <= 0:
                continue
            estimate = 40 * width_hint / probe
            candidates = [float(estimate * f) for f in np.arange(0.93, 1.071, 0.01)]
        else:
            candidates = sizes
        best = None
        for size in candidates:
            if size < 6:
                continue
            mask, base, left = render_mask(text, path, size)
            ys, xs = np.nonzero(mask > 0.02)
            if not len(ys):
                continue
            t0, t1, l0, l1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            tight = cv2.GaussianBlur(np.ascontiguousarray(mask[t0:t1, l0:l1]), blur, 0.8)
            if tight.shape[0] > padded.shape[0] or tight.shape[1] > padded.shape[1]:
                continue
            score = cv2.matchTemplate(padded, tight, cv2.TM_CCORR_NORMED)
            _, value, _, loc = cv2.minMaxLoc(score)
            # stroke weight: the rendered ink must match the observed ink (Light vs Regular)
            window = padded[loc[1]:loc[1] + tight.shape[0], loc[0]:loc[0] + tight.shape[1]]
            ratio = float(tight.sum()) / max(1e-6, float(window.sum()))
            if DEBUG_FIT:
                print('fit', Path(path).name, round(size, 2), round(value, 3), round(ratio, 3))
            # observed alpha reads about 5% light against a same-weight rendering
            rank = value - WEIGHT_PENALTY * abs(np.log(max(ratio / 0.95, 1e-6)))
            if best is None or rank > best[-1]:
                best = (float(value), path, size, loc[0] - pad + (left - l0), loc[1] - pad + (base - t0), rank)
        if best:
            results.append(best)
    if not results:
        return None
    top = max(results, key=lambda r: r[-1])
    own = [r for r in results if 'OpenSans' in r[1] and r[-1] >= top[-1] - 0.03]
    return (max(own, key=lambda r: r[-1]) if own else top)[:5]


def coverage_gamma(alpha, text, font_path, size, x, baseline):
    """Exponent g with observed alpha ~ rendered coverage ** (1 / g) (text-renderer gamma)."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, size)
    layer = Image.new('L', (alpha.shape[1], alpha.shape[0]), 0)
    ImageDraw.Draw(layer).text((x, baseline), text, font=font, fill=255, anchor='ls')
    rendered = np.asarray(layer, np.float32) / 255.0
    edge = (rendered > 0.08) & (rendered < 0.92)
    if edge.sum() < 30:
        return 1.0
    grid = np.arange(0.8, 2.61, 0.1)
    errors = [float(((alpha[edge] - rendered[edge] ** (1 / g)) ** 2).mean()) for g in grid]
    return float(grid[int(np.argmin(errors))])


def fit_color_gamma(patch, bg, text, font_path, size, x, baseline, cols):
    """Text colour and renderer gamma that best explain the patch as bg + coverage**(1/g) * (colour - bg).

    Coverage comes from rendering text with the fitted font at its fitted
    position; for each gamma the colour is solved by least squares, and the
    pair with the smallest residual wins. Low-contrast (dim) text is matched
    far better than by picking the brightest glyph pixels.
    """
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, size)
    layer = Image.new('L', (patch.shape[1], patch.shape[0]), 0)
    ImageDraw.Draw(layer).text((x, baseline), text, font=font, fill=255, anchor='ls')
    rendered = np.asarray(layer, np.float32) / 255.0
    use = (rendered > 0.02) & cols[None, :]
    if use.sum() < 30:
        return None
    b = bg.astype(np.float32)[use]
    d = patch.astype(np.float32)[use] - b
    best = None
    for g in np.arange(0.8, 2.61, 0.1):
        a = rendered[use][:, None] ** (1 / g)
        color = (a * d + a * a * b).sum(0) / max(1e-6, float((a * a).sum()))
        error = float(((d - a * (color[None, :] - b)) ** 2).sum())
        if best is None or error < best[0]:
            best = (error, np.clip(color, 0, 255), float(g))
    return best[1], best[2]


def draw_text(bg, text, font_path, size, x, baseline, color, clip, gamma=1.0):
    """bg with text composited at (x, baseline), clipped to clip=(x0, x1)."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, size)
    layer = Image.new('L', (bg.shape[1], bg.shape[0]), 0)
    ImageDraw.Draw(layer).text((x, baseline), text, font=font, fill=255, anchor='ls')
    a = (np.asarray(layer, np.float32)[..., None] / 255.0) ** (1 / gamma)
    a[:, :max(0, clip[0])] = 0
    a[:, clip[1]:] = 0
    out = bg.astype(np.float32) * (1 - a) + np.asarray(color, np.float32) * a
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), a[..., 0] > 0.02


@dataclass
class Redaction:
    box: list
    kind: str
    font: str
    size: float
    fit_score: float
    shift: int
    clipped_left: bool
    replacement_len: int
    original_len: int
    gamma: float = 1.0
    verified: bool = False


MIN_FIT = 0.85
WEIGHT_PENALTY = 0.5
DEBUG_FIT = False


def span_geometry(line, start, end):
    chars = line.get('chars') or []
    if len(chars) != len(line['text']):
        # CTC boxes align to non-space glyphs; fall back to proportional position.
        x0, x1 = line['box'][0], line['box'][2]
        n = max(1, len(line['text']))
        return (int(x0 + (x1 - x0) * start / n), int(x0 + (x1 - x0) * end / n))
    return chars[start][0], chars[end - 1][2]


def ink_columns(text, font_path, size, pen_x):
    """First and last ink column of text drawn with its pen origin at pen_x."""
    mask, _, left = render_mask(text, font_path, size)
    xs = np.nonzero((mask > 0.02).any(0))[0]
    return pen_x + xs.min() - left, pen_x + xs.max() - left + 1


def band_rows(text, font_path, size, baseline):
    """Rows (relative to the patch) that the text's glyphs can occupy, with a 4-pixel fringe."""
    mask, base, _ = render_mask(text + '/|_gjpqyÅ', font_path, size)
    ys = np.nonzero((mask > 0.02).any(1))[0]
    return int(baseline + ys.min() - base - 4), int(baseline + ys.max() - base + 5)


def ink_band(mask, c0, c1, center):
    """Rows of the text line through row center: its contiguous ink block (bridging two blank
    rows, so underscores stay attached) plus a 2-row fringe that never reaches the next line."""
    profile = mask[:, c0:c1].sum(1) > 0
    if not profile.any():
        return None
    rows = np.nonzero(profile)[0]
    center = int(rows[np.argmin(np.abs(rows - center))])
    top = bottom = center
    def inked(row):
        return 0 <= row < len(profile) and profile[row]

    # bridge up to two blank rows (an underscore below the letters stays attached)
    while any(inked(top - k) for k in (1, 2, 3)):
        top -= 1
    while any(inked(bottom + k) for k in (1, 2, 3)):
        bottom += 1
    above = top - 1
    while above >= 0 and not profile[above] and top - above <= 3:
        above -= 1
    below = bottom + 1
    while below < len(profile) and not profile[below] and below - bottom <= 3:
        below += 1
    # fringe of 2 rows, keeping at least one blank row next to a neighbouring line
    b0 = max(top - 2, above + 2 if above >= 0 and profile[above] else 0)
    b1 = min(bottom + 3, below - 1 if below < len(profile) and profile[below] else len(profile))
    return int(min(b0, top)), int(max(b1, bottom + 1))


def rule_columns(mask, band=None):
    """Columns that are straight vertical rules (field borders, dividers), not glyphs.

    With band=(b0, b1) and the full-height mask, a rule must also continue at
    least 3 rows above or below the text band; a text cursor, which is only
    as tall as the text, is then treated as a glyph and moves with the text.
    """
    if band is None:
        return mask.mean(0) >= 0.9
    b0, b1 = max(0, band[0]), min(mask.shape[0], band[1])
    full = mask[b0:b1].mean(0) >= 0.9
    above = mask[max(0, b0 - 6):b0].sum(0) >= 3
    below = mask[b1:b1 + 6].sum(0) >= 3
    return full & (above | below)


def wipe(pixels, y0, y1, band, x0, x1, polarity):
    """Clear the text in rows band=(b0, b1) and columns x0:x1 of pixels[y0:y1] (in place).

    The cleared area is filled from the surrounding background: glyphs
    anywhere in the patch are first inpainted away so they do not bleed in,
    then the band is inpainted from its (glyph-free) edges, which leaves no
    anti-aliased ghost of the old text. Straight vertical rules (field
    borders, scrollbars), horizontal borders and everything outside the band
    keep their exact values. Returns (a0, patch before, patch after, glyph
    mask) for pixels[y0:y1, a0:...].
    """
    import cv2
    width = pixels.shape[1]
    pad = 10
    a0, a1 = max(0, x0 - pad), min(width, x1 + pad)
    patch = pixels[y0:y1, a0:a1].copy()
    mask, hat, _ = glyph_mask(patch, polarity, level=0.08)
    b0, b1 = max(0, band[0]), min(patch.shape[0], band[1])
    inside = np.zeros(mask.shape, bool)
    inside[b0:b1, max(0, x0 - a0):max(0, x1 - a0)] = True
    rules = rule_columns(mask, (b0, b1))
    inside[:, rules] = False
    span = mask[:, max(0, x0 - a0):max(1, x1 - a0)]
    inside[span.mean(1) >= 0.9, :] = False  # horizontal borders keep their pixels
    glyphs = mask & ~rules[None, :]
    clean, grown = background(patch, glyphs, grow_by=3)
    fill = cv2.inpaint(np.ascontiguousarray(clean), inside.astype(np.uint8), 4, cv2.INPAINT_TELEA)
    out = patch.copy()
    out[inside] = fill[inside]
    pixels[y0:y1, a0:a1] = out
    return a0, patch, out, grown & inside


def text_run_end(mask, band, start, gap, stop=None):  # noqa: C901
    """Last ink column (exclusive) of the text run that starts at column start.

    The run ends at a blank gap wider than gap columns, a straight vertical
    rule, or stop.
    """
    rows = mask[max(0, band[0]):band[1]]
    rules = rule_columns(mask, band)
    tall = rows.mean(0) >= 0.9  # a full-height bar: a cursor if it touches the text, else a UI edge
    ink = rows.any(0) & ~rules
    stop = rows.shape[1] if stop is None else min(stop, rows.shape[1])
    last, blank = start, 0
    for column in range(start, stop):
        if rules[column] or (tall[column] and blank > max(3, gap // 4)):
            break
        if ink[column]:
            last, blank = column + 1, 0
        else:
            blank += 1
            if blank > gap:
                break
    return last


def fit_segment(alpha, line, start, count, px0, shift_total, fonts=None, sizes=None):
    """Fit count characters of line text from start; returns fit_font's tuple in patch coordinates."""
    x0, x1 = span_geometry(line, start, start + count)
    x0, x1 = x0 - shift_total - px0, x1 - shift_total - px0
    seg = alpha.copy()
    seg[:, :max(0, x0 - 3)] = 0
    seg[:, x1 + 3:] = 0
    return fit_font(seg, line['text'][start:start + count], max(8, x1 - x0), fonts=fonts, sizes=sizes)


def redact_line(pixels, line, clip_right=None):
    """Repaint the local part of every path in one OCR line of pixels (in place).

    Only the local root or account part of each path is redrawn (as
    /data/example, /home/user or user) in the fitted font; the trailing file
    and folder names and any text after them keep their captured pixels and
    are moved to follow the shorter root. Raises ValueError when a fit is
    poor, so the frame is listed for review instead of being painted badly.
    """
    records = []
    height = pixels.shape[0]
    _, ly0, _, ly1 = line['box']
    line_h = ly1 - ly0
    y0, y1 = max(0, ly0 - line_h // 2), min(height, ly1 + line_h // 2)
    shift_total = 0
    failures = []
    for start, end, token in path_spans(line['text']):
        try:
            shift = _redact_span(pixels, line, start, end, token, shift_total, y0, y1, clip_right, records)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        shift_total += shift
    if failures:
        raise SpanFailures(records, failures)
    return records


class SpanFailures(ValueError):
    """Some spans of a line could not be painted; records holds the ones that were."""

    def __init__(self, records, failures):
        super().__init__('; '.join(failures))
        self.records = records


def _redact_span(pixels, line, start, end, token, shift_total, y0, y1, clip_right, records):
    """Paint one path span (see redact_line); appends its record and returns its shift."""
    from PIL import ImageFont
    text = line['text']
    height, width = pixels.shape[:2]
    lx0, ly0, lx1, ly1 = line['box']
    cut, new_prefix, kind = neutral_cut(token)
    if cut is None:
        return 0
    if line.get('continuation') and not token.startswith('/') and new_prefix == NEUTRAL_ROOT:
        # a wrapped row continuing a path whose root the row above already shows: drop the
        # private folders here instead of repeating the neutral root
        new_prefix, kind = '', 'wrapped_local_root->(removed)'
        if cut < len(token) and token[cut] == '/':
            cut += 1
    if path_kinds(new_prefix + token[cut:]):
        raise ValueError(f'{kind} replacement would still name a local path')
    prefix_end = start + cut  # index in text of the first kept character
    tx0 = span_geometry(line, start, start + 1)[0] - shift_total
    tx1 = span_geometry(line, prefix_end - 1, prefix_end)[1] - shift_total
    prefix = text[:start].strip()
    # a path whose start is cut off (scrolled field, panel edge): the partial glyphs before
    # the first readable character belong to it
    clipped_left = ('/' in token and not token.startswith(('/', '~', '.'))
                    and len(prefix) <= 3 and ' ' not in prefix and not line.get('continuation'))
    pad = 24
    px0 = max(0, (min(tx0, lx0 - shift_total) if clipped_left else tx0) - pad)
    px1 = min(width, tx1 + pad)
    patch = pixels[y0:y1, px0:px1].copy()
    mask, hat, polarity = glyph_mask(patch)
    full_mask = mask.copy()
    cols = np.zeros(mask.shape[1], bool)
    cols[max(0, tx0 - px0 - 2):tx1 - px0 + 3] = True
    mask &= cols[None, :]
    if mask.sum() < 10:
        raise ValueError(f'no glyph pixels found for a {kind} span')
    bg, _ = background(patch, mask)
    color = text_color(patch, bg, mask)
    alpha = alpha_of(patch, bg, color)
    alpha[:, ~cols] = 0
    # Font, size and pen position from the head of the replaced part.
    head_n = min(cut, 40)
    fit = fit_segment(alpha, line, start, head_n, px0, shift_total)
    if fit is None or fit[0] < MIN_FIT:
        raise ValueError(f'font fit {None if fit is None else round(fit[0], 3)} below {MIN_FIT}')
    score, font_path, size, fx, fbase = fit
    font = ImageFont.truetype(font_path, size)
    pen_x = px0 + fx
    gamma = coverage_gamma(alpha, text[start:start + head_n], font_path, size, fx, fbase)
    joint = fit_color_gamma(patch, bg, text[start:start + head_n], font_path, size, fx, fbase, cols)
    if joint is not None:
        color, gamma = joint
    cut_x = pen_x + font.getlength(text[start:prefix_end])
    if cut > head_n:
        # the end of a long root is placed from its own fit (kerning drift)
        tail_n = min(cut, 24)
        tail = fit_segment(alpha, line, prefix_end - tail_n, tail_n, px0, shift_total,
                           fonts=[font_path], sizes=[size])
        if tail is None or tail[0] < MIN_FIT:
            raise ValueError(f'root-end fit {None if tail is None else round(tail[0], 3)} below {MIN_FIT}')
        score = min(score, tail[0])
        cut_x = px0 + tail[3] + font.getlength(text[prefix_end - tail_n:prefix_end])
    # the text band from the whole line's ink (the kept names can have descenders and
    # underscores that the replaced root lacks); the font's band if that is implausible
    font_band = band_rows(token + text[end:], font_path, size, fbase)
    lx_end = min(width, int(lx1 - shift_total + 10))
    line_mask, _, _ = glyph_mask(pixels[y0:y1, max(0, int(tx0) - 4):max(int(tx0) - 3, lx_end)], polarity)
    band = ink_band(line_mask, 0, line_mask.shape[1], (ly0 + ly1) // 2 - y0)
    if band is None or band[1] - band[0] > 1.4 * (font_band[1] - font_band[0]):
        band = font_band
    # snap the cut to the least-inked column between the last replaced and first kept glyph
    c = int(round(cut_x)) - px0
    rows = full_mask[max(0, band[0]):band[1]]
    window = range(max(0, c - 3), min(rows.shape[1], c + 4))
    if len(window):
        c = min(window, key=lambda k: (int(rows[:, k].sum()), abs(k - (cut_x - px0))))
    cut_x = px0 + c
    if clipped_left:
        look0 = max(0, lx0 - shift_total - 4 - px0)
        region = full_mask[:, look0:max(look0 + 1, int(pen_x - px0))]
        strip = region[max(0, band[0]):band[1]] & ~rule_columns(region, band)[None, :]
        inked = np.nonzero(strip.any(0))[0]
        ink0 = px0 + look0 + (int(inked.min()) if len(inked) else 0)
        start_x = ink0
    else:
        ink0 = ink_columns(text[start:start + head_n], font_path, size, pen_x)[0]
        start_x = pen_x
    new_w = font.getlength(new_prefix)
    # the kept remainder: trailing names plus the rest of this text run
    field_right = clip_right if clip_right is not None else width
    wide = pixels[y0:y1, :]
    wmask, _, _ = glyph_mask(wide[:, cut_x:min(width, field_right)], polarity, level=0.08)
    # the run ends at a word-size gap, a rule, or just past the OCR line (never the next column)
    line_right = int(lx1 - shift_total + 0.5 * size) - cut_x
    rest_end = cut_x + text_run_end(wmask, band, 0, gap=int(1.2 * size), stop=max(1, line_right))
    if rest_end - cut_x >= line_right - int(size):
        # OCR can end a line early: keep following glyphs spaced like one word run
        rest_end = cut_x + text_run_end(wmask, band, rest_end - cut_x, gap=max(3, int(0.5 * size)))
    rest_w = rest_end - cut_x
    shift = int(round(cut_x - (start_x + new_w)))  # >0 moves the remainder left
    if rest_w > 0 and rest_end - shift > field_right:
        raise ValueError('replacement is longer than the space available')
    a0, before, after, glyph_px = wipe(pixels, y0, y1, band, int(ink0) - 1, int(max(rest_end, cut_x)) + 2,
                                       polarity)
    clip = (int(ink0) if clipped_left else 0, field_right)
    painted, _ = draw_text(pixels[y0:y1].copy(), new_prefix, font_path, size, start_x, fbase, color, clip, gamma)
    pixels[y0:y1] = painted
    if rest_w > 0:
        r0, r1 = cut_x - a0, rest_end + 2 - a0
        # only the glyphs (and their anti-aliased fringe) travel, not the background under them
        glyphs = (before[:, r0:r1].astype(np.int16) - after[:, r0:r1].astype(np.int16)) * glyph_px[:, r0:r1, None]
        t0 = cut_x - shift
        target = pixels[y0:y1, t0:t0 + glyphs.shape[1]].astype(np.int16)
        pixels[y0:y1, t0:t0 + glyphs.shape[1]] = np.clip(target + glyphs[:, :target.shape[1]], 0, 255).astype(np.uint8)
    right = max(int(max(rest_end, cut_x)) + 2, int(rest_end - shift) + 2)
    records.append(Redaction(box=[int(ink0) - 12, int(y0), right + 12, int(y1)],
                             kind=kind, font=Path(font_path).name, size=round(size, 2),
                             fit_score=round(score, 3), shift=int(shift), clipped_left=clipped_left,
                             replacement_len=len(new_prefix), original_len=cut, gamma=gamma))
    return shift


def cv2_dilate(mask, iterations):
    import cv2
    return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=iterations).astype(bool)


# -------------------------------------------------------------- stage paths

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def scene_images(stage, lesson):
    """Unique (resolved path, [scene indexes]) of a staged lesson's visual.json."""
    folder = Path(stage) / 'production' / lesson
    spec = json.loads((folder / 'visual.json').read_text(encoding='utf-8'))
    images = {}
    for index, scene in enumerate(spec['scenes']):
        images.setdefault(os.path.normpath(folder / scene['image']), []).append(index)
    return spec, images


def cmd_detect(args):
    from PIL import Image
    spec, images = scene_images(args.stage, args.lesson)
    out = {'lesson': args.lesson, 'images': []}
    for path, scenes in images.items():
        pixels = np.asarray(Image.open(path).convert('RGB'))
        regions = detect(pixels)
        out['images'].append({'image': path, 'sha256': sha256(path), 'scenes': scenes, 'regions': regions})
        print(args.lesson, Path(path).name, len(regions), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'{args.lesson}.detect.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    return 0


def is_continuation(line, lines):
    """True when line starts with a path fragment and a path line sits right above it
    (a path wrapped over several rows of a narrow panel or terminal)."""
    spans = path_spans(line['text'])
    if not spans or spans[0][0] > 2 or spans[0][2].startswith('/'):
        return False
    x0, y0, _, y1 = line['box']
    height = y1 - y0
    for other in lines:
        ox0, oy0, _, oy1 = other['box']
        if (other is not line and 0 < y0 - oy0 < 1.8 * height and abs(ox0 - x0) < 2 * height
                and '/' in other['text']):
            return True
    return False


def redact_image(pixels, regions):
    """Redact all regions of one frame. Returns (new pixels, records, problems)."""
    out = pixels.copy()
    records, problems = [], []
    all_lines = [line for region in regions for line in (region.get('lines') or region.get('rows') or [])]
    for region in regions:
        box = region['box']
        seen = (region.get('rows') or []) + (region.get('lines') or [])
        if seen and not any(path_kinds(r['text']) for r in seen):
            continue  # flagged by a broader earlier rule (e.g. the organism name), not a path
        lines = [line for line in region.get('lines') or []
                 if line['box'][3] - line['box'][1] >= 0.6 * (box[3] - box[1])]
        lines = [line for line in lines if path_kinds(line['text'])] or read_region(out, box)
        if not lines:
            lines = [dict(row) for row in region.get('rows', [])]
        if not lines:
            problems.append({'box': region['box'], 'problem': 'region could not be re-read'})
            continue
        for line in lines:
            if not path_kinds(line['text']):
                continue  # detected under a broader earlier rule; names no local path
            line['continuation'] = is_continuation(line, all_lines)
            if not path_spans(line['text']):
                problems.append({'box': line['box'], 'problem': 'path kind without a path token'})
                continue
            try:
                records.extend(redact_line(out, line))
            except SpanFailures as exc:
                records.extend(exc.records)
                problems.append({'box': line['box'], 'problem': str(exc)})
            except ValueError as exc:
                problems.append({'box': line['box'], 'problem': str(exc)})
    return out, records, problems


def changed_outside(before, after, records):
    """Number of changed pixels outside every recorded box (must be zero)."""
    changed = np.any(before != after, axis=-1)
    for record in records:
        x0, y0, x1, y1 = record.box
        changed[max(0, y0):y1, max(0, x0):x1] = False
    return int(changed.sum())


def residual_paths(pixels, records):
    """Path-naming OCR lines left around the painted boxes."""
    left = []
    for record in records:
        for pad in ((20, 30),):
            for line in ocr_crop(pixels, record.box, pad):
                if path_kinds(line['text']):
                    left.append({'box': line['box'], 'kinds': path_kinds(line['text'])})
    return left


def sweep_rows(pixels, scales=(1.0, 2 / 3)):
    """Path-naming OCR rows seen the way the acceptance sweep sees a frame (lifted, 1080p tiles),
    at 4K and at the 1440p web-copy scale; OCR finds different lines at different scales."""
    import cv2
    from sample_tutorial_frames import OCR_OVERLAP, OCR_TILE, lifted
    from PIL import Image
    rows = []
    for scale in scales:
        view = lifted(Image.fromarray(pixels))
        if scale != 1.0:
            view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        height, width = view.shape[:2]
        for top in range(0, max(1, height - OCR_OVERLAP[1]), OCR_TILE[1] - OCR_OVERLAP[1]):
            for left in range(0, max(1, width - OCR_OVERLAP[0]), OCR_TILE[0] - OCR_OVERLAP[0]):
                tile = np.ascontiguousarray(view[top:top + OCR_TILE[1], left:left + OCR_TILE[0]])
                for r in _rows(engine()(tile)[0], left, top):
                    if path_kinds(r['text']):
                        r['box'] = [int(round(v / scale)) for v in r['box']]
                        rows.append(r)
    return rows


def inside_box(box, outer, tolerance=4):
    return (box[0] >= outer[0] - tolerance and box[1] >= outer[1] - tolerance
            and box[2] <= outer[2] + tolerance and box[3] <= outer[3] + tolerance)


def sliver(box, painted):
    """A short OCR box straddling a painted line (halves of two rows read as one)."""
    line_h = (painted[3] - painted[1]) / 2
    overlap_x = min(box[2], painted[2]) - max(box[0], painted[0])
    overlap_y = min(box[3], painted[3]) - max(box[1], painted[1])
    return overlap_x > 0 and overlap_y > 0 and (box[3] - box[1]) < 0.75 * line_h


def redact_until_clean(before, regions, rounds=3):
    """Redact detected regions, then re-read the result as the sweep does and around every
    painted box, redacting anything newly found; returns (after, records, issues, residual)."""
    after, records, issues = before, [], []
    for _ in range(rounds):
        if regions:
            after, found, problems = redact_image(after, regions)
            records += found
            issues += problems
        rows = sweep_rows(after) + [dict(r, text='') for r in residual_paths(after, records)]
        # a sliver read inside a painted line is OCR noise on text we set ourselves
        rows = [r for r in rows if not any(inside_box(r['box'], rec.box) or sliver(r['box'], rec.box)
                                           for rec in records)]
        boxes = merge_boxes([r['box'] for r in rows])
        # a box that could not be painted before is not retried forever
        tried = {tuple(p['box']) for p in issues}
        regions = [{'box': b, 'rows': [r for r in rows if r['text'] and r['box'][0] >= b[0] - 1
                                       and r['box'][2] <= b[2] + 1]} for b in boxes if tuple(b) not in tried]
        if not regions:
            break
    residual = [{'box': r['box'], 'kinds': path_kinds(r['text']) if r['text'] else r.get('kinds', [])}
                for r in rows]
    return after, records, issues, residual


def cmd_apply(args):
    from PIL import Image
    detected = json.loads((args.out / f'{args.lesson}.detect.json').read_text(encoding='utf-8'))
    stage = Path(args.stage)
    folder = stage / 'production' / args.lesson
    target = stage / 'captures' / 'path_redacted_447' / args.lesson
    spec, images = scene_images(stage, args.lesson)
    names = {}
    frames, problems = [], []
    for item in detected['images']:
        source = item['image']
        if sha256(source) != item['sha256']:
            raise ValueError(f'{Path(source).name} changed since detection')
        before = np.asarray(Image.open(source).convert('RGB'))
        after, records, issues, residual = redact_until_clean(before, item['regions'])
        if not records and not residual and not issues:
            print(args.lesson, Path(source).name, 'clean', flush=True)
            continue
        outside = changed_outside(before, after, records)
        entry = {'frame': Path(source).name, 'capture_folder': Path(source).parent.name, 'source': source,
                 'scenes': item['scenes'], 'source_sha256': item['sha256'],
                 'redactions': [vars(r) | {'verified': not residual} for r in records],
                 'changed_pixels_outside_boxes': outside, 'residual_path_lines': residual,
                 'problems': issues}
        if outside or residual or issues:
            problems.append(entry)
        if records and not outside:
            name = Path(source).name
            if names.get(name, source) != source:
                name = f'{Path(source).parent.name}__{name}'
            names[name] = source
            target.mkdir(parents=True, exist_ok=True)
            destination = target / name
            Image.fromarray(after).save(destination, compress_level=6)
            entry['redacted_image'] = str(destination.relative_to(stage))
            entry['redacted_sha256'] = sha256(destination)
            if args.review:
                save_review(before, after, records, args.review / f'{args.lesson}__{name}')
        frames.append(entry)
        print(args.lesson, Path(source).name, len(records), 'redactions',
              'PROBLEM' if entry in problems else 'ok', flush=True)
    receipt = {'lesson': args.lesson, 'frames': frames, 'problem_frames': len(problems)}
    (args.out / f'{args.lesson}.apply.json').write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    if problems and not args.allow_problems:
        print(args.lesson, 'has', len(problems), 'problem frames; visual.json unchanged', flush=True)
        return 2
    # point visual.json at the redacted copies
    by_source = {os.path.normpath(f['source']): f for f in frames if 'redacted_image' in f}
    changed = 0
    for scene in spec['scenes']:
        resolved = os.path.normpath(folder / scene['image'])
        frame = by_source.get(resolved)
        if frame:
            scene['image'] = os.path.relpath(stage / frame['redacted_image'], folder)
            scene['capture_sha256'] = frame['redacted_sha256']
            changed += 1
    spec['path_redaction_447'] = {
        'tool': 'tools/tutorials/redact_frame_paths.py',
        'receipt': 'tools/tutorials/evidence/2026-09-26-path-redactions.json',
        'sources': {f['redacted_sha256']: f['source_sha256'] for f in frames if 'redacted_sha256' in f}}
    (folder / 'visual.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(args.lesson, changed, 'scenes now use redacted frames', flush=True)
    return 0


def save_review(before, after, records, stem):
    """Before/after strips of every painted box for visual review."""
    from PIL import Image
    stem.parent.mkdir(parents=True, exist_ok=True)
    strips = []
    for record in records:
        x0, y0, x1, y1 = record.box
        x0, y0 = max(0, x0 - 30), max(0, y0 - 14)
        x1, y1 = min(before.shape[1], x1 + 30), min(before.shape[0], y1 + 14)
        strips.append(np.vstack([before[y0:y1, x0:x1], np.full((4, x1 - x0, 3), 255, np.uint8),
                                 after[y0:y1, x0:x1], np.full((10, x1 - x0, 3), 128, np.uint8)]))
    width = max(s.shape[1] for s in strips)
    sheet = np.vstack([np.pad(s, ((0, 0), (0, width - s.shape[1]), (0, 0))) for s in strips])
    Image.fromarray(sheet).save(f'{stem}.review.png')


def cmd_verify(args):
    from PIL import Image
    bad = 0
    for path in args.image:
        pixels = np.asarray(Image.open(path).convert('RGB'))
        hits = [r for r in ocr_lines(pixels) if path_kinds(r['text'])]
        bad += bool(hits)
        print(Path(path).name, 'PATH' if hits else 'clean', [r['box'] for r in hits], flush=True)
    return 1 if bad else 0


def cmd_record(args):
    """Add one lesson's redactions, media hashes and checks to the evidence receipt."""
    stage = Path(args.stage)
    folder = stage / 'production' / args.lesson
    applied = json.loads((args.out / f'{args.lesson}.apply.json').read_text(encoding='utf-8'))
    master = folder / 'video' / f'{args.lesson}_silent.mp4'
    web = stage / 'web-renditions' / args.lesson / 'video' / master.name
    rendition = json.loads((stage / 'web-renditions' / args.lesson / 'rendition-checks.json').read_text())
    sweep = json.loads(args.sweep.read_text(encoding='utf-8'))
    browser = {}
    for case, report in (('sentence_cues', stage / 'browser' / args.lesson / 'en-af_heart-sentence-cues'),
                         ('web_rendition', stage / 'browser-web' / args.lesson / 'en-af_heart')):
        report = report / 'playback-checks.json'
        result = json.loads(report.read_text())
        browser[case] = {'passed': result.get('passed'), 'report_sha256': sha256(report)}
        if case == 'web_rendition' and result.get('loaded_video_sha256') not in (None, sha256(web)):
            raise ValueError('web playback report is for another rendition')
    keep = ('box', 'kind', 'font', 'size', 'fit_score', 'shift', 'clipped_left', 'gamma', 'verified')
    entry = {
        'lesson': args.lesson,
        'frames': [{'frame': f['frame'], 'capture_folder': f['capture_folder'], 'scenes': f['scenes'],
                    'source_sha256': f['source_sha256'], 'redacted_image': f.get('redacted_image'),
                    'redacted_sha256': f.get('redacted_sha256'),
                    'changed_pixels_outside_boxes': f['changed_pixels_outside_boxes'],
                    'redactions': [{k: r[k] for k in keep} for r in f['redactions']]}
                   for f in applied['frames']],
        'media': {'master_sha256': sha256(master), 'poster_sha256': sha256(folder / 'poster.jpg'),
                  'web_sha256': sha256(web), 'web_bytes': web.stat().st_size,
                  'web_rendition_accepted': rendition.get('accepted'),
                  'visual_json_sha256': sha256(folder / 'visual.json'),
                  'narration_and_timing': 'unchanged (audio/en/af_heart.m4a and .json reused)',
                  'af_heart_timing_sha256': sha256(folder / 'audio/en/af_heart.json')},
        'checks': {'browser': browser,
                   'frame_sweep': {'images_checked': sweep['images_checked'],
                                   'path_offenders': sweep['path_offenders'],
                                   'frames_sampled': {'web_copy': args.frames, 'master': args.master_frames},
                                   'poster_checked': True}},
        'published': False,
    }
    import fcntl
    evidence = args.evidence
    lock = open(str(evidence) + '.lock', 'w')  # parallel lessons update one receipt
    fcntl.flock(lock, fcntl.LOCK_EX)
    receipt = json.loads(evidence.read_text(encoding='utf-8')) if evidence.exists() else {
        'item': '447', 'date': '2026-09-26',
        'scope': ('Local path text in tutorial scene frames replaced by a neutral path (/data/example/..., '
                  '/home/user) in the fitted font, size and colour over an inpainted background; trailing '
                  'file/folder names and following text keep their captured pixels. Original path text is not '
                  'recorded here. Masters re-rendered with unchanged narration timing; poster and 1440p web '
                  'copy regenerated; not published.'),
        'tool': 'tools/tutorials/redact_frame_paths.py', 'lessons': {}, 'rerecord': {}}
    receipt['lessons'][args.lesson] = entry
    evidence.write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    print(args.lesson, entry['media']['master_sha256'], entry['media']['web_sha256'], flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    det = sub.add_parser('detect')
    det.add_argument('--lesson', required=True)
    det.add_argument('--stage', type=Path, required=True)
    det.add_argument('--out', type=Path, required=True)
    det.set_defaults(func=cmd_detect)
    app = sub.add_parser('apply')
    app.add_argument('--lesson', required=True)
    app.add_argument('--stage', type=Path, required=True)
    app.add_argument('--out', type=Path, required=True, help='Folder holding <lesson>.detect.json')
    app.add_argument('--review', type=Path, help='Write before/after strips here')
    app.add_argument('--allow-problems', action='store_true')
    app.set_defaults(func=cmd_apply)
    rec = sub.add_parser('record')
    rec.add_argument('--lesson', required=True)
    rec.add_argument('--stage', type=Path, required=True)
    rec.add_argument('--out', type=Path, required=True)
    rec.add_argument('--sweep', type=Path, required=True)
    rec.add_argument('--frames', type=int, default=12, help='frames sampled from the web copy')
    rec.add_argument('--master-frames', type=int, default=6, help='frames sampled from the 4K master')
    rec.add_argument('--evidence', type=Path, default=HERE / 'evidence/2026-09-26-path-redactions.json')
    rec.set_defaults(func=cmd_record)
    ver = sub.add_parser('verify')
    ver.add_argument('--image', nargs='+', required=True)
    ver.set_defaults(func=cmd_verify)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
