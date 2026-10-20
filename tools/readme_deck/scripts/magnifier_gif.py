"""Animated Make Masks magnifier for the deck.

Replays one real Cellpose-SAM result (stock cpsam_v2 on a cell-mask field) the way the magnifier's whole-image
scope and press-and-drag work: a box follows the cursor, shows the region zoomed with the model's outlines
(blue), and every object the box fully covers is added to the mask, where it stays (coloured fill).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import find_boundaries

D = '/mnt/firecuda2/Claude/toxoplasma_projects/presentation/images/magnifier_gif/'
# Dense field: the parasite-free cell-mask crop used on the Nuclei-from-Cell-Mask slide, with stock
# Cellpose-SAM's result on it (model_predictions/*_stock.npz), which is what the magnifier runs by default.
PR = '/mnt/firecuda2/Claude/toxoplasma_projects/presentation/model_predictions/'
img = np.load(PR + 'nuclei_from_cellmask_v1.npz')['image'].astype(np.float32)
masks = np.load(PR + 'nuclei_from_cellmask_v1_stock.npz')['pred'].astype(np.int32)
lo, hi = np.percentile(img, (1, 99.7))
g = np.clip((img - lo) / (hi - lo), 0, 1) ** 0.8
H, W = g.shape
S = 640 / W                      # display scale
BOX = 250                        # region the model box covers, image px
ZOOM = 1.5
BG = (14, 18, 23)
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
FONT_S = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)

ids = [i for i in np.unique(masks) if i]
rng = np.random.default_rng(3)
palette = np.array([(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178),
                    (213, 94, 0), (204, 121, 167), (120, 200, 120), (255, 130, 130), (170, 140, 255)])
color = {i: palette[rng.integers(len(palette))] for i in ids}
bbox = {}
for i in ids:
    ys, xs = np.nonzero(masks == i)
    bbox[i] = (ys.min(), ys.max(), xs.min(), xs.max())
edges = find_boundaries(masks, mode='inner')

# serpentine path through the field
path = []
for k, yy in enumerate(np.linspace(H * 0.45, H - 125, 3)):
    xs = np.linspace(110, W - 110, 18)
    if k % 2: xs = xs[::-1]
    path += [(yy, xx) for xx in xs]
    if k < 2:
        path += [(yy + t * (H * 0.55 - 125) / 2 / 5, xs[-1]) for t in range(1, 5)]

base_rgb = np.stack([g] * 3, -1)
committed = set()


def render_field():
    out = base_rgb.copy()
    if committed:
        sel = np.isin(masks, list(committed))
        cmap = np.zeros((masks.max() + 1, 3));
        for i in committed: cmap[i] = color[i] / 255
        out[sel] = out[sel] * 0.45 + cmap[masks[sel]] * 0.55
        eb = edges & sel
        out[eb] = cmap[masks[eb]]
    return out


frames = []
for fi, (cy, cx) in enumerate(path + [path[-1]] * 3):
    cy, cx = int(cy), int(cx)
    y0, y1 = max(0, cy - BOX // 2), min(H, cy + BOX // 2)
    x0, x1 = max(0, cx - BOX // 2), min(W, cx + BOX // 2)
    inside = [i for i in ids if bbox[i][0] >= y0 and bbox[i][1] < y1 and bbox[i][2] >= x0 and bbox[i][3] < x1]
    field = render_field()
    im = Image.fromarray((field * 255).astype(np.uint8)).resize((round(W * S), round(H * S)), Image.BILINEAR)
    # lens: region zoomed, model outlines in blue for objects fully inside, committed ones keep their colour
    reg = field[y0:y1, x0:x1].copy()
    lab = masks[y0:y1, x0:x1]
    blue = find_boundaries(np.where(np.isin(lab, inside), lab, 0), mode='inner')
    reg[blue] = (0.35, 0.62, 1.0)
    lw, lh = round((x1 - x0) * S * ZOOM), round((y1 - y0) * S * ZOOM)
    lens = Image.fromarray((reg * 255).astype(np.uint8)).resize((lw, lh), Image.NEAREST)
    px, py = round(cx * S - lw / 2), round(cy * S - lh / 2)
    px = min(max(px, 0), im.width - lw); py = min(max(py, 0), im.height - lh)
    im.paste(lens, (px, py))
    dr = ImageDraw.Draw(im)
    dr.rectangle((px, py, px + lw - 1, py + lh - 1), outline=(91, 156, 245), width=3)
    ax, ay = round(cx * S), round(cy * S)
    dr.polygon([(ax, ay), (ax + 13, ay + 11), (ax + 6, ay + 12), (ax + 9, ay + 20), (ax + 6, ay + 21), (ax + 3, ay + 13), (ax, ay + 17)], fill='white', outline='black')
    committed.update(inside)
    canvas = Image.new('RGB', (im.width, im.height + 44), BG)
    canvas.paste(im, (0, 44))
    cd = ImageDraw.Draw(canvas)
    cd.text((10, 11), 'Magnifier: press and drag to add objects', fill=(232, 237, 242), font=FONT)
    cd.text((canvas.width - 160, 13), f'objects added: {len(committed)}', fill=(154, 166, 178), font=FONT_S)
    frames.append(canvas.quantize(colors=80, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))

frames[0].save(D + 'magnifier.gif', save_all=True, append_images=frames[1:], duration=[140] * (len(frames) - 1) + [3000], loop=0, optimize=True)
frames[-1].convert('RGB').save(D + 'magnifier_last.png')
print(len(frames), 'frames', len(committed), 'of', len(ids), 'objects')
