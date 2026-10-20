"""Paper panels for the deck: generous margins so no axis text is clipped; panel letters covered with a box
matching the background. Boxes are in the coordinates of the crop, as fractions (x0, y0, x1, y1)."""
import json, sys
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
D = '/home/olafsson/Downloads/source/'
SC = '/tmp/claude-1000/-mnt-firecuda2-Claude-repo/d76975b5-7a10-46fa-bf01-dd62a1467de7/scratchpad/paper/'
O = '/mnt/firecuda2/Claude/toxoplasma_projects/presentation/images/manuscript/'
SRC = {'F1': D + 'figures/Figure_1_spaCR_and_spaCRPower_workflows.png', 'F2': D + 'figures/Figure_2_spaCR_screen_experimental_workflow.png',
       'F3': SC + 'fig3hi.png', 'F4': D + 'figures/Figure_4_XGBoost_feature_attribution_and_MaxViT_saliency_2.png',
       'F5': SC + 'fig5hi.png', 'S7': D + 'supplementary_figures/Figure_S7_gene_level_OLS_diagnostics_and_permutation_2.png'}
# box in 1100-px-wide preview coordinates of the full page
JOBS = json.load(open(sys.argv[1]))
cache = {}
for name, j in JOBS.items():
    src = j['src']
    if src not in cache: cache[src] = Image.open(SRC[src]).convert('RGB')
    im = cache[src]; f = im.width / 1100
    x0, y0, x1, y1 = j['box']
    c = im.crop((round(x0 * f), round(y0 * f), round(x1 * f), round(y1 * f)))
    dr = ImageDraw.Draw(c)
    for cov in j.get('cover', []):
        a, b, cc, d = cov[:4]
        bx = (max(0, round((a - x0) * f)), max(0, round((b - y0) * f)), min(c.width - 1, round((cc - x0) * f)), min(c.height - 1, round((d - y0) * f)))
        if len(cov) > 4: fill = cov[4]
        else:  # background colour = median of the pixels just outside the box
            import numpy as np
            arr = np.asarray(c); ring = []
            for yy in (max(0, bx[1] - 3), min(c.height - 1, bx[3] + 3)): ring.append(arr[yy, bx[0]:bx[2] + 1])
            for xx in (max(0, bx[0] - 3), min(c.width - 1, bx[2] + 3)): ring.append(arr[bx[1]:bx[3] + 1, xx])
            fill = tuple(int(v) for v in np.median(np.concatenate(ring), axis=0))
        dr.rectangle(bx, fill=fill)
    if c.width > 3000: c = c.resize((3000, round(c.height * 3000 / c.width)), Image.LANCZOS)
    c.save(O + name + '.png')
    # review copy with a 10-px preview grid
    r = c.copy(); r.thumbnail((1400, 1400)); g = ImageDraw.Draw(r); s = r.width / (x1 - x0)
    for k in range(0, int(x1 - x0) + 1, 10):
        g.line((k * s, 0, k * s, 6), fill='red'); 
        if k % 50 == 0: g.text((k * s + 1, 7), str(int(x0 + k)), fill='red')
    for k in range(0, int(y1 - y0) + 1, 10):
        g.line((0, k * s, 6, k * s), fill='red')
        if k % 50 == 0: g.text((8, k * s), str(int(y0 + k)), fill='red')
    r.save(SC + 'rev_' + name + '.jpg')
    print(name, c.size)
