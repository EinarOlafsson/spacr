"""Crops of the Regression app screenshots (fractions of the full 4800x2800 window)."""
from PIL import Image
R = '/mnt/firecuda2/Claude/toxoplasma_projects/presentation/images/regression_app/'
jobs = {'crop_models': ('settings_models', (0.085, 0.455, 0.715, 0.862)),
        'crop_volcano': ('results_volcano', (0.625, 0.145, 0.995, 0.80)),
        'crop_cells': ('cells_for_gene', (0.172, 0.145, 0.625, 0.80)),
        'crop_diagnostics': ('diagnostics', (0.172, 0.395, 0.625, 0.715)),
        'crop_diag_plots': ('diagnostics', (0.628, 0.19, 0.995, 0.62))}
for out, (src, (a, b, c, d)) in jobs.items():
    im = Image.open(R + src + '.png').convert('RGB'); W, H = im.size
    cr = im.crop((round(a * W), round(b * H), round(c * W), round(d * H)))
    if cr.width > 3000: cr = cr.resize((3000, round(cr.height * 3000 / cr.width)), Image.LANCZOS)
    cr.save(R + out + '.png'); print(out, cr.size)
