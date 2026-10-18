from PIL import Image
Image.MAX_IMAGE_PIXELS = None
D = '/home/olafsson/Downloads/source/'
O = '/mnt/firecuda2/Claude/toxoplasma_projects/presentation/images/manuscript/'
f = 9922 / 1100
jobs = {'fig1A_concept': ('figures/Figure_1_spaCR_and_spaCRPower_workflows.png', (15, 5, 1095, 175)),
        'fig1DE_power': ('figures/Figure_1_spaCR_and_spaCRPower_workflows.png', (540, 185, 1095, 405)),
        'figS1G_power': ('supplementary_figures/Figure_S1_spaCRPower_simulation_and_empirical_screen_comparison.png', (30, 660, 365, 910))}
for n, (src, b) in jobs.items():
    im = Image.open(D + src).convert('RGB')
    c = im.crop(tuple(round(v * f) for v in b))
    if c.width > 3000: c = c.resize((3000, round(c.height * 3000 / c.width)), Image.LANCZOS)
    c.save(O + n + '.png'); print(n, c.size)
