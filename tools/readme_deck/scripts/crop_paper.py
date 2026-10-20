from PIL import Image
Image.MAX_IMAGE_PIXELS = None
D = '/home/olafsson/Downloads/source/'
O = '/mnt/firecuda2/Claude/toxoplasma_projects/presentation/images/manuscript/'
F2 = 'figures/Figure_2_spaCR_screen_experimental_workflow.png'
F3 = 'figures/Figure_3_GFP_TSG101_single_cell_phenotype_classification_2.png'
F4 = 'figures/Figure_4_XGBoost_feature_attribution_and_MaxViT_saliency_2.png'
S7 = 'supplementary_figures/Figure_S7_gene_level_OLS_diagnostics_and_permutation_2.png'
jobs = {  # boxes in 1100-px-wide preview coordinates (Fig 3 is small: native coords, scale 1)
    'fig2_mlr': (F2, (705, 280, 1098, 494), None),
    'fig2_classify': (F2, (515, 10, 1095, 275), None),
    'figS7_volcano': (S7, (583, 1125, 1100, 1402), None),
    'fig3A_segmentation': (F3, (0, 0, 780, 326), 1.0),
    'fig3C_accuracy': (F3, (190, 326, 540, 578), 1.0),
    'fig4BCD_features': (F4, (228, 8, 1098, 232), None),
    'fig4E_crops': (F4, (8, 238, 395, 610), None),
    'fig4F_saliency': (F4, (400, 238, 835, 610), None),
}
for n, (src, b, sc) in jobs.items():
    im = Image.open(D + src).convert('RGB')
    f = sc or im.width / 1100
    c = im.crop(tuple(round(v * f) for v in b))
    if c.width > 3000: c = c.resize((3000, round(c.height * 3000 / c.width)), Image.LANCZOS)
    c.save(O + n + '.png'); print(n, c.size)
