# model_compare - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **model_compare_01** - One cell wearing two rival contours, solid and broken, that disagree.
2. **model_compare_02** - The same field wiped down the middle: one model's outlines each side.
3. **model_compare_03** - Two touching cells: one model calls them one object, the other two.
4. **model_compare_04** - Same three cells, but one model found a fourth: the extra one ringed.
5. **model_compare_05** - The sliver where the two outlines disagree, filled in solid.
6. **model_compare_06** - One cell, a tight outline and a loose one, the gap between measured.
7. **model_compare_07** - A cell that one model outlined and the other missed entirely.
8. **model_compare_08** - The same cells outlined twice, and the two object counts beneath.
9. **model_compare_09** - Objects matched up one to one between the two labellings.
10. **model_compare_10** - A checkerboard of the same field, alternating whose outline is drawn.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_jobs_models.py`
