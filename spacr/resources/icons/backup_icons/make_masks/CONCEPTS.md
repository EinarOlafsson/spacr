# make_masks - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **make_masks_01** - Paintbrush laying mask paint inside a cell outline (half painted)
2. **make_masks_02** - Eraser lifting a wrong region off an existing mask
3. **make_masks_03** - Scalpel splitting an over-merged doublet along a cut line
4. **make_masks_04** - Merging two masks into one: shared boundary dissolves
5. **make_masks_05** - Polygon lasso with draggable vertex handles and a cursor
6. **make_masks_06** - Edit history: ghost previous outlines under an undo arc
7. **make_masks_07** - Paint-bucket flood fill pouring mask into a cell
8. **make_masks_08** - Mask layer lifted off the image layer (layer stack)
9. **make_masks_09** - Pencil hand-drawing a contour: dashed ahead, solid behind
10. **make_masks_10** - Magic wand selecting a cell: marching ants plus sparkles

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/masks_measure_group.py`
