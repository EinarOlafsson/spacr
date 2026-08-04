# external_masks - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **external_masks_01** - The pair that arrives: a field of cells, and its labels already solid.
2. **external_masks_02** - The segmentation step struck out: images and labels go straight to measure.
3. **external_masks_03** - An outline sheet drawn elsewhere, dropped onto the image beneath it.
4. **external_masks_04** - Labels that arrive already numbered, each object carrying its own id.
5. **external_masks_05** - A label from somewhere else flying in and landing on the cell it fits.
6. **external_masks_06** - Every image tile matched to the mask file that came with it.
7. **external_masks_07** - A ready-made label being measured, not drawn: calipers across a blob.
8. **external_masks_08** - Two layers handed over together: pixels below, their label map on top.
9. **external_masks_09** - Supplied labels going straight into the measurement table.
10. **external_masks_10** - An image and its label map travelling together into a spaCR project.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_data_io.py`
