# mask - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **mask_01** - Split field: raw objects on one side, contours on the other
2. **mask_02** - Trained model turning an image into contours (network to cell)
3. **mask_03** - Watershed: touching cells separated inside a field of view
4. **mask_04** - Instance labels: every object filled and given an ID
5. **mask_05** - Active contour shrinking onto a cell
6. **mask_06** - Nested classes: cell, nucleus and pathogen vacuole
7. **mask_07** - Pixel-wise labelling: contour rasterised onto the image grid
8. **mask_08** - Intensity threshold turning a histogram into a binary silhouette
9. **mask_09** - Automatic detection: corner brackets snapped onto each object
10. **mask_10** - Stencil sheet with the objects punched out

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/masks_measure_group.py`
