# image_scatter - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **image_scatter_01** - A point on the axes blown up into a framed crop of the cell behind it.
2. **image_scatter_02** - A cursor resting on a point, the cell popping up beside it.
3. **image_scatter_03** - The points themselves are crops: thumbnails sitting where their cells fall.
4. **image_scatter_04** - A lens over the cloud, magnifying one point into the cell it stands for.
5. **image_scatter_05** - Crops on the left becoming points on the axes on the right.
6. **image_scatter_06** - A lasso thrown round a few points, and the crop of one of them.
7. **image_scatter_07** - A window split in two: the plot on one side, the picked cell previewed on the other.
8. **image_scatter_08** - Two clusters, one crop opened out of each, so the groups can be compared.
9. **image_scatter_09** - A gallery strip under the plot, each crop tied back to its point.
10. **image_scatter_10** - The plot binned into tiles, each tile showing the cell that stands for it.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_imaging_explore.py`
