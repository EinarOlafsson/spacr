# measure - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **measure_01** - Ruler laid over a cell (the existing icon's idea)
2. **measure_02** - Vernier calipers closed onto a cell
3. **measure_03** - Object above a calibrated scale bar
4. **measure_04** - Bounding box with width and height dimension callouts
5. **measure_05** - Intensity contour rings with a radial profile line
6. **measure_06** - Population histogram whose bars are stacks of cells
7. **measure_07** - Micrometer screw gauge closed on a cell
8. **measure_08** - Crosshair readout: object position on calibrated axes
9. **measure_09** - Extracted feature table beside the object
10. **measure_10** - Protractor measuring an angle across the object

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/masks_measure_group.py`
