# layer_viewer - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **layer_viewer_01** - Three sheets floating apart: the points, the mask contour, the image.
2. **layer_viewer_02** - A layer list: three rows, each with its own eye, the last one switched off.
3. **layer_viewer_03** - The stack pulled apart, dashed guides keeping the sheets in register.
4. **layer_viewer_04** - The mask overlay peeled back off the image at one corner.
5. **layer_viewer_05** - Seen edge-on: an eye looking down through three separate sheets.
6. **layer_viewer_06** - One frame carrying every kind at once: the contour, the points and an ROI box.
7. **layer_viewer_07** - A slider dimming the sheet on top of the one below it.
8. **layer_viewer_08** - A deck of sheets with one pulled out sideways to be worked on.
9. **layer_viewer_09** - Sheets fanned out like cards, each carrying a different kind of mark.
10. **layer_viewer_10** - One world: a pin driven through all three sheets, holding them in register.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_imaging_explore.py`
