# curate - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **curate_01** - A brush pushing a wrong mask boundary back onto the cell.
2. **curate_02** - A pointer dragging one square control handle of a contour into place.
3. **curate_03** - A hand-drawn stroke cutting one wrongly merged blob into two objects.
4. **curate_04** - Two fragments pushed together and stitched into a single object.
5. **curate_05** - A broken track re-linked by hand across the gap it lost.
6. **curate_06** - An eraser lifting a false object off the field, the real ones left alone.
7. **curate_07** - One mask kept and one thrown out: a tick on the good outline, a cross on the bad.
8. **curate_08** - Stepping back through revisions: an undo arc over the mask, versions as dots.
9. **curate_09** - The corrected object beside the log line the correction was written into.
10. **curate_10** - The finished mask signed off: an approval stamp pressed onto the object.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_imaging_explore.py`
