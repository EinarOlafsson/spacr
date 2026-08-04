# lineage - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **lineage_01** - Concentric rings: the pathogen solid at the core, inside the vacuole, inside the cell.
2. **lineage_02** - A containment tree: the cell node above, its nucleus and vacuole hanging under it.
3. **lineage_03** - Three frames nested one inside the next, the innermost filled solid.
4. **lineage_04** - A wedge cut out of the cell, showing the nucleus and the vacuole inside it.
5. **lineage_05** - 'Is inside' read as a chain: the pathogen belongs to the vacuole belongs to the cell.
6. **lineage_06** - The cell taken apart: its nucleus and vacuole pulled out on dashed leaders.
7. **lineage_07** - An indented object list: the cell, the nucleus under it, the pathogen under that.
8. **lineage_08** - Brackets inside brackets: the outer holds the inner holds the object.
9. **lineage_09** - One cell holding two vacuoles, and each vacuole holding its own pathogen.
10. **lineage_10** - Zooming inward: the cell, blown up into the vacuole it contains.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_imaging_explore.py`
