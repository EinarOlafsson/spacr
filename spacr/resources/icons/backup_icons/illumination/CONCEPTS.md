# illumination - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **illumination_01** - A vignetted field's contour rings, corrected to a field with none.
2. **illumination_02** - The intensity profile across the field, pulled down onto a level line.
3. **illumination_03** - Field divided by its flat-field: two tiles, a division sign, a clean tile.
4. **illumination_04** - A corner falling away: arcs crowding one corner of the frame.
5. **illumination_05** - A buckled surface over the field, pressed down into a flat plane.
6. **illumination_06** - One field shown twice across a hard edge: vignetted, then flat.
7. **illumination_07** - The field estimated from the plate itself: many wells averaged into one.
8. **illumination_08** - Spots shrinking toward the corners, then all the same size again.
9. **illumination_09** - Well means arching up in the middle of the plate, then levelled.
10. **illumination_10** - The lamp's hot spot pooled in the centre of the frame.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_data_io.py`


Pulsar-map set, added on request: thin rays of unequal
length from one source, after the Voyager plaque.

11. **illumination_11** - 24 thin rays of unequal length from one point.
12. **illumination_12** - 36 rays, denser, closest to the Voyager plaque's crowding.
13. **illumination_13** - 16 rays, sparser, the most legible at 16 px.
14. **illumination_14** - 24 rays inside the field circle they are correcting.
15. **illumination_15** - An uneven field: the long rays all fall to one side.
16. **illumination_16** - 28 rays leaving a solid core.
