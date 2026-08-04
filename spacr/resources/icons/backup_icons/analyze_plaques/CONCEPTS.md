# analyze_plaques - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **analyze_plaques_01** - A solid lawn with three plaque-shaped holes knocked clean through it.
2. **analyze_plaques_02** - A square monolayer of packed cells with three bare gaps in it.
3. **analyze_plaques_03** - Each clearing ringed by a detection marker, with the running count below.
4. **analyze_plaques_04** - The clearings redrawn largest to smallest along a size axis.
5. **analyze_plaques_05** - A scan line across the field, its trace dropping to the floor over each clearing.
6. **analyze_plaques_06** - Control against treated: one field full of clearings, one with a single clearing.
7. **analyze_plaques_07** - The raw field on one side, its clearings picked out as solid shapes on the other.
8. **analyze_plaques_08** - One clearing sized from its centre out: a radius arrow and the sizing rings.
9. **analyze_plaques_09** - The clearings lifted out of the lawn into a row of separate objects.
10. **analyze_plaques_10** - A coverage bar reading off how much of the monolayer has been cleared.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_imaging_explore.py`
