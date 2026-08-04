# small multiples - candidate concepts

**The app key is `trellis`**, so whichever you pick installs as
`resources/icons/trellis.png`. This folder exists under the name you
actually use; `trellis/` holds the identical set under the key.

1. **small_multiples_01** - A 3x3 of identical frames, the same mark standing at a different height.
2. **small_multiples_02** - Shared axes: one y axis and one x axis serving the whole grid.
3. **small_multiples_03** - Two-way faceting: condition across the top, plate down the side.
4. **small_multiples_04** - One reference level ruled straight through every panel.
5. **small_multiples_05** - Every panel prints its n, so a panel of three cannot pass for a panel of thousands.
6. **small_multiples_06** - The same shape at two very different levels, comparable only because the axis is shared.
7. **small_multiples_07** - A long strip of levels wrapping onto the next row.
8. **small_multiples_08** - An empty panel is still drawn: 'measured, nothing survived' stays its own picture.
9. **small_multiples_09** - A brush on one panel picks the same objects out of every other panel.
10. **small_multiples_10** - The same chart, many times over: identical frames stepping back into depth.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_trellis_gate_feature_napari.py`
