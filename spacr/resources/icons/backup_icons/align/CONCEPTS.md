# align - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **align_01** - Two camera fields overlapping, the strip they share filled solid.
2. **align_02** - The shift measured from one speck to the same speck in the next tile.
3. **align_03** - One peak on the shift map: crosshairs on the offset that won.
4. **align_04** - A registration target where two tile corners have to land together.
5. **align_05** - The stage's path strung through the tile centres in visiting order.
6. **align_06** - A cell cut in half by the seam, whole again once the halves register.
7. **align_07** - The mosaic filling in tile by tile, one tile held at a time.
8. **align_08** - Corner brackets pulling a loose tile square onto the grid.
9. **align_09** - A heap of skewed tiles above, squared into a clean grid below.
10. **align_10** - Tiles laid down one after another, every overlap solid where they meet.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_data_io.py`
