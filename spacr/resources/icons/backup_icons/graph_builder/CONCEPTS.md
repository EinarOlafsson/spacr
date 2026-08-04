# graph_builder - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **graph_builder_01** - Three column chips on the left, and the chart they build on the right.
2. **graph_builder_02** - A cursor dragging a column chip into a dashed drop target.
3. **graph_builder_03** - Table header cells lifted out of the table and flown onto the axes.
4. **graph_builder_04** - Four empty shelves, one for each encoding: x, y, colour and size.
5. **graph_builder_05** - Bars snapping into an empty chart frame out of a stack of blocks.
6. **graph_builder_06** - A column dropped on the facet shelf splits one chart into four small ones.
7. **graph_builder_07** - A column on the size shelf: the points come out big and small.
8. **graph_builder_08** - A column on the colour shelf: the marks split into filled and hollow.
9. **graph_builder_09** - An empty chart blueprint: dashed slots waiting on the x and the y axis.
10. **graph_builder_10** - The finished chart with the chips that made it still docked on its axes.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_imaging_explore.py`
