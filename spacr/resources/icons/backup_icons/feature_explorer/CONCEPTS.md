# feature_explorer - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **feature_explorer_01** - Every row a feature, every feature two humps, sorted by how far apart they sit.
2. **feature_explorer_02** - One feature in detail: the distance between the two classes, measured.
3. **feature_explorer_03** - The table's column heads lifted off and stacked into ranked order.
4. **feature_explorer_04** - Hundreds of columns scored, and the two that actually separate ringed.
5. **feature_explorer_05** - Separation on a bounded scale: a coin flip at one end, nearly gateable at the other.
6. **feature_explorer_06** - The blind spot: same centre, different spread - scores nothing, obviously informative.
7. **feature_explorer_07** - Each ranked row also says which of the two classes is the higher one.
8. **feature_explorer_08** - Ranked by separation, not by size: the small clean feature beats the big blurred one.
9. **feature_explorer_09** - What the score means: the two classes interleaved, and the two fully ordered.
10. **feature_explorer_10** - Every continuous column scored against the one class column.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_trellis_gate_feature_napari.py`
