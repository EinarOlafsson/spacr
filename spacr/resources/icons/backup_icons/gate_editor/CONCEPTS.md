# gate_editor - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **gate_editor_01** - A threshold swept across a histogram by hand; the bars beyond it are kept.
2. **gate_editor_02** - A polygon closed vertex by vertex round the cloud, a grab handle on each.
3. **gate_editor_03** - Gates chained: each hand-drawn boundary sits inside the one before it.
4. **gate_editor_04** - The gating hierarchy: each gate a row, each row the fraction that survived it.
5. **gate_editor_05** - A rectangle dragged across a two-parameter scatter, both extents read.
6. **gate_editor_06** - A gate is named: the shape carries the label it becomes a filter under.
7. **gate_editor_07** - A predicate, not a list of objects: the same drawn shape laid onto the next plate.
8. **gate_editor_08** - One drawn shape, and every open view narrowed to what is inside it.
9. **gate_editor_09** - The strategy as a chain: each canvas shows only what the last gate kept.
10. **gate_editor_10** - Re-drawn, a gate replaces its older self instead of stacking on it.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_trellis_gate_feature_napari.py`


Flow-plot set, drawn from a real FSC-A / SSC-A density
plot the user supplied. Where 01-10 each pick one idea
about gating, these reproduce the picture a flow user
already recognises.

11. **gate_editor_11** - Density cloud, a hand-drawn polygon gate and the percentage inside it -- the flow plot in full.
12. **gate_editor_12** - Cloud, axes and gate, without the percentage.
13. **gate_editor_13** - The gate filled: what it selects, solid over the cloud.
14. **gate_editor_14** - Half-drawn -- the polygon still open, vertex handles showing the hand.
15. **gate_editor_15** - A gate drawn inside a gate, on what the first one kept.
16. **gate_editor_16** - Cloud and gate with no axes; the most whitespace and the best of these at 16 px.


Flow-plot set, drawn from a real FSC-A / SSC-A density
plot the user supplied. Where 01-10 each pick one idea
about gating, these reproduce the picture a flow user
already recognises.

11. **gate_editor_11** - Density cloud, a hand-drawn polygon gate and the percentage inside it -- the flow plot in full.
12. **gate_editor_12** - Cloud, axes and gate, without the percentage.
13. **gate_editor_13** - The gate filled: what it selects, solid over the cloud.
14. **gate_editor_14** - Half-drawn -- the polygon still open, vertex handles showing the hand.
15. **gate_editor_15** - A gate drawn inside a gate, on what the first one kept.
16. **gate_editor_16** - Cloud and gate with no axes; the most whitespace and the best of these at 16 px.
