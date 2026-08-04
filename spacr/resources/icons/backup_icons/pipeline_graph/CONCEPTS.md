# pipeline_graph - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **pipeline_graph_01** - A three-step chain whose last artefact was never produced.
2. **pipeline_graph_02** - One source forking to two products, one of them flagged bad.
3. **pipeline_graph_03** - Two branches converging on a join that is now out of date.
4. **pipeline_graph_04** - File to file: two artefacts made, the third one still blank.
5. **pipeline_graph_05** - The link between two steps snapped in half.
6. **pipeline_graph_06** - Along one branch: made, gone stale, never made at all.
7. **pipeline_graph_07** - A node marked for rebuild, with everything below it waiting on it.
8. **pipeline_graph_08** - A node carrying a clock: its inputs moved on without it.
9. **pipeline_graph_09** - A step struck out, and everything it fed left dangling.
10. **pipeline_graph_10** - Provenance: one output traced back up to the two inputs that made it.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_jobs_models.py`
