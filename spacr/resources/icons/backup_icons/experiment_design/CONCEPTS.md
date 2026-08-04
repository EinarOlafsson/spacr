# experiment_design - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **experiment_design_01** - Condition marks assigned across the plate, with a key beside it.
2. **experiment_design_02** - The plate cut into three treatment blocks, each braced as one.
3. **experiment_design_03** - Controls parked in the outside columns, samples in the middle.
4. **experiment_design_04** - One condition chip copied into three replicate wells.
5. **experiment_design_05** - A pen dropping the next condition into an empty well.
6. **experiment_design_06** - Condition chips waiting in a palette, one dragged onto the plate.
7. **experiment_design_07** - Randomisation: two crossed arrows shuffling the marks about.
8. **experiment_design_08** - A half-authored plate: assigned wells solid, the rest still blank.
9. **experiment_design_09** - A dose series laid along a row, the mark growing well by well.
10. **experiment_design_10** - The finished layout exported as a table for the pipeline.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_results_qc.py`
