# train_compare - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **train_compare_01** - Three losses falling together on one axis, one settling lower.
2. **train_compare_02** - Loss falling and accuracy rising on the same frame.
3. **train_compare_03** - Two runs' curves, and beside them the setting that differed.
4. **train_compare_04** - A cursor dropped at one epoch, reading every run's loss there.
5. **train_compare_05** - The gap between two runs' curves, filled in to show how far apart.
6. **train_compare_06** - One run stopped early at its best epoch while the other ran on.
7. **train_compare_07** - A legend of three runs, tied to the three traces beside it.
8. **train_compare_08** - Overlaid curves above, and the score each run finished on.
9. **train_compare_09** - A whole sweep of runs fanning apart from the same starting loss.
10. **train_compare_10** - Two runs whose curves cross, the crossing point ringed.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_jobs_models.py`
