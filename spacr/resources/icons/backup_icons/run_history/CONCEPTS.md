# run_history - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **run_history_01** - A magnifier held over a deep stack of finished run records.
2. **run_history_02** - A timeline of past runs, each marked passed or failed.
3. **run_history_03** - A search field over the run rows it matched.
4. **run_history_04** - A drawer of run folders with one pulled up out of the file.
5. **run_history_05** - A clock wound backwards over the list of runs already done.
6. **run_history_06** - A ledger of past runs with a pass/fail column down the right.
7. **run_history_07** - A calendar month with the days that were run marked off.
8. **run_history_08** - A run log unrolled, with the warning line flagged in it.
9. **run_history_09** - How long every past run took, plotted run after run.
10. **run_history_10** - Boxed-up past runs on the shelf, the newest one still open.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_jobs_models.py`
