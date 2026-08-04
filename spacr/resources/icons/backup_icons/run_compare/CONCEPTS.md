# run_compare - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **run_compare_01** - Two run records side by side, the one row that differs flagged in both.
2. **run_compare_02** - A diff gutter: rows removed on the left, added on the right.
3. **run_compare_03** - The same three counts from two runs, plotted at different heights.
4. **run_compare_04** - Two hit lists as overlapping sets: shared hits, and hits unique to one.
5. **run_compare_05** - A balance with a run record in each pan, tipped to the better one.
6. **run_compare_06** - Two run cards swapped back and forth against each other.
7. **run_compare_07** - A delta between one run stacked over the other.
8. **run_compare_08** - Two ranked hit lists, with one hit that jumped up the order.
9. **run_compare_09** - One settings sheet cut down the middle: before on the left, after on the right.
10. **run_compare_10** - Two records checked line for line: three agree, one does not.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_jobs_models.py`
