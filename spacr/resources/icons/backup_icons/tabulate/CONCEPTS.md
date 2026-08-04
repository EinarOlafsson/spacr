# tabulate - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **tabulate_01** - The pivot: keys down the side, keys across the top, summaries between.
2. **tabulate_02** - A stack of raw rows collapsed into one summary cell.
3. **tabulate_03** - Margins: the totals along the far edge, ruled off from the body.
4. **tabulate_04** - One group, three statistics of it: the count, the middle and the spread.
5. **tabulate_05** - The long raw table beside the short summary it becomes.
6. **tabulate_06** - A row key and a column key crossing on the one cell they share.
7. **tabulate_07** - Rows braced into groups, each brace giving out one row.
8. **tabulate_08** - A group opened into the subgroups it is made of.
9. **tabulate_09** - A hole in the grid: the combination no row was found for.
10. **tabulate_10** - The summary above, and the chart of it drawn column for column below.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_pca_tabulate_dict_outliers_dose.py`
