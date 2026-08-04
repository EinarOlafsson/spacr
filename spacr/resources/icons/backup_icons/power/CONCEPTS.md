# power - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **power_01** - A power curve with the required n dropped onto the x axis.
2. **power_02** - Two effect sizes, two curves, two required n's on the same axis.
3. **power_03** - The effect size itself: two humps with the gap measured between them.
4. **power_04** - Cells per well: a well packed with cells feeding a rising curve.
5. **power_05** - A slider set to an effect size, the curve above answering with n.
6. **power_06** - Error bars shrinking as n grows until one clears the effect line.
7. **power_07** - How many wells: a run of wells counted off under one brace.
8. **power_08** - A trade-off curve: pick an effect size, read the n it costs.
9. **power_09** - Bars of growing n against a detectable line: the first to clear wins.
10. **power_10** - Nested power contours with the chosen n and effect size pinned on.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_results_qc.py`
