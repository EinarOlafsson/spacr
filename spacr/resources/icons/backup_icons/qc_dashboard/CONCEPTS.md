# qc_dashboard - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **qc_dashboard_01** - Four check panels, three ticked and one crossed, under one verdict.
2. **qc_dashboard_02** - Three small dials read at once, answered by one big dial.
3. **qc_dashboard_03** - A checklist of QC rows, ticked or crossed, totalled at the bottom.
4. **qc_dashboard_04** - One frame holding four unlike mini-panels and a verdict badge.
5. **qc_dashboard_05** - Three checks feeding a traffic light that shows the overall call.
6. **qc_dashboard_06** - Separate ticks converging into a single overall tick.
7. **qc_dashboard_07** - A pass badge assembled from four separate checks around it.
8. **qc_dashboard_08** - A scorecard: four segments filled bar one, and the resulting call.
9. **qc_dashboard_09** - One dial split into a pass side and a fail side, needle on pass.
10. **qc_dashboard_10** - A deck of check cards with the summed verdict on the front one.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_results_qc.py`
