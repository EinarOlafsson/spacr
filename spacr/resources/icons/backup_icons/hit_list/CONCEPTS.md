# hit_list - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **hit_list_01** - Rows sorted longest first, the leaders starred.
2. **hit_list_02** - A volcano with the two far corners picked out as hits.
3. **hit_list_03** - A podium: the top three hits ranked one, two, three.
4. **hit_list_04** - A ranked list cut by the FDR line: kept above, dropped below.
5. **hit_list_05** - A long list filtered down through a funnel into a shortlist.
6. **hit_list_06** - Guides agreeing on the top hit and scattering on the one below.
7. **hit_list_07** - One hit lifted out of the ranking for a closer look.
8. **hit_list_08** - Effect sizes fanned either side of no-effect, biggest at the top.
9. **hit_list_09** - The ranking column being sorted, top to bottom.
10. **hit_list_10** - A shortlist flagged out of the ranking, hit by hit.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_results_qc.py`
