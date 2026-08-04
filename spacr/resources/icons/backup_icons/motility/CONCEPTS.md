# motility - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **motility_01** - A parasite at the head of the circular trail it has glided along.
2. **motility_02** - A trajectory with a speed dial reading its velocity off underneath.
3. **motility_03** - Equal time steps marked along one track: wide gaps where it moves fast.
4. **motility_04** - Two tracks compared: one long and straight, one short and tangled.
5. **motility_05** - A wandering path with the straight start-to-end displacement struck across it.
6. **motility_06** - The moving cell above, its speed plotted against time below.
7. **motility_07** - Every track redrawn from one origin: displacement spokes of unequal length.
8. **motility_08** - The path length read off against a graduated scale bar beneath it.
9. **motility_09** - The cell carrying its velocity vector: one big arrow off the object it moves.
10. **motility_10** - Same elapsed time, two lanes: the fast object far past the slow one.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_imaging_explore.py`
