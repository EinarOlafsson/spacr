# outliers - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **outliers_01** - A tight crowd, and the one that is nowhere near it.
2. **outliers_02** - A plate of wells all alike, and the one that is not.
3. **outliers_03** - Past the whisker: a robust fence built out of the crowd itself.
4. **outliers_04** - A tolerance ellipse drawn round the crowd, and the point left outside.
5. **outliers_05** - A column written, not a row dropped.
6. **outliers_06** - The bulk of the distribution, and the one stranded far out in the tail.
7. **outliers_07** - Two questions, two answers: the odd object and the odd well.
8. **outliers_08** - How far out it is: the distance from the crowd measured.
9. **outliers_09** - Belonging to neither group -- odd without being extreme.
10. **outliers_10** - A whole well shifted together, though not one object in it looks wrong.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_pca_tabulate_dict_outliers_dose.py`
