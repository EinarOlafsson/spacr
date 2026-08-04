# feature_dict - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **feature_dict_01** - A column name, and the meaning it resolves to.
2. **feature_dict_02** - A search run over the column names, and the one it lands on.
3. **feature_dict_03** - The dictionary itself: the reference, opened.
4. **feature_dict_04** - What is this column? -- asked of the header of a results table.
5. **feature_dict_05** - Looked up by the idea, not by the name.
6. **feature_dict_06** - One entry's card: the name, then the fields the name is explained by.
7. **feature_dict_07** - A term list with its sections tabbed down the edge.
8. **feature_dict_08** - The same name checked against the objects it actually exists for.
9. **feature_dict_09** - Which channel the feature is about, picked out of the stack.
10. **feature_dict_10** - A long name pulled apart into the object, the channel and the statistic.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_pca_tabulate_dict_outliers_dose.py`
