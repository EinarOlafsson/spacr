# pca - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **pca_01** - Many feature columns collapsed onto two derived axes.
2. **pca_02** - A scree plot: component variance falling away, the elbow ringed.
3. **pca_03** - A loadings biplot: the features drawn as vectors off the origin.
4. **pca_04** - The cloud's own long and short directions, drawn through it.
5. **pca_05** - The measured frame turned into the derived one.
6. **pca_06** - Points dropped out of many dimensions onto one flat plane.
7. **pca_07** - How much of the picture each component is: shares of one whole.
8. **pca_08** - Two groups that only come apart once the axes are derived.
9. **pca_09** - One component alone: the cloud folded down onto a single direction.
10. **pca_10** - Features on wildly different scales brought to one before decomposing.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_pca_tabulate_dict_outliers_dose.py`
