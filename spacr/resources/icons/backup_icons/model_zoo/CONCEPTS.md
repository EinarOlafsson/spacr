# model_zoo - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **model_zoo_01** - A shelf of model cards with the middle one pulled out.
2. **model_zoo_02** - A model downloading out of the cloud onto the local shelf.
3. **model_zoo_03** - A model card stamped verified after its checksum matched.
4. **model_zoo_04** - A leaderboard: three models ranked by how well they benched.
5. **model_zoo_05** - A wall of model tiles, each a different kind of model.
6. **model_zoo_06** - One model from the shelf tried out on three of your own fields.
7. **model_zoo_07** - A model being slotted into an empty bay.
8. **model_zoo_08** - Two shelves: the segmentation models above, the classifiers below.
9. **model_zoo_09** - A model card timed on the bench by a stopwatch.
10. **model_zoo_10** - Models fanned out like a hand of cards, one lifted to be chosen.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_jobs_models.py`
