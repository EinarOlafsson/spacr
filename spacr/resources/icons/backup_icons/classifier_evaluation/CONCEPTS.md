# classifier_evaluation - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **classifier_evaluation_01** - An ROC curve bulging away from the chance diagonal.
2. **classifier_evaluation_02** - Calibration: predicted against observed, points sagging off the line.
3. **classifier_evaluation_03** - Cross-validation folds: a different block held out in every row.
4. **classifier_evaluation_04** - A test set sealed away from training and only opened to score.
5. **classifier_evaluation_05** - Two score humps pulling apart either side of the decision point.
6. **classifier_evaluation_06** - A precision-recall curve falling away as recall is pushed.
7. **classifier_evaluation_07** - Leakage: the same item found in both the train and the test block.
8. **classifier_evaluation_08** - Held-out accuracy plate by plate, one plate falling off the line.
9. **classifier_evaluation_09** - A decision boundary with the few points landing on the wrong side.
10. **classifier_evaluation_10** - The same curve redrawn for every fold: how much the score wobbles.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_results_qc.py`
