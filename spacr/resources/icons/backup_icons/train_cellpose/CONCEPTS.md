# train_cellpose -- candidate concepts

1. **train_cellpose_01** - pencil hand-annotating an outline (making the training label)
2. **train_cellpose_02** - image/label pairs: raw tile in, mask tile out
3. **train_cellpose_03** - loss curve falling over epochs
4. **train_cellpose_04** - layered neural net learning cell -> mask
5. **train_cellpose_05** - epoch loop: iterate until the mask fits
6. **train_cellpose_06** - Cellpose flow field: vectors converging on the cell centre
7. **train_cellpose_07** - ground truth vs prediction outlines, mismatch area hatched
8. **train_cellpose_08** - a labelled training set (deck of tiles) feeding the model
9. **train_cellpose_09** - graduation cap on a cell: teaching the model
10. **train_cellpose_10** - model chip with a training feedback loop

_All 1024x1024 RGBA, white on transparent, house style (flat, thin strokes + solid fills), matching `plaque.png` / `measure.png`._

_`_sheet_dark.png` shows the PNGs as they are. `_sheet_light.png` re-inks the same alpha masks in dark grey, because pure-white artwork is invisible on a light background -- that is the point of the second sheet._
