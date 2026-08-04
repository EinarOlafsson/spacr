# anndata_export - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **anndata_export_01** - The h5ad layout: the X block with var along the top and obs down the side.
2. **anndata_export_02** - A sparse matrix: most of the grid empty, a scatter of cells filled.
3. **anndata_export_03** - Rows are cells, columns are features: the two margins spelled out.
4. **anndata_export_04** - One file, three layers: the matrix with obs and var stacked behind it.
5. **anndata_export_05** - The matrix handed over and read back as a cloud of cells.
6. **anndata_export_06** - The file's tree: one root holding X, obs and var.
7. **anndata_export_07** - The whole matrix folded down into one file on disk.
8. **anndata_export_08** - The shape being written: so many cells by so many features.
9. **anndata_export_09** - One cell's row of features lifted out of the matrix as a vector.
10. **anndata_export_10** - The measurement rows turned on their side into the X matrix.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_data_io.py`
