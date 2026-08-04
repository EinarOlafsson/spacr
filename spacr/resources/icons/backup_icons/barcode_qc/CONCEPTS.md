# barcode_qc - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **barcode_qc_01** - A rank-abundance curve falling through the abundance cutoff.
2. **barcode_qc_02** - Counts sorted tallest first, the tail below the cutoff crossed off.
3. **barcode_qc_03** - The mapping run came back clean: a barcode with a tick on it.
4. **barcode_qc_04** - A funnel: all the reads narrowing to the ones that pass.
5. **barcode_qc_05** - Reads stacked onto one barcode: how deep the coverage went.
6. **barcode_qc_06** - The count histogram with the knee where background stops.
7. **barcode_qc_07** - A threshold handle dragged along the abundance axis.
8. **barcode_qc_08** - Reads splitting into the mapped pile and the unmapped one.
9. **barcode_qc_09** - Two barcodes side by side: one abundant, one down at background.
10. **barcode_qc_10** - A waterline across the barcodes: above it kept, below it dropped.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_results_qc.py`
