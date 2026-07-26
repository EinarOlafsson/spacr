# map_barcodes - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **map_barcodes_01** - Barcode read resolving to a single well of a plate
2. **map_barcodes_02** - Sequence spooling out of a DNA helix into plate wells
3. **map_barcodes_03** - Row barcode x column barcode intersecting on one well
4. **map_barcodes_04** - Every well carrying its own barcode
5. **map_barcodes_05** - Lookup table: barcodes on the left wired to wells on the right
6. **map_barcodes_06** - Pipette dispensing a barcoded sample into a well
7. **map_barcodes_07** - One read, three barcode segments, three plate coordinates
8. **map_barcodes_08** - Plate as a map with a location pin dropped on the matched well
9. **map_barcodes_09** - Barcode as the key that opens one well
10. **map_barcodes_10** - Sequencer cluster tile mapped onto the plate

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/masks_measure_group.py`
