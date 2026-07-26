# logo_spacr -- 20 candidate marks

Twenty different metaphors for the same idea, not one drawing restyled twenty times.
White on transparent, flat, 1024x1024 RGBA, same treatment as `plaque.png` / `measure.png`.
Numbering matches `_sheet_dark.png`, `_sheet_light.png` and `_sheet_small.png`.

 1. **wordmark_lockup** -- Monoline geometric 'spaCR' wordmark under a grid-cell mark; the 'a' bowl is a cell.
 2. **monogram_cr** -- 'CR' monogram -- the two capitals of spaCR, the C read as a membrane around a nucleus. **[16px-safe]**
 3. **well_plate** -- Microtitre plate with the notched A1 corner; four hit wells filled solid.
 4. **dish_coords** -- Petri dish used as a coordinate frame -- crosshair axes and ticks locate one colony.
 5. **cas9_guide** -- Cas9 as a notched clamp biting a DNA duplex, the sgRNA threaded through the body.
 6. **helix_roundel** -- A double helix contained in a disc -- the sequencing/CRISPR half of the name, as a roundel. **[16px-safe]**
 7. **scissors_dna** -- The edit: scissors closing on a duplex with the double-strand break already open.
 8. **screen_array** -- The screen: a 3x3 array of cells with one filled hit picked out by a selection bracket.
 9. **pin_on_cell** -- A map pin dropped on a cell -- the phenotype, located in space. **[16px-safe]**
10. **radar_scan** -- A spatial scan: concentric range rings, a sweep wedge, and detected objects as dots.
11. **hex_tissue** -- A segmented monolayer as packed hexagonal cells, the centre one solid (called).
12. **nucleus_orbit** -- Spatial relationships: a solid nucleus with satellites travelling two orbits. **[16px-safe]**
13. **objective_slide** -- The instrument: a microscope objective over a gridded slide carrying one cell.
14. **barcode_cell** -- Barcode to phenotype: sequencing bars on the left wired across into a cell on the right.
15. **aperture_c** -- The minimal mark: one heavy C-membrane around a solid nucleus. Built for 16px. **[16px-safe]**
16. **plasmid_guide** -- The vector: a plasmid ring carrying one heavy highlighted guide cassette with an arrow.
17. **phenotype_space** -- Phenotype space: a measured scatter with one population gated and filled.
18. **roi_box** -- The spatial primitive: a cell inside its measured bounding box with dimension ticks.
19. **z_stack** -- The data: a stack of imaged planes, only the front one resolved into a cell.
20. **guide_entry** -- The perturbation: a guide strand threaded through a gap in the membrane to the nucleus.

## Sheets

* `_sheet_dark.png` -- all 20 on #14161a, as shipped.
* `_sheet_light.png` -- all 20 on #f5f6f8, **re-inked dark**. The artwork itself is white-on-transparent and is invisible on a light background (the known open bug); the sheet re-inks the alpha mask so the form can still be judged.
* `_sheet_small.png` -- every variant at 16 / 32 / 48 px, nearest-neighbour zoomed 3x. This is the favicon and title-bar case and the hardest constraint.

Marks tagged **[16px-safe]** still read at 16x16: 02 monogram_cr, 06 helix_roundel, 09 pin_on_cell, 12 nucleus_orbit, 15 aperture_c.

Regenerate: `QT_QPA_PLATFORM=offscreen python3 _generators/logo_spacr.py`
