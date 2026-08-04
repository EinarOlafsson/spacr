# logo_spacr - 30 candidates for the spaCR mark itself

Brief, in the user's words: **streamlined, simple, intuitive, elegant**.

The mark has to work as an app icon, a favicon and a splash, so the governing
constraint is **16 px**, not 1024 px.  At 16 px one unit of a 1024 design is
1/64 px, so a stroke only survives if it is roughly 0.07-0.10 of the canvas.
Group A is drawn to that budget; group B inherits more structure from the
current `logo_spacr.png` and is honestly weaker small - the verdicts below say
which is which, measured off `_sheet_small.png`, not guessed.

All 30 are 1024x1024 RGBA.  Groups A-C are pure white on transparent (house
style); group D ships colour.  No gradients, no bevels, no shadows anywhere.

The **16 px** column: `yes` = reads at 16, `soft` = present but mushy at 16 and
clean by 32, `32 px` = a smudge at 16, use 32 px as the floor.

---

## A. New concepts - 10, black and white

Ten different ideas about what the mark could be, each one idea only.

| # | file | 16 px | idea |
|---|------|-------|------|
| 1 | `concept_01_c_dot` | yes | One open C-membrane around one solid nucleus - the C of spaCR and a cell read as the same shape. |
| 2 | `concept_02_quadrant` | soft | A field split in four with exactly one cell filled - the screen, and its hit. |
| 3 | `concept_03_orbit` | soft | One body, one ring - the smallest statement of a spatial relationship. |
| 4 | `concept_04_plate` | yes | A plate with its notched A1 corner and one well called; the only rectilinear form in the set. |
| 5 | `concept_05_matrix` | soft | A 3x3 array with one well called - the screen as a rhythm, with no container at all. |
| 6 | `concept_06_split` | yes | One object, half raw and half resolved - segmentation in a single gesture. |
| 7 | `concept_07_crescent` | yes | The organism as one shape: a solid crescent, nothing else. |
| 8 | `concept_08_pin` | yes | A map pin whose counter is the cell - a phenotype, located. |
| 9 | `concept_09_cut` | yes | One body, one clean cut, the halves slid apart - the edit. |
| 10 | `concept_10_focus` | soft | Two corner brackets and the one object between them - a region of interest. |

## B. Refinements of the existing logo - 10, black and white

The current `logo_spacr.png` needs five things to read: an irregular cell
outline, a spatial grid, a nucleus, two vacuoles and a fan of arcs.  Every
variant below keeps the identity - the irregular silhouette, the grid, the
offset nucleus - and removes or reverses the rest.  In all of them the grid is
cut clear of the bodies, which is what the parent achieves by greying its grid
down; here it is done with negative space instead, so nothing depends on alpha.

| # | file | 16 px | idea |
|---|------|-------|------|
| 11 | `variant_01_grid_nucleus` | 32 px | The parent stripped to three parts: cell outline, spatial grid, solid nucleus. |
| 12 | `variant_02_grid_vacuole` | 32 px | As 11, plus the one parasitophorous vacuole - keeps the biology, drops the rest. |
| 13 | `variant_03_bare` | yes | Cell outline and nucleus only, drawn heavy - the parent with the grid removed. |
| 14 | `variant_04_cross` | soft | The grid reduced to a single cross - two lines instead of four. |
| 15 | `variant_05_solid` | yes | The cell as solid mass with the nucleus knocked out of it. |
| 16 | `variant_06_solid_grid` | 32 px | Solid cell with the grid and nucleus knocked out as negative space. |
| 17 | `variant_07_satellite` | yes | Outline, nucleus and one satellite - the parent's asymmetry without its grid. |
| 18 | `variant_08_regular` | 32 px | The silhouette regularised: the same cell, calmer, with grid and nucleus. |
| 19 | `variant_09_grid_hit` | 32 px | One square of the spatial grid filled solid, nucleus reversed out of it - the hit, inside the cell. |
| 20 | `variant_10_ticks` | yes | The grid implied by ticks straddling the membrane rather than drawn across it. |

## C. Thinner lines - 1, black and white

| # | file | 16 px | idea |
|---|------|-------|------|
| 21 | `thin_01_grid_nucleus` | 32 px | Variant 11 redrawn at 60% stroke weight throughout. |

Worth comparing against 11 side by side: it is more delicate at splash size and
measurably worse below 48 px.  It is the direction the current logo already
takes, and the reason the current logo dissolves in a title bar.

## D. Colour - 9

Four single-colour, five multi-colour.  Every ink is chosen so **one file works
on both backgrounds** - relative luminance held between 0.13 and 0.27, giving at
least 3:1 against the dark `#14161a` *and* the light `#f5f6f8`:

| ink | hex | vs dark | vs light |
|-----|-----|---------|----------|
| teal   | `#0E9488` | 4.84:1 | 3.46:1 |
| indigo | `#5B63D6` | 3.62:1 | 4.62:1 |
| coral  | `#E0533A` | 4.72:1 | 3.55:1 |
| slate  | `#6F7B8A` | 4.21:1 | 3.98:1 |

| # | file | colours | 16 px | idea |
|---|------|---------|-------|------|
| 22 | `colour_01_teal_c_dot` | one | yes | Concept 1 in a single teal. |
| 23 | `colour_02_teal_blob` | one | 32 px | Variant 11 in a single teal. |
| 24 | `colour_03_slate_coral_blob` | two | 32 px | Slate membrane and grid, coral nucleus. |
| 25 | `colour_04_teal_coral_c_dot` | two | yes | Teal membrane, coral nucleus. |
| 26 | `colour_05_indigo_orbit` | one | soft | Concept 3 in a single indigo. |
| 27 | `colour_06_slate_coral_matrix` | two | soft | Slate array, coral hit. |
| 28 | `colour_07_teal_indigo_split` | two | yes | Teal solid half, indigo resolved half. |
| 29 | `colour_08_coral_crescent` | one | yes | Concept 7 in a single coral. |
| 30 | `colour_09_trio_blob` | three | 32 px | Teal membrane, slate grid, coral nucleus and vacuole - the full parent, in colour. |

---

## Behaviour on light and dark

* **Groups A-C** are pure white on transparent, matching the rest of the shipped
  icon set - which means, exactly like the current `logo_spacr.png`, they are
  invisible on a light background.  Every one of them is a flat single-colour
  silhouette, so the fix is a one-line re-ink of the alpha mask (or shipping a
  second dark file); none of them relies on internal colour or shading to hold
  together, and none has a light-coloured detail that would disappear when
  inverted.  `_sheet_light.png` shows them re-inked dark, which is exactly what
  a light-theme build would do.
* **Group D** needs no such treatment: the same file is legible on both
  backgrounds, which is why the palette is restricted to mid-tone inks.  On
  `_sheet_light.png` the colour candidates are drawn untouched, so what you see
  there is literally what ships.
* The knockout marks (15, 16, 19, and 8/4 in group A) read the background
  through their negative space, so they take on whatever they sit on - they are
  the only ones whose *appearance* changes between themes, and both readings
  were checked.

## Sheets

* `_sheet_dark.png` - all 30 on `#14161a`, artwork exactly as it ships.
* `_sheet_light.png` - all 30 on `#f5f6f8`; white candidates re-inked dark,
  colour candidates untouched.
* `_sheet_small.png` / `_sheet_small_light.png` - every candidate at 16 / 32 /
  48 px, nearest-neighbour zoomed 4x, on each background.  **This is the sheet
  that decides it.**

Regenerate: `QT_QPA_PLATFORM=offscreen python3 _generators/logo_spacr_v2.py`

---

## Appendix - the earlier round of 20

`logo_spacr_01.png` .. `logo_spacr_20.png` are still here; their contact sheets
were moved aside to `_sheet_v1_dark.png`, `_sheet_v1_light.png` and
`_sheet_v1_small.png` when this round took over the standard filenames.
Regenerate with `_generators/logo_spacr.py`.

 1. **wordmark_lockup** - Monoline geometric 'spaCR' wordmark under a grid-cell mark; the 'a' bowl is a cell.
 2. **monogram_cr** - 'CR' monogram, the C read as a membrane around a nucleus. **[16px-safe]**
 3. **well_plate** - Microtitre plate with the notched A1 corner; four hit wells filled solid.
 4. **dish_coords** - Petri dish as a coordinate frame; crosshair axes and ticks locate one colony.
 5. **cas9_guide** - Cas9 as a notched clamp biting a DNA duplex, the sgRNA threaded through the body.
 6. **helix_roundel** - A double helix contained in a disc. **[16px-safe]**
 7. **scissors_dna** - Scissors closing on a duplex with the double-strand break already open.
 8. **screen_array** - A 3x3 array of cells with one filled hit picked out by a selection bracket.
 9. **pin_on_cell** - A map pin dropped on a cell. **[16px-safe]**
10. **radar_scan** - Concentric range rings, a sweep wedge, and detected objects as dots.
11. **hex_tissue** - A segmented monolayer as packed hexagonal cells, the centre one solid.
12. **nucleus_orbit** - A solid nucleus with satellites travelling two orbits. **[16px-safe]**
13. **objective_slide** - A microscope objective over a gridded slide carrying one cell.
14. **barcode_cell** - Sequencing bars on the left wired across into a cell on the right.
15. **aperture_c** - One heavy C-membrane around a solid nucleus. **[16px-safe]**
16. **plasmid_guide** - A plasmid ring carrying one highlighted guide cassette with an arrow.
17. **phenotype_space** - A measured scatter with one population gated and filled.
18. **roi_box** - A cell inside its measured bounding box with dimension ticks.
19. **z_stack** - A stack of imaged planes, only the front one resolved into a cell.
20. **guide_entry** - A guide strand threaded through a gap in the membrane to the nucleus.
