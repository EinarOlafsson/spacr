# logo_spacr - 30 candidates for the spaCR mark itself

Brief, in the user's words: **streamlined, simple, intuitive, elegant** - and,
after the previous round was rejected in full, *"lines should be thinner,
shapes should be simpler."*

That previous round let one constraint drive everything: *must survive 16 px*.
It drew heavy to satisfy it, and then piled on parts to keep meaning once the
drawing had gone blunt - nine elements in one candidate, five in a whole family
of ten. This round **inverts the priority: draw thin and elegant first, then
measure what survives.**

* The primary stroke is **24 canvas units** against the last round's 76-96.
  That is 3.0 px at 128, 1.5 px at 64, 0.75 px at 32.
* **Two elements is the common case, one is reachable.** Eight of the thirty
  are a single form; nothing here has more than two parts.
* Directions the last round never tried and this one does: pure negative space
  (2, 17), one unbroken line (1, 11), a single filled form with one knockout
  (6), deliberate asymmetry (3, 7, 10, 19, 20), and marks that occupy 45-60% of
  the frame instead of filling it (10, 14, 20).

All 30 are 1024x1024 RGBA. Groups A-C are pure white on transparent (house
style); group D ships colour. No gradients, no bevels, no shadows.

## The small-size column, and whose decision it is

`16 px` is read off `_sheet_small.png`, not guessed:

* **yes** - reads at 16 px.
* **soft** - present and identifiable at 16 px, but grey and thin; crisp by 32.
* **32 px** / **48 px** - a smudge below that; use it as the floor.

Several of the best marks here are **soft**. That is the honest cost of thin
lines, and it is deliberate: a mark that is beautiful at 128 px and soft at 16
may still be the right answer, and only you can decide whether the favicon or
the splash wins. If a soft favourite is chosen, the fix is a second file at a
heavier weight for 16 px use, not a heavier logo everywhere. Where a candidate
is soft, the note says what softens - usually the outline, while the solid
element keeps carrying it.

---

## A. New concepts - 10, black and white

| # | file | parts | 16 px | idea |
|---|------|-------|-------|------|
| 1 | `concept_01_trace` | 1 | soft | One unbroken line: a contour drawn past its own start, crossing where it began - a boundary while it is being drawn. |
| 2 | `concept_02_counter_c` | 1 | **yes** | Pure negative space: one disc, and the C is only what has been cut out of it. |
| 3 | `concept_03_offset_nucleus` | 2 | soft | A thin membrane and one small nucleus, deliberately off centre. The dot holds at 16; the ring greys. |
| 4 | `concept_04_open_c` | 1 | soft | A single arc, nothing else - the C of spaCR and a membrane in one stroke. |
| 5 | `concept_05_chord` | 2 | 32 px | One boundary and one cut across it, off centre. The chord and the ring merge below 32. |
| 6 | `concept_06_solid_cell` | 1 | **yes** | One filled cell with the nucleus knocked out of it - no outline anywhere. The most robust mark in the set. |
| 7 | `concept_07_pair` | 2 | soft | Two unequal cells, touching - the smallest picture of segmentation. |
| 8 | `concept_08_taper` | 1 | soft | One closed continuous line whose weight swells and thins once. The taper itself is gone by 32 px; below that it is simply a ring. |
| 9 | `concept_09_two_arcs` | 2 | 32 px | One boundary drawn in two strokes; the cell is the space they enclose. The two gaps close up at 16. |
| 10 | `concept_10_cropped` | 2 | 32 px | A cell running out of the field of view - the frame reads as a crop, not a stage. |

## B. New variants - 10, black and white

Variants of the group A ideas, including the two that carry the current
`logo_spacr.png`'s identity - the irregular silhouette and the off-centre
nucleus - in **two** parts rather than five (15, 16).

| # | file | parts | 16 px | idea |
|---|------|-------|-------|------|
| 11 | `variant_01_trace_open` | 1 | soft | The trace left open: one pass, a gap where it started. |
| 12 | `variant_02_trace_dot` | 2 | soft | The trace with its nucleus. |
| 13 | `variant_03_c_dot` | 2 | soft | The open C with one small nucleus inside it - membrane and body, and nothing else. |
| 14 | `variant_04_c_wide` | 1 | 32 px | The C at 47% of the frame with a wide mouth - the white-space study. |
| 15 | `variant_05_blob` | 1 | soft | The parent silhouette, thin, and nothing inside it. |
| 16 | `variant_06_blob_dot` | 2 | soft | The parent silhouette with one off-centre nucleus - its identity, in two parts. |
| 17 | `variant_07_reverse` | 1 | **yes** | Reversed: a disc with the cell silhouette cut out of it, off centre. |
| 18 | `variant_08_cut_fill` | 2 | **yes** | The cut, with the smaller side filled - one cell resolved out of one boundary. The fill carries it at 16; the ring greys. |
| 19 | `variant_09_pair_solid` | 2 | soft | The pair with the smaller cell solid, the two just touching. |
| 20 | `variant_10_arc_dot` | 2 | 32 px | A fragment of membrane and the body it belongs to - the most reduced mark here, 40% of the frame. It asks the most of the viewer; at 16 px only the dot is left. |

## C. Thinner lines - 1, black and white

| # | file | parts | 16 px | idea |
|---|------|-------|-------|------|
| 21 | `thin_01_c_dot` | 2 | 48 px | Candidate 13 at hairline weight - 11 units, under half the primary stroke. |

Compare 21 against 13 side by side. At 256 px and above it is the most refined
thing in the folder and the closest in spirit to the current logo; at 48 px it
is a whisper, and below that it is gone. This is the trade in its purest form.

## D. Colour - 9

Four single-colour, five two-colour. The inks are inherited unchanged from the
previous round, which got this part right: relative luminance held between 0.13
and 0.27, so **one file works on both backgrounds** at 3:1 or better.

| ink | hex | vs dark `#14161a` | vs light `#f5f6f8` |
|-----|-----|---------|----------|
| teal   | `#0E9488` | 4.84:1 | 3.46:1 |
| indigo | `#5B63D6` | 3.62:1 | 4.62:1 |
| coral  | `#E0533A` | 4.72:1 | 3.55:1 |
| slate  | `#6F7B8A` | 4.21:1 | 3.98:1 |

A mid-tone ink at 3-5:1 is quieter than white-on-black, so a coloured thin line
loses roughly one step at small sizes against its black-and-white parent. The
verdicts below already account for that.

| # | file | colours | 16 px | idea |
|---|------|---------|-------|------|
| 22 | `colour_01_teal_trace` | one | 32 px | The trace in one teal. |
| 23 | `colour_02_teal_open_c` | one | soft | The open C in one teal. |
| 24 | `colour_03_indigo_offset` | one | 32 px | Membrane and off-centre nucleus in one indigo. |
| 25 | `colour_04_coral_solid_cell` | one | **yes** | The filled cell in one coral. |
| 26 | `colour_05_teal_coral_c_dot` | two | soft | Teal C, coral nucleus. |
| 27 | `colour_06_slate_coral_trace` | two | 32 px | Slate trace, coral nucleus. |
| 28 | `colour_07_teal_coral_blob` | two | 32 px | Teal silhouette, coral nucleus. |
| 29 | `colour_08_indigo_counter_c` | one | **yes** | The negative-space C in one indigo. |
| 30 | `colour_09_teal_indigo_pair` | two | 32 px | Teal cell, indigo cell. |

---

## Behaviour on light and dark

* **Groups A-C** are pure white on transparent, matching the rest of the
  shipped icon set - which means, exactly like the current `logo_spacr.png`,
  they are invisible on a light background. Every one is a flat single-colour
  silhouette, so the fix is a one-line re-ink of the alpha mask (or a second
  dark file); none relies on internal colour to hold together.
  `_sheet_light.png` shows them re-inked dark, which is what a light-theme
  build would do.
* **Group D** needs no such treatment: the same file is legible on both, which
  is why the palette is restricted to mid-tone inks. On `_sheet_light.png` the
  colour candidates are drawn untouched - what you see there is what ships.
* The knockout marks (2, 6, 17, 25, 29) read the background through their
  negative space, so they take on whatever they sit on. Both readings were
  checked; the C in 2/29 and the nucleus in 6/25 invert cleanly.

## Sheets

* `_sheet_dark.png` - all 30 on `#14161a`, artwork exactly as it ships.
* `_sheet_light.png` - all 30 on `#f5f6f8`; white candidates re-inked dark,
  colour candidates untouched.
* `_sheet_small.png` / `_sheet_small_light.png` - every candidate at 16 / 32 /
  48 px, nearest-neighbour zoomed 4x. **This is where the verdicts above come
  from** - but it is no longer the sheet that decides it. Look at the large
  sheet first, then come here to learn what the choice costs.

Regenerate: `QT_QPA_PLATFORM=offscreen python3 _generators/logo_spacr_v3.py`

---

## Appendix - the rejected round of 30

`v2_concept_*`, `v2_variant_*`, `v2_thin_*`, `v2_colour_*` and the sheets
`_sheet_v2_dark.png`, `_sheet_v2_light.png`, `_sheet_v2_small.png`,
`_sheet_v2_small_light.png` are the previous round, kept so nothing is lost.
Rejected in full for heavy lines and busy shapes. Regenerated by
`_generators/logo_spacr_v2.py` - note that re-running it writes the *unprefixed*
names and would collide with this round's files.

 1. **concept_01_c_dot** - One open C-membrane around one solid nucleus.
 2. **concept_02_quadrant** - A field split in four with exactly one cell filled.
 3. **concept_03_orbit** - One body, one ring.
 4. **concept_04_plate** - A plate with its notched A1 corner and one well called.
 5. **concept_05_matrix** - A 3x3 array with one well called.
 6. **concept_06_split** - One object, half raw and half resolved.
 7. **concept_07_crescent** - The organism as one solid crescent.
 8. **concept_08_pin** - A map pin whose counter is the cell.
 9. **concept_09_cut** - One body, one clean cut, the halves slid apart.
10. **concept_10_focus** - Two corner brackets and the one object between them.
11. **variant_01_grid_nucleus** - Cell outline, spatial grid, solid nucleus.
12. **variant_02_grid_vacuole** - As 11, plus the parasitophorous vacuole.
13. **variant_03_bare** - Cell outline and nucleus only, drawn heavy.
14. **variant_04_cross** - The grid reduced to a single cross.
15. **variant_05_solid** - The cell as solid mass, nucleus knocked out.
16. **variant_06_solid_grid** - Solid cell, grid and nucleus as negative space.
17. **variant_07_satellite** - Outline, nucleus and one satellite.
18. **variant_08_regular** - The silhouette regularised.
19. **variant_09_grid_hit** - One square of the grid filled solid.
20. **variant_10_ticks** - The grid implied by ticks straddling the membrane.
21. **thin_01_grid_nucleus** - Variant 11 at 60% stroke weight.
22-30. **colour_01..09** - colour treatments of the above.

## Appendix - the earlier round of 20

`logo_spacr_01.png` .. `logo_spacr_20.png`, with sheets `_sheet_v1_dark.png`,
`_sheet_v1_light.png` and `_sheet_v1_small.png`. Regenerate with
`_generators/logo_spacr.py`.

 1. **wordmark_lockup** - Monoline geometric 'spaCR' wordmark under a grid-cell mark.
 2. **monogram_cr** - 'CR' monogram, the C read as a membrane around a nucleus.
 3. **well_plate** - Microtitre plate with the notched A1 corner.
 4. **dish_coords** - Petri dish as a coordinate frame.
 5. **cas9_guide** - Cas9 as a notched clamp biting a DNA duplex.
 6. **helix_roundel** - A double helix contained in a disc.
 7. **scissors_dna** - Scissors closing on a duplex.
 8. **screen_array** - A 3x3 array of cells with one filled hit.
 9. **pin_on_cell** - A map pin dropped on a cell.
10. **radar_scan** - Range rings, a sweep wedge, detected objects as dots.
11. **hex_tissue** - A segmented monolayer as packed hexagonal cells.
12. **nucleus_orbit** - A solid nucleus with satellites on two orbits.
13. **objective_slide** - A microscope objective over a gridded slide.
14. **barcode_cell** - Sequencing bars wired across into a cell.
15. **aperture_c** - One heavy C-membrane around a solid nucleus.
16. **plasmid_guide** - A plasmid ring carrying one highlighted guide cassette.
17. **phenotype_space** - A measured scatter with one population gated.
18. **roi_box** - A cell inside its measured bounding box.
19. **z_stack** - A stack of imaged planes, only the front one resolved.
20. **guide_entry** - A guide strand threaded through a gap in the membrane.
