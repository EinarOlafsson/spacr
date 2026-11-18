# Item 970: pathogen and organelle objects in the OPS phenotype path

- **Number:** 970 (assigned by master 2026-11-17). **Priority:** medium. **Alpha.**
- **Asked by:** the maintainer, 2026-11-17: "just like in mask generation. OPS should allow the user to define pathogen and organelle objects just like in mask generation." Scope decision: masks + per-cell features + gene profiles.
- **Gap:** the OPS phenotype path (`_ph_fields` → `_ph_segment_field` → `_ph_measure_table` → `_ph_aggregate`) segments only nuclei and cells. Mask Generation segments pathogens (Cellpose on `pathogen_channel` with `pathogen_model`/`pathogen_diameter`) and organelles (`generate_organelle_masks_sam`: `organelle_morphology` × `organelle_method`, classical or Cellpose), and Measure summarises organelles per parent cell (`spacr.measure._summarize_organelles_per_parent`). None of that reaches an OPS run.

## Spec
1. **Settings** (`spacr/ops_settings.py`, alpha, ALPHA_FEATURES[970], 9-locale labels and tooltips), named after Mask Generation's with the `ops_ph_` prefix:
   - `ops_ph_pathogen_channel` (None = off), `ops_ph_pathogen_model` ('cpsam', any model-zoo name or path, as `pathogen_model`), `ops_ph_pathogen_diameter` (None), `ops_ph_pathogen_min_area` (0);
   - `ops_ph_organelle_channel` (None = off), `ops_ph_organelle_morphology` ('spots'), `ops_ph_organelle_method` ('otsu'), `ops_ph_organelle_model` ('cpsam', used by method 'cellpose'), `ops_ph_organelle_diameter` (None), `ops_ph_organelle_min_area` (10, Mask Generation's default).
   - Channels are names from `ops_ph_channels`, as `ops_ph_cell_channel` is.
   - Proposed subcategory (item 969 places it): "Phenotype segmentation: pathogens and organelles".
2. **Segmentation** (`ph_segment`, and the large-site path `run_large_site_segmentation`): at full field resolution, after nuclei and cells.
   - Pathogens: Mask Generation's Cellpose route (`_cellpose_model` + `_governed_eval` on the pathogen channel), then `spacr.object.merge_split_filter_masks` for the minimum area.
   - Organelles: Mask Generation's organelle route, `_set_organelle_defaults` + `_validate_organelle_settings` + `_preprocess_batch` + `_segment_single_image` for the classical methods, Cellpose for 'cellpose', then `merge_split_filter_masks`.
   - The labels are saved in the field's mask file as `pathogen` / `organelle` arrays (large-site: `labels/<site>_pathogen.npy`, `_organelle.npy`), only when switched on.
3. **Per-cell features** (`ph_measure`): `spacr.measure._summarize_organelles_per_parent` for each object class against the cell mask (each object belongs to the cell holding most of its pixels, `_map_child_to_parent`), renamed `cell_<class>_*`:
   - count, total area, area fraction of the cell, mean and SD area, eccentricity, solidity and axis lengths;
   - mean and SD intensity of each channel over the cell's objects.
   - The intensity loop of `_summarize_organelles_per_parent` is restricted to each object's bounding box (`_label_bounding_boxes`, same pixels in the same order, so bit-identical values) so a field with thousands of objects is not O(objects × field).
   - These are cheap, so no extra flag.
4. **Profiles:** the `cell_` prefix makes the columns per-cell features (`_ph_feature_columns`), so the aggregate phase, per-gene profiles and clustering inputs carry them with no further change.
5. **Defaults:** both channels None. The mask files, part markers, cell tables, profiles and clusters of a default run are byte-identical to nightly. No schema bump, no DEFAULT_CHANGES row. Turning objects on over masks made without them is refused, with the fix named, rather than mixed.

## Proof check (pre-set)
- **Default identity:** Funk A1 test subset (smallest `ops_testdata_funk2022_A1_*` fixture, CPU): every phenotype output (`phenotype_masks/*.npz`, `phenotype_cells/*.parquet`, the well cell table) has the same md5 on nightly and on this branch with defaults.
- **Feature check:** a synthetic field with planted pathogens and organelles of known count, area and intensity per cell. The masks, the per-cell counts, areas, area fractions and intensities, and the gene-profile columns must equal the planted values exactly. Includes an object outside every cell (dropped) and a cell with none (count 0).
- **Real-channel smoke:** Funk A1 with one real stain (e.g. the mitochondria-like channel) as the organelle channel, 'spots'/'otsu', CPU: masks, non-trivial per-cell counts and finite profile columns.
- **Profiles:** the new columns reach the per-gene profile table and the clustering input. Control: profiles and clusters with the feature off are identical to nightly.
- **Report:** `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/item970_proof/REPORT.md` with tables, controls and a glossary.
