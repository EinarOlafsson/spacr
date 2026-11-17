# Item spec 969: OPS settings panel layout, model-zoo choices and the OPS power curve in Power

- **Lifecycle:** GUI improvement of the OPS module (alpha module); no new pipeline behaviour.
- **Number:** 969. Lands on nightly after 966 and 959 (both touch `spacr/ops_settings.py`).
- **Rule:** pipeline output unchanged. A default OPS run on the Funk A1 core tier
  gives the same tables, masks and images before and after; an old settings
  file naming `cellpose_model` loads and runs identically.
- **Code basis:** nightly a884551a8b.

## Maintainer requests (2026-11-17)
1. The nuclei model setting names what it is for (nuclei segmentation). `cpsam`
   stays the default; any model from the model zoo can be chosen, as in Mask
   generation.
2. The OPS module hides the leading "ops" of each setting label (keys keep
   it; category titles keep "OPS"). Translated labels follow the same rule.
3. `ops_ph_cell_model` (and `ops_ph_nucleus_model`) offer the model zoo too.
4. No category or subcategory holds more than 10-20 settings: the large OPS
   categories get subcategories, organised after Brieflow's stages
   (preprocess, sbs, phenotype, merge, aggregate, cluster).
5. Plate layout and controls sit inside OPS input.
6. OPS performance becomes a subcategory.
7. (Added later the same day) The OPS power settings (`ops_power_*`) leave
   the OPS panel and become an "OPS power curve" section of the Power module,
   the future home of item 968 (OPS power simulation).

## Design
- **Rename.** `cellpose_model` (read only by OPS) becomes
  `ops_sbs_nucleus_model`, label "Ops SBS nucleus segmentation model". The
  old key folds onto the new one through `spacr.settings.RENAMED_SETTINGS`
  and `spacr.validate.RETIRED_SETTINGS`, the mechanism every earlier rename
  used, so old CSVs load unchanged. `run_ops` folds the old name too, for
  Python callers passing a dict. No schema bump: the default is unchanged.
- **Model zoo.** The three OPS Cellpose keys join
  `AppScreen._MODEL_ZOO_KEYS` (the "Model zoo..." button Mask generation's
  model fields use) and `spacr.model_zoo._model_input_kinds("ops")`, so a
  zoo choice is pinned in `model_metadata` exactly as in Mask generation.
- **Prefix hiding.** `SettingsWidgets._label_for` drops a leading "Ops " for
  the `ops` module (`_APP_LABEL_PREFIXES`); the catalog builder harvests the
  result as `ops.<key>` label records, which are translated in nine languages.
- **Subcategories.** The existing nesting (`_APP_CATEGORY_PARENTS`, as Mask
  generation's Image Preprocessing) is reused; the OPS map lives in
  `spacr.ops_settings.OPS_CATEGORY_PARENTS`. "Plate Layout & Controls" nests
  under OPS input and gains the guide design, control prefix and genotype
  table. New top-level groups OPS merge, OPS aggregate, OPS cluster and OPS
  evaluation follow Brieflow's stages.
- **Power.** `ops_power_*` stay OPS settings (old files and `run_ops(...,
  phases=["etest"])` are unchanged) but are hidden from the OPS panel. The
  Power module gets an "OPS power curve" group that runs the same
  `run_ops(settings, phases=["etest"])` from an OPS settings file, the cells
  folder and the five power settings.

## Proof
- Default OPS run on the A1 core tier, CPU: base vs branch (old CSV through
  the rename, and a CSV with the new key): every table, mask, image and
  report identical apart from timing fields.
- Power run: OPS-module `run_ops(phases=["etest"])` vs the Power section on
  the etest fixture: identical outputs.
- Screenshots of the OPS settings panel and the Power module, before and
  after, light and dark: `R/item969_proof/`.

## Tests
Prefix hiding, subcategory caps (<= 20), model-zoo kinds for the three model
keys, the alias load, plate layout and controls under OPS input, the power
keys hidden from OPS and present in Power.

## Tutorials
Lessons that show the OPS settings panel or the Power module need
re-recording (listed in the item report); nothing is re-recorded here.
