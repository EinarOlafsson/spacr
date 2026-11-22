# Item 1008 Part A integration: translation handover

Branch: `ws-offload-codex-work-1008a-int`.
Base: `0f183ac5bb99f4f955964d868b26e0ae88f26131`.
Merged scientific owners: 1008B `917e4bd2f5cc4c502dac58181a2102c356ae6630` and 1008C `915c91819959844e1ded19b5f139a5d5bc6dc850`.

The FINAL status identifies the exact implementation commit. The accompanying `tr_1008a2_sources.json` carries source hashes, canonical blocks and locations.

One existing public API docstring changes to describe the optional cell-only cytoskeleton channel selection. No new runtime messages, settings labels or tooltips are introduced by this integration. Inherited B/C sources remain in their committed branch-root handovers; all two B and five C scientific function bodies and docstrings match their finalized commits exactly. Catalog rebuilds and locale records remain owned by Home.

Required locales: sv, de, es, zh_CN, pt, hi, ko, is, fr.

## Changed canonical API source

`spacr.ops_sample.measure_phenotype_field` at `spacr/ops_sample.py:438`.

Previous body SHA256: `f03d3e9ea04ed9fa64f54f838043a1b8c3be6374d916bbb2bd90791d98aeeb69`.
Current body SHA256: `4373a0a91af1f265d683a98c94ae6d952030cbaf772949f2557dd32d28f3ffcb`.

Admission verified by `build_documentation_i18n.public_docstrings()`.

```rst
Per-cell features of one phenotype field, with spaCR's own measurement.

:param nuclei: nucleus labels, paired with ``cells`` by
    :func:`pair_nuclei_with_cells`, so nucleus ``k`` is in cell ``k``.
:param cells: cell labels.
:param image: ``(H, W, C)`` intensities.
:param channel_names: one short name per channel, used in the column
    names in place of spaCR's ``channel_<i>``.
:param texture: include the GLCM homogeneity block.
:param funk_features: add Funk et al. 2022's features: the mass
    displacement of every channel, the correlation of every channel
    pair and the form factor, ``4 * pi * area / perimeter ** 2``
    (``shape_features['form_factor']`` of their ``ops/cp_emulator.py``),
    in the nucleus, the cell and the cytoplasm. They come from
    :func:`spacr.measure._add_funk_frames`, the code Measure's
    ``funk_object_features`` runs, with the names
    :func:`funk_object_features` gives (``<compartment>_<ch>_mass_displacement``,
    ``<compartment>_correlation_<a>_<b>``, ``<compartment>_form_factor``).
:param zernike: add the Zernike moments of degree 8
    (``zernike_0`` to ``zernike_24``) of the cell and the nucleus, as
    :func:`spacr.measure._calculate_zernike` computes them with mahotas;
    the cytoplasm gets none. Needs ``spacr[zernike]``. Default False.
:param gpu: measure intensity and texture on the CUDA card
    (:func:`_measurement_settings`).
:param profile_features: the profiling families spaCR's measure adds
    (:func:`spacr.measure.measure_profile_features`): ``'profiling'``
    for the default set, ``'all'``, or family names. ``None``, the
    default, measures exactly as before.
:param granularity_subsample: the ``subsample`` of the granularity
    family (:func:`spacr.measure.measure_granularity`); 1.0 sieves at
    full resolution, 0.25 is CellProfiler's default.
:param cytoskeleton_channels: image channel indices or names for the
    cell-only ``cytoskeleton`` block in ``profile_features``. None
    selects every channel. The block is absent unless selected.
:returns: one row per cell, keyed by ``label``, with ``nucleus_``,
    ``cell_`` and ``cytoplasm_`` columns and the nucleus centroid as
    ``centroid_row``/``centroid_col`` in field pixels.

The features are the ones spaCR's measure writes -- morphology from
:func:`spacr.measure._morphological_measurements`, intensity and texture
from :func:`spacr.measure._intensity_measurements` -- so an OPS cell and
a measured well's cell carry the same columns and nothing is
re-derived. The cytoplasm is the cell minus its nucleus, under the
cell's label. The object labels the colocalisation block carries
(``*_label_correlation``) are left out, as spaCR's database write
leaves them out, so no column repeats the label.
```

Canonical translation blocks:

Block 0; SHA256 `571c88fc06c1e2d1d4e9f9ddeacf3facfe6bd884f84c69d770f5472966dad0a6`.

```rst
Per-cell features of one phenotype field, with spaCR's own measurement.
```

Block 1; SHA256 `1c657bab03ecde0e2ae743218df4372efa6b20dcc0ed02db1bd445373883b417`.

```rst
nucleus labels, paired with ``cells`` by :func:`pair_nuclei_with_cells`, so nucleus ``k`` is in cell ``k``.
```

Block 2; SHA256 `9f40e34fed58c3dcee8dff5ca6e0971e7fffb4169b632bf0e99e4dccbec0066f`.

```rst
cell labels.
```

Block 3; SHA256 `7719b7d94eb9e746032cb37791b520a033fe972b6815598c400f992970508f46`.

```rst
``(H, W, C)`` intensities.
```

Block 4; SHA256 `866f3aa7ea0ec71fa4e043819482326a550bd57c4172bd3977c46360156e7fb3`.

```rst
one short name per channel, used in the column names in place of spaCR's ``channel_<i>``.
```

Block 5; SHA256 `2b6e27e08a166680010842957b52eaa0377149ca84641848fa71eff945e0c918`.

```rst
include the GLCM homogeneity block.
```

Block 6; SHA256 `d851a1d7ab542d4bfe27b35d47766ca5aed114416775ba13181ef1191425b465`.

```rst
add Funk et al. 2022's features: the mass displacement of every channel, the correlation of every channel pair and the form factor, ``4 * pi * area / perimeter ** 2`` (``shape_features['form_factor']`` of their ``ops/cp_emulator.py``), in the nucleus, the cell and the cytoplasm. They come from :func:`spacr.measure._add_funk_frames`, the code Measure's ``funk_object_features`` runs, with the names :func:`funk_object_features` gives (``<compartment>_<ch>_mass_displacement``, ``<compartment>_correlation_<a>_<b>``, ``<compartment>_form_factor``).
```

Block 7; SHA256 `17650eb4334eda7c9467bb436c171d5ac8c856d38d593d5c271ddf94cc7c3acb`.

```rst
add the Zernike moments of degree 8 (``zernike_0`` to ``zernike_24``) of the cell and the nucleus, as :func:`spacr.measure._calculate_zernike` computes them with mahotas; the cytoplasm gets none. Needs ``spacr[zernike]``. Default False.
```

Block 8; SHA256 `89aaf3c0ff84226f300726b3241c3fc17c8dfd4d007999f3e2798328a89552dc`.

```rst
measure intensity and texture on the CUDA card (:func:`_measurement_settings`).
```

Block 9; SHA256 `33943118d22d91e93fe9c8055f733948315968115766e48b068c1c2d9864198f`.

```rst
the profiling families spaCR's measure adds (:func:`spacr.measure.measure_profile_features`): ``'profiling'`` for the default set, ``'all'``, or family names. ``None``, the default, measures exactly as before.
```

Block 10; SHA256 `b9b8a47b20c53594f7227da5073b7ab0074fda9781d5d643b9705d60c049f4a0`.

```rst
the ``subsample`` of the granularity family (:func:`spacr.measure.measure_granularity`); 1.0 sieves at full resolution, 0.25 is CellProfiler's default.
```

Block 11; SHA256 `5a6c5eab36963de3a6d896055a3f31828e4c30ef03128b69a97b8f64774e08de`.

```rst
image channel indices or names for the cell-only ``cytoskeleton`` block in ``profile_features``. None selects every channel. The block is absent unless selected.
```

Block 12; SHA256 `ab2fb1269405def9b4c64ed5536264ebb71d656dc75d1dad590e13508397280f`.

```rst
one row per cell, keyed by ``label``, with ``nucleus_``, ``cell_`` and ``cytoplasm_`` columns and the nucleus centroid as ``centroid_row``/``centroid_col`` in field pixels.
```

Block 13; SHA256 `6db25ce944b655bbeb4af71273d857721e60f757fef1475f1f5a13b0202ea510`.

```rst
The features are the ones spaCR's measure writes -- morphology from :func:`spacr.measure._morphological_measurements`, intensity and texture from :func:`spacr.measure._intensity_measurements` -- so an OPS cell and a measured well's cell carry the same columns and nothing is re-derived. The cytoplasm is the cell minus its nucleus, under the cell's label. The object labels the colocalisation block carries (``*_label_correlation``) are left out, as spaCR's database write leaves them out, so no column repeats the label.
```

The unchanged historical fixture is `ops_phenotype_path_api_arrivals_2026-11-05.json`. Its revision is bound by `tests/data/release_contracts/1008a2_phenotype_cytoskeleton_api_change_2026-11-21.json`. The source-only helper checks the exact positional and keyword-only names, required parameters, default values and types, source body and both hashes. Negative drift checks cover keyword names, defaults and source records.

## Private and test documentation

These bodies need no API translation records.

- `_intensity_measurements` at `spacr/measure.py:1854`: none: private or test documentation.
- `_measurement_settings` at `spacr/ops_sample.py:332`: none: private or test documentation.
- `test_the_blocks_are_names_of_profile_features_and_not_of_its_presets` at `tests/test_measure_ops_parity_1008.py:74`: none: private or test documentation.
- `test_final_list_parity_in_both_directions` at `tests/test_measure_ops_parity_1008.py:218`: none: private or test documentation.
- `test_ops_cytoskeleton_selection_reaches_the_shared_cell_only_measurement` at `tests/test_measure_ops_parity_1008.py:299`: none: private or test documentation.
- `test_cytoskeleton_api_revision_rejects_unrecorded_source_drift` at `tests/test_measure_ops_parity_1008.py:323`: none: private or test documentation.
- `test_profile_features_declares_its_columns_and_reruns_identically` at `tests/test_measure_profile_families.py:225`: none: private or test documentation.
- `test_profiling_preset_leaves_the_slow_families_out` at `tests/test_measure_optional_families_941.py:55`: none: private or test documentation.
- `test_optional_families_parse_and_only_widen_profiling` at `tests/test_measure_optional_families_941.py:66`: none: private or test documentation.
- `test_ops_settings_resolve_with_the_flag` at `tests/test_measure_optional_families_941.py:110`: none: private or test documentation.
- `_current_1008a2_ops_api` at `tests/test_documentation_i18n.py:5500`: none: private or test documentation.
- `test_otsu_colocalisation_has_explicit_shared_aggregation_rules` at `tests/test_aggregation_feature_defaults_937.py:385`: none: test documentation.

## Validation

Complete-family P1/P2/P9 and name/index/None channel selection passed. The first focused run recorded 78 passed and one historical 941 all-family assertion failure; the corrected 941 module passed all 9 tests. API revision and full parity follow-up passed 23 tests. Canonical extraction verified one changed API, 14 blocks and the unchanged scientific owners. Full functional, source and ratchet results, including exact deferred record failures, are reported in FINAL. No raw failing result is relabeled as passing.

## Approved shared aggregation rules

WORK_TASKS v82 approves the three existing regex clauses in `spacr/merge_tables.py`. Summary aggregation uses the existing mean for `raw_overlap`, `otsu_m1` and `otsu_m2`, and the existing median for `lstsq_slope_ij` and `lstsq_slope_ji`. All five have explicit existing mean rules for child-to-parent aggregation. The legacy rule table and all existing method explanations are unchanged. Regex identifiers need no translation records. Six planted cases cover all nine statistics, two naming conventions and three compartments. The proposal passed seven in-memory proof cases, and the actual-source aggregation/merge/parity follow-up passed 78 tests.

P9 records 1,788 columns on the three-cell, three-channel 934 synthetic field with texture, Funk and Zernike enabled and every final family selected; six columns are cell-only cytoskeleton statistics. This is a fixture count. The accompanying JSON preserves the column-name hash.
