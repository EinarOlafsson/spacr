"""ops-measure-shared: OPS measures a phenotype field with Measure's per-field core.

The maintainer's rule (2026-11-21, ONE OPERATION, ONE IMPLEMENTATION): the OPS
phenotype measurement (``_ph_measure_field`` / ``_ph_measure_table`` in
:mod:`spacr.ops_engine`, through :func:`spacr.ops_sample.measure_phenotype_field`)
goes through :func:`spacr.measure._measure_field_objects`, the core Measure's
``_measure_crop_core`` and its GPU check run. These tests hold that:

* the OPS table of a field equals, value for value, the core's tables for the
  same field and settings, the per-cell object summary included;
* no measurement implementation is left in ``ops_engine``, and the OPS and
  Measure entry points reach the feature functions only through the core
  (an AST check, so a re-added copy fails here, not in a review).
"""
from __future__ import annotations

import ast
import pathlib
import sys

import numpy as np
import pytest

from spacr import measure as M
from spacr import ops_sample

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_measure_ops_features_934 import _ops_field  # noqa: E402

SPACR = pathlib.Path(M.__file__).resolve().parent

#: The functions that compute per-object features. Only the shared core may
#: call them on behalf of a module's per-field measurement.
FEATURE_FUNCTIONS = {
    "_morphological_measurements", "_intensity_measurements",
    "_summarize_organelles_per_parent", "_calculate_zernike",
    "_add_funk_frames", "measure_profile_features", "regionprops",
    "regionprops_table", "center_of_mass",
}


def _called_names(node):
    """Every name a function body calls or imports (``f()``, ``m.f()``, ``from m import f``)."""
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
        elif isinstance(sub, ast.ImportFrom):
            names.update(alias.name for alias in sub.names)
    return names


def _functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_no_measurement_implementation_remains_in_ops_engine():
    """``ops_engine``'s phenotype measurement only reads, corrects and hands over."""
    functions = _functions(SPACR / "ops_engine.py")
    for name in ("_ph_measure_table", "_ph_measure_field", "_ph_check_table"):
        found = _called_names(functions[name]) & FEATURE_FUNCTIONS
        assert not found, f"{name} computes features itself: {sorted(found)}"
    text = (SPACR / "ops_engine.py").read_text(encoding="utf-8")
    for name in ("_morphological_measurements", "_intensity_measurements",
                 "_summarize_organelles_per_parent", "regionprops_table"):
        assert name not in text, f"ops_engine names {name}"


def test_ops_and_measure_reach_the_feature_functions_only_through_the_core():
    """One implementation: the OPS adapter, Measure's field loop and its GPU
    check call the core, never the feature functions."""
    sample = _functions(SPACR / "ops_sample.py")
    measure = _functions(SPACR / "measure.py")
    for functions, name in ((sample, "measure_phenotype_field"),
                            (sample, "_measure_core"),
                            (sample, "_cell_object_features"),
                            (sample, "_cell_object_table"),
                            (measure, "_measure_crop_core"),
                            (measure, "_field_tables_for_check")):
        called = _called_names(functions[name])
        found = called & FEATURE_FUNCTIONS
        assert not found, f"{name} computes features itself: {sorted(found)}"
    assert "_measure_field_objects" in _called_names(sample["_measure_core"])
    assert "_measure_field_objects" in _called_names(measure["_measure_crop_core"])
    assert "_measure_field_objects" in _called_names(measure["_field_tables_for_check"])


def _core_tables(nuclei, cells, image, names, **options):
    """The core's per-role tables for the OPS field, with the OPS settings."""
    settings = ops_sample._measurement_settings(
        image.shape[2], options.get("texture", False),
        funk_features=options.get("funk", False),
        profile_features=options.get("profile_features"),
        channel_names=names)
    cytoplasm = np.where(nuclei > 0, 0, cells).astype(np.int32)
    measured = M._measure_field_objects(
        {"cell": cells, "nucleus": nuclei, "cytoplasm": cytoplasm}, image,
        settings, periphery=False, outside=False, summarize={})
    return M._field_role_tables(measured["morphology"], measured["intensity"],
                               ("cell", "nucleus", "cytoplasm"))


@pytest.mark.parametrize("options", [
    {},
    {"texture": True, "funk": True,
     "profile_features": ["edge_intensity", "intensity_moments"]},
])
def test_ops_values_equal_the_core_values_for_the_same_field(options):
    """Every OPS feature column is the core's column of its compartment, exactly."""
    nuclei, cells, image, names = _ops_field()
    table = ops_sample.measure_phenotype_field(
        nuclei, cells, image, names, texture=options.get("texture", False),
        funk_features=options.get("funk", False),
        profile_features=options.get("profile_features"))
    core = _core_tables(nuclei, cells, image, names, **options)
    checked = 0
    for column in table.columns:
        if column in ("label", "centroid_row", "centroid_col"):
            continue
        prefix = column.split("_", 1)[0]
        frame = core[prefix].set_index("label")
        source = column if column in frame.columns else column[len(prefix) + 1:]
        assert source in frame.columns, column
        expected = frame[source].reindex(table["label"]).to_numpy(np.float64)
        np.testing.assert_array_equal(table[column].to_numpy(np.float64), expected,
                                      err_msg=column)
        checked += 1
    assert checked == len(table.columns) - 3


def test_ops_object_summary_is_the_core_summary_measure_writes():
    """The per-cell object columns (item 970) are Measure's
    ``cell_organelle_summary`` values under the OPS names, and
    ``_ph_measure_table`` appends exactly them."""
    nuclei, cells, image, names = _ops_field()
    spots = np.zeros_like(cells)
    ys, xs = np.nonzero(nuclei)
    for k, (y, x) in enumerate(zip(ys[::97], xs[::97]), start=1):
        spots[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = k
    objects = {"pathogen": spots}
    table = ops_sample._cell_object_features(cells, objects, image, names)
    settings = {**ops_sample._measurement_settings(image.shape[2], False,
                                                   summarize=True),
                "channel_names": None}
    summary = M._measure_field_objects({"cell": cells}, image, settings,
                                      summarize=objects, tables=False)["summaries"]["cell"]
    summary = summary.set_index("label").reindex(table["label"])
    assert (table["cell_pathogen_count"] > 0).any()
    np.testing.assert_array_equal(
        table["cell_pathogen_count"].to_numpy(np.float64),
        summary["organelle_summary_pathogen_count"].fillna(0).to_numpy(np.float64))
    for index, name in enumerate(n.lower() for n in names):
        np.testing.assert_array_equal(
            table[f"cell_pathogen_{name}_mean_intensity"].to_numpy(np.float64),
            summary[f"organelle_summary_pathogen_channel_{index}_mean_intensity_per_cell"]
            .fillna(0).to_numpy(np.float64))


def test_ops_measures_a_field_in_one_call_of_the_core(monkeypatch):
    """One OPS field is one call of the core, without Measure's legacy
    periphery and outside rings (OPS never wrote them)."""
    calls = []
    original = M._measure_field_objects

    def _spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(M, "_measure_field_objects", _spy)
    nuclei, cells, image, names = _ops_field()
    ops_sample.measure_phenotype_field(nuclei, cells, image, names)
    assert len(calls) == 1 and calls[0]["periphery"] is False


def _legacy_merge(mask1, mask2):
    """The whole-field form of ``utils._merge_overlapping_objects`` before it
    examined each object in its bounding box (nightly fb838fb3b4)."""
    from skimage.measure import label

    labeled_1 = label(mask1)
    for m1_id in range(1, int(np.max(labeled_1)) + 1):
        current = labeled_1 == m1_id
        found = np.unique(mask2[current])
        found = found[found != 0]
        if len(found) > 1:
            shares = [np.sum(current & (mask2 == m)) / np.sum(current) * 100 for m in found]
            best = found[np.argmax(shares)]
            if max(shares) >= 90:
                for m in found:
                    if m != best:
                        mask1[current & (mask2 == m)] = 0
            else:
                for m in found[1:]:
                    mask2[mask2 == m] = found[0]
    return mask1, mask2


@pytest.mark.parametrize("seed", range(6))
def test_windowed_ownership_rule_equals_the_whole_field_rule(seed):
    """The bounding-box form gives exactly the masks of the old whole-field loop."""
    from skimage.segmentation import expand_labels
    from spacr.utils import _merge_overlapping_objects

    rng = np.random.default_rng(seed)
    nuclei = np.zeros((96, 96), np.int32)
    for k, (y, x) in enumerate(rng.integers(4, 92, size=(14, 2)), start=1):
        nuclei[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4] = k
    seeds = np.zeros_like(nuclei)
    for k, (y, x) in enumerate(rng.integers(0, 96, size=(10, 2)), start=1):
        seeds[y, x] = k
    cells = expand_labels(seeds, 40).astype(np.int32)
    want = _legacy_merge(nuclei.copy(), cells.copy())
    got = _merge_overlapping_objects(nuclei.copy(), cells.copy())
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def test_ops_pairing_merges_cells_a_nucleus_spans_as_measure_does():
    """Maintainer decision 2026-11-23: a nucleus split under 90 % between two
    cells merges them (Measure's rule); at 90 % or more it is trimmed. Since
    schema 976 (maintainer decision 2026-11-25) this is the replay arm
    ops_ph_spanning_merge='merge'; the default is 'best_parent'."""
    cells = np.zeros((20, 40), np.int32)
    cells[:, :20], cells[:, 20:] = 1, 2
    nuclei = np.zeros_like(cells)
    nuclei[8:12, 16:24] = 1                       # half in each cell
    out_n, out_c, counts = ops_sample._pair_nuclei_with_cells(nuclei, cells, spanning="merge")[:3]
    assert np.unique(out_c[out_c > 0]).tolist() == [1]
    assert (out_c > 0).sum() == cells.size and counts["merged_cells"] == 1
    assert (out_n > 0).sum() == 32
    nuclei[:] = 0
    nuclei[8:12, 0:19] = 1
    nuclei[8:12, 19:21] = 1                       # 76 px: 72 in cell 1, 4 in cell 2 -> 94.7 %
    nuclei[2:5, 30:33] = 2
    out_n, out_c, counts = ops_sample._pair_nuclei_with_cells(nuclei, cells, spanning="merge")[:3]
    assert counts["merged_cells"] == 0 and np.unique(out_c[out_c > 0]).size == 2
    assert (out_n[8:12, 20] == 0).all()           # trimmed back into cell 1
    assert (out_n[8:12, :19] > 0).all()
