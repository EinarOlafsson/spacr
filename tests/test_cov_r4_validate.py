"""The branches pre-flight takes when the answer is "that one is fine".

:mod:`spacr.validate` is written to report what is wrong, so its well-trodden
paths are the failing ones. This file pins the other side of a dozen of its
decisions: the settings helper whose return value is not a key mapping, the
``stack/`` folder that holds no arrays (and the one that holds arrays *and*
must not overwrite what ``merged/`` already said), the external-mask entry
that is neither a group nor a path, the bracketed ``src`` that does not parse
to one list, the ``custom_model`` and U-Net checkpoints that are actually on
disk, the Explain CV inputs that exist, the stack whose planes are all gone,
the size sweep with no extension filter, the one-dimensional field that
projects no mask stack, and the card that has no source to report free disk
for.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr import validate as V


@pytest.fixture()
def plate(tmp_path):
    """Build plate folders on demand: ``plate("p1", merged=(4, 4, 5))``."""

    def _build(name, *, merged=None, stack=None, raw=()):
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        for sub, shape in (("merged", merged), ("stack", stack)):
            if shape is None:
                continue
            folder = root / sub
            folder.mkdir(exist_ok=True)
            if shape == "empty":
                (folder / "readme.txt").write_text("no arrays here")
            else:
                np.save(folder / "field_0.npy", np.zeros(shape, dtype=np.uint16))
        for filename in raw:
            (root / filename).write_bytes(b"")
        return str(root)

    return _build


@pytest.fixture()
def existing_file(tmp_path):
    """Create and return a real file path."""

    def _make(name, payload=b"x"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)

    return _make


def _messages(problems):
    return "\n".join(str(p) for p in problems)


# ---------------------------------------------------------------------------
# building the known-key universe
# ---------------------------------------------------------------------------

def test_a_category_that_is_not_a_list_of_keys_contributes_nothing(monkeypatch):
    """A ``categories`` value that is a label, not a key list, is stepped over.

    ``categories`` maps a heading to the keys shown under it. Iterating a
    value that is a bare string would fold every letter of it into the
    known-key universe and poison typo detection, so only real containers are
    read.
    """
    from spacr import settings as S

    monkeypatch.setattr(S, "categories", {
        "Real group": ["zz_key_from_a_real_group"],
        "Heading": "zz_not_a_group_at_all",
    }, raising=False)
    monkeypatch.setattr(V, "_KNOWN_KEYS_CACHE", None)

    keys = V._known_setting_keys()
    assert "zz_key_from_a_real_group" in keys
    assert "zz_not_a_group_at_all" not in keys
    assert not {"z", "n", "_"} & keys

    # And the key that did survive is the one typo detection can suggest.
    problems = V.validate_settings({"zz_key_from_a_real_grouq": 1}, "mask")
    assert "did you mean 'zz_key_from_a_real_group'" in _messages(problems)


def test_a_settings_helper_that_returns_a_list_contributes_no_keys(monkeypatch):
    """Only the helpers that hand back a settings mapping widen the universe.

    The sweep calls every ``get_*``/``set_*`` helper in :mod:`spacr.settings`
    and harvests the keys of whatever comes back. A helper that returns a
    list of names rather than a dict must contribute nothing, or its elements
    would be mistaken for setting keys.
    """
    from spacr import settings as S

    monkeypatch.setattr(S, "get_zz_list_helper",
                        lambda settings: ["zz_list_element"], raising=False)
    monkeypatch.setattr(S, "get_zz_dict_helper",
                        lambda settings: {"zz_dict_key": 7}, raising=False)
    monkeypatch.setattr(V, "_KNOWN_KEYS_CACHE", None)

    keys = V._known_setting_keys()
    assert "zz_dict_key" in keys
    assert "zz_list_element" not in keys
    assert "src" in keys


# ---------------------------------------------------------------------------
# what the inventory reads off disk
# ---------------------------------------------------------------------------

def test_a_stack_folder_holding_no_arrays_leaves_the_channels_to_the_raw_scan(plate):
    """An empty ``stack/`` says nothing, so the filenames are read instead."""
    empty_stack = plate(
        "p_empty_stack", stack="empty",
        raw=("plate1_A01_T0001F001L01A01Z01C01.tif",
             "plate1_A01_T0001F001L01A01Z01C02.tif"))
    plan = V.describe_plan({"src": empty_stack}, "mask")
    assert "2 raw image files" in plan
    assert "channel stack" not in plan
    # The channel count came from the filenames, not from any array.
    assert "channels      2 (01, 02)" in plan

    # Same folder with one real array in stack/: now the stack is what counts.
    stacked = plate("p_real_stack", stack=(4, 4, 3))
    stacked_plan = V.describe_plan({"src": stacked}, "mask")
    assert "1 channel stack (.npy)" in stacked_plan
    assert "channels      3 (0, 1, 2)" in stacked_plan


def test_the_merged_planes_survive_a_stack_of_a_different_depth(plate):
    """``merged/`` establishes the array depth; ``stack/`` must not overwrite it.

    ``merged/`` arrays carry the image channels *and* the mask planes appended
    later, so they are deeper than the raw channel stack. The plan has to
    report the channel count from ``stack/`` and the plane count from
    ``merged/`` — reporting three planes for a five-plane array is how a
    ``cell_mask_dim`` of 4 gets waved through.
    """
    both = plate("p_both", merged=(4, 4, 5), stack=(4, 4, 3))
    plan = V.describe_plan({"src": both}, "mask")
    assert "channels      3 (0, 1, 2)" in plan
    assert "array planes  5 (indices 0-4)" in plan

    # Without merged/, the stack is what sets the plane count.
    stack_only = plate("p_stack_only", stack=(4, 4, 3))
    assert "array planes  3 (indices 0-2)" in V.describe_plan(
        {"src": stack_only}, "mask")


def test_a_stack_of_zero_planes_reports_a_count_but_no_channel_ids(plate):
    """An array whose last axis is empty has a channel count and no ids.

    ``(4, 4, 0)`` is what a truncated or mis-written stack looks like: the
    file loads, the shape is real, and there is not a single channel index to
    name. The plan must not print an empty ``()`` after the count.
    """
    empty_planes = plate("p_zero", stack=(4, 4, 0))
    plan = V.describe_plan({"src": empty_planes}, "mask")
    assert "channels      0" in plan
    assert "channels      0 (" not in plan

    # A stack with planes does name them, in the same shape of line.
    filled = plate("p_two", stack=(4, 4, 2))
    assert "channels      2 (0, 1)" in V.describe_plan({"src": filled}, "mask")


# ---------------------------------------------------------------------------
# resolving the source setting
# ---------------------------------------------------------------------------

def test_an_external_mask_entry_that_is_neither_group_nor_path_is_dropped(
        existing_file):
    """External Masks takes dicts and strings; anything else contributes no root.

    The drop target hands over a list built from whatever the user dragged in,
    and a stray ``None`` (or a number from a hand-edited settings CSV) must
    not become a folder to inventory.
    """
    mask_file = existing_file("drop/mask_a.tif")
    roots = V._src_values(
        {"inputs": [7, None, {"root": "/data/run"}, mask_file]},
        "external_masks")
    assert roots == ["/data/run", os.path.dirname(mask_file)]


def test_a_bracketed_src_that_is_not_one_list_stays_a_single_path():
    """``src`` is split only when the brackets really are one list literal.

    A string that starts with ``[`` and ends with ``]`` is parsed, but the
    parse can hand back a tuple of lists rather than the list of paths the
    caller meant. That is not a source list, so the original string is kept
    intact rather than silently reinterpreted.
    """
    assert V._src_values({"src": "['/a'],['/b']"}) == ["['/a'],['/b']"]
    # The shape it IS meant to split.
    assert V._src_values({"src": "['/a', '/b']"}) == ["/a", "/b"]


# ---------------------------------------------------------------------------
# paths that are actually on disk
# ---------------------------------------------------------------------------

def test_a_custom_model_that_exists_is_not_reported_missing(plate, existing_file):
    """The ``custom_model`` check fires on the path, not on its presence."""
    src = plate("p_custom", merged=(4, 4, 4))
    model = existing_file("models/cellpose_finetuned.pt")

    good = V.validate_settings({"src": src, "cell_channel": 0,
                                "custom_model": model}, "cellpose_masks")
    assert "does not exist" not in _messages(good)

    missing = os.path.join(os.path.dirname(model), "gone.pt")
    bad = V.validate_settings({"src": src, "cell_channel": 0,
                               "custom_model": missing}, "cellpose_masks")
    assert any(p.setting == "custom_model" and p.is_error
               and "does not exist" in p.message for p in bad)


def test_a_unet_checkpoint_that_exists_is_not_reported_missing(plate,
                                                               existing_file):
    """Each organelle slot is judged on its own U-Net path.

    Both slots ask for ``method='unet'``; only the one whose checkpoint is
    missing may be reported, and the loop has to carry on to reach it.
    """
    src = plate("p_unet", merged=(4, 4, 4))
    unet = existing_file("models/organelle_unet.pth")
    settings = {
        "src": src, "cell_channel": 0,
        "organelle_method": "unet", "organelle_unet_model_path": unet,
        "organelleb_method": "unet",
        "organelleb_unet_model_path": os.path.join(
            os.path.dirname(unet), "absent.pth"),
    }
    problems = V._check_required_paths(settings, "mask")
    reported = {p.setting for p in problems}
    assert "organelleb_unet_model_path" in reported
    assert "organelle_unet_model_path" not in reported


def test_explain_cv_accepts_the_inputs_that_are_on_disk(existing_file):
    """Explain CV names only the input it cannot find, and only when absent."""
    db = existing_file("run/measurements.db")
    predictions = existing_file("run/predictions.csv")

    ok = V._check_app_specific({
        "db_path": db, "predictions_file": predictions,
        "surrogate_model": "xgboost", "surrogate_split_by": "plate",
    }, "explain_cv")
    assert ok == []

    half = V._check_app_specific({
        "db_path": db,
        "predictions_file": os.path.join(os.path.dirname(db), "gone.csv"),
        "surrogate_model": "xgboost", "surrogate_split_by": "plate",
    }, "explain_cv")
    assert [p.setting for p in half] == ["predictions_file"]
    assert "does not exist" in half[0].message


# ---------------------------------------------------------------------------
# the resource card
# ---------------------------------------------------------------------------

def test_the_size_sweep_with_no_extension_filter_counts_every_file(tmp_path):
    """``_dir_bytes`` defaults to every file in the folder, not to none of it.

    Reached directly: both in-module call sites pass an explicit suffix
    tuple, so the documented no-filter default has no caller to drive it.
    """
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "a.npy").write_bytes(b"0" * 10)
    (folder / "notes.txt").write_bytes(b"0" * 4)

    assert V._dir_bytes(str(folder)) == (14, 2, False)
    assert V._dir_bytes(str(folder), (".npy",)) == (10, 1, False)


def test_a_one_dimensional_field_projects_no_mask_stack(plate):
    """A field with no height and width cannot be given a label-plane estimate.

    The mask projection is ``height × width × 2 bytes`` per object. An array
    that has only one axis has no such plane, so the card projects the merged
    arrays and reports zero for the masks rather than inventing a number from
    the wrong axis.
    """
    flat = plate("p_flat")
    merged_dir = os.path.join(flat, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    np.save(os.path.join(merged_dir, "field_0.npy"),
            np.arange(5, dtype=np.uint16))

    card = V.describe_resources({"src": flat, "cell_channel": 0}, "mask")
    assert "one field    5 uint16 = 10 B in memory" in card
    assert "1 mask stack(s) 0 B" in card

    # A real two-dimensional field does get a mask projection.
    solid = plate("p_solid", merged=(8, 8, 3))
    solid_card = V.describe_resources({"src": solid, "cell_channel": 0}, "mask")
    assert "1 mask stack(s) 0 B" not in solid_card
    assert "mask stack(s) 128 B" in solid_card


def test_a_source_that_resolves_to_nothing_gets_no_disk_row(plate):
    """With no folder to write into, the card reports no free space at all.

    External Masks carries its sources inside ``inputs``; a group with no
    root and no paths resolves to nothing, and a "disk free" row for the
    current directory would be an answer about the wrong volume.
    """
    empty = V.describe_resources({"inputs": [{"paths": []}]}, "external_masks")
    assert "disk free" not in empty
    assert empty.startswith("Resources — nothing to project")

    real = V.describe_resources(
        {"src": plate("p_disk", merged=(4, 4, 3)), "cell_channel": 0}, "mask")
    assert "disk free" in real
