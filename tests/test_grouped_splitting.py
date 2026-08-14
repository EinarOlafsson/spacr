"""Strict train/test grouping shared by ML, crops, GUI, and surrogates."""

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import pytest

from spacr.classifier_evaluation import (
    SPLIT_LEVELS,
    grouped_split,
    normalize_split_level,
    split_columns_for,
    split_group_values,
)


def _measurements(wells=8, cells_per_well=5):
    rows = []
    for well in range(wells):
        for cell in range(cells_per_well):
            rows.append({
                "plateID": "p1",
                "rowID": f"r{well // 4 + 1}",
                "columnID": f"c{well % 4 + 1}",
                "fieldID": f"f{cell % 2 + 1}",
                "label": cell % 2,
            })
    return pd.DataFrame(rows)


def test_the_user_facing_ladder_and_legacy_cell_aliases_are_exact():
    assert SPLIT_LEVELS == ("cell", "field", "well", "plate")
    for alias in (None, False, "none", "off", "cell"):
        assert normalize_split_level(alias) == "cell"
    with pytest.raises(ValueError, match="group_by.*lower-case"):
        normalize_split_level("Well")


def test_partial_well_columns_are_not_a_coarser_silent_fallback():
    with pytest.raises(ValueError, match="complete identity.*rowID"):
        split_columns_for("well", ["plateID", "columnID"], "screen")


def test_well_split_has_no_overlap_and_reports_groups_and_cells():
    frame = _measurements(wells=8, cells_per_well=5)
    level, groups = split_group_values(group_by="well", frame=frame)
    train, test, report = grouped_split(
        groups, frame["label"], 0.25, seed=4, group_by=level)

    assert not set(groups[train]) & set(groups[test])
    assert report.group_by == "well"
    assert report.test_groups + report.train_groups == 8
    assert report.test_cells + report.train_cells == len(frame)
    assert report.group_fraction == pytest.approx(report.test_groups / 8)
    assert report.cell_fraction == pytest.approx(len(test) / len(frame))
    assert "wells" not in report.summary()  # grammar stays unit-neutral
    assert "cells" in report.summary()


def test_prcfo_is_a_strict_fallback_when_component_columns_were_filtered():
    frame = pd.DataFrame({
        "prcfo": [
            f"exp_plate_r1_c{well}_f1_o{cell}"
            for well in range(1, 5) for cell in range(1, 5)
        ]
    })
    level, groups = split_group_values(group_by="well", frame=frame)
    assert level == "well"
    assert len(np.unique(groups)) == 4
    assert all("exp_plate" in str(group) for group in groups)


def test_crop_filenames_keep_sibling_objects_together():
    paths = [
        f"/crops/p1_A0{well}_f1_o{cell}.png"
        for well in range(1, 5) for cell in range(1, 7)
    ]
    labels = [cell % 2 for _well in range(4) for cell in range(6)]
    level, groups = split_group_values(group_by="well", paths=paths)
    train, test, _report = grouped_split(
        groups, labels, 0.25, seed=1, group_by=level)
    assert not set(groups[train]) & set(groups[test])


@pytest.mark.parametrize("bad_frame", [
    pd.DataFrame({"plateID": ["p1"], "columnID": ["c1"]}),
    pd.DataFrame({
        "plateID": ["p1"], "rowID": [None], "columnID": ["c1"],
    }),
])
def test_missing_or_null_group_metadata_is_refused(bad_frame):
    with pytest.raises(ValueError, match="Cannot split.*well"):
        split_group_values(group_by="well", frame=bad_frame, table="screen")


def test_anonymous_crop_names_are_not_invented_as_wells():
    with pytest.raises(ValueError, match="does not encode a well"):
        split_group_values(group_by="well", paths=["class_001.png"])


def test_one_group_is_refused_with_the_memorisation_explanation():
    with pytest.raises(ValueError, match="one well.*memorised"):
        grouped_split(["w1"] * 20, [0, 1] * 10, 0.2, group_by="well")


def test_one_independent_well_per_class_is_refused_not_randomised():
    groups = np.array((["w0"] * 10) + (["w1"] * 10), dtype=object)
    labels = np.array(([0] * 10) + ([1] * 10))
    with pytest.raises(ValueError, match="every class in both train and test"):
        grouped_split(groups, labels, 0.25, group_by="well")


def test_whole_groups_can_expand_the_holdout_to_keep_every_class():
    groups = np.repeat(["w0", "w1", "w2", "w3"], 6)
    labels = np.repeat([0, 0, 1, 1], 6)
    train, test, report = grouped_split(
        groups, labels, 0.25, seed=0, group_by="well")
    assert set(labels[train]) == {0, 1}
    assert set(labels[test]) == {0, 1}
    assert report.requested_fraction == 0.25
    assert report.group_fraction == 0.5
    assert report.cell_fraction == 0.5


def test_cell_alias_is_reproducible_and_reported_as_a_deliberate_choice():
    labels = np.array([0, 1] * 20)
    groups = np.arange(len(labels), dtype=object)
    first = grouped_split(groups, labels, 0.2, seed=7, group_by="none")
    second = grouped_split(groups, labels, 0.2, seed=7, group_by="cell")
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2].group_by == "cell"
    assert "sibling cells may cross" in first[2].rule


def test_cell_split_keeps_repeated_rows_of_the_same_object_together():
    frame = pd.DataFrame({
        "prcfo": [f"p1_r1_c{i // 4 + 1}_f1_o{i % 4}" for i in range(24)],
    })
    # Repeat the complete table as a second measurement/crop view.
    frame = pd.concat([frame, frame], ignore_index=True)
    labels = np.tile([0, 1, 0, 1], 12)
    level, groups = split_group_values(group_by="cell", frame=frame)
    train, test, report = grouped_split(
        groups, labels, 0.25, seed=3, group_by=level)
    assert not set(groups[train]) & set(groups[test])
    assert "repeated rows of one object stay together" in report.rule


@pytest.mark.parametrize("fraction", [0, 1, -0.1, np.nan, np.inf])
def test_invalid_holdout_fractions_are_refused(fraction):
    with pytest.raises(ValueError, match="strictly between"):
        grouped_split(range(10), [0, 1] * 5, fraction, group_by="cell")


def test_group_and_label_lengths_must_match():
    with pytest.raises(ValueError, match="one group per label"):
        grouped_split(["a"], [0, 1], 0.2, group_by="well")


def test_a_single_rare_cell_cannot_make_a_two_sided_class_split():
    with pytest.raises(ValueError, match="class counts"):
        grouped_split(range(10), [0] * 9 + [1], 0.2, group_by="cell")


def test_generated_crop_dataset_writes_split_provenance_and_keeps_wells(
        tmp_path):
    from spacr.io import generate_dataset_from_lists

    class_data = [[], []]
    for class_index in range(2):
        for well in range(4):
            for cell in range(3):
                path = tmp_path / (
                    f"p1_{chr(65 + class_index)}0{well + 1}_f1_o{cell}.png")
                path.write_bytes(b"crop")
                class_data[class_index].append(str(path))
    root = tmp_path / "dataset"
    train, test = generate_dataset_from_lists(
        str(root), class_data, ["negative", "positive"], test_split=0.25)

    train_wells = {
        "_".join(name.split("_")[:2])
        for cls in os.listdir(train)
        for name in os.listdir(os.path.join(train, cls))
    }
    test_wells = {
        "_".join(name.split("_")[:2])
        for cls in os.listdir(test)
        for name in os.listdir(os.path.join(test, cls))
    }
    assert not train_wells & test_wells
    provenance = json.loads((root / ".spacr_split.json").read_text())
    assert provenance["group_by"] == "well"
    assert provenance["group_fraction"] == pytest.approx(0.25)
    assert provenance["cell_fraction"] == pytest.approx(0.25)


def test_augmented_crops_from_one_well_move_to_only_one_side(tmp_path):
    from spacr.utils import augment_classes

    for class_name, row in (("aug_nc", "A"), ("aug_pc", "B")):
        folder = tmp_path / class_name
        folder.mkdir()
        for well in range(4):
            for cell in range(2):
                (folder / f"p1_{row}0{well + 1}_f1_o{cell}_aug1.png").write_bytes(
                    b"crop")

    augment_classes(
        str(tmp_path), nc=["unused"], pc=["unused"], generate=False,
        move=True, group_by="well", test_size=0.25)
    train_names = list((tmp_path / "aug" / "train" / "nc").iterdir()) + \
        list((tmp_path / "aug" / "train" / "pc").iterdir())
    test_names = list((tmp_path / "aug" / "test" / "nc").iterdir()) + \
        list((tmp_path / "aug" / "test" / "pc").iterdir())
    well = lambda path: "_".join(path.name.split("_")[:2])
    assert not {well(path) for path in train_names} & {
        well(path) for path in test_names}
