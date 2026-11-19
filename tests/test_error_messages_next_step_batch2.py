"""Import and evaluation refusals preserve data and explain the next step."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from spacr import classifier_evaluation as evaluation
from spacr import convert
from spacr.errors import ConfigurationError


def test_incomplete_group_identity_names_the_column_and_recovery():
    """Refuse missing acquisition metadata without changing the input frame."""
    frame = pd.DataFrame({"plateID": ["plate1"], "rowID": [""], "columnID": ["c1"]})
    before = frame.copy(deep=True)
    with pytest.raises(ValueError) as caught:
        evaluation.split_group_values(group_by="well", frame=frame, table="cells")
    assert "row 0 has no rowID identity" in str(caught.value)
    assert "Fill in the missing acquisition identity from the source records" in str(caught.value)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("problem, recovery", [
    ("missing_column", "Choose a metadata column present in this table"),
    ("conflict", "Correct the conflicting entries using the acquisition records"),
])
def test_well_metadata_refusal_keeps_the_file_and_explains_recovery(
        tmp_path, problem, recovery):
    """A missing column or conflicting well remains a refusal with usable advice."""
    path = tmp_path / "metadata.csv"
    content = "plateID,wellID,compound\nplate1,A01,control\n"
    if problem == "conflict":
        content += "plate1,A01,treated\n"
    path.write_text(content)
    column = "missing" if problem == "missing_column" else "compound"
    with pytest.raises(ValueError) as caught:
        evaluation._well_metadata_lookup(path, [column])
    assert str(path) in str(caught.value)
    assert recovery in str(caught.value)
    assert path.read_text() == content


def test_missing_crop_metadata_names_the_well_to_complete(tmp_path):
    """The missing crop value is refused rather than assigned an invented fold."""
    path = tmp_path / "metadata.csv"
    path.write_text("plateID,wellID,compound\nplate1,A01,control\n")
    with pytest.raises(ValueError) as caught:
        evaluation._crop_metadata_values(
            ["plate1_A02_f1_o1.png"], "compound", path)
    assert "plate1_r1_c2" in str(caught.value)
    assert "Complete the requested metadata column for every crop well" in str(caught.value)


def test_unparseable_crop_keeps_the_original_cause_and_names_recovery():
    """A damaged crop name keeps its parsing cause and original diagnostic."""
    with pytest.raises(ValueError) as caught:
        evaluation._crop_well_keys(["unlabelled.png"], "training crops")
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))
    assert "unlabelled.png" in str(caught.value)
    assert "Use the original spaCR crop filenames" in str(caught.value)


@pytest.mark.parametrize("held_out, recovery", [
    (["plate9"], "Choose hold-out groups listed in the input data"),
    (["plate1", "plate2"], "leaving labelled groups for training"),
])
def test_named_holdout_refusals_keep_group_names_and_give_a_next_step(
        held_out, recovery):
    """Unknown and all-inclusive holdouts remain invalid and explain a repair."""
    with pytest.raises(ValueError) as caught:
        evaluation.grouped_split(
            ["plate1", "plate1", "plate2", "plate2"], [0, 1, 0, 1],
            0.5, group_by="plate", hold_out_groups=held_out)
    assert held_out[0] in str(caught.value)
    assert recovery in str(caught.value)


def test_one_group_cannot_fall_back_to_random_cells(monkeypatch):
    """Keep the grouped-split exception and recommend independent labelled data."""
    from sklearn import model_selection
    from unittest.mock import Mock

    random_split = Mock(side_effect=AssertionError("random fallback called"))
    monkeypatch.setattr(model_selection, "train_test_split", random_split)
    with pytest.raises(evaluation._GroupedSplitImpossible) as caught:
        evaluation.grouped_split(["plate1", "plate1"], [0, 1], 0.5, group_by="plate")
    assert "every labelled cell comes from one plate" in str(caught.value)
    assert "Add labelled data from another independent group" in str(caught.value)
    random_split.assert_not_called()


def test_probability_schema_refusal_names_the_model_inputs():
    """A mismatched class schema is refused without altering probability values."""
    probabilities = np.array([[0.1, 0.9]])
    with pytest.raises(ValueError) as caught:
        evaluation.normalize_probabilities(probabilities, n_classes=3)
    assert "2 columns but 3 classes" in str(caught.value)
    assert "from the same trained model" in str(caught.value)
    np.testing.assert_array_equal(probabilities, [[0.1, 0.9]])


def test_invalid_import_regex_keeps_the_parser_cause_and_recovery(tmp_path):
    """Preview rejects an invalid pattern before scanning or modifying images."""
    with pytest.raises(ConfigurationError) as caught:
        convert.scan(str(tmp_path), metadata_type="custom", custom_regex="[")
    assert isinstance(caught.value.__cause__, re.error)
    assert "custom_regex does not compile" in str(caught.value)
    assert "Correct the reported syntax error" in str(caught.value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_scanr_mapping_refusal_names_the_correction(tmp_path, problem):
    """Invalid well maps are refused by the same plan used for GUI previews."""
    path = tmp_path / "well_map.csv"
    content = "plate,index,well\n"
    if problem == "duplicate":
        content += "plateA,W00001,A01\nplateA,W00001,A02\n"
    else:
        content += "plateA,W00002,A02\n"
    path.write_text(content)
    source = convert.SourceImage(
        "unused.tif", "plateA", "W00001", "field1",
        meta={"metadata_type": "scanr_indexed"})
    plan = convert.plan([source], well_map=str(path))
    assert not plan.ok
    assert plan.mappings == []
    assert len(plan.errors) == 1
    message = plan.errors[0]
    assert "well map" in message.lower()
    assert "preview" in message.lower()
    assert path.read_text() == content


def test_changed_npy_preview_must_be_repeated_before_conversion(tmp_path):
    """A changed source remains blocked and the recovery message names the path."""
    path = tmp_path / "field.npy"
    np.save(path, np.zeros((2, 2), dtype=np.uint8))
    source, = convert.scan(str(tmp_path), extensions=(".npy",))
    np.save(path, np.zeros((3, 3), dtype=np.uint8))
    changed = path.read_bytes()
    with pytest.raises(ConfigurationError) as caught:
        convert._read_npy(source)
    assert str(path) in str(caught.value)
    assert "preview the import again before converting" in str(caught.value)
    assert path.read_bytes() == changed
