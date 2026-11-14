"""Every refusal a named derived-table merge can give, and why it gives it."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from spacr import derived_tables as dt
from spacr.merge_tables import MergeError
from tests.test_derived_tables_requested_merges import (
    database, standard_frames)


@pytest.fixture
def standard(tmp_path):
    return database(tmp_path, standard_frames())


def test_identity_columns_are_the_ones_present():
    frame = pd.DataFrame(columns=["plateID", "rowID", "time_id", "area"])
    assert dt.identity_columns(frame) == ["plateID", "rowID", "time_id"]


def test_a_named_base_joins_the_table_list(standard):
    definition = dt.default_definition(standard, ["pathogen"], base="cell")
    assert definition["base"] == "cell"
    with pytest.raises(MergeError, match="Missing tables: nope"):
        dt.default_definition(standard, ["cell", "nope"])


def test_a_definition_from_another_version_or_with_a_repeat_is_refused(standard):
    definition = dt.default_definition(standard, ["cell", "pathogen"])
    with pytest.raises(MergeError, match="Unsupported"):
        dt.validate_source(standard, dict(definition, version=2))
    repeated = copy.deepcopy(definition)
    repeated["joins"].append(copy.deepcopy(repeated["joins"][0]))
    with pytest.raises(MergeError, match="only once"):
        dt.validate_source(standard, repeated)


def test_text_columns_aggregate_only_by_order_and_count():
    assert dt.allowed_methods(pd.Series(["a", "b"])) == (
        "first", "last", "count", "nunique", "min", "max")


@pytest.mark.parametrize("keys, message", [
    ([], "nonempty list"),
    (["image", "image"], "distinct join columns"),
])
def test_join_keys_must_be_distinct_and_present(keys, message):
    with pytest.raises(MergeError, match=message):
        dt._require_keys(pd.DataFrame({"image": [1]}), keys, "t")


def test_base_keys_must_be_complete():
    with pytest.raises(MergeError, match="missing values"):
        dt._require_keys(pd.DataFrame({"image": [1, None]}), ["image"], "t",
                         unique=True)


def test_saved_definitions_from_another_database_are_refused(standard, tmp_path):
    sidecar = dt.sidecar_path(standard)
    sidecar.write_text(json.dumps({"source": "/elsewhere.db", "definitions": {}}), encoding="utf-8")
    with pytest.raises(MergeError, match="different database"):
        dt.load_definitions(standard)


def test_a_result_named_like_a_source_table_is_refused(standard):
    definition = dt.default_definition(standard, ["cell", "pathogen"], name="cell")
    with pytest.raises(MergeError, match="nonempty name"):
        dt.save_definition(standard, definition)


def test_a_failed_save_leaves_no_temporary_file(standard, monkeypatch):
    definition = dt.default_definition(standard, ["cell", "pathogen"], name="Merged")

    def refuse(*_args):
        raise OSError("read-only")

    monkeypatch.setattr(dt.os, "replace", refuse)
    with pytest.raises(OSError):
        dt.save_definition(standard, definition)
    sidecar = dt.sidecar_path(standard)
    assert not [p for p in sidecar.parent.iterdir()
                if p.name.startswith(sidecar.name + ".")]


def test_a_conflicting_primary_policy_cannot_expand_the_named_base(tmp_path):
    """A recipe cannot replace cell observations with its more numerous children."""
    from spacr.merge_tables import MergePolicy

    identity = {"plateID": "p1", "rowID": "r1", "columnID": "c1", "fieldID": "f1"}
    frames = {
        "cell": pd.DataFrame([{**identity, "object_label": 1, "area": 100.0}]),
        "pathogen": pd.DataFrame([{**identity, "object_label": label,
                                   "cell_id": 1, "area": float(label)}
                                  for label in (1, 2, 3)])}
    path = database(tmp_path, frames)
    before = Path(path).read_bytes()
    recipe = dt.default_definition(path, ["cell", "pathogen"], base="cell",
                                   policy=MergePolicy(primary="pathogen"))
    assert recipe["base"] == "cell"
    assert recipe["policy"]["primary"] == "pathogen"

    with pytest.raises(MergeError, match="increased the base observation count"):
        dt.execute(path, recipe)

    assert Path(path).read_bytes() == before
    assert not dt.sidecar_path(path).exists()
