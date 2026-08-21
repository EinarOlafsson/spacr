"""Images are streamed, not only loaded (instruction 230).

TRAINING SHOULD BE POSSIBLE ON AS MANY COMBINATIONS OF THE DATA AS POSSIBLE.
Streaming makes the combination a setting rather than a directory.

THE TABLE COMES FIRST, AND IT IS SAVED: a training set decided at run time
and never written down cannot be re-made, compared against, or audited when
a model turns out to have learned something it should not have.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from spacr.settings import deep_spacr_defaults
from spacr.stream_dataset import (COORDINATE_COLUMNS, METHOD_SETTINGS,
                                  SELECTION_COLUMNS, SELECTION_FILE,
                                  STREAM_METHODS, build_selection,
                                  coordinate_column, crop_name, cut,
                                  selection_from_arrays,
                                  settings_for_method)


@pytest.fixture
def objects():
    return pd.DataFrame({
        "plateID": ["p1"] * 20, "rowID": ["r1"] * 20,
        "columnID": ["c1"] * 20, "fieldID": ["1"] * 20,
        "cell_id": range(1, 21),
    })


# --------------------------------------------------------------- A: settings

class TestTheSettingsAreRenamedAndMerged:

    def test_image_source_replaces_crop_source(self):
        got = deep_spacr_defaults({})
        assert got["image_source"] in ("load_images", "stream_images")

    def test_the_two_options_are_the_two_choices(self):
        assert deep_spacr_defaults(
            {"crop_source": "merged"})["image_source"] == "stream_images"
        assert deep_spacr_defaults(
            {"crop_source": "pre_generated"})["image_source"] == \
            "load_images"

    def test_an_unknown_source_still_opens_the_module(self):
        """A settings file naming a source spaCR never had should not stop
        the module opening; the panel shows what it resolved to."""
        assert deep_spacr_defaults(
            {"crop_source": "nonsense"})["image_source"] == "load_images"

    def test_three_settings_became_one(self):
        """file_metadata, path_string and file_type described one pattern
        between them, which is three chances to describe it
        inconsistently."""
        got = deep_spacr_defaults({"path_string": "nucleus_png"})
        assert got["load_path_regex"] == "nucleus_png"

    def test_an_old_file_type_still_loads(self):
        got = deep_spacr_defaults({"file_type": "pathogen_png"})
        assert got["load_path_regex"] == "pathogen_png"

    def test_tables_defaults_to_the_four_objects(self):
        assert deep_spacr_defaults({})["tables"] == [
            "cell", "nucleus", "pathogen", "cytoplasm"]

    def test_extract_channels_is_gone(self):
        from spacr.settings import expected_types, tooltips

        assert "extract_channels" not in expected_types
        assert "extract_channels" not in tooltips

    def test_and_its_value_moves_to_train_channels(self):
        """A settings file that set the old key MEANT those channels."""
        got = deep_spacr_defaults({"extract_channels": [0, 1]})
        assert got["train_channels"] == [0, 1]
        assert "extract_channels" not in got

    def test_a_set_train_channels_is_not_overwritten(self):
        got = deep_spacr_defaults({"extract_channels": [0, 1],
                                   "train_channels": ["r", "g"]})
        assert got["train_channels"] == ["r", "g"]


class TestTheCoordinateColumnIsDerived:
    """"coordinate column will always be the same so figure that out from
    object array"."""

    def test_every_object_has_one(self):
        for name in ("cell", "nucleus", "pathogen", "cytoplasm"):
            assert coordinate_column(name)

    def test_it_is_not_asked_for(self):
        got = deep_spacr_defaults({"object_array": "nucleus"})
        assert got["coordinate_columns"] == ["nucleus_id"]

    def test_changing_the_object_changes_it(self):
        assert deep_spacr_defaults(
            {"object_array": "pathogen"})["coordinate_columns"] == \
            ["pathogen_id"]

    def test_an_object_spacr_does_not_measure_raises(self):
        """Guessing a column name produces a table that joins to nothing and
        reports no error."""
        with pytest.raises(KeyError):
            coordinate_column("mitochondrion")


# ------------------------------------------------------- B: the stream method

class TestTheStreamMethod:

    def test_there_are_two(self):
        assert [m for m, _ in STREAM_METHODS] == ["column", "array"]

    def test_column_needs_the_object_array_and_the_channels(self):
        assert set(settings_for_method("column")) == {"object_array",
                                                      "channel_arrays"}

    def test_array_needs_the_mask_the_channels_and_the_box(self):
        assert set(settings_for_method("array")) == {
            "mask_array", "channel_arrays", "bounding_box"}

    def test_the_two_read_different_settings(self):
        """Which is why it is one control rather than a pair of flags."""
        assert set(METHOD_SETTINGS["column"]) != set(METHOD_SETTINGS["array"])

    def test_an_unknown_method_raises(self):
        """Returning an empty tuple would grey every setting and look like a
        UI bug."""
        with pytest.raises(KeyError):
            settings_for_method("telepathy")

    def test_every_named_setting_exists(self):
        from spacr.settings import expected_types

        got = deep_spacr_defaults({})
        for method, _ in STREAM_METHODS:
            for key in settings_for_method(method):
                assert key in got, f"{method} reads {key}, which has no default"
                assert key in expected_types or key == "object_array"


# ------------------------------------------------------ C: the table is first

class TestTheSelectionTableComesFirst:

    def test_it_is_written_before_any_image(self, objects, tmp_path):
        table, path = build_selection(str(tmp_path), objects=objects)
        assert os.path.isfile(path)
        assert os.path.basename(path) == SELECTION_FILE
        assert len(table) == 20

    def test_it_carries_the_train_test_annotation(self, objects, tmp_path):
        table, _ = build_selection(str(tmp_path), objects=objects,
                                   test_split=0.25)
        assert set(table["split"]) == {"train", "test"}
        assert int((table["split"] == "test").sum()) == 5

    def test_it_holds_the_columns_it_promises(self, objects, tmp_path):
        table, _ = build_selection(str(tmp_path), objects=objects)
        assert list(table.columns) == list(SELECTION_COLUMNS)

    def test_the_same_settings_give_the_same_table(self, objects, tmp_path):
        """A shuffle nobody can reproduce turns the saved table into a record
        of one run rather than a recipe."""
        first, _ = build_selection(str(tmp_path / "a"), objects=objects,
                                   seed=7)
        second, _ = build_selection(str(tmp_path / "b"), objects=objects,
                                    seed=7)
        assert list(first["split"]) == list(second["split"])

    def test_a_different_seed_gives_a_different_one(self, objects, tmp_path):
        first, _ = build_selection(str(tmp_path / "a"), objects=objects,
                                   seed=1)
        second, _ = build_selection(str(tmp_path / "b"), objects=objects,
                                    seed=2)
        assert list(first["split"]) != list(second["split"])

    def test_an_object_table_that_names_no_object_raises(self, tmp_path):
        with pytest.raises(ValueError, match="names no object"):
            build_selection(str(tmp_path),
                            objects=pd.DataFrame({"area": [1, 2]}))


class TestTheNpyFallback:
    """"if that does not exist another method must read all npy files in the
    merged folder and record the object numbers in the chosen mask array"."""

    @pytest.fixture
    def merged(self, tmp_path):
        folder = tmp_path / "merged"
        folder.mkdir()
        for field in (1, 2):
            stack = np.zeros((20, 20, 3), dtype=np.int32)
            stack[2:6, 2:6, 2] = 1
            stack[8:12, 8:12, 2] = 2
            stack[14:18, 14:18, 2] = 3
            np.save(folder / f"plate1_A01_{field}_0.npy", stack)
        return str(folder)

    def test_it_finds_the_objects_with_no_table_at_all(self, merged,
                                                       tmp_path):
        table, path = build_selection(str(tmp_path), merged_folder=merged)
        assert len(table) == 6      # three labels in each of two fields
        assert os.path.isfile(path)

    def test_background_is_not_an_object(self, merged):
        table = selection_from_arrays(merged)
        assert "0" not in set(table["objectID"])

    def test_it_records_which_file_each_came_from(self, merged):
        table = selection_from_arrays(merged)
        assert all(str(s).startswith("npy:") for s in table["source"])

    def test_it_annotates_the_split_too(self, merged):
        table = selection_from_arrays(merged, test_split=0.5)
        assert set(table["split"]) == {"train", "test"}

    def test_an_empty_folder_raises(self, tmp_path):
        """A selection table with no rows would stream nothing and report
        success."""
        empty = tmp_path / "nothing"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            selection_from_arrays(str(empty))

    def test_the_object_table_is_preferred_when_it_exists(self, objects,
                                                          merged, tmp_path):
        table, _ = build_selection(str(tmp_path), objects=objects,
                                   merged_folder=merged)
        assert set(table["source"]) == {"object table"}


class TestTheNamingMatchesMeasureCrop:
    """"make sure the images are saved with the same naming convention as
    measure crop"."""

    def test_it_is_measure_crops_own_function(self):
        from spacr.utils import _generate_names

        expected, _folder, _table = _generate_names(
            "plate1_r1_c1_f1", np.asarray([7]), np.asarray([0]),
            np.asarray([0]), "", crop_mode="cell", object_id=7)
        assert crop_name("plate1_r1_c1_f1", 7) == expected

    def test_the_object_id_is_in_the_name(self):
        assert "_7." in crop_name("plate1_r1_c1_f1", 7)

    def test_it_is_a_png(self):
        assert crop_name("plate1_r1_c1_f1", 7).endswith(".png")


class TestBoundingBoxOrTheMaskAlone:

    @pytest.fixture
    def field(self):
        mask = np.zeros((12, 12), dtype=np.int32)
        mask[3:7, 4:9] = 5
        mask[4, 4] = 0          # a notch, so the two cuts differ
        stack = np.ones((12, 12, 3), dtype=float)
        return stack, mask

    def test_the_box_keeps_its_corners(self, field):
        stack, mask = field
        out = cut(stack, mask, 5, bounding_box=True)
        assert out.shape[:2] == (4, 5)
        assert out[1, 0, 0] == 1.0

    def test_without_it_only_the_mask_survives(self, field):
        """"just the ppixels that overlap with the mask"."""
        stack, mask = field
        out = cut(stack, mask, 5, bounding_box=False)
        assert out.shape[:2] == (4, 5)
        assert out[1, 0, 0] == 0.0, "the notch should be zeroed"

    def test_they_are_different_training_sets(self, field):
        stack, mask = field
        box = cut(stack, mask, 5, bounding_box=True)
        shaped = cut(stack, mask, 5, bounding_box=False)
        assert box.sum() != shaped.sum()

    def test_an_absent_object_is_none_not_an_empty_array(self, field):
        stack, mask = field
        assert cut(stack, mask, 99) is None

    def test_it_is_the_shared_cutter(self):
        """A second cutter would drift from the one the on-demand path
        already uses."""
        import inspect

        from spacr import stream_dataset

        assert "crop_object" in inspect.getsource(stream_dataset.cut)


class TestTheStreamingPass:
    """"after the table is generated the streamin begins to generate the
    datasets on disk"."""

    @pytest.fixture
    def screen(self, tmp_path):
        merged = tmp_path / "merged"
        merged.mkdir()
        for field in (1, 2):
            stack = np.zeros((24, 24, 3), dtype=np.int32)
            stack[..., 0] = 5
            stack[..., 1] = 6
            stack[2:8, 2:8, 2] = 1
            stack[10:16, 10:16, 2] = 2
            np.save(merged / f"plate1_A01_{field}_0.npy", stack)
        return str(merged), str(tmp_path / "dataset")

    def test_it_writes_every_selected_object(self, screen):
        from spacr.stream_dataset import stream_dataset

        merged, dst = screen
        report = stream_dataset({"merged_folder": merged, "test_split": 0.5,
                                 "random_seed": 1}, dst)
        assert report["written"] == 4
        assert report["missing"] == 0

    def test_the_table_is_written_first(self, screen):
        """A pass that streamed first and recorded afterwards would record
        what it happened to write rather than what it set out to."""
        from spacr.stream_dataset import stream_dataset

        merged, dst = screen
        report = stream_dataset({"merged_folder": merged}, dst)
        assert os.path.isfile(report["selection"])

    def test_the_splits_become_folders(self, screen):
        from spacr.stream_dataset import stream_dataset

        merged, dst = screen
        stream_dataset({"merged_folder": merged, "test_split": 0.5,
                        "random_seed": 1}, dst)
        assert os.path.isdir(os.path.join(dst, "train"))
        assert os.path.isdir(os.path.join(dst, "test"))

    def test_the_crops_carry_the_object_id(self, screen):
        from spacr.stream_dataset import stream_dataset

        merged, dst = screen
        stream_dataset({"merged_folder": merged, "test_split": 0.5,
                        "random_seed": 1}, dst)
        names = []
        for split in ("train", "test"):
            names += os.listdir(os.path.join(dst, split))
        assert any(n.endswith("_1.npy") for n in names)
        assert any(n.endswith("_2.npy") for n in names)

    def test_a_field_with_no_stack_is_counted_not_skipped(self, tmp_path):
        """A dataset short by a field is a dataset trained on a different
        screen from the one the table describes."""
        from spacr.stream_dataset import stream

        table = pd.DataFrame({
            "plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"],
            "fieldID": ["9"], "objectID": ["1"], "object_array": ["cell"],
            "split": ["train"], "source": ["object table"],
        })
        report = stream(table, str(tmp_path), str(tmp_path / "out"))
        assert report["missing"] == 1
        assert report["trouble"]

    def test_an_empty_selection_says_so(self, tmp_path):
        from spacr.stream_dataset import stream

        report = stream(pd.DataFrame(), str(tmp_path), str(tmp_path / "o"))
        assert "nothing to stream" in " ".join(report["trouble"])

    def test_the_stack_is_read_once_per_field(self, screen, monkeypatch):
        """A field holds hundreds of objects and a merged stack is tens of
        megabytes; re-reading per crop is a minute against an afternoon."""
        from spacr import stream_dataset as module

        merged, dst = screen
        reads = []
        original = np.load

        def counted(path, *args, **kwargs):
            reads.append(str(path))
            return original(path, *args, **kwargs)

        monkeypatch.setattr(module.np, "load", counted)
        module.stream_dataset({"merged_folder": merged}, dst)
        # Two fields, and each .npy opened once for the selection scan and
        # once for the cutting pass.
        assert len(reads) <= 4, reads

    def test_the_recorded_source_file_beats_a_rebuilt_name(self, screen):
        """A merged file is named plate1_A01_1_0.npy while the parsed parts
        come back as r1/c1, so a rebuilt stem matches nothing."""
        from spacr.stream_dataset import stream_dataset

        merged, dst = screen
        report = stream_dataset({"merged_folder": merged}, dst)
        assert report["fields"] == 2, report["trouble"]
