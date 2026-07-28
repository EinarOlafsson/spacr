"""An empty training dataset must say which folder was empty, not `zip(*[])`.

Reported by the user from a real run: generate_training_dataset selected no
rows, wrote empty class folders, and spacrDataset then died in
shuffle_dataset with

    ValueError: not enough values to unpack (expected 2, got 0)

which names neither the directory nor the classes.
"""
import os
import pytest


def _dirs(tmp_path, counts):
    root = tmp_path / "ds"
    for cls, n in counts.items():
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"img_{i}.png").write_bytes(b"x")
    return str(root)


def test_an_empty_dataset_raises_with_the_folder_and_the_classes(tmp_path):
    from spacr.io import spacrDataset
    root = _dirs(tmp_path, {"nc": 0, "pc": 0})
    with pytest.raises(ValueError) as e:
        spacrDataset(root, ["nc", "pc"])
    msg = str(e.value)
    assert "not enough values to unpack" not in msg, "the old opaque error survived"
    assert root in msg, "the message does not name the directory"
    assert "nc" in msg and "pc" in msg, "the message does not name the classes"
    assert "0 file" in msg, "the message does not say the folders were empty"


def test_a_missing_class_folder_is_named_as_missing(tmp_path):
    from spacr.io import spacrDataset
    root = _dirs(tmp_path, {"nc": 0})
    with pytest.raises(ValueError) as e:
        spacrDataset(root, ["nc", "pc"])
    assert "NO SUCH FOLDER" in str(e.value)


def test_the_message_points_at_the_settings_that_cause_it(tmp_path):
    """A user needs the next action, not just the failure."""
    from spacr.io import spacrDataset
    root = _dirs(tmp_path, {"nc": 0, "pc": 0})
    with pytest.raises(ValueError) as e:
        spacrDataset(root, ["nc", "pc"])
    msg = str(e.value)
    assert "class_metadata" in msg and "metadata_type_by" in msg


def test_a_non_empty_dataset_is_unaffected(tmp_path):
    from spacr.io import spacrDataset
    root = _dirs(tmp_path, {"nc": 3, "pc": 2})
    ds = spacrDataset(root, ["nc", "pc"])
    assert len(ds) == 5
    assert set(ds.labels) == {0, 1}


def test_one_empty_class_still_loads(tmp_path):
    """Imbalance is a warning sign, not a reason to refuse to train."""
    from spacr.io import spacrDataset
    root = _dirs(tmp_path, {"nc": 4, "pc": 0})
    ds = spacrDataset(root, ["nc", "pc"])
    assert len(ds) == 4
