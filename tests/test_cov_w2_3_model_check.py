"""Answering "will this model work?" before an hour of training finds out.

:mod:`spacr.model_check` never raises -- it runs from a click. What it does
instead is turn each way a chosen model can be wrong into a sentence naming
the setting that would fix it, and the sentences are what this drives.
"""
from __future__ import annotations

import builtins

import pytest

from spacr import model_check as MC


def test_a_channel_count_that_is_not_a_number_falls_through_to_the_next_key():
    """One unreadable channels entry does not hide the readable one after it."""
    assert MC.expected_channels({"train_channels": "all of them",
                                 "extract_channels": [0, 1, 2]}) == 3
    assert MC.expected_channels({"train_channels": "3"}) == 3
    assert MC.expected_channels({"train_channels": "all",
                                 "extract_channels": None,
                                 "channels": object()}) is None


def test_a_plain_dictionary_is_named_as_not_being_a_model(tmp_path):
    """Weights with no architecture cannot be run, and the fix is named."""
    torch = pytest.importorskip("torch")
    path = tmp_path / "weights.pth"
    torch.save({"layer.weight": torch.zeros(2, 2)}, path)

    with pytest.raises(ValueError) as excinfo:
        MC._load_custom(str(path))
    assert "plain dictionary" in str(excinfo.value)
    assert "resume_checkpoint" in str(excinfo.value)


def test_a_checkpoint_is_told_apart_from_a_bare_state_dict(tmp_path):
    """A wrapper with ``state_dict`` inside gets the checkpoint wording."""
    torch = pytest.importorskip("torch")
    path = tmp_path / "checkpoint.pth"
    torch.save({"state_dict": {}, "epoch": 3}, path)

    with pytest.raises(ValueError) as excinfo:
        MC._load_custom(str(path))
    assert "checkpoint of weights" in str(excinfo.value)


def test_a_file_holding_something_that_cannot_run_says_what_it_holds(tmp_path):
    """An object with no ``forward`` is named by its type, not by a guess."""
    torch = pytest.importorskip("torch")
    path = tmp_path / "a_list.pth"
    torch.save([1, 2, 3], path)

    with pytest.raises(ValueError) as excinfo:
        MC._load_custom(str(path))
    assert "holds a list" in str(excinfo.value)


def test_a_model_whose_head_cannot_be_walked_reports_no_size_rather_than_a_guess():
    """Returning None is the honest answer; a made-up number is worse."""
    class Unwalkable:
        def modules(self):
            raise RuntimeError("this model does not expose its modules")

    assert MC._head_size(Unwalkable()) is None


def test_a_custom_model_cannot_be_checked_without_pytorch(tmp_path, monkeypatch):
    """Without torch there is nothing that can read the file, and it says so."""
    path = tmp_path / "model.pth"
    path.write_bytes(b"anything")

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    report = MC.check_model({"custom_model_path": str(path),
                             "classes": ["a", "b"]})
    assert report.ok is False
    assert report.source == "model.pth"
    assert "PyTorch is not installed" in report.problems[0]


def test_a_builtin_model_spacr_does_not_know_is_offered_the_ones_it_does():
    """An unrecognised backbone name is a problem, not a silent pass.

    ``model_type`` is a TorchVision architecture name. A misspelt one trains
    nothing -- ``choose_model`` prints "Invalid model_type" and returns None
    -- so the check exists to catch it here, at the click, and to name the
    alternatives.
    """
    report = MC.check_model({"model_type": "not_a_real_backbone",
                             "classes": ["a", "b"]})
    assert report.ok is False
    assert any("not a model spaCR knows" in problem
               for problem in report.problems)


def test_a_recognised_backbone_raises_nothing_and_reports_its_channels():
    """A named backbone with two classes and three channels is compatible."""
    report = MC.check_model({"model_type": "maxvit_t", "classes": ["a", "b"],
                             "train_channels": [0, 1, 2]})
    assert report.channels == 3 and report.classes == 2
    assert "will not work" not in report.summary()


def test_a_single_class_is_not_a_classifier():
    """Two classes is the minimum; one is named as the problem it is."""
    report = MC.check_model({"model_type": "maxvit_t", "classes": ["only"]})
    assert report.ok is False
    assert "a classifier needs two" in report.problems[0]


def test_an_unusual_channel_count_is_a_note_rather_than_a_refusal():
    """A pretrained backbone adapts its first layer; that is worth saying."""
    report = MC.check_model({"model_type": "maxvit_t", "classes": ["a", "b"],
                             "train_channels": [0, 1]})
    assert report.ok is True
    assert any("2 channels" in note for note in report.notes)
    assert "looks compatible" in report.summary()


def test_no_model_at_all_names_both_settings_that_would_fix_it():
    """Neither a type nor a path is a choice nobody has made yet."""
    report = MC.check_model({})
    assert report.ok is False
    assert report.source == "no model"
    assert "custom_model_path" in report.problems[0]


def test_a_malformed_classes_dict_is_reported_not_raised():
    """A dialog that crashes the screen is a worse answer than a sentence.

    ``check_model`` promises never to raise. A ``classes`` entry that is a
    bare number rather than a column/value pair has to come back as a
    problem naming the setting, like every other way the choice can be wrong.
    """
    report = MC.check_model({"model_type": "resnet50", "classes": {"a": 1}})
    assert report.ok is False
    assert report.classes is None
    assert report.problems == (
        "the classes setting is invalid: class 'a' is defined as 1; it needs "
        "a column and a value, or random_complement",
    )
    assert "no classes are defined" not in report.summary()


def test_a_bad_custom_file_does_not_hide_the_malformed_classes_problem(
        tmp_path, monkeypatch):
    """Independent input defects are both useful; neither should win."""
    path = tmp_path / "broken.pth"
    path.write_bytes(b"not a model")

    def refuse_model(_path):
        raise ValueError("this file is not a saved model")

    monkeypatch.setattr(MC, "_load_custom", refuse_model)
    report = MC.check_model({"custom_model_path": str(path),
                             "classes": {"a": 1}})

    assert report.source == "broken.pth"
    assert len(report.problems) == 2
    assert report.problems[0].startswith("the classes setting is invalid:")
    assert report.problems[1] == "this file is not a saved model"


def test_a_saved_model_is_checked_against_the_classes_that_were_chosen(tmp_path):
    """A two-output head on a three-class problem is the silent half-failure."""
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Flatten(),
                                torch.nn.Linear(4, 8),
                                torch.nn.Linear(8, 2))
    path = tmp_path / "trained.pth"
    torch.save(model, path)

    assert MC._head_size(MC._load_custom(str(path))) == 2

    report = MC.check_model({"custom_model_path": str(path),
                             "model_type": "maxvit_t",
                             "classes": ["a", "b", "c"],
                             "train_channels": [0, 1, 2]})
    assert report.ok is False
    assert "2 output(s) but 3 classes" in report.problems[0]
    assert "model_type is not used" in report.notes[0]
    assert report.source == "trained.pth"


def test_a_custom_path_that_does_not_exist_falls_back_to_the_model_type(tmp_path):
    """A missing path is a mistake, not a preference for the built-in."""
    kind, name = MC.resolve_model_source(
        {"custom_model_path": str(tmp_path / "gone.pth"),
         "model_type": "maxvit_t"})
    assert (kind, name) == ("builtin", "maxvit_t")
