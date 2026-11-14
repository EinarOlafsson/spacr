"""Table-driven tests of ``spacr.model_check``: loading a custom file and the full report.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: :func:`spacr.model_check._load_custom` (each kind of
file that is not a runnable model is refused with a sentence saying what it
holds) and :func:`spacr.model_check.check_model` (every way a chosen model
can be wrong becomes an exact :class:`~spacr.model_check.ModelReport`, never
an exception).  Saved models are tiny CPU torch objects under ``tmp_path``.
"""
from __future__ import annotations

import builtins

import pytest

from spacr import model_check as MC
from spacr.model_check import ModelReport

torch = pytest.importorskip("torch")

NO_CHANNELS = "no channels are chosen yet, so the input was not checked"
CUSTOM_NOTE = "a custom model was loaded, so model_type is not used"
BAD_CLASSES = ("the classes setting is invalid: class 'a' is defined as 1; it "
               "needs a column and a value, or random_complement")


def test_load_custom_refuses_files_that_cannot_run_and_returns_ones_that_can(subtests, tmp_path):
    """Plain dicts, checkpoints, non-runnable objects and unreadable bytes are named; models load."""
    cases = [
        # (case id, object saved with torch.save or raw bytes, exact ValueError text or None)
        ("plain state dict", {"layer.weight": torch.zeros(2, 2)},
         "this file holds a plain dictionary, not a model; if it is a state "
         "dict, set model_type and use resume_checkpoint"),
        ("checkpoint wrapper with state_dict", {"state_dict": {}, "epoch": 3},
         "this is a checkpoint of weights, not a model; set model_type to the "
         "architecture and use resume_checkpoint to load these weights into it"),
        ("checkpoint wrapper with model_state_dict", {"model_state_dict": {}},
         "this is a checkpoint of weights, not a model; set model_type to the "
         "architecture and use resume_checkpoint to load these weights into it"),
        ("a list is not runnable", [1, 2, 3],
         "this file holds a list, which is not something that can be run on an image"),
        ("a string is not runnable", "weights",
         "this file holds a str, which is not something that can be run on an image"),
        ("a saved module loads", torch.nn.Sequential(torch.nn.Flatten(),
                                                     torch.nn.Linear(4, 8),
                                                     torch.nn.Linear(8, 2)), None),
    ]
    for index, (case_id, obj, message) in enumerate(cases):
        with subtests.test(msg=case_id):
            path = tmp_path / f"case{index}.pth"
            torch.save(obj, path)
            if message is None:
                loaded = MC._load_custom(str(path))
                assert isinstance(loaded, torch.nn.Sequential)
                assert MC._head_size(loaded) == 2
            else:
                with pytest.raises(ValueError) as caught:
                    MC._load_custom(str(path))
                assert str(caught.value) == message
    with subtests.test(msg="bytes torch cannot read"):
        path = tmp_path / "junk.pth"
        path.write_bytes(b"not a model")
        with pytest.raises(ValueError) as caught:
            MC._load_custom(str(path))
        assert str(caught.value).startswith(
            "this file cannot be read as a PyTorch model: ")
        assert caught.value.__cause__ is not None


def _saved_head(tmp_path, outputs=2):
    """A small runnable model with ``outputs`` logits, saved whole."""
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 8),
                                torch.nn.Linear(8, outputs))
    path = tmp_path / "trained.pth"
    torch.save(model, path)
    return str(path)


def test_check_model_turns_every_wrong_choice_into_an_exact_report(subtests, tmp_path,
                                                                   monkeypatch):
    """Each settings shape yields one exact ModelReport and the matching summary line."""
    trained = _saved_head(tmp_path)
    broken = tmp_path / "broken.pth"
    broken.write_bytes(b"not a model")
    known = MC._known_builtin_models()
    unknown_problem = ("'not_a_real_backbone' is not a known spaCR model name; "
                       f"choose one of {', '.join(sorted(known)[:8])}…")
    cases = [
        # (case id, settings, patch, expected report)
        ("no model at all names both settings", {}, None,
         ModelReport(ok=False, source="no model", problems=(
             "no model is chosen: set model_type, or point custom_model_path "
             "at a saved model",))),
        ("a recognised backbone with three channels is compatible",
         {"model_type": "maxvit_t", "classes": ["a", "b"], "train_channels": [0, 1, 2]},
         None, ModelReport(ok=True, source="maxvit_t", channels=3, classes=2)),
        ("an unusual channel count is a note",
         {"model_type": "maxvit_t", "classes": ["a", "b"], "train_channels": [0, 1]},
         None, ModelReport(ok=True, source="maxvit_t", notes=(
             "2 channels: a pretrained backbone expects 3, so its first layer "
             "will be adapted",), channels=2, classes=2)),
        ("one class is not a classifier",
         {"model_type": "maxvit_t", "classes": ["only"]}, None,
         ModelReport(ok=False, source="maxvit_t", problems=(
             "only 1 class is defined; a classifier needs two",),
             notes=(NO_CHANNELS,), classes=1)),
        ("no classes at all",
         {"model_type": "maxvit_t", "train_channels": [0]}, None,
         ModelReport(ok=False, source="maxvit_t", problems=(
             "no classes are defined, so the model's output cannot be checked; "
             "set the Classes dict",), channels=1)),
        ("an unknown backbone is offered the known ones",
         {"model_type": "not_a_real_backbone", "classes": ["a", "b"]}, None,
         ModelReport(ok=False, source="not_a_real_backbone",
                     problems=(unknown_problem,), notes=(NO_CHANNELS,), classes=2)),
        ("a malformed classes dict is reported, not raised",
         {"model_type": "resnet50", "classes": {"a": 1}}, None,
         ModelReport(ok=False, source="resnet50", problems=(BAD_CLASSES,),
                     notes=(NO_CHANNELS,))),
        ("a two-output head on three classes",
         {"custom_model_path": trained, "model_type": "maxvit_t",
          "classes": ["a", "b", "c"], "train_channels": [0, 1, 2]}, None,
         ModelReport(ok=False, source="trained.pth", problems=(
             "the model has 2 output(s) but 3 classes are defined; its final "
             "layer has to be replaced or the classes changed",),
             notes=(CUSTOM_NOTE,), channels=3, classes=3)),
        ("a matching custom head is compatible",
         {"custom_model_path": trained, "classes": ["a", "b"], "train_channels": [0]},
         None, ModelReport(ok=True, source="trained.pth", notes=(CUSTOM_NOTE,),
                           channels=1, classes=2)),
        ("a bad custom file does not hide the malformed classes",
         {"custom_model_path": str(broken), "classes": {"a": 1}}, "refuse",
         ModelReport(ok=False, source="broken.pth", problems=(
             BAD_CLASSES, "this file is not a saved model"))),
        ("a custom model cannot be checked without pytorch",
         {"custom_model_path": str(broken), "classes": ["a", "b"]}, "no torch",
         ModelReport(ok=False, source="broken.pth", problems=(
             "PyTorch is not installed in this environment, so a saved model "
             "cannot be checked",), classes=2)),
    ]
    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    def refuse_model(_path):
        raise ValueError("this file is not a saved model")

    for case_id, settings, patch, expected in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as mp:
            if patch == "refuse":
                mp.setattr(MC, "_load_custom", refuse_model)
            elif patch == "no torch":
                mp.setattr(builtins, "__import__", no_torch)
            report = MC.check_model(settings)
            mp.undo()
            assert report == expected
            if expected.ok:
                assert report.summary() == "; ".join(
                    [f"{expected.source} looks compatible", *expected.notes])
            else:
                assert report.summary() == (f"{expected.source} will not work: "
                                            + "; ".join(expected.problems))
