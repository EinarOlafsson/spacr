"""What the settings panel makes of a settings dict.

WHAT THIS FILE USED TO BE: CPU coverage for the database half of
``spacr.gui_elements.AnnotateApp`` -- its background writer thread, paging
helpers, ``train_and_classify`` and multi-annotation builders. AnnotateApp
was a Tk window; the Tkinter interface is gone and the class has no
definition left in the tree, so those forty tests guarded nothing and came
out. Annotation now happens in the Qt screens, which have their own tests
under ``tests/qt/``.

What outlived it is ``convert_settings_dict_for_gui``. AnnotateApp once
carried a stale private copy of it, and this file reached the real one
through ``spacr.gui_utils``, which only ever re-exported it. It lives in
``spacr.settings_spec`` -- a module that deliberately imports no GUI
toolkit -- and it is what the Qt settings model reads to decide whether a
setting reaches the panel as a combo, a check or an entry.

These tests stay because they are the ones holding each combo's option list
to what the pipeline will actually accept: ``dataset_mode`` against
``training_basis.TRAINING_BASES``, ``class_balance`` against
``io.CLASS_BALANCE_MODES``, ``cv_group_by`` against ``io.CV_GROUP_LEVELS``,
``seg_qc`` against ``seg_qc.MODES``, and every offered ``loss_type``
against a real ``utils.build_loss`` call. An option the panel offers that
the pipeline cannot dispatch on is a setting the user can pick and then get
silence from, and these assertions are what make the two drift apart loudly
instead of quietly.
"""
from __future__ import annotations

import pytest


# ===========================================================================
# widget-kind classification
# ===========================================================================

def test_convert_settings_dict_classifies_widget_kinds():
    """bool -> check, numbers/lists/strings -> entry, known keys -> combo."""
    from spacr import settings_spec as GU

    out = GU.convert_settings_dict_for_gui({
        "verbose": True,
        "epochs": 10,
        "lr": 0.001,
        "channels_list": [1, 2, 3],
        "name": "abc",
        "nothing": None,
        "metadata_type": "cq1",
        "channels": "[0,1]",
    })

    assert out["verbose"] == ("check", None, True)
    assert out["epochs"] == ("entry", None, 10)
    assert out["lr"] == ("entry", None, 0.001)
    assert out["channels_list"] == ("entry", None, "[1, 2, 3]")
    assert out["name"] == ("entry", None, "abc")
    assert out["nothing"] == ("entry", None, None)
    # special cases ignore the supplied value and use the canned spec
    assert out["metadata_type"] == ("combo",
                                    ["cellvoyager", "cq1", "auto", "custom"],
                                    "cellvoyager")
    kind, options, initial = out["channels"]
    assert kind == "combo" and initial == "[0,1,2,3]" and "[0,1]" in options


def test_convert_settings_dict_uses_real_torchvision_model_list():
    """model_type options extend to the full zoo once torchvision is loaded."""
    pytest.importorskip("torchvision.models")
    from spacr import settings_spec as GU

    kind, options, initial = GU.convert_settings_dict_for_gui(
        {"model_type": "resnet50"}
    )["model_type"]
    assert kind == "combo" and initial == "resnet50"
    assert "resnet50" in options
    assert options == sorted(options)
    # the curated fallback is short; torchvision exposes far more
    assert len(options) > 5


def test_convert_settings_dict_falls_back_when_torchvision_unloaded(monkeypatch):
    """With torchvision absent from sys.modules the curated list is used.

    settings_spec deliberately never *imports* torchvision here -- enumerating
    the zoo costs ~5 s and made the first settings screen sluggish to open --
    it only reads sys.modules. So the fallback is exercised by hiding the
    module, not by blocking the import.
    """
    import sys
    from spacr import settings_spec as GU

    monkeypatch.delitem(sys.modules, "torchvision.models", raising=False)
    out = GU.convert_settings_dict_for_gui({"model_type": "resnet50"})

    kind, options, initial = out["model_type"]
    assert kind == "combo" and initial == "resnet50"
    assert options == list(GU._TORCHVISION_MODELS_CURATED)


# ===========================================================================
# the combos must offer exactly what the pipeline accepts
# ===========================================================================

def test_dataset_mode_combo_matches_the_modes_io_dispatches_on():
    """'recruitment' was offered here and is not a real mode.

    io.generate_training_dataset dispatches on metadata|annotation and
    returns (None, None) for anything else -- so picking 'recruitment' in
    the panel silently produced no dataset at all, with no error to read.

    'measurement' was a real mode and has been retired. It is not offered,
    and this asserts against TRAINING_BASES rather than a hand-written set
    so the two cannot drift: an option the panel offers that the pipeline
    does not dispatch on is the exact bug this test exists for.
    """
    from spacr import settings_spec as GU
    from spacr.training_basis import TRAINING_BASES
    kind, options, initial = GU.convert_settings_dict_for_gui(
        {"dataset_mode": "metadata"})["dataset_mode"]
    assert kind == "combo"
    assert set(options) == set(TRAINING_BASES)
    assert initial == "metadata"


def test_class_balance_combo_matches_io_class_balance_modes():
    from spacr import settings_spec as GU
    from spacr.io import CLASS_BALANCE_MODES
    kind, options, initial = GU.convert_settings_dict_for_gui(
        {"class_balance": "none"})["class_balance"]
    assert kind == "combo"
    assert set(options) == set(CLASS_BALANCE_MODES)
    assert initial == "none"


def test_cv_group_by_combo_matches_io_cv_group_levels():
    from spacr import settings_spec as GU
    from spacr.io import CV_GROUP_LEVELS
    kind, options, initial = GU.convert_settings_dict_for_gui(
        {"cv_group_by": "well"})["cv_group_by"]
    assert kind == "combo"
    assert set(options) == set(CV_GROUP_LEVELS)
    # well is the safe default: crops from one well are not independent
    assert initial == "well"


def test_seg_qc_combo_matches_seg_qc_modes():
    from spacr import settings_spec as GU
    from spacr.seg_qc import MODES
    kind, options, initial = GU.convert_settings_dict_for_gui(
        {"seg_qc": "report"})["seg_qc"]
    assert kind == "combo"
    assert set(options) == set(MODES)
    # report, not flag: surface the problem, do not silently filter fields
    assert initial == "report"


def test_optimizer_and_loss_combos_are_all_accepted_by_the_pipeline():
    """Every offered option must survive the real dispatch, not just look right."""
    from spacr import settings_spec as GU
    specs = GU.convert_settings_dict_for_gui(
        {"optimizer_type": "adamw", "loss_type": "auto"})

    _, opt_options, _ = specs["optimizer_type"]
    assert set(opt_options) == {
        "adamw", "adam", "adamax", "adagrad", "adadelta", "asgd",
        "sgd", "rmsprop", "nadam", "radam",
    }

    torch = pytest.importorskip("torch")
    from spacr.utils import build_loss
    _, loss_options, _ = specs["loss_type"]
    counts = torch.tensor([80.0, 20.0])
    for name in loss_options:
        n_classes = 1 if name == "binary_cross_entropy_with_logits" else 2
        fn = build_loss(name, num_classes=n_classes, class_counts=counts)
        assert callable(fn), name
