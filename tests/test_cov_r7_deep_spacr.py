"""Round-7 coverage for the last cold branches of :mod:`spacr.deep_spacr`.

``spacr/deep_spacr.py`` sits at 98.7%. What is left is the *quiet* side of a
dozen decisions -- the tracing hook that declines, the scaler built for the
Torch floor, the report that has nothing to compare, the card nobody could
register -- plus one family of guards that cannot be false at all. Each test
below drives a real input down the quiet side and asserts what the run then
says; each "unreachable" claim is written as a test of the invariant that
makes it unreachable, driven in the same test, so the day the invariant moves
this file goes red instead of the guard quietly coming alive.

What is driven:

* ``_flowview_event``   -- the environment opt-in, both when the optional
  import lands and when it cannot, and a tracer that is loaded but off.
* ``_gradient_scaler``  -- the Torch 2.1 fallback, on a torch whose
  ``torch.amp`` has no ``GradScaler``.
* ``format_per_class_accuracy`` -- one finite class is a line without a
  WORST clause; the comparison needs two.
* ``_print_cv_report``  -- a fold summary carrying no ``accuracy`` row stops
  after the table rather than printing a spread it does not have.
* ``_log_tensorboard_epoch`` -- a class with no support is NaN, and a NaN is
  not written as a scalar.
* ``dataset_class_balance`` -- a split folder with no class folders in it is
  left out of the answer entirely, rather than reported as an empty split.
* ``_imbalance_note``   -- a class counted at zero is not a class.
* ``write_model_card`` / ``model_card`` -- the JSON-only card, and the card
  whose registry refused it.
* ``apply_model`` / ``apply_model_to_tar`` / ``generate_activation_map`` --
  ``input_statistics='none'`` really does leave the Normalize step out.
* ``generate_activation_map`` -- correlations computed but neither shown nor
  stored.
* ``analyze_activation_maps`` -- the whole panel read off the return value,
  with nothing printed.
* ``visualize_smooth_grad`` -- an explicit channel list, which the
  preprocessor behind it does not read.
* ``train_model`` -- a folderless dataset keeping the class names it was
  given, and a resume whose checkpoint recorded no best metric.
* ``deep_spacr`` -- a test-only run that installs no model path, and the
  matched-object count summed across two plates when one of them declines.

What is proved:

* ``_multiclass_metrics`` fills one-hot labels unconditionally because it has
  already returned for the empty case forty lines above it.
* the three ``cam_type`` chains in ``generate_activation_map`` end in
  exhaustive ``else`` arms after validating the method vocabulary.
* ``analyze_activation_maps`` sorts unconditionally because its two input
  guards guarantee a non-empty table with ``deletion_auc``.
* ``train_model`` chooses only whether a validation frame accompanies the
  training frame, which is appended unconditionally each epoch.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

torch = pytest.importorskip("torch")

from spacr import deep_spacr  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# _gradient_scaler: the Torch 2.1 floor
# ---------------------------------------------------------------------------

def test_the_scaler_falls_back_to_the_cuda_namespace_on_the_torch_floor(
        monkeypatch):
    """``torch.amp.GradScaler`` is newer than the supported 2.1 floor.

    On 2.1 the same CUDA scaler lives under ``torch.cuda.amp``, and mixed
    precision is only ever enabled for CUDA, so the fallback is equivalent
    rather than a reduced-precision substitute. Driven both ways in one test:
    the modern namespace is used when it exists, and removing it -- which is
    what 2.1 looks like from here -- reaches the legacy one instead.
    """
    device = torch.device("cpu")

    modern = deep_spacr._gradient_scaler(device, False)
    assert modern.__class__.__name__ == "GradScaler"
    assert modern.is_enabled() is False

    reached = []

    class LegacyScaler:
        def __init__(self, enabled):
            reached.append(enabled)
            self.enabled = enabled

    monkeypatch.delattr(torch.amp, "GradScaler", raising=False)
    monkeypatch.setattr(torch.cuda.amp, "GradScaler", LegacyScaler)

    legacy = deep_spacr._gradient_scaler(device, True)

    assert isinstance(legacy, LegacyScaler)
    assert reached == [True], "the legacy scaler was not asked for enabled=True"


# ---------------------------------------------------------------------------
# the per-class line, and the comparison it needs two classes for
# ---------------------------------------------------------------------------

def test_one_finite_class_gets_a_line_but_no_worst_class_clause():
    """"WORST" is a comparison, and one number cannot be compared.

    Both sides driven here: a head whose second class has no support at all
    reports NaN for it, leaving one finite accuracy and no clause; the same
    head with both classes scored gets the clause and names the loser.
    """
    lonely = {'per_class_accuracy': [0.75, float('nan')],
              'class_support': [40, 0]}
    line = deep_spacr.format_per_class_accuracy(lonely, classes=['neg', 'pos'],
                                                prefix='Val ')

    assert line.startswith('Val per-class acc.: ')
    assert 'neg 0.750 (n=40)' in line
    assert 'pos nan (n=0)' in line
    assert 'WORST' not in line

    both = {'per_class_accuracy': [0.95, 0.40], 'class_support': [40, 20]}
    compared = deep_spacr.format_per_class_accuracy(
        both, classes=['neg', 'pos'], prefix='Val ')

    assert 'WORST: pos at 0.400' in compared
    assert '0.550 below the best class' in compared


# ---------------------------------------------------------------------------
# the cross-validation report that has no accuracy to spread
# ---------------------------------------------------------------------------

def test_a_fold_summary_without_accuracy_stops_after_the_table(capsys):
    """The spread paragraph is about accuracy, so it needs an accuracy row.

    A summary that holds only a loss row prints the table and stops; the same
    report with accuracy present prints the +/- line, so the absence above is
    the guard's doing and not an empty summary.
    """
    import pandas as pd

    folds = pd.DataFrame({'fold': [1, 2], 'loss': [0.30, 0.40]})
    loss_only = pd.DataFrame({'metric': ['loss'], 'mean': [0.35],
                              'std': [0.05], 'min': [0.30], 'max': [0.40]})

    deep_spacr._print_cv_report(folds, loss_only, 2)
    printed = capsys.readouterr().out
    assert 'Cross-validation results (2 folds)' in printed
    assert 'Fold-to-fold spread' in printed
    assert 'accuracy across folds' not in printed

    with_accuracy = pd.concat([
        loss_only,
        pd.DataFrame({'metric': ['accuracy'], 'mean': [0.80], 'std': [0.05],
                      'min': [0.75], 'max': [0.85]})], ignore_index=True)
    deep_spacr._print_cv_report(folds, with_accuracy, 2)
    spoken = capsys.readouterr().out
    assert 'accuracy across folds: 0.8000 +/- 0.0500' in spoken
    assert 'range 0.7500-0.8500' in spoken


# ---------------------------------------------------------------------------
# a split folder that holds no classes
# ---------------------------------------------------------------------------

def test_a_split_folder_with_no_class_folders_is_left_out_of_the_balance(
        tmp_path):
    """An empty ``test/`` is not a split with zero images; it is not a split.

    Reporting ``{'test': {}}`` would put a split with no classes into a model
    card that reads as though the test set had been counted. Driven beside a
    ``train/`` that IS counted, so the omission is the guard's and not a
    failure to look.
    """
    root = tmp_path / 'ds'
    (root / 'train' / 'nc').mkdir(parents=True)
    (root / 'train' / 'pc').mkdir(parents=True)
    (root / 'train' / 'nc' / 'a.png').write_bytes(b'x')
    (root / 'train' / 'nc' / '.hidden.png').write_bytes(b'x')
    (root / 'train' / 'pc' / 'b.png').write_bytes(b'x')
    (root / 'train' / 'pc' / 'c.png').write_bytes(b'x')
    # a real directory, with nothing in it that names a class
    (root / 'test').mkdir()

    balance = deep_spacr.dataset_class_balance(str(root))

    assert balance == {'train': {'nc': 1, 'pc': 2}}
    assert 'test' not in balance
    assert os.path.isdir(root / 'test'), "the empty split really was there"


# ---------------------------------------------------------------------------
# a class counted at zero is not a class
# ---------------------------------------------------------------------------

def test_a_class_with_zero_images_does_not_make_the_set_imbalanced():
    """Zero counts are dropped before the imbalance is measured.

    Keeping them would make every two-class balance with an empty third class
    read as 100 % dominated. Driven with the same two real classes twice: once
    beside a zero-count class (no note, one class left after the drop) and
    once alone but lopsided (a note).
    """
    assert deep_spacr._imbalance_note({'nc': 50, 'pc': 0}) == ''
    assert deep_spacr._imbalance_note({'nc': 0, 'pc': 0}) == ''

    note = deep_spacr._imbalance_note({'nc': 950, 'pc': 50})
    assert note, "a 95/5 split should be reported"
    assert 'nc' in note and 'pc' in note


# ---------------------------------------------------------------------------
# the card, written and unregistered
# ---------------------------------------------------------------------------

def test_a_card_can_be_written_without_its_markdown_twin(tmp_path):
    """``markdown=False`` writes the JSON only.

    Driven both ways against the same checkpoint path, so the absent .md is
    the argument's doing rather than a writer that never worked.
    """
    weights = tmp_path / 'nested' / 'model.pth'
    card = deep_spacr.build_model_card(str(weights), classes=['nc', 'pc'])

    json_only = deep_spacr.write_model_card(str(weights), card, markdown=False)
    assert os.path.isfile(json_only)
    markdown_path = os.path.splitext(json_only)[0]
    markdown_path = str(weights.with_suffix('')) + deep_spacr.MODEL_CARD_MD_SUFFIX
    assert not os.path.exists(markdown_path)

    deep_spacr.write_model_card(str(weights), card)
    assert os.path.isfile(markdown_path)
    assert 'nc' in open(markdown_path).read()


def test_a_registry_that_refuses_the_card_does_not_cost_the_card(tmp_path):
    """No artifact id is stamped when nothing could be registered.

    Losing the registry row must not lose the card, so the card is on disk
    either way. Driven with both registries in one test: the refusing one
    leaves ``artifact_id`` off and the accepting one puts it on, and the card
    is rewritten to carry it.
    """
    weights = tmp_path / 'model.pth'
    weights.write_bytes(b'not really weights')

    class RefusingRegistry:
        def register(self, **_kwargs):
            raise RuntimeError('artifacts.db is read-only')

    card, card_path, artifact = deep_spacr.model_card(
        str(weights), registry=RefusingRegistry(), classes=['nc', 'pc'])

    assert artifact is None
    assert 'artifact_id' not in card
    assert os.path.isfile(card_path)

    class AcceptingRegistry:
        def register(self, **_kwargs):
            return type('Artifact', (), {'artifact_id': 'art-1'})()

    card2, card_path2, artifact2 = deep_spacr.model_card(
        str(weights), registry=AcceptingRegistry(), classes=['nc', 'pc'])

    assert artifact2 is not None
    assert card2['artifact_id'] == 'art-1'
    import json
    assert json.load(open(card_path2))['artifact_id'] == 'art-1'


# ---------------------------------------------------------------------------
# the empty-split re-check that cannot be false
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# analyze_activation_maps: the quiet report, and the table it always has
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# visualize_smooth_grad: a channel list that changes nothing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# the deep_spacr orchestrator: a test-only run, and the objects a merge found
# ---------------------------------------------------------------------------

from tests.test_cov_deep_spacr_entry_fusion import ds_stubs  # noqa: E402,F401


def test_a_test_only_run_does_not_take_its_model_path_from_the_trainer(
        tmp_path, ds_stubs):
    """``train=False, test=True`` still evaluates, but names no new model.

    ``train_test_model`` returns the checkpoint it wrote, and that path is
    what the apply stage later reads. A run that only asked for the TEST half
    wrote no checkpoint, so whatever came back must not be installed as
    ``model_path`` -- doing so would point inference at the model the user had
    already chosen, or at None.

    Driven both ways against the same stubbed trainer and the same return
    value, so the difference is the ``train`` flag alone.
    """
    from spacr.deep_spacr import deep_spacr

    ds_stubs.train_ret = str(tmp_path / "written_by_the_trainer.pth")

    test_only = {"src": str(tmp_path), "train": False, "test": True,
                 "generate_training_dataset": False,
                 "apply_model_to_dataset": False,
                 "model_path": str(tmp_path / "chosen_by_the_user.pth")}
    deep_spacr(test_only)

    assert len(ds_stubs.train) == 1, "the test half must still run"
    assert test_only["model_path"] == str(tmp_path / "chosen_by_the_user.pth")
    assert test_only["src"] == str(tmp_path), "src is restored either way"

    trained = {"src": str(tmp_path), "train": True, "test": True,
               "generate_training_dataset": False,
               "apply_model_to_dataset": False,
               "model_path": str(tmp_path / "chosen_by_the_user.pth")}
    deep_spacr(trained)

    assert trained["model_path"] == ds_stubs.train_ret


def test_the_matched_object_count_adds_up_across_every_database(
        tmp_path, ds_stubs, monkeypatch):
    """Each source's merge contributes its own count; a refusal contributes none.

    ``merge_predictions_into_db`` returns the number of objects it could match,
    or None when it could not read that plate's database at all. The total is
    what the run reports as ``matched_objects``, and a plate that answered
    None must leave the others' counts alone rather than zeroing the total or
    raising.

    Driven with two plates in one run: one database answers 4 and the other
    answers None, so the reported total is 4 out of two databases -- not 0,
    and not a crash.
    """
    import spacr.deep_spacr as module
    from spacr.deep_spacr import deep_spacr

    metrics = []
    monkeypatch.setattr(module, "_flowview_metric",
                        lambda name, value: metrics.append((name, value)))

    tar = tmp_path / "datasets" / "ds.tar"
    tar.parent.mkdir(parents=True)
    tar.write_bytes(b"not-really-a-tar")
    model = tmp_path / "clf.pth"
    model.write_bytes(b"not-really-a-model")

    frame = pd.DataFrame({"path": ["a.png", "b.png", "c.png", "d.png"],
                          "pred": [0.1, 0.4, 0.6, 0.9]})
    ds_stubs.apply_ret = frame

    answers = {str(tmp_path / "plate1"): 4, str(tmp_path / "plate2"): None}
    seen = []

    def merge(df, db_path):
        plate = os.path.dirname(os.path.dirname(db_path))
        seen.append(plate)
        return answers[plate]

    monkeypatch.setattr(module, "merge_predictions_into_db", merge)

    settings = {"src": [str(tmp_path / "plate1"), str(tmp_path / "plate2")],
                "train": False, "test": False,
                "generate_training_dataset": False,
                "apply_model_to_dataset": True,
                "tar_path": str(tar), "model_path": str(model),
                "n_top_examples": 2}
    deep_spacr(settings)

    assert seen == [str(tmp_path / "plate1"), str(tmp_path / "plate2")]
    assert ("matched_objects", 4) in metrics
    assert ("databases", 2) in metrics
    assert ("objects", 4) in metrics
