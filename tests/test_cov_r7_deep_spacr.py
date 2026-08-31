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

* ``_multiclass_metrics``'s second ``if len(y_true)`` is dead: the function
  has already returned for the empty case forty lines above it.
* the three ``elif`` chains over ``cam_type`` in ``generate_activation_map``
  are exhaustive partitions of a validated four-name vocabulary, so their
  final else-arcs cannot be taken.
* ``analyze_activation_maps``'s ``not table.empty and 'deletion_auc' in
  table.columns`` is answered by the two guards above it.
* ``train_model``'s ``elif accumulated_train_dicts:`` is answered by an
  unconditional append earlier in the same loop body.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

torch = pytest.importorskip("torch")

from spacr import deep_spacr  # noqa: E402
# Imported here, at collection time, so the package is already in
# ``sys.modules`` when a test below replaces ``spacr.flowview.trace`` with a
# double: a replacement made first turns the package's own
# ``from .trace import ...`` into an ImportError.
import spacr.flowview  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# _flowview_event: the optional tracer, three ways of not being there
# ---------------------------------------------------------------------------

@pytest.fixture
def _flowview_unloaded(monkeypatch):
    """Run with ``spacr.flowview`` absent from ``sys.modules``, and restore it.

    The branches under test are reached only when the tracer has NOT already
    been imported, and importing it is exactly what one of them does -- so
    both the module table AND ``spacr.flowview``, the attribute the fresh
    import rebinds on the parent package, have to be put back. Restoring only
    ``sys.modules`` leaves ``from .flowview import ...`` resolving to the
    throwaway copy, and the next test in the file then patches a module
    nothing reads.
    """
    import spacr as spacr_package

    before = {name: module for name, module in sys.modules.items()
              if name.startswith("spacr.flowview")}
    package_attribute = getattr(spacr_package, "flowview", None)
    for name in before:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in [n for n in sys.modules if n.startswith("spacr.flowview")]:
        sys.modules.pop(name, None)
    sys.modules.update(before)
    if package_attribute is not None:
        spacr_package.flowview = package_attribute


def test_the_tracer_is_not_imported_until_the_environment_asks_for_it(
        monkeypatch, _flowview_unloaded):
    """``SPACR_FLOWVIEW`` unset means no import at all; set, it is traced.

    Both halves in one test, because the interesting claim is the ASYMMETRY:
    an ordinary Classify run must not pay for an optional subsystem, and a run
    that opted in must actually get it. ``spacr.flowview.trace`` reads the
    same variable when it is first imported, so the opt-in both admits the
    import and switches the tracer on -- and the stage really lands on the
    graph, which is what the True says.
    """
    monkeypatch.delenv("SPACR_FLOWVIEW", raising=False)
    assert deep_spacr._flowview_event("advance", "scores") is False
    assert "spacr.flowview.trace" not in sys.modules, (
        "an unset SPACR_FLOWVIEW imported the tracer anyway")

    monkeypatch.setenv("SPACR_FLOWVIEW", "TRUE")
    assert deep_spacr._flowview_event("advance", "scores") is True
    assert "spacr.flowview.trace" in sys.modules
    assert sys.modules["spacr.flowview.trace"].is_enabled() is True
    # ...and the graph really moved: the same boundary a second time is
    # refused, because the pipeline graph is monotonic.
    assert deep_spacr._flowview_event("advance", "scores") is False


def test_an_opted_in_run_whose_tracer_will_not_import_still_runs(
        monkeypatch, _flowview_unloaded):
    """A broken optional dependency costs the tracing, not the run.

    ``SPACR_FLOWVIEW=1`` is set, so the guard above lets the import be
    attempted; the package is then made unimportable, which is what a partial
    install looks like from here. The event must come back False rather than
    taking the classification down with it.
    """
    monkeypatch.setenv("SPACR_FLOWVIEW", "1")
    # None in sys.modules is how Python spells "this import must fail"; it is
    # the cheapest faithful stand-in for a package that is not installed.
    monkeypatch.setitem(sys.modules, "spacr.flowview", None)

    with pytest.raises(ImportError):
        from spacr.flowview import trace  # noqa: F401

    assert deep_spacr._flowview_event("advance", "scores") is False


def test_a_loaded_but_disabled_tracer_declines_before_the_stage_lookup(
        monkeypatch):
    """``is_enabled()`` False stops at the guard, without reaching the stages.

    This is the live-panel case: the tracer module is already imported, so the
    environment branch above is skipped entirely and the only thing left to
    ask is whether tracing is on. Driven from both sides on one double:
    disabled returns False and the stage function is never called; enabled
    returns what the stage function said, so the False above is the guard's
    answer and not an accident of the action name.
    """
    calls = []

    class Tracer:
        def __init__(self, enabled):
            self._enabled = enabled

        def is_enabled(self):
            return self._enabled

    class Stages:
        @staticmethod
        def _advance(node_id):
            calls.append(node_id)
            return True

    monkeypatch.setitem(sys.modules, "spacr.flowview.trace", Tracer(False))
    monkeypatch.setattr(sys.modules["spacr.flowview"], "_classify_stages",
                        Stages, raising=False)
    assert deep_spacr._flowview_event("advance", "scores") is False
    assert calls == [], "a disabled tracer still reached the stage table"

    monkeypatch.setitem(sys.modules, "spacr.flowview.trace", Tracer(True))
    assert deep_spacr._flowview_event("advance", "scores") is True
    assert calls == ["scores"]


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

    monkeypatch.delattr(torch.amp, "GradScaler")
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
# tensorboard: a class with no support has no scalar
# ---------------------------------------------------------------------------

class _RecordingWriter:
    """The two SummaryWriter methods ``_log_tensorboard_epoch`` calls."""

    def __init__(self):
        self.scalars = {}
        self.flushes = 0

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = (value, step)

    def flush(self):
        self.flushes += 1


def test_a_class_with_no_support_is_not_written_as_a_scalar():
    """A NaN accuracy would draw a broken line, so it is skipped.

    Driven on one epoch that has both: class ``neg`` was scored and gets its
    scalar, class ``pos`` had no true examples in the split and does not.
    """
    writer = _RecordingWriter()
    train = {'loss': 0.4, 'accuracy': 0.9, 'f1_macro': 0.88, 'lr': 1e-4,
             'per_class_accuracy': [0.9, float('nan')],
             'class_support': [40, 0]}

    deep_spacr._log_tensorboard_epoch(writer, train, {}, epoch=3,
                                      classes=['neg', 'pos'])

    assert writer.scalars['accuracy_neg/train'] == (0.9, 3)
    assert 'accuracy_pos/train' not in writer.scalars
    assert writer.scalars['loss/train'] == (0.4, 3)
    assert writer.scalars['learning_rate'] == (1e-4, 3)
    assert writer.flushes == 1


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

def test_the_one_hot_build_is_never_asked_about_an_empty_split():
    """``_multiclass_metrics``'s second ``if len(y_true)`` is dead.

    The function returns a fully NaN metrics dict for an empty split at its
    top (deep_spacr.py:637-652), so by the time the one-hot matrix is built
    (:637 having returned) ``y_true`` is non-empty by construction and the
    guard at :670 is always True.

    Both halves driven: the empty call returns before any array work and
    reports the class schema it still knows, and a non-empty call comes back
    with a real one-vs-rest average precision -- which is the value the
    guarded statement exists to make computable.
    """
    empty = deep_spacr._multiclass_metrics(
        np.array([], dtype=int), np.zeros((0, 3), dtype=float))

    assert empty['num_classes'] == 3
    assert empty['per_class_accuracy'] == [0.0, 0.0, 0.0]
    assert empty['class_support'] == [0, 0, 0]
    assert np.isnan(empty['accuracy'])
    # the early return is what makes :670 unreachable -- it never reaches the
    # confusion matrix, which is what sklearn 1.7 refuses for an empty split.
    assert 'confusion_matrix' not in empty

    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6],
                      [0.7, 0.2, 0.1]])
    scored = deep_spacr._multiclass_metrics(np.array([0, 1, 2, 0]), probs)

    assert scored['accuracy'] == 1.0
    assert scored['class_support'] == [2, 1, 1]
    assert scored['prauc'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# input_statistics='none': the Normalize step that is really left out
# ---------------------------------------------------------------------------

from tests.test_cov_deep_spacr_apply_model import (  # noqa: E402,F401
    _expected_prob, _save_model as _save_logit_model, _scalar, _tar_settings,
    const_png_dir, const_png_tar)   # the last two are fixtures, used by name


def test_apply_model_normalizes_nothing_when_the_statistics_are_none(
        const_png_dir, tmp_path):
    """``normalize=True`` with ``input_statistics='none'`` is not normalising.

    The two settings mean different things: ``normalize`` is the on/off switch
    a settings file has always carried, and ``input_statistics`` is WHICH
    statistics. 'none' is the one value of the second that empties the first,
    and the proof is arithmetic: the probabilities come back at exactly the
    un-normalised values, and the default 'symmetric' moves every one of them.
    """
    from spacr.deep_spacr import apply_model

    src, values = const_png_dir
    model_path = _save_logit_model(tmp_path / "m.pth", n_out=1)

    plain = apply_model(src, model_path, image_size=32, batch_size=8,
                        normalize=True, n_jobs=0, input_statistics='none')
    shifted = apply_model(src, model_path, image_size=32, batch_size=8,
                          normalize=True, n_jobs=0,
                          input_statistics='symmetric')

    plain_map = {path: _scalar(pred)
                 for path, pred in zip(plain["path"], plain["pred"])}
    shifted_map = {path: _scalar(pred)
                   for path, pred in zip(shifted["path"], shifted["pred"])}
    assert set(plain_map) == set(values)
    for path, value in values.items():
        assert plain_map[path] == pytest.approx(
            _expected_prob(value, normalize=False), abs=1e-5)
        assert shifted_map[path] == pytest.approx(
            _expected_prob(value, normalize=True), abs=1e-5)
    # ...and the two really are different runs: only the mid-grey fixed point
    # of Normalize(0.5, 0.5) may agree.
    moved = [p for p in values if abs(plain_map[p] - shifted_map[p]) > 1e-4]
    assert len(moved) >= len(values) - 1


def test_apply_model_to_tar_normalizes_nothing_when_the_statistics_are_none(
        const_png_tar, tmp_path):
    """The tar reader honours ``input_statistics='none'`` the same way.

    Same claim as above on the other entry point, because these are two
    separately written transform pipelines and a settings key that worked on
    one of them is exactly the kind of thing that quietly does not work on the
    other.
    """
    from spacr.deep_spacr import apply_model_to_tar

    tar_path, values = const_png_tar
    model_path = _save_logit_model(tmp_path / "binary.pth", n_out=1)

    plain = apply_model_to_tar(_tar_settings(
        tar_path, model_path, normalize=True, input_statistics='none'))
    shifted = apply_model_to_tar(_tar_settings(
        tar_path, model_path, normalize=True, input_statistics='symmetric'))

    plain_map = dict(zip(plain["path"], plain["pred"]))
    shifted_map = dict(zip(shifted["path"], shifted["pred"]))
    for path, value in values.items():
        assert plain_map[path] == pytest.approx(
            _expected_prob(value, normalize=False), abs=1e-5)
        assert shifted_map[path] == pytest.approx(
            _expected_prob(value, normalize=True), abs=1e-5)
    assert any(abs(plain_map[p] - shifted_map[p]) > 1e-3 for p in values)


# ---------------------------------------------------------------------------
# generate_activation_map: the transform, and correlations nobody asked to see
# ---------------------------------------------------------------------------

from tests.test_cov_deep_spacr_activation_maps import (  # noqa: E402,F401
    _project, _settings, model_path)   # model_path is a fixture, used by name


def _recorded_transforms(monkeypatch):
    """Spy on the transform ``generate_activation_map`` hands its dataset.

    The Compose is built inside the function and never returned, so the only
    place its steps are observable is the dataset it is given. The spy
    delegates to the real ``TarImageDataset``, so the run itself is unchanged.
    """
    from spacr import io as io_module

    seen = []
    real = io_module.TarImageDataset

    def spy(path, transform=None, **kwargs):
        seen.append(transform)
        return real(path, transform=transform, **kwargs)

    # patched on spacr.io: generate_activation_map imports the class lazily
    # from there on every call, so the module it is defined in is the only
    # place a spy is seen.
    monkeypatch.setattr(io_module, "TarImageDataset", spy)
    return seen


def test_the_activation_transform_drops_normalize_for_statistics_none(
        tmp_path, model_path, monkeypatch):
    """``normalize_input=True`` still composes no Normalize under 'none'.

    Driven both ways in one test against the same tar: 'symmetric' puts a
    ``Normalize`` in the pipeline the dataset is read through and 'none'
    leaves the rest of the pipeline exactly as it was, minus that step.
    """
    from torchvision import transforms as tv
    from spacr.deep_spacr import generate_activation_map

    _root, tar_path, _names = _project(tmp_path, n_images=3)
    seen = _recorded_transforms(monkeypatch)

    generate_activation_map(_settings(tar_path, model_path,
                                      normalize_input=True,
                                      input_statistics='none'))
    generate_activation_map(_settings(tar_path, model_path,
                                      normalize_input=True,
                                      input_statistics='symmetric'))

    assert len(seen) == 2
    without, with_stats = (list(t.transforms) for t in seen)
    assert not any(isinstance(step, tv.Normalize) for step in without)
    normalizers = [step for step in with_stats if isinstance(step, tv.Normalize)]
    assert len(normalizers) == 1
    assert tuple(normalizers[0].mean) == (0.5, 0.5, 0.5)
    # everything else about the two pipelines is the same
    assert ([type(step).__name__ for step in without]
            == [type(step).__name__ for step in with_stats
                if not isinstance(step, tv.Normalize)])


def test_correlations_can_be_computed_without_being_shown_or_stored(
        tmp_path, model_path, capsys):
    """``correlation=True`` with ``plot`` and ``save`` off writes nothing.

    The correlation table is the expensive part; displaying it and pushing it
    into the measurements database are two separate choices below it, and a
    batch run wants neither. Driven against a run with ``save=True``, which
    DOES create the database tables, so the emptiness above is the two guards
    and not a correlation that never ran.
    """
    import sqlite3

    from spacr.deep_spacr import generate_activation_map

    root, tar_path, _names = _project(tmp_path, n_images=3)
    # the maps and their correlations are stored beside the dataset they came
    # from, in a database named after it
    db_path = str(root / "measurements" / "ds.db")

    quiet = _settings(tar_path, model_path, correlation=True, plot=False,
                      save=False)
    quiet["src"] = str(root)
    generate_activation_map(quiet)

    assert not os.path.exists(db_path), "nothing may be stored with save=False"
    grids = root / "datasets" / "ds" / "saliency_image" / "batch_grids"
    assert not os.path.exists(grids), "nothing may be drawn with plot=False"

    stored = _settings(tar_path, model_path, correlation=True, plot=False,
                       save=True)
    stored["src"] = str(root)
    generate_activation_map(stored)

    assert os.path.isfile(db_path), "save=True must reach the database"
    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert any("correlation" in name for name in tables), tables


def test_the_cam_type_chain_is_an_exhaustive_partition_of_four_names(
        tmp_path, model_path):
    """The three ``elif`` ladders over ``cam_type`` have no else to reach.

    ``generate_activation_map`` validates ``cam_type`` at the top
    (deep_spacr.py:3161-3170): anything that is not one of the four legacy
    names must be a registered attribution method, and everything else is
    refused by name. ``use_attribution`` is then exactly "not a legacy name",
    so each of the three ladders --

      * :3235/:3241/:3243  which generator to build,
      * :3253/:3255/:3257  which generator call to make,
      * :3293/:3304        greyscale map or per-channel RGB,

    -- covers the whole vocabulary between its branches, and the final
    else-arc of each cannot be taken. Were one taken, the run would die on an
    unbound local rather than produce a map.

    Driven: the refusal really does fire for a name outside the vocabulary,
    and each of the four legacy names really does produce maps, so the
    partition is complete rather than merely asserted.
    """
    from spacr.attribution import ATTRIBUTION_METHODS
    from spacr.deep_spacr import generate_activation_map

    legacy = ('gradcam', 'gradcam_pp', 'saliency_image', 'saliency_channel')
    # the two elif tests of the first ladder, spelled out
    assert set(legacy) == ({'gradcam', 'gradcam_pp'}
                           | {'saliency_image', 'saliency_channel'})
    # the third ladder's two arms partition the same four names
    assert set(legacy) == ({'saliency_image', 'gradcam', 'gradcam_pp'}
                           | {'saliency_channel'})
    # the legacy names are matched BEFORE the registry is consulted, so the
    # two gradcam entries the registry also carries never route a legacy run
    # down the attribution arm.
    assert {'gradcam', 'gradcam_pp'} <= set(ATTRIBUTION_METHODS)
    assert not {'saliency_image', 'saliency_channel'} & set(ATTRIBUTION_METHODS)

    _root, tar_path, _names = _project(tmp_path, n_images=2)
    with pytest.raises(ValueError, match="unknown cam_type"):
        generate_activation_map(_settings(tar_path, model_path,
                                          cam_type='not_a_cam'))

    for name in ('saliency_image', 'saliency_channel'):
        settings = _settings(tar_path, model_path, cam_type=name, save=True)
        generate_activation_map(settings)
        written = os.path.join(os.path.dirname(tar_path), 'ds', name)
        assert os.path.isdir(written), f"{name} produced no maps"


# ---------------------------------------------------------------------------
# analyze_activation_maps: the quiet report, and the table it always has
# ---------------------------------------------------------------------------

from tests.test_deep_spacr_activation_analysis import (  # noqa: E402
    FAST, IMG, TinyCNN)


def test_the_attribution_report_can_be_read_from_the_return_value_alone(
        capsys):
    """``verbose=False`` prints nothing; the same run with it on prints.

    The whole panel -- the caveat, the sanity verdicts and the agreement --
    is on the returned mapping either way, so a batch caller that does not
    want a page of text per image loses nothing by asking for silence. Both
    runs here are the same model, the same image and the same methods, so the
    only difference in what reaches stdout is the flag.
    """
    from spacr.attribution import NOT_AN_EXPLANATION
    from spacr.deep_spacr import analyze_activation_maps

    model = TinyCNN()
    model.eval()
    torch.manual_seed(7)
    image = torch.rand(3, IMG, IMG)

    quiet = analyze_activation_maps(model, image, methods=FAST, n_steps=3,
                                    sanity_check=True, verbose=False)
    silence = capsys.readouterr().out
    assert silence == ''
    assert NOT_AN_EXPLANATION in quiet['notes']
    assert set(quiet['sanity']) == set(FAST)

    loud = analyze_activation_maps(model, image, methods=FAST, n_steps=3,
                                   sanity_check=True, verbose=True)
    spoken = capsys.readouterr().out
    assert NOT_AN_EXPLANATION in spoken
    assert spoken.count('\n') >= 1 + len(FAST)
    assert set(loud['sanity']) == set(quiet['sanity'])


def test_the_attribution_table_always_has_a_deletion_auc_to_sort_on():
    """``if not table.empty and 'deletion_auc' in table.columns`` is always True.

    ``analyze_activation_maps`` refuses an empty image list at :3398, falls
    back to four default methods when ``methods`` is falsy at :3402, and
    ``compare_methods`` returns one :class:`Attribution` per requested name --
    a FAILED one, not a missing one, for a method that cannot run. So the row
    list is ``len(images) * len(methods) >= 1`` entries long, and every row
    carries ``deletion_auc`` on both sides of the :3419 branch. The sort at
    :3448 can never be skipped.

    All three legs driven here: the empty batch raises rather than reaching
    the table, a bogus method still produces its row WITH the column, and
    ``methods=[]`` fills itself in rather than emptying the table.
    """
    from spacr.deep_spacr import analyze_activation_maps

    model = TinyCNN()
    model.eval()
    torch.manual_seed(7)
    image = torch.rand(3, IMG, IMG)

    with pytest.raises(ValueError, match="at least one image"):
        analyze_activation_maps(model, [], methods=FAST)

    with_failure = analyze_activation_maps(
        model, image, methods=['saliency', 'not_a_method'], n_steps=3,
        sanity_check=False)
    table = with_failure['table']
    assert len(table) == 2
    assert 'deletion_auc' in table.columns
    assert bool(table.loc[table['method'] == 'not_a_method', 'failed'].iloc[0])
    # ...and the failed row still carries the column, as NaN
    assert np.isnan(
        table.loc[table['method'] == 'not_a_method', 'deletion_auc'].iloc[0])

    filled_in = analyze_activation_maps(model, image, methods=[], n_steps=3,
                                        sanity_check=False)
    assert set(filled_in['table']['method']) == {
        'gradcam', 'saliency', 'integrated_gradients', 'occlusion'}
    assert 'deletion_auc' in filled_in['table'].columns
    # sorted by the column the guard asks about, ascending, NaN last
    ordered = filled_in['table']['deletion_auc']
    assert ordered.dropna().is_monotonic_increasing


# ---------------------------------------------------------------------------
# visualize_smooth_grad: a channel list that changes nothing
# ---------------------------------------------------------------------------

def test_an_explicit_channel_list_is_accepted_and_makes_no_difference(
        tmp_path, monkeypatch):
    """``channels`` is threaded through to a preprocessor that ignores it.

    ``visualize_smooth_grad`` defaults ``channels`` to ``[1, 2, 3]`` and hands
    it to :func:`spacr.utils.preprocess_image`, which documents the parameter
    as reserved and never reads it: the image is converted to RGB and used
    whole. So an explicit list takes the other side of that default and must
    produce the identical map -- which is what makes the default assignment
    the only thing the parameter does here.

    Both sides driven on the same image, and the preprocessor is asked
    directly with two different lists so the equality above is its doing.
    """
    from PIL import Image

    from spacr.deep_spacr import visualize_smooth_grad
    from spacr.utils import preprocess_image

    size = 8
    rng = np.random.default_rng(11)
    src = tmp_path / "src"
    src.mkdir()
    pixels = rng.integers(0, 256, (size, size, 3), dtype=np.uint16).astype(np.uint8)
    Image.fromarray(pixels).save(src / "a.png")

    _image, default_tensor = preprocess_image(str(src / "a.png"),
                                              image_size=size, channels=None)
    _image, chosen_tensor = preprocess_image(str(src / "a.png"),
                                             image_size=size, channels=[0, 2])
    assert torch.equal(default_tensor, chosen_tensor)

    model = torch.nn.Sequential(torch.nn.Flatten(),
                                torch.nn.Linear(3 * size * size, 2))
    torch.manual_seed(3)
    model_path = tmp_path / "m.pth"
    torch.save(model, str(model_path))

    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    for channels, folder in ((None, "default"), ([0, 2], "explicit")):
        visualize_smooth_grad(str(src), str(model_path), 1, image_size=size,
                              channels=channels, save_smooth_grad=True,
                              save_dir=str(tmp_path / folder))

    default_map = np.array(Image.open(tmp_path / "default" / "smooth_grad_a.png"))
    explicit_map = np.array(Image.open(tmp_path / "explicit" / "smooth_grad_a.png"))
    assert default_map.shape == (size, size)
    assert np.array_equal(default_map, explicit_map)


# ---------------------------------------------------------------------------
# train_model: the class names a folderless dataset cannot supply
# ---------------------------------------------------------------------------

def _tiny_loader():
    return [(torch.zeros(2, 2), torch.tensor([0, 1]), ["a.png", "b.png"])]


def _tiny_training(monkeypatch, saved, accuracy=0.5):
    """A one-batch CPU training run whose ``_save_model`` calls are recorded."""
    import torch.nn as nn

    import spacr.deep_spacr as ds
    import spacr.io as spacr_io
    import spacr.utils as utils

    model = nn.Linear(2, 2)
    counted = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "choose_model", lambda *_a, **_k: model)
    monkeypatch.setattr(utils, "build_loss", lambda **_k: nn.CrossEntropyLoss())

    def count(loaders, head_dim, src=None, classes=None):
        counted.append({'head_dim': head_dim, 'src': src, 'classes': classes})
        return None

    monkeypatch.setattr(utils, "estimate_class_counts", count)
    monkeypatch.setattr(utils, "suggest_training_changes",
                        lambda *_a, **_k: {"summary": {}, "flags": [],
                                           "suggestions": []})
    monkeypatch.setattr(spacr_io, "_save_progress", lambda *_a, **_k: None)

    def record(_model, _model_type, _train_dict, dst, epoch, _epochs, **kwargs):
        saved.append({'epoch': epoch, **kwargs})
        return os.path.join(dst, f"epoch_{epoch}.pth")

    monkeypatch.setattr(spacr_io, "_save_model", record)
    monkeypatch.setattr(
        ds, "evaluate_model_performance",
        lambda _m, _l, epoch, **_k: ({"epoch": epoch, "loss": 1.0,
                                      "accuracy": accuracy, "f1_macro": 0.5},
                                     [[], []]))
    monkeypatch.setattr(ds, "_open_tensorboard_writer",
                        lambda dst, enabled=True: (None, dst))
    return model, counted


def test_a_dataset_with_no_train_folder_keeps_the_class_names_it_was_given(
        tmp_path, monkeypatch):
    """A tar-backed run's only class naming is the caller's list.

    The folder names of ``src/train/`` are the head order when there IS a
    folder tree, and they win over anything passed in. With no tree there is
    nothing to read, and throwing the caller's list away is what used to make
    every checkpoint and per-class report say ``class_0``/``class_1``.

    Driven both ways against the same folderless ``src``: names supplied are
    carried into the checkpoint and into the class counting, and no names at
    all leaves both at None.
    """
    from spacr.deep_spacr import train_model

    src = tmp_path / "tarred"
    src.mkdir()
    assert not (src / "train").exists()

    named_saves = []
    _model, named_counts = _tiny_training(monkeypatch, named_saves)
    train_model(str(src), str(tmp_path / "named"), "tiny", _tiny_loader(),
                epochs=1, num_classes=2, classes=['nc', 'pc'], schedule=None,
                tensorboard=False, write_card=False)

    assert named_saves and named_saves[-1]['classes'] == ['nc', 'pc']
    assert named_counts == [{'head_dim': 2, 'src': os.path.join(str(src), 'train'),
                             'classes': ['nc', 'pc']}]

    bare_saves = []
    _model, bare_counts = _tiny_training(monkeypatch, bare_saves)
    train_model(str(src), str(tmp_path / "bare"), "tiny", _tiny_loader(),
                epochs=1, num_classes=2, classes=None, schedule=None,
                tensorboard=False, write_card=False)

    assert bare_saves and bare_saves[-1]['classes'] is None
    assert bare_counts == [], "no class names means nothing to count"


def test_a_resume_without_a_best_metric_starts_the_comparison_from_scratch(
        tmp_path, monkeypatch):
    """A checkpoint that recorded no best metric must not pin the bar at zero.

    ``best_val_acc`` starts at the -1.0 sentinel so the FIRST scored epoch is
    always an improvement. A resume restores the previous bar when the
    checkpoint carries one -- and a checkpoint written before the field
    existed carries None, which must leave the sentinel alone rather than
    become ``float(None)``.

    Both driven: resuming with 0.9 recorded leaves an epoch at 0.5 not-best,
    and resuming with nothing recorded makes the same 0.5 epoch the best.
    """
    import torch.nn as nn

    import spacr.deep_spacr as ds
    from spacr.deep_spacr import train_model

    checkpoint = tmp_path / "resume.pth"
    checkpoint.write_bytes(b"stub")

    def run(state, dst):
        saved = []
        _tiny_training(monkeypatch, saved, accuracy=0.5)
        loaded = nn.Linear(2, 2)
        loaded.num_classes = 2
        monkeypatch.setattr(
            ds, "load_model_artifact",
            lambda *_a, **_k: (loaded, {"optimizer_state_dict": {"x": 1}}))
        monkeypatch.setattr(ds, "restore_training_state",
                            lambda *_a, **_k: state)
        train_model(str(tmp_path), str(dst), "tiny", _tiny_loader(),
                    epochs=4, resume_checkpoint=str(checkpoint), num_classes=2,
                    schedule=None, tensorboard=False, write_card=False)
        return saved

    with_bar = run({"epoch": 1, "best_metric": 0.9}, tmp_path / "with_bar")
    assert with_bar[0]['best_metric'] == 0.9
    assert with_bar[0]['is_best'] is False

    without_bar = run({"epoch": 1, "best_metric": None}, tmp_path / "no_bar")
    assert without_bar[0]['best_metric'] == 0.5
    assert without_bar[0]['is_best'] is True


def test_the_epoch_flush_always_has_a_training_row_to_write(tmp_path,
                                                            monkeypatch):
    """``elif accumulated_train_dicts:`` at :3011 cannot be False.

    Every pass of the epoch loop appends the training metrics unconditionally
    at :2911, and the only two places the list is emptied (:3010 and :3013)
    are BELOW the test that reads it -- with no ``continue`` between the
    append and the test. So by the time :3007 is reached the training list is
    always non-empty and the second question has one answer; what :3007
    actually decides is whether there is a VALIDATION frame beside it.

    Both live sides driven, on the same loaders: with a validation loader
    the flush gets two frames, without one it gets a train frame and None --
    and never neither.
    """
    from spacr.deep_spacr import train_model

    flushes = []

    def record(dst, train_df, validation_df):
        flushes.append((len(train_df),
                        None if validation_df is None else len(validation_df)))

    saved = []
    _tiny_training(monkeypatch, saved)
    monkeypatch.setattr("spacr.io._save_progress", record)

    train_model(str(tmp_path), str(tmp_path / "trainonly"), "tiny",
                _tiny_loader(), epochs=2, num_classes=2, val_loaders=None,
                schedule=None, tensorboard=False, write_card=False)
    assert flushes == [(1, None), (1, None)]

    flushes.clear()
    saved = []
    _tiny_training(monkeypatch, saved)
    monkeypatch.setattr("spacr.io._save_progress", record)
    train_model(str(tmp_path), str(tmp_path / "withval"), "tiny",
                _tiny_loader(), epochs=2, num_classes=2,
                val_loaders=_tiny_loader(), schedule=None, tensorboard=False,
                write_card=False)
    assert flushes == [(1, 1), (1, 1)]


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
