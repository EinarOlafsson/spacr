"""The drop branches that only fire when the *registry* already knows.

Every case below is a path through :mod:`spacr.qt.dnd_handlers` that the
existing dnd files never take, and each one is a different way for a drop to
be quietly wrong rather than loudly broken:

* Measure resolving nothing and having to use the folder as dropped, with and
  without a trailing ``merged`` to hop.
* Classify being handed a plate it is already pointing at -- the second drop
  of the same folder must not list it twice.
* The Data Manager taking a project on a screen that has no ``scan``.
* The Prediction Profiler and Curate being told by
  :func:`spacr.chaining.resolve_drop` *which* file inside a folder is the one,
  which has to beat the alphabetical scan of the folder they would otherwise
  do.
* A coefficients CSV and a settings snapshot dropped by themselves, which must
  go straight in rather than re-searching the folder they sit in.

The registry's answer is supplied by the ``resolves`` fixture and the "which
one did you mean?" dialog by ``picks``, both shaped like the ones in
``tests/qt/test_cov_dnd_handlers.py``; everything else is a real tree on disk,
because every assertion here is about what was found on it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from spacr.qt import dnd_handlers as dh

# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class _Console:
    def __init__(self):
        self.text = ""

    def append_stdout(self, s):
        self.text += s


class _Screen:
    """A plain, non-Qt screen: what a tool screen looks like to a handler."""

    def __init__(self, **attrs):
        self._console = _Console()
        for name, value in attrs.items():
            setattr(self, name, value)

    @property
    def log(self) -> str:
        return self._console.text


class _Model:
    """A settings model that reports one ``src`` and records what is set."""

    def __init__(self, current=None):
        self.current = current
        self.set_values = {}

    def collect(self):
        return {"src": self.current}

    def set_value_for_key(self, key, value):
        self.set_values[key] = value
        return True


class _Field:
    """The one-line path field a screen keeps for a chosen file."""

    def __init__(self):
        self.value = ""

    def setText(self, value):
        self.value = value


class _ModulePicker:
    """The module combo Distributed Jobs preselects from the snapshot name."""

    def __init__(self):
        self.value = ""

    def setCurrentText(self, value):
        self.value = value


class _Target:
    def __init__(self, value, location=None, kind="measurements-db",
                 source="registry", paths=()):
        self.value = value
        self.location = location if location is not None else value
        self.kind = kind
        self.source = source
        self.paths = tuple(paths)


class _Resolution:
    def __init__(self, targets=(), choices=(), root="/root", reason="",
                 ok=True, ambiguous=False):
        self.targets = tuple(targets)
        self.choices = tuple(choices)
        self.root = root
        self.reason = reason
        self.ok = ok
        self.ambiguous = ambiguous

    def target_for(self, kind):
        for target in self.targets:
            if target.kind == kind:
                return target
        return None


@pytest.fixture
def resolves(monkeypatch):
    """Make spacr.chaining answer whatever the case under test needs."""
    def _install(resolution):
        monkeypatch.setattr(dh._ch, "resolve_drop", lambda *a, **k: resolution)
        return resolution
    return _install


@pytest.fixture
def picks(monkeypatch):
    """Answer -- or record the absence of -- the "which one?" chooser."""
    from spacr.qt import dnd as dnd_mod
    asked = []

    def _install(answer):
        def _dialog(screen, headline, question, options):
            asked.append((headline, question, list(options)))
            return answer(options) if callable(answer) else answer
        monkeypatch.setattr(dnd_mod, "choose_one_dialog", _dialog)
        return asked
    return _install


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Measure: nothing resolved, so the folder as dropped
# ---------------------------------------------------------------------------

def test_measure_hops_a_dropped_merged_folder_but_keeps_a_plate_folder(
        tmp_path, resolves):
    """Measure's ``src`` is the PLATE, never ``<plate>/merged``.

    Two strings for one project is how a settings CSV written from a drop
    stops matching one written by the auto-chain -- both run, because
    ``project_root`` hops a trailing ``merged``, so the disagreement only
    surfaces later when the two CSVs are compared. When the registry cannot
    answer, the handler still has to trim ``merged`` off the drop and leave a
    plate folder exactly as the user dropped it.
    """
    resolves(_Resolution(targets=()))          # the registry knows nothing
    plate = tmp_path / "plate1"
    (plate / "merged").mkdir(parents=True)

    dropped_merged = _Model()
    dh.MeasureDropHandler().apply(
        plate / "merged", _Screen(_settings_model=dropped_merged))

    dropped_plate = _Model()
    dh.MeasureDropHandler().apply(plate, _Screen(_settings_model=dropped_plate))

    assert dropped_merged.set_values["src"] == str(plate)
    assert dropped_plate.set_values["src"] == str(plate)


def test_measure_uses_the_resolved_merged_arrays_when_the_registry_has_them(
        tmp_path, resolves):
    """The registry is asked first, so a plate whose merged arrays were
    written somewhere unusual resolves to where the producer says they are --
    and the console line names the source, which is the only way a user can
    tell a resolved drop from a guessed one."""
    elsewhere = tmp_path / "scratch" / "plate1"
    elsewhere.mkdir(parents=True)
    resolves(_Resolution(targets=[
        _Target(str(elsewhere), kind=dh._kinds.MERGED_ARRAYS,
                location=str(elsewhere / "merged"), source="registry")]))
    model = _Model()
    screen = _Screen(_settings_model=model)

    dh.MeasureDropHandler().apply(tmp_path / "plate1", screen)

    assert model.set_values["src"] == str(elsewhere)
    assert "merged arrays" in screen.log
    assert "from the registry" in screen.log


def test_measure_suggests_child_plates_when_the_parent_cannot_be_listed(
        tmp_path):
    """A drop is dispatched inside Qt's event loop, so an exception here is a
    crashed window rather than a "did you mean…?" prompt.

    The sibling scan reads the *parent* of the dropped folder, which can stop
    being a directory between the two reads -- a network share going away
    mid-drag. The child hits already found must still be offered.
    """
    class _ParentRemovedMidDrop(type(Path("."))):
        @property
        def parent(self):
            return Path(f"{self}-was-removed")

    root = tmp_path / "screen1"
    (root / "plateA" / "merged").mkdir(parents=True)
    (tmp_path / "screen2" / "merged").mkdir(parents=True)

    with_parent = dh.MeasureDropHandler().suggest_alternatives(root)
    parent_gone = dh.MeasureDropHandler().suggest_alternatives(
        _ParentRemovedMidDrop(root))

    assert with_parent == [root / "plateA" / "merged",
                           tmp_path / "screen2" / "merged"]
    assert [Path(hit) for hit in parent_gone] == [root / "plateA" / "merged"]


# ---------------------------------------------------------------------------
# Classify: the same plate dropped twice
# ---------------------------------------------------------------------------

def test_classify_keeps_one_entry_per_plate_however_often_it_is_dropped(
        tmp_path):
    """Classify's ``src`` is a SET of plates, and a drop ADDS to it.

    Dropping a second plate has to grow the list; dropping the one already
    there must not, because a duplicated plate is trained on twice and
    silently doubles that plate's weight in the model.
    """
    plate1 = tmp_path / "plate1"
    plate2 = tmp_path / "plate2"
    for plate in (plate1, plate2):
        (plate / "data").mkdir(parents=True)

    grew = _Model(current=[str(plate1)])
    grew_screen = _Screen(_settings_model=grew)
    dh.ClassifyDropHandler().apply(plate2, grew_screen)

    again = _Model(current=[str(plate1)])
    again_screen = _Screen(_settings_model=again)
    dh.ClassifyDropHandler().apply(plate1, again_screen)

    assert grew.set_values["src"] == [str(plate1), str(plate2)]
    assert "classify plates" in grew_screen.log
    assert again.set_values["src"] == [str(plate1)]
    assert "classify plates" not in again_screen.log


# ---------------------------------------------------------------------------
# Data Manager: a project, then a measurement of it -- if it can measure
# ---------------------------------------------------------------------------

def test_the_data_manager_still_takes_a_project_on_a_screen_that_cannot_scan(
        tmp_path, resolves):
    """The scan is a follow-up, not a precondition.

    ``DataManagerDropHandler`` measures the project it was just handed, but
    the same handler is reachable from a screen built without that button --
    an older data manager, or one still constructing. Requiring ``scan`` there
    would turn a working drop into an AttributeError inside Qt's drop
    dispatch, which is a crash and not a dialog.
    """
    resolves(_Resolution(targets=[_Target(str(tmp_path))]))
    handler = dh.DataManagerDropHandler("data_manager")

    full = []
    handler.apply(tmp_path, _Screen(set_project=lambda p: full.append("project"),
                                    scan=lambda: full.append("scan")))
    without_scan = []
    handler.apply(tmp_path, _Screen(set_project=without_scan.append))

    assert full == ["project", "scan"]
    assert without_scan == [str(tmp_path)]


# ---------------------------------------------------------------------------
# Prediction Profiler: which CSV holds the coefficients
# ---------------------------------------------------------------------------

def test_the_profiler_loads_the_table_the_registry_names_not_the_first_on_disk(
        tmp_path, resolves, picks):
    """A results folder holds several CSVs and only one is the coefficients.

    When the run that wrote them registered which, that answer has to win over
    the alphabetical scan of the folder -- otherwise the profiler quietly
    plots ``all_gene_scores.csv`` as if it were the model's coefficients, and
    nothing about the screen says it did.
    """
    folder = tmp_path / "results"
    _touch(folder / "all_gene_scores.csv", "gene,score\n")
    coefficients = _touch(folder / "regression_coefficients.csv", "gene,beta\n")
    resolves(_Resolution(targets=[
        _Target(str(folder), kind=dh._kinds.REGRESSION_RESULTS,
                paths=(str(coefficients),))]))
    asked = picks(None)
    seen = []

    dh.CoefficientsDropHandler("profiler").apply(
        folder, _Screen(load_coefficients=seen.append))

    assert seen == [str(coefficients)]
    assert asked == []          # the registry answered, so nobody was asked


def test_a_coefficients_csv_dropped_by_itself_is_not_second_guessed(
        tmp_path, resolves, picks):
    """Dropping the file IS the answer to "which one?".

    The same folder dropped whole is genuinely ambiguous and must ask; the
    single file must not, because re-searching its folder would put a chooser
    in front of a user who already chose.
    """
    folder = tmp_path / "results"
    wanted = _touch(folder / "regression_coefficients.csv", "gene,beta\n")
    other = _touch(folder / "all_gene_scores.csv", "gene,score\n")
    resolves(_Resolution(targets=[
        _Target(str(folder), kind=dh._kinds.REGRESSION_RESULTS)]))
    asked = picks(lambda options: str(wanted))
    handler = dh.CoefficientsDropHandler("profiler")

    direct = []
    handler.apply(wanted, _Screen(load_coefficients=direct.append))
    assert direct == [str(wanted)]
    assert asked == []

    whole_folder = []
    handler.apply(folder, _Screen(load_coefficients=whole_folder.append))
    assert whole_folder == [str(wanted)]
    assert asked[0][2] == [str(other), str(wanted)]


# ---------------------------------------------------------------------------
# Curate: which mask
# ---------------------------------------------------------------------------

def test_curate_opens_the_mask_the_registry_names_not_the_alphabetical_first(
        tmp_path, resolves):
    """``masks/`` holds one file per field of view, and Curate opens one.

    The run that produced them can say which array this screen is for. Falling
    back to ``sorted(iterdir())`` there would open a different field of view
    than the one the pipeline just wrote, and the user would annotate it
    without ever being told a choice was made.
    """
    masks = tmp_path / "masks"
    _touch(masks / "aaa_A01_cell_mask.tif")
    wanted = _touch(masks / "zzz_B02_cell_mask.tif")
    resolves(_Resolution(targets=[
        _Target(str(masks), kind=dh._kinds.MASKS, paths=(str(wanted),))]))
    seen = []

    dh.LabelMaskDropHandler("curate").apply(
        tmp_path, _Screen(set_paths=lambda mask=None: seen.append(mask)))

    assert seen == [str(wanted)]
    assert sorted(p.name for p in masks.iterdir())[0] != Path(seen[0]).name


# ---------------------------------------------------------------------------
# Distributed Jobs: which settings snapshot
# ---------------------------------------------------------------------------

def test_a_settings_snapshot_dropped_by_itself_is_submitted_as_dropped(
        tmp_path, picks):
    """Distributed Jobs submits a settings snapshot and preselects the module
    that snapshot runs.

    A plate holding two snapshots has to ask which; the snapshot itself must
    not be re-searched out of its own ``settings/`` folder, or dropping
    ``measure_settings.csv`` would pop a chooser and could come back with
    ``mask_settings.csv`` -- submitting a different module than the file the
    user dragged.
    """
    plate = tmp_path / "plate"
    wanted = _touch(plate / "settings" / "measure_settings.csv", "Key,Value\n")
    other = _touch(plate / "settings" / "mask_settings.csv", "Key,Value\n")
    asked = picks(lambda options: str(wanted))
    handler = dh.SubmissionSettingsDropHandler("distributed_jobs")

    field, module = _Field(), _ModulePicker()
    screen = _Screen(_settings_path=field, _module=module)
    assert handler.can_accept(wanted) is True
    handler.apply(wanted, screen)
    assert field.value == str(wanted)
    assert module.value == "measure"
    assert asked == []
    assert "distributed_jobs settings" in screen.log

    plate_field = _Field()
    handler.apply(plate, _Screen(_settings_path=plate_field))
    assert plate_field.value == str(wanted)
    assert asked[0][2] == [str(other), str(wanted)]
