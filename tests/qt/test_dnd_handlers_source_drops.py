"""Table-driven tests of the drop plumbing and the source-field drop handlers.

Covers the guards every drop runs through (``_is_alive``, the inline fallback
of ``_scan_then``, the one-walk ``scan_folder_structure``, the per-drop memo
``_resolve_for`` and ``table_names``) and the handlers whose job is to put the
right string into a screen's ``src``: the generic source drop, the
measurements, Measure and Classify drops, the multi-source add, the paired
plate-row attach and Regression's two-sided drop.  Every case drives a real
path on disk against a screen double and asserts exactly what reached it.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QWidget

from spacr.qt import dnd_handlers as dh

pytestmark = pytest.mark.qt

NO_FIELD = "This module has no source field to receive the drop."


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


class _Field:
    """A one-line ``src`` field the handler writes with ``setText``."""

    def __init__(self):
        self.value = None

    def setText(self, value):
        self.value = value


class _Model:
    """A settings model: refuses ``set_value_for_key`` so the field is written."""

    def __init__(self, current=None, accept=False, widgets=None):
        self.current = current
        self.accept = accept
        self.set_values = {}
        self._widgets = {"src": _Field()} if widgets is None else widgets

    def collect(self):
        return {"src": self.current}

    def set_value_for_key(self, key, value):
        if self.accept:
            self.set_values[key] = value
        return self.accept


class _Target:
    def __init__(self, value, location=None, kind="measurements-db",
                 source="registry"):
        self.value = value
        self.location = location if location is not None else value
        self.kind = kind
        self.source = source


class _Resolution:
    def __init__(self, targets=()):
        self.targets = tuple(targets)

    def target_for(self, kind):
        return next((t for t in self.targets if t.kind == kind), None)


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _database(path: Path, tables=("png_list",)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        for name in tables:
            conn.execute(f'CREATE TABLE "{name}" (a INTEGER)')
    return path


@pytest.fixture
def resolver(monkeypatch):
    """Make ``spacr.chaining.resolve_drop`` answer per case and record calls."""
    calls = []

    def install(answer):
        def resolve(app_key, path, kinds=(), form=None):
            calls.append((app_key, Path(path), tuple(kinds), form))
            if isinstance(answer, BaseException):
                raise answer
            return answer
        monkeypatch.setattr(dh._ch, "resolve_drop", resolve)
        calls.clear()
        return calls
    return install


# ---------------------------------------------------------------------------
# The guards every drop runs through
# ---------------------------------------------------------------------------

def test_is_alive_reads_a_destroyed_or_unanswerable_qt_object_as_gone(
        subtests, monkeypatch):
    """Non-Qt objects are alive; a deleted widget or a failing shiboken is gone."""
    import shiboken6

    class _Broken:
        @staticmethod
        def isValid(obj):
            raise RuntimeError("shiboken is unhappy")

    with subtests.test(msg="a screen that is not a Qt object is alive"):
        assert dh._is_alive(_Screen()) is True
        assert dh._is_alive(None) is True

    with subtests.test(msg="shiboken answers for a live and a deleted object"):
        widget = QWidget()          # not given to qtbot: deleted on purpose
        assert dh._is_alive(widget) is True
        shiboken6.delete(widget)
        assert dh._is_alive(widget) is False

    with subtests.test(msg="without shiboken the object is poked instead"):
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "shiboken6", None)
            widget = QWidget()
            assert dh._is_alive(widget) is True
            shiboken6.delete(widget)
            assert dh._is_alive(widget) is False

    with subtests.test(msg="a shiboken that refuses to answer reads as gone"):
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "shiboken6", _Broken)
            assert dh._is_alive(QObject()) is False


def test_scan_then_runs_inline_when_no_thread_can_take_it(subtests):
    """Sealed screens and refused threads scan inline; failures go to on_error."""

    class _Sealed:
        __slots__ = ()

    class _RefusingScanner:
        def submit(self, fn, on_done):
            raise RuntimeError("cannot create thread")

    def refusing_screen():
        screen = _Screen()
        screen._dnd_scanner = _RefusingScanner()
        return screen

    boom = OSError("permission denied")

    def raises():
        raise boom

    cases = [
        # (id, screen factory, fn, with on_error?, done, errors)
        ("a screen that cannot hold a scanner scans inline", _Sealed,
         lambda: "walked", False, ["walked"], []),
        ("a thread qt refuses to start falls back inline", refusing_screen,
         lambda: "walked", True, ["walked"], []),
        ("a scan that raises reports nothing without on_error", _Sealed,
         raises, False, [], []),
        ("a scan that raises reaches on_error", refusing_screen, raises,
         True, [], [boom]),
    ]
    for case_id, make, fn, with_error, done, errors in cases:
        with subtests.test(msg=case_id):
            seen, failed = [], []
            threaded = dh._scan_then(make(), fn, seen.append,
                                     failed.append if with_error else None)
            assert (threaded, seen, failed) == (False, done, errors)


def test_scan_folder_structure_walks_once_and_reports_plain_data(
        subtests, tmp_path, monkeypatch):
    """Exact report per layout outcome; an unknown layout stops at the probe."""
    from spacr.qt import folder_metadata as fm
    from spacr.qt import ingest_preview as ip

    class _Template:
        def __init__(self, labels):
            self.depth_labels = labels

    def plan_raising(exc):
        def plan(*a, **k):
            raise exc
        return plan

    files = [f"{i}.tif" for i in range(5)]
    cases = [
        # (id, iterator, detector answer, planner, expected report, walked)
        ("no recognisable layout is never fully walked",
         lambda: (f"{i}.tif" for i in range(1000)), None, None,
         {"labels": (), "rows": [], "error": ""}, dh._FOLDER_PROBE),
        ("a layout with no labels reports nothing", lambda: iter(files),
         _Template(()), None, {"labels": (), "rows": [], "error": ""}, 5),
        ("a layout detector that raises reports nothing",
         OSError("unreadable"), None, None,
         {"labels": (), "rows": [], "error": ""}, 0),
        ("a recognised layout is planned from the same walk",
         lambda: iter(files), _Template(("plate", "well")), "plan",
         {"labels": ("plate", "well"), "rows": [{"plate": "p1"}],
          "error": ""}, 5),
        ("a failing plan reports its reason beside the labels",
         lambda: iter([]), _Template(["plate"]),
         plan_raising(ValueError("two plates, one row")),
         {"labels": ("plate",), "rows": [], "error": "two plates, one row"},
         0),
        ("a failing plan with no message names its type", lambda: iter([]),
         _Template(("plate",)), plan_raising(KeyError()),
         {"labels": ("plate",), "rows": [], "error": "KeyError"}, 0),
    ]
    for case_id, iterator, template, planner, expected, walked_n in cases:
        with subtests.test(msg=case_id):
            walked, planned, probed = [], [], []

            def iter_paths(path, iterator=iterator):
                if isinstance(iterator, BaseException):
                    raise iterator
                for item in iterator():
                    walked.append(item)
                    yield item

            def detect(path, files=None, template=template):
                probed.append(list(files))
                return template

            def plan(path, files=None, template=None):
                planned.append(([str(f) for f in files], template))
                return [{"plate": "p1"}]

            with monkeypatch.context() as patch:
                patch.setattr(fm, "_iter_image_paths", iter_paths)
                patch.setattr(fm, "detect_folder_metadata", detect)
                patch.setattr(ip, "plan_folder_extraction",
                              plan if planner == "plan" else planner)
                assert dh.scan_folder_structure(tmp_path) == expected
            assert len(walked) == walked_n
            if planner == "plan":
                assert probed == [[Path(f) for f in files]]
                assert planned == [(files, template)]


def test_resolve_for_memoises_one_drop_until_the_folder_changes(
        subtests, tmp_path, resolver):
    """One resolution per unchanged path; a new entry or a failure re-asks."""
    with subtests.test(msg="a path that cannot be stat-ed is memoised too"):
        answer = _Resolution()
        calls = resolver(answer)
        handler = dh.SourceDropHandler()
        gone = tmp_path / "gone"
        assert dh._resolve_for(handler, "mask", gone) is answer
        assert dh._resolve_for(handler, "mask", gone) is answer
        assert calls == [("mask", gone, (), "path")]
        assert handler._last_resolution == (("mask", str(gone), 0, -1), answer)

    with subtests.test(msg="another module key is resolved again"):
        calls = resolver(_Resolution())
        handler = dh.SourceDropHandler()
        dh._resolve_for(handler, "mask", gone)
        dh._resolve_for(handler, "measure", gone)
        assert [c[0] for c in calls] == ["mask", "measure"]

    with subtests.test(msg="a folder that gained a file is resolved again"):
        calls = resolver(_Resolution())
        handler = dh.SourceDropHandler()
        folder = tmp_path / "plate"
        folder.mkdir()
        dh._resolve_for(handler, "mask", folder)
        dh._resolve_for(handler, "mask", folder)
        _touch(folder / "measurements.db")
        dh._resolve_for(handler, "mask", folder)
        assert len(calls) == 2
        assert handler._last_resolution[0][3] == 1

    with subtests.test(msg="a resolution that raises is no answer, not cached"):
        calls = resolver(KeyError("no such module"))
        handler = dh.SourceDropHandler()
        assert dh._resolve_for(handler, "mask", tmp_path) is None
        assert dh._resolve_for(handler, "mask", tmp_path) is None
        assert len(calls) == 2
        assert getattr(handler, "_last_resolution", None) is None


def test_a_cached_answer_is_replaced_only_by_a_question_asked_no_earlier(
        monkeypatch, subtests):
    """A late answer to an older question never overwrites a newer one."""
    for case, first, second, kept in (
            ("a late older answer is dropped", ("yes", 5), ("no", 3), "yes"),
            ("a newer answer replaces the older", ("no", 3), ("yes", 5), "yes"),
            ("the same question may answer again", ("no", 4), ("yes", 4), "yes")):
        with subtests.test(msg=case):
            monkeypatch.setattr(dh, "_decisions", {})
            key = ("handler", "/share/plate")
            dh._remember(key, first[0], first[1])
            dh._remember(key, second[0], second[1])
            assert [(v[1], v[2]) for v in dh._decisions.values()] == [
                (kept, max(first[1], second[1]))]


def test_table_names_lists_a_databases_tables_or_nothing(subtests, tmp_path):
    """Sorted table names for a real database; [] for anything else."""
    odd = tmp_path / "plate?1#2"
    cases = [
        ("a dropped csv is not asked for its tables",
         lambda: _touch(tmp_path / "scores.csv"), []),
        ("a database lists its tables sorted",
         lambda: _database(tmp_path / "a" / "measurements.db",
                           ("png_list", "cell")), ["cell", "png_list"]),
        ("a .sqlite suffix in capitals still counts",
         lambda: _database(tmp_path / "b" / "M.SQLITE", ("x",)), ["x"]),
        ("a file named like a database that is not one lists nothing",
         lambda: _touch(tmp_path / "broken.db", "not sqlite"), []),
        ("a database that cannot be opened at all lists nothing",
         lambda: tmp_path / "unmounted" / "measurements.db", []),
    ]
    if sys.platform != "win32":
        cases.append(("a folder with ? and # in its name still opens",
                      lambda: _database(odd / "measurements.db",
                                        ("png_list", "cell")),
                      ["cell", "png_list"]))
    for case_id, make, expected in cases:
        with subtests.test(msg=case_id):
            assert dh.table_names(make()) == expected
    assert not (tmp_path / "unmounted").exists()


# ---------------------------------------------------------------------------
# Handlers that fill src
# ---------------------------------------------------------------------------

def test_source_drops_write_the_plate_into_src(subtests, tmp_path, resolver):
    """Each handler writes one exact src value and console text per drop shape."""
    plate = tmp_path / "plate"
    _database(plate / "measurements" / "measurements.db")
    merged = plate / "merged"
    _touch(merged / "a.npy")
    elsewhere = tmp_path / "scratch" / "plate1"
    elsewhere.mkdir(parents=True)
    other = tmp_path / "plate_b"
    other.mkdir()
    image = _touch(tmp_path / "imgs" / "a.tif")
    source = dh.SourceDropHandler
    measurements = dh.MeasurementsDropHandler
    measure = dh.MeasureDropHandler
    classify = dh.ClassifyDropHandler
    p = str(plate)
    cases = [
        # (id, handler, app_key, resolution, dropped, model, src, log, asked)
        ("generic drop uses the auto-chain answer", source, "tabulate",
         _Resolution([_Target("/p", "/p/measurements", "measurements_db",
                              "layout")]), tmp_path, _Model(), "/p",
         "[drop] src = /p\n[drop] resolved measurements_db → /p/measurements "
         "(from the layout)\n", ["tabulate"]),
        ("generic drop of a file sets the folder holding it", source, "",
         None, image, _Model(), str(image.parent),
         f"[drop] src = {image.parent}\n", []),
        ("generic drop of the db file sets the plate", source, "", None,
         plate / "measurements" / "measurements.db", _Model(), p,
         f"[drop] src = {p}\n", []),
        ("generic drop of measurements/ sets the plate", source, "", None,
         plate / "measurements", _Model(), p, f"[drop] src = {p}\n", []),
        ("generic unresolved drop on a declared module", source, "mask",
         _Resolution(), merged, _Model(accept=True), str(merged),
         f"[drop] src = {merged}\n", ["mask"]),
        ("measurements drop prefers where the run wrote", measurements,
         "umap", _Resolution([_Target("/p", "/p/measurements")]), tmp_path,
         _Model(), "/p",
         "[drop] src = /p\n[drop] measurements → /p/measurements "
         "(from the registry)\n", ["umap"]),
        ("measurements drop of the db sets the plate above it", measurements,
         "", None, plate / "measurements" / "measurements.db", _Model(), p,
         f"[drop] src = {p}\n", []),
        ("measure trims merged/ when nothing resolves", measure, None,
         _Resolution(), merged, _Model(accept=True), p,
         f"[drop] measure src = {p}\n", ["measure"]),
        ("measure trims an array file and merged/", measure, None,
         _Resolution(), merged / "a.npy", _Model(), p,
         f"[drop] measure src = {p}\n", ["measure"]),
        ("measure keeps a plate folder as dropped", measure, None,
         _Resolution(), plate, _Model(accept=True), p,
         f"[drop] measure src = {p}\n", ["measure"]),
        ("measure uses the registry's merged arrays", measure, None,
         _Resolution([_Target(str(elsewhere), str(elsewhere / "merged"),
                              dh._kinds.MERGED_ARRAYS)]),
         plate, _Model(accept=True), str(elsewhere),
         f"[drop] measure src = {elsewhere}\n[drop] merged arrays → "
         f"{elsewhere / 'merged'} (from the registry)\n", ["measure"]),
        ("classify adds a second plate beside the first", classify, None,
         None, other, _Model(current=p), repr([p, str(other)]),
         f"[drop] classify src = {other}\n"
         f"[drop] classify plates = {[p, str(other)]}\n", []),
        ("classify keeps one entry per plate", classify, None, None, plate,
         _Model(current=[p], accept=True), [p],
         f"[drop] classify src = {p}\n", []),
        ("classify grows a list model", classify, None, None, other,
         _Model(current=[p, " "], accept=True), [p, str(other)],
         f"[drop] classify src = {other}\n"
         f"[drop] classify plates = {[p, str(other)]}\n", []),
        ("classify drops the placeholder src", classify, None, None, plate,
         _Model(current="path"), p, f"[drop] classify src = {p}\n", []),
    ]
    for (case_id, handler, app_key, resolution, dropped, model, src, log,
         asked) in cases:
        with subtests.test(msg=case_id):
            calls = resolver(resolution)
            attrs = {"_settings_model": model}
            if app_key is not None:
                attrs["app_key"] = app_key
            screen = _Screen(**attrs)
            handler().apply(dropped, screen)
            written = (model.set_values.get("src") if model.accept
                       else model._widgets["src"].value)
            assert written == src
            assert screen.log == log
            assert [c[0] for c in calls] == asked
            assert all(c[1] == dropped for c in calls)


def test_a_screen_with_no_source_field_refuses_the_drop(subtests, tmp_path,
                                                        resolver):
    """Resolved or not, a screen with nowhere to write src raises TypeError."""
    cases = [
        ("a resolved drop", _Resolution([_Target("/p")]), "tabulate"),
        ("an unresolvable drop", None, ""),
        ("a model whose src field cannot be written", _Resolution(), "x"),
    ]
    for case_id, resolution, app_key in cases:
        with subtests.test(msg=case_id):
            resolver(resolution)
            attrs = {"app_key": app_key}
            if case_id.startswith("a model"):
                attrs["_settings_model"] = _Model(widgets={"src": object()})
            screen = _Screen(**attrs)
            with pytest.raises(TypeError) as raised:
                dh.SourceDropHandler().apply(tmp_path, screen)
            assert str(raised.value) == NO_FIELD
            assert screen.log == ""


def test_add_to_source_set_adds_without_replacing(subtests):
    """True/False when every path is present after the add; None when not ours."""

    class _Sources:
        def __init__(self, keep=None, add_error=None, read_error=None):
            self.added = []
            self.keep = keep
            self.add_error = add_error
            self.read_error = read_error

        def add_sources(self, values):
            if self.add_error:
                raise self.add_error
            self.added.extend(values)

        def sources(self):
            if self.read_error:
                raise self.read_error
            return self.added if self.keep is None else self.keep

    def screen(widgets):
        return _Screen(_settings_model=_Model(widgets=widgets))

    cases = [
        # (id, screen, path, answer, added)
        ("one path is added and confirmed", _Sources(), "/plate", True,
         ["/plate"]),
        ("a list of paths is added as strings", _Sources(),
         [Path("/p1"), "/p2"], True, ["/p1", "/p2"]),
        ("a tuple is a list too", _Sources(), ("/p1",), True, ["/p1"]),
        ("a control that loses a path reports False",
         _Sources(keep=["/p1"]), ["/p1", "/p2"], False, ["/p1", "/p2"]),
        ("a control that will not say what it holds counts as added",
         _Sources(read_error=RuntimeError("being rebuilt")),
         ["/plate1", "/plate2"], True, ["/plate1", "/plate2"]),
        ("a control that refuses the add hands back",
         _Sources(add_error=RuntimeError("read-only")), "/plate", None, []),
        ("a src field with no add_sources is not ours", _Field(), "/plate",
         None, None),
        ("a screen with no source control is not ours", None, "/plate", None,
         None),
        ("a screen with no settings model is not ours", "bare", "/plate",
         None, None),
    ]
    for case_id, widget, path, answer, added in cases:
        with subtests.test(msg=case_id):
            if widget == "bare":
                target = object()
            elif widget is None:
                target = screen({})
            else:
                target = screen({"src": widget})
            assert dh._add_to_source_set(target, path) is answer
            if added is not None:
                assert widget.added == added


def test_a_database_drop_goes_to_the_plate_row_or_the_sweep_card(
        subtests, tmp_path, monkeypatch):
    """Paired screens attach the db; Regression files CSVs by their header."""

    class _Paired:
        def __init__(self):
            self.attached = []

        def attach_database(self, path):
            self.attached.append(path)
            return f"attached {Path(path).name} to plate1"

    class _List:
        def __init__(self):
            self.paths = []

        def add_paths(self, values):
            self.paths.extend(values)

    class _Sweep:
        def __init__(self):
            self.score_data = _List()
            self.count_data = _List()

    root = tmp_path / "plate1"
    _touch(root / "measurements" / "measurements.db", "SQLite format 3")
    db = root / "measurements" / "measurements.db"
    counts = _touch(tmp_path / "table_one.csv",
                    "prc,grna,count\nplate1_r1_c1,A,7\n")
    scores = _touch(tmp_path / "table_two.csv", "prc,pred\nplate1_r1_c1,0.5\n")
    attached = "[drop] attached measurements.db to plate1\n"
    monkeypatch.setattr(dh._ch, "resolve_drop",
                        lambda *a, **k: pytest.fail("must not resolve"))
    cases = [
        # (id, handler, dropped, has sweep?, attached, counts, scores, log)
        ("measurements drop attaches the plate's db", dh.MeasurementsDropHandler,
         root, False, [str(db)], None, None, attached),
        ("measurements drop of the db file itself", dh.MeasurementsDropHandler,
         db, False, [str(db)], None, None, attached),
        ("regression's database half wins for a plate",
         dh.RegressionDropHandler, root, True, [str(db)], [], [], attached),
        ("a count csv lands on the count side", dh.RegressionDropHandler,
         counts, True, [], [str(counts)], [],
         f"[drop] parameter_sweep count += {counts}\n"),
        ("a score csv lands on the score side", dh.RegressionDropHandler,
         scores, True, [], [], [str(scores)],
         f"[drop] parameter_sweep score += {scores}\n"),
    ]
    for case_id, handler, dropped, sweep, att, cnt, scr, log in cases:
        with subtests.test(msg=case_id):
            table = _Paired()
            screen = _Screen(app_key="regression", _settings_model=_Model(
                widgets={"paired_data": table, "src": object()}))
            if sweep:
                screen._sweep = _Sweep()
            handler().apply(dropped, screen)
            assert table.attached == att
            if sweep:
                assert screen._sweep.count_data.paths == cnt
                assert screen._sweep.score_data.paths == scr
            assert screen.log == log

    with subtests.test(msg="a regression screen with no sweep card says so"):
        screen = _Screen(app_key="regression", _settings_model=_Model(widgets={}))
        with pytest.raises(TypeError) as raised:
            dh.RegressionDropHandler().apply(counts, screen)
        assert str(raised.value) == "Regression has no parameter-sweep inputs."
        assert screen.log == ""
