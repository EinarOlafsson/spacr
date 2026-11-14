"""Table-driven tests of the layout-aware tool-screen drops and ``get_handler``.

The table, coefficient, label-mask and settings-snapshot screens resolve a
dropped project to the one artifact they read and *ask* (never guess) when a
folder holds several; Investigate Hit, External Masks and Methods & Results
route each drop to the field it fits by what the path is.  ``get_handler``
hands out the registered handler, a plugin's own, or the generic source drop.
Every case drives a real path on disk against a screen double and asserts the
exact value delivered, the question asked and the console line written.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from spacr.qt import dnd_handlers as dh

pytestmark = pytest.mark.qt


class _Console:
    def __init__(self):
        self.text = ""

    def append_stdout(self, s):
        self.text += s


class _Screen:
    def __init__(self, **attrs):
        self._console = _Console()
        for name, value in attrs.items():
            setattr(self, name, value)

    @property
    def log(self) -> str:
        return self._console.text


class _Field:
    def __init__(self, value=""):
        self.value = value

    def setText(self, value):
        self.value = value

    def text(self):
        return self.value


class _Target:
    def __init__(self, value, kind, paths=(), source="registry"):
        self.value = value
        self.location = value
        self.kind = kind
        self.source = source
        self.paths = tuple(paths)


class _Resolution:
    def __init__(self, targets=(), root="/root", reason="nothing here"):
        self.targets = tuple(targets)
        self.root = root
        self.reason = reason
        self.choices = ()
        self.ok = True
        self.ambiguous = False


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _database(path: Path, tables=("png_list",)) -> Path:
    import sqlite3
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        for name in tables:
            conn.execute(f'CREATE TABLE "{name}" (a INTEGER)')
    return path


@pytest.fixture
def resolver(monkeypatch):
    """Make ``resolve_drop`` answer per case; record (app, path, kinds, form)."""
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


@pytest.fixture
def picks(monkeypatch):
    """Answer the "which one did you mean?" chooser and record each question."""
    from spacr.qt import dnd as dnd_mod
    asked = []

    def install(answer):
        def dialog(screen, headline, question, options):
            asked.append((headline, question, list(options)))
            if isinstance(answer, BaseException):
                raise answer
            return answer(options) if callable(answer) else answer
        monkeypatch.setattr(dnd_mod, "choose_one_dialog", dialog)
        asked.clear()
        return asked
    return install


def _resolved_log(app, location, kind, root="/root"):
    return (f"[drop] {app} ← {location}\n[drop] resolved {kind} in {root} "
            f"(from the registry)\n")


def test_table_screens_load_one_table_and_ask_when_there_are_several(
        subtests, tmp_path, picks, resolver):
    """Exact (path, table) loaded, question asked and log line per drop."""
    one = _database(tmp_path / "one" / "measurements.db")
    many = _database(tmp_path / "many" / "measurements.db",
                     ("cell", "nucleus", "png_list"))
    empty = _database(tmp_path / "empty" / "measurements.db", ())
    csv = _touch(tmp_path / "scores.csv", "a,b\n")
    plate = tmp_path / "plate"
    plate.mkdir()
    question = ("measurements.db holds 3 tables.", "Which one should be loaded?",
                ["cell", "nucleus", "png_list"])
    cases = [
        # (id, dropped, resolution, chooser answer, loaded, asked, log)
        ("a database with one table is read without asking", one, None,
         None, [(str(one), "png_list")], [], f"[drop] tabulate ← {one}\n"),
        ("a database with several tables is asked about", many, None,
         lambda options: options[1], [(str(many), "nucleus")], [question],
         f"[drop] tabulate ← {many}\n"),
        ("cancelling the table chooser loads nothing", many, None, None, [],
         [question], f"[drop] tabulate ← {many}\n"),
        ("a chooser that cannot be shown loads nothing", many, None,
         RuntimeError("no display"), [], [question],
         f"[drop] tabulate ← {many}\n"),
        ("a database with no tables is read with no table name", empty,
         None, None, [(str(empty), None)], [], f"[drop] tabulate ← {empty}\n"),
        ("a csv is read with no table name", csv, None, None,
         [(str(csv), None)], [], f"[drop] tabulate ← {csv}\n"),
        ("a project resolves to its database", plate,
         _Resolution([_Target(str(one), "measurements-db")]), None,
         [(str(one), "png_list")], [],
         _resolved_log("tabulate", one, "measurements-db")),
    ]
    for case_id, dropped, resolution, answer, loaded, asked_q, log in cases:
        with subtests.test(msg=case_id):
            calls = resolver(resolution)
            asked = picks(answer)
            seen = []
            screen = _Screen(load_path=lambda path, table=None:
                             seen.append((path, table)))
            dh.TableDropHandler("tabulate").apply(dropped, screen)
            assert seen == loaded
            assert asked == asked_q
            assert screen.log == log
            assert calls == ([("tabulate", dropped, ("measurements-db",),
                               "path")] if resolution else [])


def test_the_profiler_loads_the_coefficient_table_the_drop_names(
        subtests, tmp_path, picks, resolver):
    """Registry paths beat the folder scan; several CSVs are asked about."""
    kind = dh._kinds.REGRESSION_RESULTS
    folders = {}
    for name, files in (("only", ["coefficients.csv"]),
                        ("two", ["a.csv", "b.csv"]),
                        ("none", []),
                        ("registered", ["all_gene_scores.csv",
                                        "regression_coefficients.csv"])):
        folder = tmp_path / name / "results"
        folder.mkdir(parents=True)
        for file_name in files:
            _touch(folder / file_name, "gene,beta\n")
        folders[name] = folder
    two = folders["two"]
    registered = folders["registered"] / "regression_coefficients.csv"
    ask_two = ("results holds 2 tables.", "Which one holds the coefficients?",
               [str(two / "a.csv"), str(two / "b.csv")])
    cases = [
        # (id, folder key or file, target paths, answer, loaded, asked, error)
        ("the only coefficient table is taken", "only", (), None,
         [str(folders["only"] / "coefficients.csv")], [], None),
        ("several tables are asked about", "two", (),
         lambda options: options[1], [str(two / "b.csv")], [ask_two], None),
        ("cancelling the chooser loads nothing", "two", (), None, [],
         [ask_two], None),
        ("the registry's table wins over the folder", "registered",
         (str(registered),), None, [str(registered)], [], None),
        ("a results folder with no table is refused by name", "none", (),
         None, [], [], f"No coefficient CSV was found in {folders['none']}."),
    ]
    for case_id, key, paths, answer, loaded, asked_q, error in cases:
        with subtests.test(msg=case_id):
            folder = folders[key]
            resolver(_Resolution([_Target(str(folder), kind, paths)]))
            asked = picks(answer)
            seen = []
            screen = _Screen(load_coefficients=seen.append)
            handler = dh.CoefficientsDropHandler("profiler")
            if error is None:
                handler.apply(folder.parent, screen)
                assert screen.log == _resolved_log("profiler", folder, kind)
            else:
                with pytest.raises(ValueError) as raised:
                    handler.apply(folder.parent, screen)
                assert str(raised.value) == error
            assert seen == loaded
            assert asked == asked_q

    with subtests.test(msg="a csv dropped by itself is not second-guessed"):
        calls = resolver(_Resolution())
        asked = picks(lambda options: options[0])
        seen = []
        screen = _Screen(load_coefficients=seen.append)
        dh.CoefficientsDropHandler("profiler").apply(two / "b.csv", screen)
        assert (seen, asked, calls) == ([str(two / "b.csv")], [], [])
        assert screen.log == f"[drop] profiler ← {two / 'b.csv'}\n"


def test_label_mask_screens_open_exactly_one_mask(subtests, tmp_path, picks,
                                                  resolver):
    """One mask opens, several are asked about, none is refused by name."""
    kind = dh._kinds.MASKS
    folders = {}
    for name, files in (("one", ["cell.tif", "notes.txt"]),
                        ("two", ["a.tif", "b.npy"]),
                        ("none", ["readme.md"]),
                        ("registered", ["aaa_A01_cell_mask.tif",
                                        "zzz_B02_cell_mask.tif"])):
        folder = tmp_path / name / "masks"
        folder.mkdir(parents=True)
        for file_name in files:
            _touch(folder / file_name)
        folders[name] = folder
    two = folders["two"]
    wanted = folders["registered"] / "zzz_B02_cell_mask.tif"
    ask_two = ("masks holds 2 masks.", "Which one should be opened?",
               [str(two / "a.tif"), str(two / "b.npy")])
    cases = [
        # (id, folder key, target paths, answer, opened, asked, error)
        ("a folder with one mask opens it without asking", "one", (), None,
         [str(folders["one"] / "cell.tif")], [], None),
        ("a folder with several masks asks which one", "two", (),
         lambda options: options[1], [str(two / "b.npy")], [ask_two], None),
        ("cancelling the mask chooser opens nothing", "two", (), None, [],
         [ask_two], None),
        ("the registry's mask wins over the alphabetical first",
         "registered", (str(wanted), str(folders["registered"] / "x.json")),
         None, [str(wanted)], [], None),
        ("a masks folder with no mask is refused by name", "none", (), None,
         [], [], f"No label mask was found in {folders['none']}."),
    ]
    for case_id, key, paths, answer, opened, asked_q, error in cases:
        with subtests.test(msg=case_id):
            folder = folders[key]
            resolver(_Resolution([_Target(str(folder), kind, paths)]))
            asked = picks(answer)
            seen = []
            screen = _Screen(set_paths=lambda mask=None: seen.append(mask))
            handler = dh.LabelMaskDropHandler("curate")
            if error is None:
                handler.apply(folder.parent, screen)
            else:
                with pytest.raises(ValueError) as raised:
                    handler.apply(folder.parent, screen)
                assert str(raised.value) == error
            assert seen == opened
            assert asked == asked_q

    with subtests.test(msg="a screen with a mask field gets the path then opens"):
        resolver(_Resolution([_Target(str(folders["one"]), kind)]))
        field = _Field()
        order = []
        screen = _Screen(_mask_edit=field,
                         open_mask=lambda: order.append(field.text()))
        dh.LabelMaskDropHandler("napari_bridge").apply(folders["one"].parent,
                                                       screen)
        assert order == [str(folders["one"] / "cell.tif")]
        assert screen.log == _resolved_log("napari_bridge", folders["one"], kind)

    with subtests.test(msg="a mask file dropped directly is opened as is"):
        calls = resolver(_Resolution())
        seen = []
        dh.LabelMaskDropHandler("curate").apply(
            two / "b.npy", _Screen(set_paths=lambda mask=None: seen.append(mask)))
        assert (seen, calls) == ([str(two / "b.npy")], [])

    with subtests.test(msg="a drop that resolves to nothing is reported"):
        resolver(_Resolution(reason="no masks were written yet"))
        seen = []
        screen = _Screen(set_paths=lambda mask=None: seen.append(mask))
        dh.LabelMaskDropHandler("curate").apply(tmp_path, screen)
        assert seen == []
        assert screen.log == (
            f"[drop rejected] {tmp_path}\nReason: no masks were written yet\n"
            "Suggestion: Drop the folder or file this screen names, or run "
            "the step that writes it into /root.\n")


class _ModulePicker:
    def __init__(self):
        self.value = ""

    def setCurrentText(self, value):
        self.value = value


def test_distributed_jobs_submits_the_snapshot_the_user_chose(
        subtests, tmp_path, picks):
    """Exact snapshot path, preselected module and log; several are asked about."""
    plate = tmp_path / "plate"
    mask = _touch(plate / "settings" / "mask_settings.csv", "Key,Value\n")
    measure = _touch(plate / "settings" / "measure_settings.csv", "Key,Value\n")
    single = tmp_path / "single"
    only = _touch(single / "classify_settings.csv", "Key,Value\n")
    empty = tmp_path / "empty"
    empty.mkdir()
    question = ("plate holds 2 snapshots.", "Which one should be submitted?",
                [str(mask), str(measure)])
    cases = [
        # (id, dropped, answer, chosen, module, asked, error)
        ("a plate asks which snapshot to submit", plate,
         lambda options: options[1], measure, "measure", [question], None),
        ("cancelling the snapshot chooser submits nothing", plate, None, None,
         "", [question], None),
        ("a snapshot dropped by itself is submitted as dropped", measure,
         lambda options: options[0], measure, "measure", [], None),
        ("a folder with one snapshot takes it", single, None, only,
         "classify", [], None),
        ("a folder with no snapshot is refused by name", empty, None, None,
         "", [], f"No settings snapshot was found in {empty}."),
    ]
    for case_id, dropped, answer, chosen, module, asked_q, error in cases:
        with subtests.test(msg=case_id):
            asked = picks(answer)
            field, picker = _Field(), _ModulePicker()
            screen = _Screen(_settings_path=field, _module=picker)
            handler = dh.SubmissionSettingsDropHandler("distributed_jobs")
            if error is None:
                handler.apply(dropped, screen)
            else:
                with pytest.raises(ValueError) as raised:
                    handler.apply(dropped, screen)
                assert str(raised.value) == error
            assert field.value == (str(chosen) if chosen else "")
            assert picker.value == module
            assert asked == asked_q
            assert screen.log == (
                f"[drop] distributed_jobs settings = {chosen}\n" if chosen
                else "")

    with subtests.test(msg="a screen with no module picker still gets the path"):
        picks(None)
        field = _Field()
        dh.SubmissionSettingsDropHandler("distributed_jobs").apply(
            only, _Screen(_settings_path=field))
        assert field.value == str(only)


class _Panel:
    def __init__(self):
        for name in ("database", "regression_folder", "fractions",
                     "predictions"):
            setattr(self, name, _Field())
        self.refreshed = 0

    def _refresh_prediction_columns(self):
        self.refreshed += 1

    def values(self):
        return {name: getattr(self, name).value for name in
                ("database", "regression_folder", "fractions", "predictions")}


def test_investigate_hit_fills_the_field_each_input_belongs_in(subtests,
                                                               tmp_path):
    """Header and shape decide the field; only predictions refresh columns."""
    fractions = _touch(tmp_path / "f.csv", "grna,fraction,well\n1,0.2,A1\n")
    abundance = _touch(tmp_path / "g.csv", "﻿Guide_ID,Abundance\n")
    predictions = _touch(tmp_path / "p.csv", "object,score\n1,0.5\n")
    guide_only = _touch(tmp_path / "q.csv", "grna,score\n")
    folder = tmp_path / "results"
    folder.mkdir()
    plate = tmp_path / "plate"
    db = _database(plate / "measurements.db")
    nested = tmp_path / "nested"
    nested_db = _database(nested / "measurements" / "measurements.db")
    empty = {"database": "", "regression_folder": "", "fractions": "",
             "predictions": ""}
    cases = [
        # (id, dropped, field, value, refreshed, log label)
        ("a guide-fraction table goes to fractions", fractions, "fractions",
         fractions, 0, "guide fractions"),
        ("an abundance table with a BOM header is fractions too", abundance,
         "fractions", abundance, 0, "guide fractions"),
        ("a prediction table goes to predictions", predictions,
         "predictions", predictions, 1, "predictions"),
        ("a guide column alone is not a fraction table", guide_only,
         "predictions", guide_only, 1, "predictions"),
        ("a folder is the regression results folder", folder,
         "regression_folder", folder, 0, "regression results"),
        ("a project is read as its database", plate, "database", db, 0,
         "database"),
        ("a plate's measurements/ database is found", nested, "database",
         nested_db, 0, "database"),
        ("a database file is its own database", db, "database", db, 0,
         "database"),
    ]
    for case_id, dropped, field, value, refreshed, label in cases:
        with subtests.test(msg=case_id):
            panel = _Panel()
            screen = _Screen(investigate=panel)
            dh.InvestigateHitInputsDropHandler().apply(dropped, screen)
            assert panel.values() == {**empty, field: str(value)}
            assert panel.refreshed == refreshed
            assert screen.log == f"[drop] investigate_hit {label} = {value}\n"

    with subtests.test(msg="a screen with no input panel says so"):
        with pytest.raises(TypeError) as raised:
            dh.InvestigateHitInputsDropHandler().apply(folder, _Screen())
        assert str(raised.value) == "Investigate Hit has no input panel."


class _FileList:
    def __init__(self, count=None):
        self.paths = []
        self.count = count

    def add_paths(self, paths):
        self.paths.extend(paths)
        return len(paths) if self.count is None else self.count


class _MaskModel:
    def __init__(self, widgets, dst_value=""):
        self._widgets = dict(widgets)
        self.written = {}
        self.read = []
        self._dst_value = dst_value

    def _read_widget(self, widget):
        self.read.append(widget)
        return self._dst_value

    def set_value_for_key(self, key, value):
        self.written[key] = value


def test_external_masks_appends_inputs_and_names_a_destination(subtests,
                                                               tmp_path):
    """Inputs are appended; dst defaults beside the drop unless already typed."""
    folder = tmp_path / "images"
    folder.mkdir()
    image = _touch(tmp_path / "loose" / "a.tif")
    cases = [
        # (id, dropped, inputs, dst typed, written, error)
        ("a folder gets a destination beside it", folder, _FileList(), "",
         {"dst": f"{folder}_spacr"}, None),
        ("a file's destination is beside its folder", image, _FileList(), "",
         {"dst": f"{image.parent}_spacr"}, None),
        ("a destination the user already typed is left alone", folder,
         _FileList(), "/somewhere/chosen", {}, None),
        ("a folder with no supported images is refused by name", folder,
         _FileList(count=0), "", {},
         (ValueError, f"No supported images were found under {folder}.")),
    ]
    for case_id, dropped, inputs, typed, written, error in cases:
        with subtests.test(msg=case_id):
            dst = object()
            model = _MaskModel({"inputs": inputs, "dst": dst}, dst_value=typed)
            screen = _Screen(_settings_model=model)
            if error is None:
                dh.ExternalMasksDropHandler().apply(dropped, screen)
                assert model.read == [dst]
                assert screen.log == (
                    f"[drop] detected 1 external image/mask file(s) from "
                    f"{dropped}; review the assignments before Run.\n")
            else:
                with pytest.raises(error[0]) as raised:
                    dh.ExternalMasksDropHandler().apply(dropped, screen)
                assert str(raised.value) == error[1]
                assert screen.log == ""
            assert inputs.paths == [str(dropped)]
            assert model.written == written

    with subtests.test(msg="a screen with no input table says so"):
        with pytest.raises(TypeError) as raised:
            dh.ExternalMasksDropHandler().apply(
                folder, _Screen(_settings_model=_MaskModel({})))
        assert str(raised.value) == (
            "The External Masks input-mapping table is unavailable.")
        assert isinstance(raised.value.__cause__, KeyError)

    with subtests.test(msg="no dst control means no destination is written"):
        model = _MaskModel({"inputs": _FileList()})
        dh.ExternalMasksDropHandler().apply(folder,
                                            _Screen(_settings_model=model))
        assert (model.written, model.read) == ({}, [])


def test_methods_and_results_fills_the_field_each_source_fits(
        subtests, tmp_path, resolver):
    """Checkpoint, results folder and project land in their own fields."""
    checkpoint = _touch(tmp_path / "best.PTH")
    keras = _touch(tmp_path / "m.keras")
    results = tmp_path / "results"
    results.mkdir()
    plate = tmp_path / "plate"
    plate.mkdir()
    root = str(tmp_path / "project")
    resolved = _Resolution([_Target(root, "measurements-db")], root=root)
    cases = [
        # (id, dropped, resolution, field, value, resolved?)
        ("a checkpoint is the model", checkpoint, resolved, "model",
         str(checkpoint), False),
        ("a keras file is the model", keras, resolved, "model", str(keras),
         False),
        ("a results folder is the results", results, resolved, "results",
         str(results), False),
        ("anything else is the project root it resolves to", plate, resolved,
         "project", root, True),
        ("an unresolvable drop falls back to the dropped path", plate,
         RuntimeError("no ports"), "project", str(plate), True),
    ]
    for case_id, dropped, resolution, field, value, asked in cases:
        with subtests.test(msg=case_id):
            calls = resolver(resolution)
            fields = {name: _Field() for name in ("model", "results",
                                                  "project")}
            screen = _Screen(_fields=fields)
            handler = dh.MethodsSourcesDropHandler("methods_export")
            assert handler.can_accept(dropped) is True
            assert handler.accepts_multiple() is True
            handler.apply(dropped, screen)
            assert {k: f.value for k, f in fields.items() if f.value} == {
                field: value}
            assert screen.log == f"[drop] methods_export {field} = {value}\n"
            assert calls == ([("methods_export", dropped, (), "root")]
                             if asked else [])

    with subtests.test(msg="a screen with no field for the drop says so"):
        resolver(resolved)
        handler = dh.MethodsSourcesDropHandler("methods_export")
        assert handler.can_accept(tmp_path / "missing") is False
        with pytest.raises(TypeError) as raised:
            handler.apply(plate, _Screen(_fields={}))
        assert str(raised.value) == (
            "Methods & Results has no field for this drop.")


def test_get_handler_returns_the_registered_plugin_or_generic_handler(
        subtests, monkeypatch):
    """Exact handler class and app key per registry, plugin and failure path."""
    import spacr.plugins as plugins

    class _PluginHandler(dh.DropHandler):
        def can_accept(self, path):
            return True

        def error_message(self, path):
            return ""

        def apply(self, path, screen):
            pass

    class _PluginLayout(dh.LayoutDropHandler):
        def deliver(self, screen, value, target):
            pass

    def app(ref):
        return lambda key: type("A", (), {"drop_handler": ref})()

    def raising(exc):
        def boom(*a, **k):
            raise exc
        return boom

    recorded = []
    cases = [
        # (id, key, get_app, load_object, record, class, app key, diagnostics)
        ("a built-in layout screen is told its key", "curate", None, None,
         None, dh.LabelMaskDropHandler, "curate", []),
        ("a built-in plain handler", "measure", None, None, None,
         dh.MeasureDropHandler, None, []),
        ("an unknown screen with no plugin gets the source drop", "nobody",
         lambda key: None, None, None, dh.SourceDropHandler, None, []),
        ("a plugin with no drop handler gets the source drop", "nobody",
         app(""), None, None, dh.SourceDropHandler, None, []),
        ("a plugin can supply its own drop handler", "my_plugin", app("x:y"),
         lambda ref: _PluginHandler, None, _PluginHandler, None, []),
        ("a plugin layout handler is told its key", "my_plugin", app("x:y"),
         lambda ref: _PluginLayout, None, _PluginLayout, "my_plugin", []),
        ("a plugin offering a non-handler class is recorded", "my_plugin",
         app("x:y"), lambda ref: dict, None, dh.SourceDropHandler, None,
         [("my_plugin", "Could not load plugin drag-and-drop handler",
           "x:y is not a DropHandler subclass")]),
        ("a plugin offering an instance is recorded", "my_plugin",
         app("x:z"), lambda ref: _PluginHandler(), None,
         dh.SourceDropHandler, None,
         [("my_plugin", "Could not load plugin drag-and-drop handler",
           "x:z is not a DropHandler subclass")]),
        ("a diagnostic that cannot be recorded still leaves a handler",
         "my_plugin", raising(RuntimeError("plugin registry down")), None,
         raising(RuntimeError("also down")), dh.SourceDropHandler, None, []),
    ]
    for case_id, key, get_app, load, record, cls, app_key, diags in cases:
        with subtests.test(msg=case_id):
            recorded.clear()
            with monkeypatch.context() as patch:
                if get_app is not None:
                    patch.setattr(plugins, "get_app", get_app)
                if load is not None:
                    patch.setattr(plugins, "load_object", load)
                patch.setattr(
                    plugins, "record_diagnostic", record or (
                        lambda k, message, exc: recorded.append(
                            (k, message, str(exc)))))
                handler = dh.get_handler(key)
            assert type(handler) is cls
            if app_key is not None:
                assert handler.app_key == app_key
            assert recorded == diags
