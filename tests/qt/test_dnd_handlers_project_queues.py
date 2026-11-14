"""Table-driven tests of the drops that import projects, queue plates and open databases.

``ForeignProjectDropHandler`` routes a drop to Import Project's mapping,
measurement-table, mask-folder or image input by its header and name;
``PlateQueueDropHandler`` and ``BatchDropHandler`` turn plate lists, settings
snapshots and saved queues into queued work and name what they could not use;
``ResultsDatabaseDropHandler`` opens a database through either setter name.
Every case drives a real path on disk against a screen double.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from spacr.qt import dnd_handlers as dh

pytestmark = pytest.mark.qt

MAPPING_HEADER = "source,target,transform,unit_in,unit_out,note\n"


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


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class _ForeignScreen(_Screen):
    """Import Project: records which of its four inputs a drop reached."""

    def __init__(self, **attrs):
        super().__init__(**attrs)
        self.calls = []

    def load_mapping(self, path):
        self.calls.append(("mapping", path))
        return getattr(self, "mapping_result", None)

    def set_measurements(self, path):
        self.calls.append(("measurements", path))

    def set_images(self, path):
        self.calls.append(("images", path))

    def add_mask_folder(self, object_type, path):
        self.calls.append(("masks", object_type, path))
        return getattr(self, "mask_result", None)


class _Box:
    def __init__(self, data):
        self.data = data

    def currentData(self):
        if isinstance(self.data, BaseException):
            raise self.data
        return self.data


def test_import_project_routes_each_drop_to_the_input_it_fits(
        subtests, tmp_path, monkeypatch):
    """Exactly one input receives each drop; a refused one raises its message."""
    mapping_csv = _touch(tmp_path / "map.csv", MAPPING_HEADER)
    mapping_json = _touch(tmp_path / "map.json", "{}")
    table = _touch(tmp_path / "cells.csv", "object,area\n")
    tsv = _touch(tmp_path / "cells.tsv", "object\tarea\n")
    image = _touch(tmp_path / "imgs" / "a.tif")
    folders = {}
    for name in ("nucleus_masks", "segmentation", "label", "LABELS", "raw"):
        folders[name] = tmp_path / name
        folders[name].mkdir()
    cases = [
        # (id, dropped, screen attrs, expected calls, error message)
        ("a mapping csv is loaded as a mapping", mapping_csv, {},
         [("mapping", str(mapping_csv))], None),
        ("a json mapping is loaded as a mapping", mapping_json, {},
         [("mapping", str(mapping_json))], None),
        ("a mapping that would not load is reported", mapping_json,
         {"mapping_result": False}, [("mapping", str(mapping_json))],
         f"Could not load mapping {mapping_json}."),
        ("a plain csv is the measurement table", table, {},
         [("measurements", str(table))], None),
        ("a tsv is the measurement table", tsv, {},
         [("measurements", str(tsv))], None),
        ("a masks folder uses the chosen object type",
         folders["nucleus_masks"], {"_object_box": _Box("nucleus")},
         [("masks", "nucleus", str(folders["nucleus_masks"]))], None),
        ("a mask folder before an object type was picked uses cell",
         folders["segmentation"], {},
         [("masks", "cell", str(folders["segmentation"]))], None),
        ("an object box that raises also means cell", folders["LABELS"],
         {"_object_box": _Box(RuntimeError("deleted"))},
         [("masks", "cell", str(folders["LABELS"]))], None),
        ("a mask folder that was refused is reported", folders["label"],
         {"mask_result": False}, [("masks", "cell", str(folders["label"]))],
         f"Could not add mask folder {folders['label']}."),
        ("any other folder is the image source", folders["raw"], {},
         [("images", str(folders["raw"]))], None),
        ("an image file sets its folder as the image source", image, {},
         [("images", str(image.parent))], None),
    ]
    for case_id, dropped, attrs, calls, error in cases:
        with subtests.test(msg=case_id):
            screen = _ForeignScreen(**attrs)
            if error is None:
                dh.ForeignProjectDropHandler().apply(dropped, screen)
            else:
                with pytest.raises(ValueError) as raised:
                    dh.ForeignProjectDropHandler().apply(dropped, screen)
                assert str(raised.value) == error
            assert screen.calls == calls

    with subtests.test(msg="a csv that cannot be read is not guessed at"):
        def refuse(self, *a, **k):
            raise PermissionError("nope")
        screen = _ForeignScreen()
        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", refuse)
            dh.ForeignProjectDropHandler().apply(mapping_csv, screen)
        assert screen.calls == [("measurements", str(mapping_csv))]


class _Queue(list):
    def add(self, item):
        self.append(item)


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, value):
        self.emitted.append(value)


class _QueueScreen(_Screen):
    def __init__(self, **attrs):
        super().__init__(**attrs)
        self._queue = _Queue()
        self.items = []
        self.refreshed = 0
        self.queue_size_changed = _Signal()

    def queue(self):
        return self._queue

    def _refresh_table(self):
        self.refreshed += 1

    def add_item(self, module, settings):
        self.items.append((module, settings))


def test_plate_queue_queues_plate_lists_and_snapshots(subtests, tmp_path,
                                                      monkeypatch):
    """Exact queued items, console line and refusal per plate-list or folder."""
    from spacr.qt import plate_queue as pq
    import spacr.utils as utils

    imported = []

    def import_plates(path, base_settings=None, app_key="mask"):
        imported.append((Path(path), base_settings, app_key))
        return ["item-a", "item-b"] if "plates" in Path(path).name else []

    monkeypatch.setattr(pq, "import_plates_from_csv", import_plates)

    with subtests.test(msg="a plate-list csv queues every row it names"):
        csv = _touch(tmp_path / "plates.csv", "src\n/a\n/b\n")
        screen = _QueueScreen()
        dh.PlateQueueDropHandler().apply(csv, screen)
        assert list(screen.queue()) == ["item-a", "item-b"]
        assert (screen.refreshed, screen.queue_size_changed.emitted) == (1, [2])
        assert imported == [(csv, {}, "mask")]

    with subtests.test(msg="a plate-list csv with no src rows is refused"):
        csv = _touch(tmp_path / "notes.csv", "note\nhello\n")
        screen = _QueueScreen()
        with pytest.raises(ValueError) as raised:
            dh.PlateQueueDropHandler().apply(csv, screen)
        assert str(raised.value) == (
            "notes.csv contains no plate rows with an src value.")
        assert (list(screen.queue()), screen.refreshed) == ([], 0)

    def strict_then_loose(answer):
        def load(path, **kwargs):
            name = Path(path).name
            load.calls.append((name, kwargs))
            if "broken" in name:
                raise ValueError("not a settings file")
            if "handmade" in name and kwargs:
                raise ValueError("no Key column")
            return answer() if callable(answer) else answer
        load.calls = []
        return load

    strict = {"setting_key": "Key", "setting_value": "Value"}
    cases = [
        # (id, snapshot files, loader answer, items, load calls, log, error)
        ("a plate's snapshots are queued against that folder",
         ["mask_settings.csv"], lambda: {"src": "/somewhere/else"},
         [("mask", {"src": "{plate}"})],
         [("mask_settings.csv", strict)], "", None),
        ("a hand-made snapshot is read with the documented default",
         ["handmade_measure_settings.csv"], lambda: {"channels": [0]},
         [("measure", {"channels": [0], "src": "{plate}"})],
         [("handmade_measure_settings.csv", strict),
          ("handmade_measure_settings.csv", {})], "", None),
        ("module names come from the snapshot name",
         ["gen_masks_settings.csv", "train_test_settings.csv",
          "other_settings.csv"], lambda: {},
         [("mask", {"src": "{plate}"}), ("mask", {"src": "{plate}"}),
          ("classify", {"src": "{plate}"})],
         [("gen_masks_settings.csv", strict),
          ("other_settings.csv", strict),
          ("train_test_settings.csv", strict)], "", None),
        ("one unreadable snapshot among several is named",
         ["mask_settings.csv", "broken_settings.csv"],
         lambda: {"channels": [0]},
         [("mask", {"channels": [0], "src": "{plate}"})],
         [("broken_settings.csv", strict), ("broken_settings.csv", {}),
          ("mask_settings.csv", strict)],
         "Queued 1 of 2 settings snapshots from plate. Skipped: "
         "broken_settings.csv.", None),
        ("a snapshot that is not a settings dict is skipped",
         ["mask_settings.csv"], ["not", "a", "dict"], [],
         [("mask_settings.csv", strict)], "",
         "No readable settings snapshots found in {plate}."),
    ]
    for index, (case_id, names, answer, items, calls, log,
                error) in enumerate(cases):
        with subtests.test(msg=case_id):
            plate = tmp_path / f"case{index}" / "plate"
            for name in names:
                _touch(plate / "settings" / name, "Key,Value\n")
            loader = strict_then_loose(answer)
            screen = _QueueScreen()
            with monkeypatch.context() as patch:
                patch.setattr(utils, "load_settings", loader)
                if error is None:
                    dh.PlateQueueDropHandler().apply(plate, screen)
                else:
                    with pytest.raises(ValueError) as raised:
                        dh.PlateQueueDropHandler().apply(plate, screen)
                    assert str(raised.value) == error.format(plate=plate)
            assert screen.items == [
                (module, {k: (v.format(plate=plate) if k == "src" else v)
                          for k, v in settings.items()})
                for module, settings in items]
            assert sorted(loader.calls, key=lambda c: c[0]) == calls
            assert screen.log == log


def test_batch_runner_loads_queues_and_adds_snapshot_jobs(subtests, tmp_path):
    """A queue file replaces via load_queue_from; snapshots add one job each."""
    plate = tmp_path / "plate"
    mask = _touch(plate / "settings" / "mask_settings.csv", "Key,Value\n")
    measure = _touch(plate / "measure_settings.csv", "Key,Value\n")
    _touch(plate / "settings" / "readme.txt")
    single = _touch(tmp_path / "classify_settings.csv", "Key,Value\n")

    def loader(answer):
        def load(path):
            load.calls.append(path)
            return answer
        load.calls = []
        return load

    def adder(answer):
        def add(module, settings):
            add.calls.append((module, settings))
            return answer
        add.calls = []
        return add

    cases = [
        # (id, dropped, loader result, last_error, adder result,
        #  loaded, added, error)
        ("a saved json queue is loaded", _touch(tmp_path / "queue.json", "[]"),
         True, None, None, ["queue.json"], None, None),
        ("a yaml queue that will not load reports the screen's reason",
         _touch(tmp_path / "queue.yaml", "[]"), False, "bad YAML", None,
         ["queue.yaml"], None, "bad YAML"),
        ("a yml queue that will not load names the path",
         _touch(tmp_path / "queue.yml", "[]"), False, None, None,
         ["queue.yml"], None, f"Could not load {tmp_path / 'queue.yml'}."),
        ("a folder adds each snapshot as a job", plate, None, None, True,
         None, [("mask", str(mask)), ("measure", str(measure))], None),
        ("a settings csv is one job", single, None, None, True, None,
         [("classify", str(single))], None),
        ("a folder whose snapshots are all refused is reported", plate, None,
         None, False, None,
         [("mask", str(mask)), ("measure", str(measure))],
         f"No runnable settings jobs found in {plate}."),
    ]
    for case_id, dropped, loaded, reason, added, load_calls, add_calls, error \
            in cases:
        with subtests.test(msg=case_id):
            load, add = loader(loaded), adder(added)
            attrs = {"load_queue_from": load, "add_job": add}
            if reason is not None:
                attrs["last_error"] = reason
            screen = _Screen(**attrs)
            if error is None:
                dh.BatchDropHandler().apply(dropped, screen)
            else:
                with pytest.raises(ValueError) as raised:
                    dh.BatchDropHandler().apply(dropped, screen)
                assert str(raised.value) == error
            assert [Path(p).name for p in load.calls] == (load_calls or [])
            assert add.calls == (add_calls or [])


def test_results_database_opens_through_either_setter(subtests, tmp_path):
    """set_database wins over open_database; a refusal names the reason or path."""
    db = tmp_path / "measurements.db"

    def opener(answer):
        def open_(path):
            open_.calls.append(path)
            return answer
        open_.calls = []
        return open_

    cases = [
        # (id, set_database, open_database, last_error, error type, message)
        ("plate viewer opens through open_database", None, opener(None),
         None, None, None),
        ("annotator agreement opens through set_database", opener(True),
         opener(True), None, None, None),
        ("a refused database reports the screen's reason", opener(False),
         None, "locked", ValueError, "locked"),
        ("a refused database without a reason names the path", None,
         opener(False), "", ValueError, f"Could not open {db}."),
        ("a screen that cannot open a database says so", None, None, None,
         TypeError, "This screen cannot open a database."),
    ]
    for case_id, setter, open_db, reason, error, message in cases:
        with subtests.test(msg=case_id):
            attrs = {}
            if setter is not None:
                attrs["set_database"] = setter
            if open_db is not None:
                attrs["open_database"] = open_db
            if reason is not None:
                attrs["last_error"] = reason
            screen = _Screen(**attrs)
            if error is None:
                dh.ResultsDatabaseDropHandler().apply(db, screen)
            else:
                with pytest.raises(error) as raised:
                    dh.ResultsDatabaseDropHandler().apply(db, screen)
                assert str(raised.value) == message
            used = setter if setter is not None else open_db
            if used is not None:
                assert used.calls == [str(db)]
            if setter is not None and open_db is not None:
                assert open_db.calls == []
