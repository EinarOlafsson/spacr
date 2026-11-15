"""Table-driven tests of saving, reading back and restoring a run workspace.

``spacr.workspace`` is best-effort by design: a panel that is not open, a file
that has gone, a provider that raises or a document from another machine must
each end as a line in the document or the restore report, never as an
exception, because the run itself has already finished.  One general test per
unit walks a case table with pytest's ``subtests`` fixture.  Everything is
written to and read back from real directories with real SHA-256 digests; the
failure paths come from real broken inputs (a file replaced by a directory, a
home directory that does not exist, a document that is not JSON).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from spacr.workspace import (CARRY_KEY, DOC_NAME, FILES_DIR, _human_size,
                             _jsonable, check_files, clear_providers, collect,
                             default_copy_limit_mb, default_mode,
                             has_workspace, hash_file, inventory,
                             inventory_text, load, providers, register,
                             restore, save, save_for_run, section_states,
                             set_default_mode)


@pytest.fixture(autouse=True)
def process_defaults_restored():
    """The module keeps process-wide state; put all of it back afterwards."""
    before = (default_mode(), default_copy_limit_mb())
    registered = providers()
    yield
    clear_providers()
    for name, provider in registered.items():
        register(name, provider)
    set_default_mode(before[0], before[1])


def _written(run):
    """The workspace document exactly as it landed on disk."""
    return json.loads((run / DOC_NAME).read_text(encoding="utf-8"))


def _carry(path, role="figure"):
    """A section that asks for ``path`` to be carried with the workspace."""
    return {CARRY_KEY: [{"role": role, "path": str(path)}]}


def _stored_name(path):
    """The bundle name ``save`` gives a carried copy of ``path``."""
    return f"{FILES_DIR}/{hash_file(path)[:16]}__{path.name}"


def _file(path, data):
    """Write ``data`` (bytes) to ``path`` and hand the path back."""
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# collecting
# ---------------------------------------------------------------------------

def test_section_states_keeps_the_good_sections_and_names_the_bad_ones(
        subtests):
    """Absent panels are silent; stateless or raising ones are one problem line each."""
    class Silent:
        pass

    def explodes():
        raise RuntimeError("the panel is half built")

    cases = [
        ("a panel that is not open is not a problem",
         {"volcano": lambda: None, "montage": lambda: {"page": 3}},
         {"montage": {"page": 3}}, []),
        ("a present panel offering no state is reported",
         {"volcano": Silent()},
         {}, [{"section": "volcano", "why": "offers no workspace state"}]),
        ("one raising provider does not take the others with it",
         {"broken": explodes, "montage": lambda: {"page": 3}},
         {"montage": {"page": 3}},
         [{"section": "broken",
           "why": "RuntimeError: the panel is half built"}]),
    ]
    for case_id, contributors, sections, problems in cases:
        with subtests.test(msg=case_id):
            assert section_states(contributors) == (sections, problems)


def test_jsonable_turns_any_state_into_bounded_json(subtests):
    """Exotic values become text, and cycles or deep nests stop at depth 12."""
    class Opaque:
        def __repr__(self):
            return "<a QColor>"

    def wrapped(key, inner, levels=13):
        for _ in range(levels):
            inner = {key: inner}
        return inner

    loop = {}
    loop["self"] = loop
    deep = {"end": Path("/tmp/x")}
    for _ in range(20):
        deep = {"d": deep}
    cut = deep
    for _ in range(13):
        cut = cut["d"]

    cases = [
        ("exotic values become text, sets sort, tuples list",
         {"colour": Opaque(), "path": Path("/tmp/x"), "seen": {3, 1, 2},
          "pair": (1, 2), "plain": [None, True, 1, 2.5, "s"], 7: "key"},
         {"colour": "<a QColor>", "path": str(Path("/tmp/x")),
          "seen": [1, 2, 3], "pair": [1, 2],
          "plain": [None, True, 1, 2.5, "s"], "7": "key"}),
        ("a state containing itself is truncated, not recursed forever",
         loop, wrapped("self", str(loop))),
        ("a value past the depth bound is stringified there",
         deep, wrapped("d", str(cut))),
    ]
    for case_id, value, expected in cases:
        with subtests.test(msg=case_id):
            out = _jsonable(value)
            assert out == expected
            assert json.loads(json.dumps(out)) == out


def test_inventory_records_the_files_a_state_points_at(tmp_path, subtests):
    """Carried entries with or without a role, path-looking strings only, foreign homes dropped."""
    figure = _file(tmp_path / "volcano.png", b"PNG")
    table = tmp_path / "coeffs.csv"
    table.write_text("gene,coef\n", encoding="utf-8")

    def record(role, path, **extra):
        stat = path.stat()
        return {"role": role, "path": str(path), "exists": True,
                "kind": "file", "size": stat.st_size,
                "mtime": round(stat.st_mtime, 3),
                "sha256": hash_file(path), **extra}

    cases = [
        ("a carried file named with or without a role is one carried record",
         {"volcano": {CARRY_KEY: [str(figure),
                                  {"role": "table", "path": str(figure)},
                                  {"no path": "here"}]}},
         [record("volcano:carried", figure, carry=True)]),
        ("a path from another machine's home is skipped, not raised",
         {"volcano": {
             "figure": "~a_user_that_does_not_exist_here/plots/volcano.png"}},
         []),
        ("only strings that look like paths are stat-ed",
         {"volcano": {"genes": ["TP53", "BRCA1", "ROCK1"],
                      "table": str(table), "short": "a/b", "initial": "/"}},
         [record("volcano.table", table)]),
        ("a section that is not a mapping carries nothing",
         {"volcano": ["not", "a", "state"]}, []),
    ]
    for case_id, sections, expected in cases:
        with subtests.test(msg=case_id):
            assert inventory(sections) == expected


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def test_save_writes_the_document_and_carries_what_it_should(
        tmp_path, subtests, monkeypatch):
    """Every mode and every reason a file is not copied ends up in the document."""
    with subtests.test(msg="off writes nothing at all"):
        doc = collect({"volcano": {"page": 1}}, saved="t")
        assert save(tmp_path / "off", doc, mode="off") is None
        assert not (tmp_path / "off").exists()

    with subtests.test(msg="a missing file or a directory is recorded, not copied"):
        run = tmp_path / "missing"
        files = [
            {"role": "figure", "path": str(tmp_path / "gone.png"),
             "exists": False, "kind": "file", "size": 10, "carry": True},
            {"role": "outdir", "path": str(tmp_path), "exists": True,
             "kind": "directory", "carry": True},
        ]
        assert save(run, {"files": files}, mode="copy") == run / DOC_NAME
        assert _written(run) == {"files": files, "mode": "copy"}
        assert not (run / FILES_DIR).exists()

    with subtests.test(msg="reference mode records a file without carrying it"):
        table = tmp_path / "coeffs.csv"
        table.write_text("gene,coef\nTP53,1.0\n", encoding="utf-8")
        run = tmp_path / "reference"
        save(run, collect({"volcano": {"table": str(table)}}, saved="t"),
             mode="reference")
        written = _written(run)
        assert written["mode"] == "reference"
        assert [(r["path"], r["sha256"], "copied" in r)
                for r in written["files"]] == [
            (str(table), hash_file(table), False)]
        assert not (run / FILES_DIR).exists()

    with subtests.test(msg="a file over the limit is skipped and says why"):
        big = _file(tmp_path / "huge.tif", b"x" * 4096)
        run = tmp_path / "limit"
        save(run, collect({"images": {"stack": str(big)}}, saved="t"),
             mode="copy", copy_limit_mb=0.001)
        record = _written(run)["files"][0]
        assert record["copied"] is None
        assert record["skipped"] == "0.0 MB is over the 0.001 MB per-file limit"
        assert not (run / FILES_DIR).exists()

    with subtests.test(msg="a carried file is copied whatever its size"):
        figure = _file(tmp_path / "volcano.png", b"y" * 4096)
        run = tmp_path / "carried"
        save(run, collect({"volcano": _carry(figure)}, saved="t"),
             mode="copy", copy_limit_mb=0.001)
        copied = _written(run)["files"][0]["copied"]
        assert copied == _stored_name(figure)
        assert (run / copied).read_bytes() == b"y" * 4096

    with subtests.test(msg="a carried file is copied in reference mode too"):
        run = tmp_path / "carried_reference"
        save(run, collect({"volcano": _carry(figure)}, saved="t"),
             mode="reference")
        assert _written(run)["files"][0]["copied"] == _stored_name(figure)

    with subtests.test(msg="a file already in the store is not copied again"):
        source = _file(tmp_path / "stack.tif", b"the same bytes both times")
        run = tmp_path / "twice"
        save(run, collect({"images": {"stack": str(source)}}, saved="t"),
             mode="copy")
        stored = sorted((run / FILES_DIR).iterdir())
        first_mtime = stored[0].stat().st_mtime_ns
        save(run, collect({"images": {"stack": str(source)}}, saved="t"),
             mode="copy")
        again = sorted((run / FILES_DIR).iterdir())
        assert [p.name for p in again] == [stored[0].name]
        assert again[0].stat().st_mtime_ns == first_mtime
        assert _written(run)["files"][0]["copied"] == _stored_name(source)

    with subtests.test(msg="a source turned directory is a recorded copy failure"):
        vanished = _file(tmp_path / "vanished.png", b"PNG")
        run = tmp_path / "vanished"
        doc = collect({"volcano": _carry(vanished)}, saved="t")
        vanished.unlink()
        vanished.mkdir()
        assert save(run, doc, mode="copy") == run / DOC_NAME
        record = _written(run)["files"][0]
        assert record["copied"] is None
        assert record["skipped"].startswith("could not copy: IsADirectoryError: ")

    with subtests.test(msg="a store that cannot be written leaves the reason"):
        def refuse(*args, **kwargs):
            raise PermissionError("read-only file system")

        monkeypatch.setattr(shutil, "copy2", refuse)
        source = _file(tmp_path / "data.tif", b"data")
        run = tmp_path / "readonly"
        save(run, collect({"images": {"stack": str(source)}}, saved="t"),
             mode="copy")
        record = _written(run)["files"][0]
        assert record["copied"] is None
        assert record["skipped"] == \
            "could not copy: PermissionError: read-only file system"


def test_save_for_run_writes_only_when_a_run_has_state_to_keep(
        tmp_path, subtests):
    """Headless or switched-off runs gain nothing; a GUI run gets its document."""
    cases = [
        ("nothing registered writes nothing", {}, None, {}),
        ("a provider with nothing open writes nothing",
         {"volcano": lambda: None}, None, {}),
        ("save_workspace=off writes nothing",
         {"volcano": lambda: {"page": 1}}, None, {"save_workspace": "off"}),
    ]
    for case_id, registry, contributors, settings in cases:
        with subtests.test(msg=case_id):
            clear_providers()
            for name, provider in registry.items():
                register(name, provider)
            run = tmp_path / case_id.replace(" ", "_")
            assert save_for_run(run, settings, contributors=contributors) is None
            assert not run.exists()

    with subtests.test(msg="a finished run gets its workspace beside it"):
        clear_providers()
        register("volcano", lambda: {"page": 4})
        run = tmp_path / "finished"
        assert save_for_run(run, {"save_workspace": "reference"},
                            app_key="regression") == run / DOC_NAME
        doc = load(run)
        assert {k: doc[k] for k in ("version", "app_key", "sections",
                                    "files", "mode")} == {
            "version": 1, "app_key": "regression",
            "sections": {"volcano": {"page": 4}}, "files": [],
            "mode": "reference"}

    with subtests.test(msg="explicit contributors replace the registry"):
        run = tmp_path / "explicit"
        save_for_run(run, {"save_workspace": "reference"},
                     contributors={"montage": {"page": 9}})
        assert load(run)["sections"] == {"montage": {"page": 9}}

    with subtests.test(msg="the settings' copy limit reaches the writer"):
        big = _file(tmp_path / "big.tif", b"z" * 4096)
        run = tmp_path / "limited"
        save_for_run(run, {"save_workspace": "copy",
                           "workspace_copy_limit_mb": 0.001},
                     contributors={"images": {"stack": str(big)}})
        assert load(run)["files"][0]["skipped"] == \
            "0.0 MB is over the 0.001 MB per-file limit"


# ---------------------------------------------------------------------------
# reading back
# ---------------------------------------------------------------------------

def test_load_finds_a_document_or_answers_none(tmp_path, subtests, caplog):
    """A folder, the document or a sibling all find it; junk and absence read as None."""
    cases = [
        ("a document that is not json reads as absent", "{ truncated", True),
        ("valid json that is not an object is refused", "[1, 2, 3]", False),
    ]
    for case_id, text, warns in cases:
        with subtests.test(msg=case_id):
            folder = tmp_path / case_id.replace(" ", "_")
            folder.mkdir()
            (folder / DOC_NAME).write_text(text, encoding="utf-8")
            caplog.clear()
            with caplog.at_level("WARNING", logger="spacr.workspace"):
                assert load(folder) is None
            assert (DOC_NAME in caplog.text) is warns

    with subtests.test(msg="a run folder with no workspace reads as none"):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert load(empty) is None
        assert has_workspace(empty) is False

    with subtests.test(msg="folder, document and sibling name the same document"):
        run = tmp_path / "run"
        doc = collect({"volcano": {"page": 1}}, saved="t")
        save(run, doc, mode="reference")
        sibling = run / "settings.csv"
        sibling.write_text("a,b\n", encoding="utf-8")
        assert has_workspace(run) is True
        expected = {**doc, "mode": "reference"}
        assert [load(run), load(run / DOC_NAME), load(sibling)] == [expected] * 3


def test_check_files_says_what_became_of_each_file(tmp_path, subtests):
    """present, changed, carried from the bundle, missing; junk entries skipped."""
    with subtests.test(msg="a file that changed since the save says so"):
        table = tmp_path / "coeffs.csv"
        table.write_text("gene,coef\nTP53,1.0\n", encoding="utf-8")
        doc = collect({"volcano": {"table": str(table)}}, saved="t")
        assert check_files(doc) == [
            {"role": "volcano.table", "path": str(table), "state": "present"}]
        table.write_text("gene,coef\nTP53,9.9\n", encoding="utf-8")
        assert check_files(doc) == [
            {"role": "volcano.table", "path": str(table), "state": "changed"}]

    with subtests.test(msg="a carried file that is gone is found in the bundle"):
        figure = _file(tmp_path / "volcano.png", b"PNG")
        run = tmp_path / "run"
        save(run, collect({"volcano": _carry(figure)}, saved="t"), mode="copy")
        stored = (run / _stored_name(figure)).resolve()
        figure.unlink()
        saved_doc = load(run)
        assert check_files(saved_doc, run_dir=run) == [
            {"role": "volcano:figure", "path": str(stored), "state": "carried"}]
        assert check_files(saved_doc) == [
            {"role": "volcano:figure", "path": str(figure), "state": "missing"}]

    with subtests.test(msg="a junk entry in the file list is skipped"):
        assert check_files({"files": ["not a record", None, 7]}) == []


# ---------------------------------------------------------------------------
# putting it back
# ---------------------------------------------------------------------------

def test_restore_reports_every_section_restored_or_why_not(subtests):
    """Every way a section can fail to come back is a named skipped line."""
    class Panel:
        def __init__(self, answer=True):
            self.got = None
            self.answer = answer

        def apply_workspace_state(self, state):
            self.got = state
            return self.answer

    class Old:
        def __init__(self):
            self.got = None

        def apply_plot_state(self, state):
            self.got = state

    class Deaf:
        pass

    class Broken:
        def apply_workspace_state(self, state):
            raise ValueError("the table is not loaded")

    def skip(section, why):
        return {"section": section, "why": why}

    provided = Panel()
    old = Old()
    declining = Panel(answer=False)
    cases = [
        ("a document with no sections says so", {}, {},
         [], [skip("", "no sections in the document")], None, None),
        ("sections that are not a mapping are no sections", {},
         {"sections": "not a mapping"},
         [], [skip("", "no sections in the document")], None, None),
        ("a section that is not a state document is skipped",
         {"volcano": Panel()}, {"sections": {"volcano": [1, 2]}},
         [], [skip("volcano", "not a state document")], None, None),
        ("a section nothing on screen owns is skipped", {},
         {"sections": {"volcano": {"page": 1}}},
         [], [skip("volcano", "nothing on screen owns it")], None, None),
        ("a provider is called and the reserved keys are stripped",
         {"volcano": lambda: provided},
         {"sections": {"volcano": {"page": 2, CARRY_KEY: [{"path": "x"}]}}},
         ["volcano"], [], provided, {"page": 2}),
        ("the legacy apply_plot_state is accepted",
         {"volcano": old}, {"sections": {"volcano": {"page": 5}}},
         ["volcano"], [], old, {"page": 5}),
        ("a panel with no setter is reported",
         {"volcano": Deaf()}, {"sections": {"volcano": {"page": 1}}},
         [], [skip("volcano", "cannot take a workspace state back")],
         None, None),
        ("a raising panel does not stop the others",
         {"volcano": Broken(), "montage": Panel()},
         {"sections": {"volcano": {"a": 1}, "montage": {"b": 2}}},
         ["montage"], [skip("volcano", "ValueError: the table is not loaded")],
         None, None),
        ("a panel declining is skipped with its own wording",
         {"volcano": declining}, {"sections": {"volcano": {"page": 1}}},
         [], [skip("volcano", "the panel declined it")],
         declining, {"page": 1}),
    ]
    for case_id, contributors, doc, restored, skipped, panel, got in cases:
        with subtests.test(msg=case_id):
            assert restore(contributors, doc) == {
                "restored": restored, "skipped": skipped, "files": []}
            if panel is not None:
                assert panel.got == got


# ---------------------------------------------------------------------------
# saying what is in one
# ---------------------------------------------------------------------------

def test_human_size_prints_binary_units_up_to_terabytes(subtests):
    """Each unit starts at 1024 of the one below; the top unit keeps counting."""
    cases = [
        (None, "?"), ("big", "?"), ("a lot", "?"),
        (0, "0 B"), (1023, "1023 B"), (1024, "1.0 KB"), (2048, "2.0 KB"),
        (1.5 * 1024 ** 2, "1.5 MB"), (3 * 1024 ** 3, "3.0 GB"),
        (5 * 1024 ** 3, "5.0 GB"), (1023 * 1024 ** 3, "1023.0 GB"),
        (7 * 1024 ** 4, "7.0 TB"), (2500 * 1024 ** 4, "2500.0 TB"),
        ("2048", "2.0 KB"),
    ]
    for value, expected in cases:
        with subtests.test(msg=repr(value)):
            assert _human_size(value) == expected


def test_inventory_text_names_sections_files_and_problems(tmp_path, subtests):
    """One readable block: what was saved, what it points at, what went wrong."""
    class Broken:
        def workspace_state(self):
            raise RuntimeError("half built")

    with subtests.test(msg="a carried file, its size and a failed section"):
        figure = _file(tmp_path / "volcano.png", b"z" * 2048)
        run = tmp_path / "run"
        doc = collect({"volcano": {"page": 1, **_carry(figure)},
                       "broken": Broken()},
                      app_key="regression", saved="2026-01-01T00:00:00+00:00")
        save(run, doc, mode="copy")
        assert inventory_text(load(run), run_dir=run).splitlines() == [
            "workspace v1 [copy] saved 2026-01-01T00:00:00+00:00",
            "  app: regression",
            "  sections (1):",
            f"    {'volcano':<24} 2 keys",
            "  files (1):",
            f"    {figure}  (present, carried, 2.0 KB)",
            "  ! broken: RuntimeError: half built",
        ]

    with subtests.test(msg="a skipped file prints the reason with the file"):
        big = _file(tmp_path / "huge.tif", b"x" * 4096)
        run = tmp_path / "skipped"
        save(run, collect({"images": {"stack": str(big)}}, saved="t"),
             mode="copy", copy_limit_mb=0.001)
        assert inventory_text(load(run), run_dir=run).splitlines() == [
            "workspace v1 [copy] saved t",
            "  sections (1):",
            f"    {'images':<24} 1 key",
            "  files (1):",
            f"    {big}  (present, skipped: 0.0 MB is over the 0.001 MB "
            "per-file limit, 4.0 KB)",
        ]
