"""A workspace is mostly promises about what happens when something is wrong.

Saving and restoring the interactive state around a run is best-effort by
design: a panel that is not open, a figure whose file has been deleted, a
provider that raises, a state document from another machine. Every one of
those has to end as a line in the report rather than as an exception, because
the run itself has already finished and its results must not be lost to a
bookkeeping failure.

Nothing here is mocked away. The documents are written to and read back from
real directories, the digests are real SHA-256 over real bytes, and the
failure paths are produced by real broken inputs -- a file replaced by a
directory, a home directory that does not exist, a document that is not JSON.
"""

import json
import os
from pathlib import Path

import pytest

from spacr import workspace
from spacr.workspace import (CARRY_KEY, DEFAULT_COPY_LIMIT_MB, DEFAULT_MODE,
                             DOC_NAME, FILES_DIR, _carried, _file_record,
                             _human_size, _jsonable, _state_of, _walk_strings,
                             check_files, clear_providers, collect,
                             copy_limit_from_settings, default_copy_limit_mb,
                             default_mode, has_workspace, hash_file, inventory,
                             inventory_text, load, mode_from_settings,
                             providers, register, report_text, restore,
                             resolve_mode, save, save_for_run, section_states,
                             set_default_mode, unregister)


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


# ---------------------------------------------------------------------------
# the process-wide defaults
# ---------------------------------------------------------------------------

def test_a_nonsense_copy_limit_leaves_the_old_one_standing():
    """An unparseable limit is ignored rather than becoming zero.

    Zero would silently stop every file being carried, which looks exactly
    like the feature being off.
    """
    set_default_mode("copy", 64)
    assert default_copy_limit_mb() == 64.0

    assert set_default_mode("copy", "not a number") == "copy"
    assert default_copy_limit_mb() == 64.0

    assert set_default_mode("copy", object()) == "copy"
    assert default_copy_limit_mb() == 64.0

    # a negative one is equally ignored
    set_default_mode("copy", -5)
    assert default_copy_limit_mb() == 64.0


def test_the_mode_aliases_a_user_actually_types_are_understood():
    """`save_workspace` is a settings field, so it arrives as text."""
    assert resolve_mode(None) == DEFAULT_MODE
    assert resolve_mode(True) == "reference"
    assert resolve_mode(False) == "off"
    assert resolve_mode("  COPY ") == "copy"
    assert resolve_mode("no") == "off"
    assert resolve_mode("yes") == "copy"
    assert resolve_mode("sometimes") == DEFAULT_MODE


# ---------------------------------------------------------------------------
# what a contributor offers
# ---------------------------------------------------------------------------

def test_a_contributor_can_offer_state_four_ways():
    """A mapping, the new method, the legacy pair, or nothing at all."""
    class New:
        def workspace_state(self):
            return {"a": 1}

    class Legacy:
        def plot_state(self):
            return {"b": 2}

    class Silent:
        pass

    assert _state_of(None) is None
    assert _state_of({"a": 1}) == {"a": 1}
    assert _state_of(New()) == {"a": 1}
    assert _state_of(Legacy()) == {"b": 2}
    assert _state_of(Silent()) is None


def test_a_contributor_that_answers_with_the_wrong_shape_offers_nothing():
    """A state getter returning a list is not a state document."""
    class Wrong:
        def workspace_state(self):
            return ["not", "a", "mapping"]

    class WrongLegacy:
        def plot_state(self):
            return 7

    assert _state_of(Wrong()) is None
    assert _state_of(WrongLegacy()) is None


def test_a_panel_that_is_not_open_is_not_a_problem():
    """Every screen registers the same panels and most build none of them.

    Reporting each absent panel would bury the one section that genuinely
    failed under a dozen that were simply not there.
    """
    sections, problems = section_states({
        "volcano": lambda: None,
        "montage": lambda: {"page": 3},
    })
    assert sections == {"montage": {"page": 3}}
    assert problems == []


def test_a_panel_that_offers_no_state_is_reported():
    """Present but stateless is a problem: it was expected to have some."""
    class Silent:
        pass

    sections, problems = section_states({"volcano": Silent()})
    assert sections == {}
    assert problems == [{"section": "volcano",
                         "why": "offers no workspace state"}]


def test_a_provider_that_raises_does_not_take_the_others_with_it():
    """One broken panel is one line in `problems`; the rest still collect."""
    def explodes():
        raise RuntimeError("the panel is half built")

    sections, problems = section_states({
        "broken": explodes,
        "montage": lambda: {"page": 3},
    })
    assert sections == {"montage": {"page": 3}}
    assert problems == [{"section": "broken",
                         "why": "RuntimeError: the panel is half built"}]


def test_state_that_json_cannot_hold_becomes_text_rather_than_failing():
    """Exotic values are stringified so one odd widget cannot lose a section."""
    class Opaque:
        def __repr__(self):
            return "<a QColor>"

    out = _jsonable({"colour": Opaque(), "path": Path("/tmp/x"),
                     "seen": {3, 1, 2}, "pair": (1, 2)})
    assert out["colour"] == "<a QColor>"
    assert out["path"] == "/tmp/x"
    assert out["seen"] == [1, 2, 3]
    assert out["pair"] == [1, 2]
    json.dumps(out)          # the point of the exercise


def test_a_state_that_contains_itself_costs_a_truncated_section():
    """A cycle is bounded by depth, not by the process running out of stack."""
    loop = {}
    loop["self"] = loop
    out = _jsonable(loop)
    assert json.dumps(out)


# ---------------------------------------------------------------------------
# the file inventory
# ---------------------------------------------------------------------------

def test_a_file_that_cannot_be_read_has_no_digest(tmp_path):
    """`hash_file` answers None rather than raising, and the record says so."""
    assert hash_file(tmp_path / "not_there") is None
    assert hash_file(tmp_path) is None          # a directory is not readable

    record = _file_record("figure", tmp_path / "not_there")
    assert record == {"role": "figure", "path": str(tmp_path / "not_there"),
                      "exists": False}


def test_a_directory_is_recorded_as_one_and_not_hashed(tmp_path):
    """A referenced folder is inventoried without a size or a digest."""
    record = _file_record("outdir", tmp_path)
    assert record["exists"] is True
    assert record["kind"] == "directory"
    assert "sha256" not in record
    assert "size" not in record


def test_a_section_that_is_not_a_mapping_carries_nothing():
    """The carry key only exists inside a state document."""
    assert _carried(None) == []
    assert _carried(["a", "b"]) == []
    assert _carried("a string") == []


def test_a_carried_file_may_be_named_with_or_without_a_role(tmp_path):
    """A bare string is accepted and gets the default role."""
    figure = tmp_path / "volcano.png"
    figure.write_bytes(b"PNG")

    records = inventory({"volcano": {CARRY_KEY: [
        str(figure),
        {"role": "table", "path": str(figure)},
        {"no path": "here"},
    ]}})
    assert len(records) == 1
    assert records[0]["carry"] is True
    assert records[0]["role"].endswith("carried")


def test_a_path_from_another_machines_home_is_skipped_not_raised(tmp_path):
    """A `~someone-else` path cannot be expanded here, and is dropped.

    A saved workspace travels; a home directory does not.
    """
    records = inventory({"volcano": {
        "figure": "~a_user_that_does_not_exist_here/plots/volcano.png"}})
    assert records == []


def test_only_strings_that_look_like_paths_are_stat_ed(tmp_path):
    """Gene names are not asked about; a path with a separator is.

    Without the filter the walk calls `stat` once per gene in a picked-cell
    list, which is the whole plot.
    """
    table = tmp_path / "coeffs.csv"
    table.write_text("gene,coef\n")

    records = inventory({"volcano": {
        "genes": ["TP53", "BRCA1", "ROCK1"],
        "table": str(table),
        "short": "a/b",
        "initial": "/",
    }})
    assert [r["path"] for r in records] == [str(table)]
    assert records[0]["sha256"] == hash_file(table)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _doc_with(tmp_path, **section):
    return collect({"volcano": section}, app_key="regression",
                   saved="2026-01-01T00:00:00+00:00")


def test_off_writes_nothing_at_all(tmp_path):
    """A run folder saved with the feature off is what it was before."""
    doc = _doc_with(tmp_path, page=1)
    assert save(tmp_path / "run", doc, mode="off") is None
    assert not (tmp_path / "run").exists()


def test_a_missing_file_is_recorded_but_not_copied(tmp_path):
    """A record for a file that has gone is left alone by the copier."""
    run = tmp_path / "run"
    doc = {"files": [
        {"role": "figure", "path": str(tmp_path / "gone.png"),
         "exists": False, "kind": "file", "size": 10, "carry": True},
        {"role": "outdir", "path": str(tmp_path), "exists": True,
         "kind": "directory", "carry": True},
    ]}
    save(run, doc, mode="copy")

    written = json.loads((run / DOC_NAME).read_text())
    assert "copied" not in written["files"][0]
    assert "copied" not in written["files"][1]
    assert not (run / FILES_DIR).exists()


def test_a_file_over_the_limit_is_skipped_and_says_why(tmp_path):
    """The limit bounds `copy`, and the skip is visible in the document."""
    big = tmp_path / "huge.tif"
    big.write_bytes(b"x" * 4096)
    run = tmp_path / "run"

    doc = collect({"images": {"stack": str(big)}}, saved="t")
    save(run, doc, mode="copy", copy_limit_mb=0.001)

    written = json.loads((run / DOC_NAME).read_text())
    record = written["files"][0]
    assert record["copied"] is None
    assert "over the 0.001 MB per-file limit" in record["skipped"]


def test_a_carried_file_is_copied_whatever_its_size(tmp_path):
    """A section asserting a figure exists nowhere else is believed.

    A silently dropped figure is worse than a large run folder.
    """
    figure = tmp_path / "volcano.png"
    figure.write_bytes(b"y" * 4096)
    run = tmp_path / "run"

    doc = collect({"volcano": {CARRY_KEY: [
        {"role": "figure", "path": str(figure)}]}}, saved="t")
    save(run, doc, mode="copy", copy_limit_mb=0.001)

    written = json.loads((run / DOC_NAME).read_text())
    copied = written["files"][0]["copied"]
    assert copied and copied.startswith(FILES_DIR)
    assert (run / copied).read_bytes() == b"y" * 4096


def test_a_copy_that_fails_is_recorded_rather_than_raised(tmp_path):
    """A file that vanishes between inventory and copy leaves a note.

    The run has already finished; losing it to a failed file copy would be
    the worst possible trade.
    """
    figure = tmp_path / "volcano.png"
    figure.write_bytes(b"PNG")
    run = tmp_path / "run"
    doc = collect({"volcano": {CARRY_KEY: [
        {"role": "figure", "path": str(figure)}]}}, saved="t")

    figure.unlink()
    figure.mkdir()           # same path, now a directory: copy2 refuses

    assert save(run, doc, mode="copy") is not None
    written = json.loads((run / DOC_NAME).read_text())
    record = written["files"][0]
    assert record["copied"] is None
    assert record["skipped"].startswith("could not copy: ")


# ---------------------------------------------------------------------------
# reading back
# ---------------------------------------------------------------------------

def test_a_document_that_is_not_json_reads_as_absent(tmp_path, caplog):
    """A truncated write is a warning and a None, not an exception."""
    (tmp_path / DOC_NAME).write_text("{ truncated")
    with caplog.at_level("WARNING"):
        assert load(tmp_path) is None
    assert DOC_NAME in caplog.text


def test_a_document_holding_something_other_than_an_object_is_refused(
        tmp_path):
    """Valid JSON that is not a document is still not a document."""
    (tmp_path / DOC_NAME).write_text("[1, 2, 3]")
    assert load(tmp_path) is None


def test_a_run_folder_with_no_workspace_reads_as_none(tmp_path):
    """Absence is None, and asking is not an error."""
    assert load(tmp_path) is None
    assert has_workspace(tmp_path) is False


def test_asking_whether_nonsense_has_a_workspace_is_false():
    """`has_workspace` answers rather than raising on a bad argument."""
    assert has_workspace(None) is False
    assert has_workspace(object()) is False


def test_a_document_can_be_named_directly_or_by_its_folder(tmp_path):
    """Both the folder and the document path find the same document."""
    doc = collect({"volcano": {"page": 1}}, saved="t")
    save(tmp_path, doc, mode="reference")

    assert has_workspace(tmp_path) is True
    by_folder = load(tmp_path)
    by_file = load(tmp_path / DOC_NAME)
    assert by_folder == by_file
    # a sibling file in the same folder resolves to the document too
    sibling = tmp_path / "settings.csv"
    sibling.write_text("a,b\n")
    assert load(sibling) == by_folder


# ---------------------------------------------------------------------------
# the state of the files, later
# ---------------------------------------------------------------------------

def test_a_file_that_changed_since_the_save_says_so(tmp_path):
    """Same path, different bytes, is `changed` -- not `present`."""
    table = tmp_path / "coeffs.csv"
    table.write_text("gene,coef\nTP53,1.0\n")
    doc = collect({"volcano": {"table": str(table)}}, saved="t")

    assert [e["state"] for e in check_files(doc)] == ["present"]

    table.write_text("gene,coef\nTP53,9.9\n")
    assert [e["state"] for e in check_files(doc)] == ["changed"]


def test_a_directory_is_present_without_a_digest_to_compare(tmp_path):
    """Nothing was hashed, so nothing can have changed."""
    outdir = tmp_path / "results"
    outdir.mkdir()
    doc = collect({"run": {"outdir": str(outdir)}}, saved="t")
    assert [e["state"] for e in check_files(doc)] == ["present"]


def test_a_file_that_is_gone_but_carried_is_found_in_the_bundle(tmp_path):
    """The bundle is the fallback, and the report points at the copy."""
    figure = tmp_path / "volcano.png"
    figure.write_bytes(b"PNG")
    run = tmp_path / "run"
    doc = collect({"volcano": {CARRY_KEY: [
        {"role": "figure", "path": str(figure)}]}}, saved="t")
    save(run, doc, mode="copy")
    saved_doc = load(run)

    figure.unlink()
    states = check_files(saved_doc, run_dir=run)
    assert states[0]["state"] == "carried"
    assert Path(states[0]["path"]).is_file()

    # with no bundle to look in, the same file is simply missing
    assert check_files(saved_doc)[0]["state"] == "missing"


def test_a_junk_entry_in_the_file_list_is_skipped():
    """A document from elsewhere may hold anything; it does not stop the walk."""
    doc = {"files": ["not a record", None, 7]}
    assert check_files(doc) == []


# ---------------------------------------------------------------------------
# putting it back
# ---------------------------------------------------------------------------

def test_a_document_with_no_sections_says_so():
    """Nothing to restore is one skipped line, not an empty success."""
    report = restore({}, {})
    assert report["restored"] == []
    assert report["skipped"] == [{"section": "",
                                  "why": "no sections in the document"}]

    assert restore({}, {"sections": "not a mapping"})["skipped"][0]["why"] == \
        "no sections in the document"


def test_a_section_that_is_not_a_state_document_is_skipped():
    """A section holding a list cannot be handed to a panel."""
    report = restore({"volcano": {}}, {"sections": {"volcano": [1, 2]}})
    assert report["skipped"] == [{"section": "volcano",
                                  "why": "not a state document"}]


def test_a_section_nothing_on_screen_owns_is_skipped():
    """A workspace from another screen names panels this one does not have."""
    report = restore({}, {"sections": {"volcano": {"page": 1}}})
    assert report["skipped"] == [{"section": "volcano",
                                  "why": "nothing on screen owns it"}]


def test_a_provider_is_called_and_its_panel_takes_the_state_back():
    """A callable contributor is resolved, and the reserved keys are stripped."""
    class Panel:
        def __init__(self):
            self.got = None

        def apply_workspace_state(self, state):
            self.got = state
            return True

    panel = Panel()
    report = restore({"volcano": lambda: panel},
                     {"sections": {"volcano": {"page": 2,
                                               CARRY_KEY: [{"path": "x"}]}}})
    assert report["restored"] == ["volcano"]
    assert panel.got == {"page": 2}


def test_the_legacy_apply_pair_is_still_accepted():
    """The regression panel had this pair before the workspace existed."""
    class Old:
        def __init__(self):
            self.got = None

        def apply_plot_state(self, state):
            self.got = state

    panel = Old()
    report = restore({"volcano": panel},
                     {"sections": {"volcano": {"page": 5}}})
    assert report["restored"] == ["volcano"]
    assert panel.got == {"page": 5}


def test_a_panel_that_cannot_take_a_state_back_is_reported():
    """No setter at all is a skipped line naming why."""
    class Panel:
        pass

    report = restore({"volcano": Panel()},
                     {"sections": {"volcano": {"page": 1}}})
    assert report["skipped"] == [
        {"section": "volcano", "why": "cannot take a workspace state back"}]


def test_a_panel_that_raises_while_restoring_is_reported():
    """One panel's exception does not stop the others being restored."""
    class Broken:
        def apply_workspace_state(self, state):
            raise ValueError("the table is not loaded")

    class Fine:
        def apply_workspace_state(self, state):
            return True

    report = restore({"volcano": Broken(), "montage": Fine()},
                     {"sections": {"volcano": {"a": 1}, "montage": {"b": 2}}})
    assert report["restored"] == ["montage"]
    assert report["skipped"] == [{"section": "volcano",
                                  "why": "ValueError: the table is not loaded"}]


def test_a_panel_declining_is_an_answer_not_a_failure():
    """`False` means "I have no table yet", and is reported as skipped.

    From the user's side nothing was put back either way, so it belongs in
    the same list -- but with its own wording, not an exception's.
    """
    class NotReady:
        def apply_workspace_state(self, state):
            return False

    report = restore({"volcano": NotReady()},
                     {"sections": {"volcano": {"page": 1}}})
    assert report["restored"] == []
    assert report["skipped"] == [{"section": "volcano",
                                  "why": "the panel declined it"}]


# ---------------------------------------------------------------------------
# saying what is in one
# ---------------------------------------------------------------------------

def test_a_size_that_is_not_a_number_prints_as_a_question_mark():
    """An unparseable size does not stop the inventory printing."""
    assert _human_size(None) == "?"
    assert _human_size("big") == "?"
    assert _human_size(0) == "0 B"
    assert _human_size(2048) == "2.0 KB"
    assert _human_size(5 * 1024 ** 3) == "5.0 GB"


def test_there_is_no_inventory_for_something_that_is_not_a_document():
    """`inventory_text` says so rather than raising on a None."""
    assert inventory_text(None) == "no workspace document"
    assert inventory_text([1, 2]) == "no workspace document"


def test_the_inventory_names_the_sections_the_files_and_the_problems(
        tmp_path):
    """One readable block: what was saved, what it points at, what went wrong."""
    figure = tmp_path / "volcano.png"
    figure.write_bytes(b"z" * 2048)
    run = tmp_path / "run"

    class Broken:
        def workspace_state(self):
            raise RuntimeError("half built")

    doc = collect({"volcano": {"page": 1,
                               CARRY_KEY: [{"role": "figure",
                                            "path": str(figure)}]},
                   "broken": Broken()},
                  app_key="regression", saved="2026-01-01T00:00:00+00:00")
    save(run, doc, mode="copy")
    saved_doc = load(run)

    text = inventory_text(saved_doc, run_dir=run)
    assert "workspace v1 [copy] saved 2026-01-01T00:00:00+00:00" in text
    assert "app: regression" in text
    assert "sections (1):" in text
    assert "volcano" in text
    assert "files (1):" in text
    assert "carried" in text
    assert "2.0 KB" in text
    assert "! broken: RuntimeError: half built" in text


def test_a_skipped_file_says_so_in_the_inventory(tmp_path):
    """The reason a file was not carried is printed with the file."""
    big = tmp_path / "huge.tif"
    big.write_bytes(b"x" * 4096)
    run = tmp_path / "run"
    doc = collect({"images": {"stack": str(big)}}, saved="t")
    save(run, doc, mode="copy", copy_limit_mb=0.001)

    text = inventory_text(load(run), run_dir=run)
    assert "skipped: " in text
    assert "per-file limit" in text


def test_a_junk_file_record_does_not_stop_the_inventory_printing():
    """A document from elsewhere may hold anything in its file list."""
    text = inventory_text({"version": 1, "sections": {}, "files": [
        "not a record", {"path": "/tmp/x", "size": 10}]})
    assert "/tmp/x" in text
    assert "not a record" not in text


def test_a_restore_report_names_what_did_not_come_back():
    """Nothing restored still prints a line; trouble is listed under it."""
    assert report_text({}) == "restored: nothing"

    text = report_text({
        "restored": ["montage"],
        "skipped": [{"section": "volcano", "why": "nothing on screen owns it"},
                    {"why": "no sections in the document"}],
        "files": [{"state": "missing", "role": "figure", "path": "/gone.png"},
                  {"state": "present", "role": "table", "path": "/here.csv"}],
    })
    assert "restored: montage" in text
    assert "not restored — volcano: nothing on screen owns it" in text
    assert "not restored — (document): no sections in the document" in text
    assert "missing — figure: /gone.png" in text
    assert "/here.csv" not in text


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

def test_a_contributor_needs_a_name_and_a_callable():
    """Both are refused loudly: a nameless section cannot be written back."""
    with pytest.raises(ValueError):
        register("", lambda: {})
    with pytest.raises(TypeError):
        register("volcano", {"page": 1})


def test_a_registered_provider_is_resolved_at_collection_time():
    """The callable is held, not the widget it returns.

    Retaining the panel itself would keep a rebuilt or closed screen alive.
    """
    clear_providers()
    built = []

    def provider():
        built.append(1)
        return {"page": len(built)}

    register("volcano", provider)
    assert providers()["volcano"] is provider
    assert built == [], "registering already built the panel"

    doc = collect(providers(), saved="t")
    assert doc["sections"]["volcano"] == {"page": 1}

    assert unregister("volcano") is True
    assert unregister("volcano") is False
    assert "volcano" not in providers()


def test_a_snapshot_of_the_registry_is_a_copy():
    """Mutating what `providers()` returned does not touch the registry."""
    register("volcano", lambda: {"page": 1})
    snapshot = providers()
    snapshot.clear()
    assert "volcano" in providers()


def test_clearing_the_registry_also_restores_the_defaults():
    """Test teardown and application shutdown both want one call for this."""
    register("volcano", lambda: {"page": 1})
    set_default_mode("copy", 1)

    clear_providers()

    assert providers() == {}
    assert default_mode() == DEFAULT_MODE
    assert default_copy_limit_mb() == float(DEFAULT_COPY_LIMIT_MB)


# ---------------------------------------------------------------------------
# what the run settings ask for
# ---------------------------------------------------------------------------

def test_settings_that_say_nothing_get_the_process_default():
    """`save_workspace` unset means "whatever this process was told"."""
    set_default_mode("copy", 32)
    assert mode_from_settings({}) == "copy"
    assert mode_from_settings({"save_workspace": None}) == "copy"
    assert mode_from_settings(None) == "copy"
    assert copy_limit_from_settings({}) == 32.0


def test_an_explicit_setting_overrides_the_process_default():
    """The run's own value wins, in every spelling `resolve_mode` accepts."""
    set_default_mode("copy", 32)
    assert mode_from_settings({"save_workspace": "off"}) == "off"
    assert mode_from_settings({"save_workspace": False}) == "off"
    assert copy_limit_from_settings({"workspace_copy_limit_mb": 8}) == 8.0
    assert copy_limit_from_settings({"workspace_copy_limit_mb": "16"}) == 16.0


def test_a_nonsense_limit_in_the_settings_falls_back_to_the_default():
    """A bad value in a settings CSV does not silently mean zero."""
    set_default_mode("copy", 32)
    assert copy_limit_from_settings({"workspace_copy_limit_mb": "big"}) == 32.0
    assert copy_limit_from_settings({"workspace_copy_limit_mb": -1}) == 32.0
    assert copy_limit_from_settings({"workspace_copy_limit_mb": None}) == 32.0


# ---------------------------------------------------------------------------
# saving beside a finished run
# ---------------------------------------------------------------------------

def test_a_headless_run_does_not_gain_an_empty_workspace(tmp_path):
    """Nothing registered, or nothing collected, writes no document."""
    clear_providers()
    assert save_for_run(tmp_path / "a") is None
    assert not (tmp_path / "a").exists()

    register("volcano", lambda: None)
    assert save_for_run(tmp_path / "b") is None
    assert not (tmp_path / "b").exists()


def test_saving_is_skipped_entirely_when_the_run_turned_it_off(tmp_path):
    """`save_workspace=off` in the settings writes nothing at all."""
    clear_providers()
    register("volcano", lambda: {"page": 1})
    assert save_for_run(tmp_path / "run",
                        {"save_workspace": "off"}) is None
    assert not (tmp_path / "run").exists()


def test_a_finished_run_gets_its_workspace_beside_it(tmp_path):
    """The document lands in the run folder with the app key on it.

    The registry is process-wide and a GUI screen may have contributed to it
    already, so this clears it first rather than assuming it is empty.
    """
    clear_providers()
    register("volcano", lambda: {"page": 4})
    path = save_for_run(tmp_path / "run", {"save_workspace": "reference"},
                        app_key="regression")

    assert path == tmp_path / "run" / DOC_NAME
    doc = load(tmp_path / "run")
    assert doc["app_key"] == "regression"
    assert doc["sections"] == {"volcano": {"page": 4}}
    assert doc["mode"] == "reference"


def test_explicit_contributors_are_used_instead_of_the_registry(tmp_path):
    """A caller with its own panels does not have to touch the registry."""
    clear_providers()
    register("volcano", lambda: {"page": 1})
    save_for_run(tmp_path / "run", {"save_workspace": "reference"},
                 contributors={"montage": {"page": 9}})

    doc = load(tmp_path / "run")
    assert doc["sections"] == {"montage": {"page": 9}}


# ---------------------------------------------------------------------------
# bounded walks
# ---------------------------------------------------------------------------

def test_the_string_walk_is_bounded_and_skips_the_reserved_keys():
    """Reserved keys are carried separately, so the walk must not re-add them.

    Depth is bounded for the same reason `_jsonable`'s is: a state that
    contains itself must cost a truncated section, not the process.
    """
    nested = {"a": "/x/y"}
    for _ in range(20):
        nested = {"deeper": nested}
    nested[CARRY_KEY] = [{"path": "/should/not/be/walked"}]

    found = dict(_walk_strings(nested))
    assert all("should/not" not in v for v in found.values())
    assert len(found) <= 1


def test_a_value_nested_past_the_depth_bound_is_stringified():
    """Beyond the bound, `_jsonable` stops descending and takes the text."""
    deep = {"end": Path("/tmp/x")}
    for _ in range(20):
        deep = {"d": deep}
    assert json.dumps(_jsonable(deep))


def test_reference_mode_records_a_file_without_carrying_its_bytes(tmp_path):
    """The default mode is an inventory, not a bundle.

    A run folder that quietly grew a copy of every figure the user had open
    would be a surprise; `copy` is the mode that says to do that.
    """
    table = tmp_path / "coeffs.csv"
    table.write_text("gene,coef\nTP53,1.0\n")
    run = tmp_path / "run"

    doc = collect({"volcano": {"table": str(table)}}, saved="t")
    save(run, doc, mode="reference")

    written = load(run)
    record = written["files"][0]
    assert record["path"] == str(table)
    assert record["sha256"] == hash_file(table)
    assert "copied" not in record
    assert not (run / FILES_DIR).exists()


def test_a_file_already_in_the_store_is_recorded_without_copying_it_again(
        tmp_path):
    """Saving twice must not rewrite bytes that are already there.

    The copy target is named by content digest, so a file already in the store
    IS the file being asked for. Copying it again is pure I/O -- and on a run
    carrying a multi-gigabyte stack that is the difference between a save that
    returns and one the user gives up on.

    The record still has to say ``copied``, because the document is what a
    later reader resolves paths through: skipping the copy and skipping the
    record would make the second save produce a document that had lost the
    file the first one carried.
    """
    source = tmp_path / "stack.tif"
    source.write_bytes(b"the same bytes both times")
    run = tmp_path / "run"

    doc = collect({"images": {"stack": str(source)}}, saved="t")
    save(run, doc, mode="copy")

    stored = sorted((run / FILES_DIR).iterdir())
    assert len(stored) == 1
    first_mtime = stored[0].stat().st_mtime_ns

    # Second save, same content, same destination name.
    doc_again = collect({"images": {"stack": str(source)}}, saved="t")
    save(run, doc_again, mode="copy")

    stored_again = sorted((run / FILES_DIR).iterdir())
    assert [p.name for p in stored_again] == [stored[0].name]
    assert stored_again[0].stat().st_mtime_ns == first_mtime, (
        "the file was copied over itself despite already being there")

    written = json.loads((run / DOC_NAME).read_text())
    assert written["files"][0]["copied"].endswith(stored[0].name)


def test_a_store_that_cannot_be_written_leaves_the_reason_in_the_document(
        tmp_path, monkeypatch):
    """A failed copy is recorded, not raised.

    A workspace is a convenience attached to a finished run. Taking the run's
    own save down because one referenced file could not be copied would lose
    the results to protect a copy of the inputs.
    """
    import shutil

    source = tmp_path / "stack.tif"
    source.write_bytes(b"data")
    run = tmp_path / "run"

    def refuse(*args, **kwargs):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(shutil, "copy2", refuse)

    doc = collect({"images": {"stack": str(source)}}, saved="t")
    save(run, doc, mode="copy")

    record = json.loads((run / DOC_NAME).read_text())["files"][0]
    assert record["copied"] is None
    assert "PermissionError" in record["skipped"]
