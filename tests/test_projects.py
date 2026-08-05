"""``N4`` — what the project browser reports, and what it refuses to claim.

Two things are asserted harder than the rest, because both were wrong in the
first draft and both are wrong in a way a user would act on.

**Stage cannot be "every declared output that exists".** ``ml_analyze``
declares exactly one output — ``measurements/measurements.db`` — which
``mask`` creates and ``measure`` fills. On the naive reading, a project that
has only been segmented reports classical ML as complete.
:func:`spacr.projects.evidence_ports` is the rule that fixes it and
``test_a_bare_measurements_database_is_not_evidence_that_ml_analyze_ran``
is what holds it in place.

**A project the registry has never seen must not read as clean.** It is
listed — that is the whole point — but with staleness *unknown*, because with
no provenance there is nothing to compare against. Zero stale artifacts and
no record of any artifact are different facts and the browser has to say
which one it has.

Everything here is filesystem and registry only: no Qt, no torch, no images.
"""
from __future__ import annotations

import os
import sqlite3
import time

import numpy as np

from spacr import artifacts, ports, projects


# ---------------------------------------------------------------------------
# Projects on disk
# ---------------------------------------------------------------------------

def _plate(root, name="plate1", *, merged=2, db=True, crops=False,
           raw=False, model=False):
    """Build a project folder with exactly the outputs asked for.

    Written against the port declarations rather than against remembered
    folder names: every path here is the one ``spacr.ports`` says the module
    writes, so a declaration that moves breaks this fixture rather than
    silently making the tests test nothing.
    """
    plate = os.path.join(str(root), name)
    os.makedirs(plate, exist_ok=True)
    if merged:
        os.makedirs(os.path.join(plate, "merged"), exist_ok=True)
        for index in range(merged):
            np.save(os.path.join(plate, "merged", f"f{index}.npy"),
                    np.zeros((4, 4, 3), dtype=np.uint16))
    if db:
        os.makedirs(os.path.join(plate, "measurements"), exist_ok=True)
        connection = sqlite3.connect(
            os.path.join(plate, "measurements", "measurements.db"))
        connection.execute("CREATE TABLE IF NOT EXISTS cell (id INTEGER)")
        connection.execute("INSERT INTO cell VALUES (1)")
        connection.commit()
        connection.close()
    if crops:
        folder = os.path.join(plate, "data", "plate1", "well_A01_png")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "0.png"), "wb") as handle:
            handle.write(b"png")
    if model:
        os.makedirs(os.path.join(plate, "model"), exist_ok=True)
        with open(os.path.join(plate, "model", "best.pth"), "wb") as handle:
            handle.write(b"weights")
    if raw:
        with open(os.path.join(plate, f"{name}_A01_T0001F001L01A01Z01C01.tif"),
                  "wb") as handle:
            handle.write(b"tif")
    return plate


# ---------------------------------------------------------------------------
# The pipeline ladder is derived, not written down
# ---------------------------------------------------------------------------

def test_pipeline_order_is_a_topological_sort_of_the_declared_graph():
    """Every module comes after everything that feeds it."""
    order = projects.pipeline_order()
    assert set(order) == set(projects.producing_modules())
    position = {module: index for index, module in enumerate(order)}
    for module in order:
        for upstream in ports.upstream_modules(module):
            if upstream in position:
                assert position[upstream] < position[module], (
                    f"{upstream} produces what {module} consumes but sorts "
                    f"after it")


def test_pipeline_order_is_stable_between_calls():
    """A browser whose ladder reshuffles between scans is unreadable."""
    assert projects.pipeline_order() == projects.pipeline_order()


def test_a_module_registered_later_takes_its_place_without_editing_projects():
    """The ladder is read off :data:`spacr.ports.PORTS`, so a plugin joins it.

    The seam that makes this file's ordering a *derivation* rather than a
    second hand-written list of stage names.
    """
    declaration = ports.ModulePorts(
        key="projects_probe",
        summary="a plugin that post-processes measurements",
        consumes=(ports.Port(ports.MEASUREMENTS_DB, "db",
                             "measurements/measurements.db"),),
        produces=(ports.Port(ports.REGRESSION_RESULTS, "out", "probe_out",
                             "*.csv"),))
    ports.register_module_ports(declaration)
    try:
        order = projects.pipeline_order()
        assert "projects_probe" in order
        assert order.index("measure") < order.index("projects_probe")
    finally:
        ports.PORTS.pop("projects_probe", None)


# ---------------------------------------------------------------------------
# Stage: which declared outputs exist, minus the ones that prove nothing
# ---------------------------------------------------------------------------

def test_a_bare_measurements_database_is_not_evidence_that_ml_analyze_ran(tmp_path):
    """The false positive this rule exists for.

    ``ml_analyze`` declares one output and it is the database ``mask``
    created. Reporting classical ML as complete on a freshly segmented plate
    is the single most misleading thing this screen could say.
    """
    plate = _plate(tmp_path, db=True, merged=1)
    states = {s.module: s for s in projects.module_states(plate)}
    assert states["mask"].state == projects.STATE_DONE
    assert states["ml_analyze"].state == projects.STATE_ABSENT
    # And it says *unknown* rather than *no*: there is no on-disk signature
    # for this module at all, which is a different claim from "did not run".
    assert states["ml_analyze"].detectable is False
    assert "only it could have written" in states["ml_analyze"].describe()


def test_evidence_ports_drop_anything_an_earlier_module_already_explains():
    """A port at a location an earlier module writes with a different kind."""
    shared = "measurements/measurements.db"
    assert any(port.path == shared
               for port in ports.module_ports("ml_analyze").produces)
    assert projects.evidence_ports("ml_analyze") == ()
    # mask is the first writer of that file, so for *it* the database counts.
    assert {port.role for port in projects.evidence_ports("mask")} >= {
        "merged", "counts"}
    # classify keeps the one output nothing else writes.
    assert {port.role for port in projects.evidence_ports("classify")} == {
        "model"}


def test_interchangeable_modules_do_not_both_claim_one_segmentation():
    """mask and timelapse declare the identical outputs; only one may claim.

    ``ports.py`` literally hands both the same tuple. Left alone, one
    segmented project would report that two different modules had run.
    """
    assert (ports.module_ports("timelapse").produces
            == ports.module_ports("mask").produces)
    assert projects.evidence_ports("timelapse") == ()


def test_stage_is_the_furthest_module_that_left_something_behind(tmp_path):
    plate = _plate(tmp_path, merged=2, db=True, crops=True, model=True)
    summary = projects.scan(plate, with_next_steps=False)
    assert summary.stage == "classify"
    assert "mask" in summary.ran and "measure" in summary.ran


def test_a_folder_of_raw_images_is_a_project_with_nothing_run(tmp_path):
    """The plate that was copied in this morning."""
    plate = _plate(tmp_path, name="fresh", merged=0, db=False, raw=True)
    assert projects.looks_like_project(plate)
    summary = projects.scan(plate)
    assert summary.stage == ""
    assert summary.stage_label == "nothing run"
    assert summary.ran == ()


def test_a_partial_run_is_reported_as_partial_not_as_done(tmp_path):
    """Required outputs missing, optional ones not: only the first counts."""
    plate = _plate(tmp_path, merged=0, db=True)
    os.makedirs(os.path.join(plate, "masks"), exist_ok=True)
    with open(os.path.join(plate, "masks", "a.tif"), "wb") as handle:
        handle.write(b"m")
    states = {s.module: s for s in projects.module_states(plate)}
    assert states["mask"].state == projects.STATE_PARTIAL
    assert "merged" in states["mask"].missing
    assert "masks" in states["mask"].found


def test_a_cleaned_up_optional_output_does_not_make_a_project_look_broken(tmp_path):
    """``masks/`` is removed by cleanup and its port says it may be absent."""
    plate = _plate(tmp_path, merged=2, db=True)
    states = {s.module: s for s in projects.module_states(plate)}
    assert states["mask"].state == projects.STATE_DONE
    assert states["mask"].missing == ()
    assert "masks" in states["mask"].optional_missing


# ---------------------------------------------------------------------------
# Finding projects
# ---------------------------------------------------------------------------

def test_discover_stops_at_a_project_and_does_not_list_its_merged_folder(tmp_path):
    """A project's own subfolders are not projects."""
    plate = _plate(tmp_path, merged=2, db=True)
    found = projects.discover([tmp_path])
    assert found == (plate,)
    assert not any(f.endswith("merged") for f in found)


def test_discover_finds_projects_one_and_two_levels_down(tmp_path):
    nested = tmp_path / "experiment_a"
    nested.mkdir()
    plate = _plate(nested, name="plate9", merged=1, db=True)
    assert projects.discover([tmp_path], depth=2) == (plate,)
    assert projects.discover([tmp_path], depth=1) == ()


def test_discover_honours_its_limit(tmp_path):
    for index in range(5):
        _plate(tmp_path, name=f"p{index}", merged=1, db=True)
    assert len(projects.discover([tmp_path], limit=3)) == 3


def test_discover_ignores_an_unreadable_folder_rather_than_raising(tmp_path):
    """A browser must survive a stale mount, not refuse to draw a table."""
    plate = _plate(tmp_path, merged=1, db=True)
    blocked = tmp_path / "locked"
    blocked.mkdir()
    os.chmod(blocked, 0o000)
    try:
        assert plate in projects.discover([tmp_path])
    finally:
        os.chmod(blocked, 0o755)


def test_a_folder_that_is_not_a_project_summarises_without_raising(tmp_path):
    """"There is nothing here" is an answer a browser has to be able to show."""
    empty = tmp_path / "not_a_project"
    empty.mkdir()
    assert projects.looks_like_project(empty) is False
    summary = projects.scan(empty)
    assert summary.exists and summary.stage == "" and summary.size_bytes == 0


def test_a_root_that_is_gone_is_reported_rather_than_raised(tmp_path):
    summary = projects.scan(tmp_path / "never_existed")
    assert summary.exists is False
    assert "gone" in summary.note()
    assert "the folder is gone" in projects.format_project(summary)


# ---------------------------------------------------------------------------
# The project the registry has never seen — the case N4 exists for
# ---------------------------------------------------------------------------

def test_an_unregistered_project_is_listed_with_everything_disk_can_answer(tmp_path):
    plate = _plate(tmp_path, merged=3, db=True)
    summary = projects.scan(plate)
    assert summary.known is False
    assert summary.has_registry is False
    assert summary.stage == "mask"
    assert summary.size_bytes > 0 and summary.n_files >= 4
    # Its bytes are all unaccounted for, which is `scan_project`'s own answer
    # rather than a second opinion computed here.
    assert summary.unregistered_bytes == summary.size_bytes


def test_an_unregistered_project_reports_staleness_as_unknown_not_as_clean(tmp_path):
    """Zero stale and no record of anything are different facts."""
    plate = _plate(tmp_path, merged=2, db=True)
    summary = projects.scan(plate)
    assert summary.n_stale == 0
    assert summary.staleness_known is False
    assert summary.staleness_note() == "unknown — nothing recorded"
    assert "not in the registry" in summary.note()
    assert "no record of this project" in projects.format_project(summary)


def test_an_unregistered_project_dates_itself_from_the_filesystem(tmp_path):
    """A weaker claim, reported as a weaker claim."""
    plate = _plate(tmp_path, merged=1, db=True)
    summary = projects.scan(plate)
    assert summary.last_run_source == projects.SOURCE_FILESYSTEM
    assert summary.last_run_utc
    assert "from the filesystem" in projects.format_project(summary)


def test_the_registry_answers_when_it_has_one_and_says_so(tmp_path):
    plate = _plate(tmp_path, merged=2, db=True)
    registry = artifacts.open_registry(plate)
    registry.register(project=plate, kind=ports.MERGED_ARRAYS, role="merged",
                      path=os.path.join(plate, "merged"), module="mask",
                      settings={"src": plate})
    summary = projects.scan(plate)
    assert summary.known is True and summary.has_registry is True
    assert summary.n_artifacts == 1
    assert summary.last_run_source == projects.SOURCE_REGISTRY
    assert summary.staleness_note() == "current"


def test_a_run_record_settles_a_stage_the_filesystem_cannot(tmp_path):
    """measure writes the database mask already made; the registry saw it run.

    This is the other half of :func:`spacr.projects.evidence_ports`: the rule
    that stops a false positive would leave a real ``measure`` invisible if
    the registry were not consulted.
    """
    plate = _plate(tmp_path, merged=2, db=True, crops=False)
    assert projects.scan(plate, with_next_steps=False).stage == "mask"
    registry = artifacts.open_registry(plate)
    registry.register(
        project=plate, kind=ports.MEASUREMENTS_DB, role="db",
        path=os.path.join(plate, "measurements", "measurements.db"),
        module="measure", settings={"src": plate})
    summary = projects.scan(plate, with_next_steps=False)
    assert summary.stage == "measure"
    state = {s.module: s for s in summary.modules}["measure"]
    assert state.evidence == projects.SOURCE_REGISTRY
    assert "from the run record" in state.describe()


def test_a_recorded_artifact_whose_file_is_gone_reports_partial(tmp_path):
    plate = _plate(tmp_path, merged=2, db=True)
    registry = artifacts.open_registry(plate)
    crops = os.path.join(plate, "data")
    os.makedirs(crops)
    registry.register(project=plate, kind=ports.CROPS, role="crops",
                      path=crops, module="measure", settings={"src": plate})
    os.rmdir(crops)
    summary = projects.scan(plate, with_next_steps=False)
    state = {s.module: s for s in summary.modules}["measure"]
    assert state.state == projects.STATE_PARTIAL
    assert summary.missing and summary.missing[0].missing is True
    assert "gone" in summary.staleness_note()
    assert "no longer on disk" in summary.note()


# ---------------------------------------------------------------------------
# What is stale
# ---------------------------------------------------------------------------

def test_a_result_whose_input_was_re_produced_is_reported_stale(tmp_path):
    """The registry's verdict, with its own sentence, not a rederived one."""
    plate = _plate(tmp_path, merged=2, db=True)
    registry = artifacts.open_registry(plate)
    merged = registry.register(
        project=plate, kind=ports.MERGED_ARRAYS, role="merged",
        path=os.path.join(plate, "merged"), module="mask",
        settings={"src": plate, "cell_diameter": 30})
    time.sleep(0.01)
    registry.register(
        project=plate, kind=ports.MEASUREMENTS_DB, role="db",
        path=os.path.join(plate, "measurements", "measurements.db"),
        module="measure", settings={"src": plate},
        inputs=[merged.artifact_id])
    time.sleep(0.01)
    np.save(os.path.join(plate, "merged", "later.npy"),
            np.zeros((4, 4, 3), dtype=np.uint16))
    registry.register(
        project=plate, kind=ports.MERGED_ARRAYS, role="merged",
        path=os.path.join(plate, "merged"), module="mask",
        settings={"src": plate, "cell_diameter": 45})

    summary = projects.scan(plate, with_next_steps=False)
    assert summary.staleness_known is True
    assert summary.n_stale == 1
    entry = summary.stale[0]
    assert entry.kind == ports.MEASUREMENTS_DB and entry.module == "measure"
    assert entry.causes and entry.reasons
    # The reasons are the registry's, verbatim — they name the path that moved.
    assert os.path.join(plate, "merged") in " ".join(entry.reasons)
    assert entry.explain()
    assert "out of date" in summary.note()


def test_missing_and_stale_are_kept_apart(tmp_path):
    """An availability problem is not a provenance one, and vice versa."""
    plate = _plate(tmp_path, merged=1, db=True)
    registry = artifacts.open_registry(plate)
    gone = os.path.join(plate, "results")
    os.makedirs(gone)
    with open(os.path.join(gone, "results_a.csv"), "w",
              encoding="utf-8") as handle:
        handle.write("a\n")
    registry.register(project=plate, kind=ports.REGRESSION_RESULTS,
                      role="results", path=gone, module="regression",
                      settings={"src": plate})
    for name in os.listdir(gone):
        os.remove(os.path.join(gone, name))
    os.rmdir(gone)
    summary = projects.scan(plate, with_next_steps=False)
    assert [entry.missing for entry in summary.missing] == [True]
    assert summary.n_stale == 0


# ---------------------------------------------------------------------------
# What could run next
# ---------------------------------------------------------------------------

def test_next_steps_are_the_offer_the_modules_own_screen_makes(tmp_path):
    plate = _plate(tmp_path, merged=2, db=True)
    summary = projects.scan(plate)
    offered = {module for module, _blocked in summary.next_steps}
    assert "measure" in offered
    ready = {module for module, blocked in summary.next_steps if not blocked}
    assert "measure" in ready


def test_next_steps_can_be_switched_off(tmp_path):
    plate = _plate(tmp_path, merged=1, db=True)
    assert projects.scan(plate, with_next_steps=False).next_steps == ()


# ---------------------------------------------------------------------------
# browse(), and the report
# ---------------------------------------------------------------------------

def test_browse_lists_every_project_most_recently_run_first(tmp_path):
    old = _plate(tmp_path, name="old_plate", merged=1, db=True)
    time.sleep(0.02)
    new = _plate(tmp_path, name="new_plate", merged=1, db=True)
    summaries = projects.browse([tmp_path], with_next_steps=False)
    assert [s.root for s in summaries] == [new, old]


def test_browse_reports_progress_for_a_caller_that_wants_it(tmp_path):
    _plate(tmp_path, name="a", merged=1, db=True)
    _plate(tmp_path, name="b", merged=1, db=True)
    seen = []
    projects.browse([tmp_path], with_next_steps=False,
                    on_progress=lambda done, total, root: seen.append(
                        (done, total)))
    assert seen == [(1, 2), (2, 2)]


def test_a_progress_callback_that_raises_does_not_lose_the_scan(tmp_path):
    _plate(tmp_path, name="a", merged=1, db=True)

    def _boom(done, total, root):
        raise RuntimeError("the caller's bug")

    assert len(projects.browse([tmp_path], with_next_steps=False,
                               on_progress=_boom)) == 1


def test_browse_reuses_scan_project_rather_than_walking_twice(tmp_path, monkeypatch):
    """The one walk rule, asserted rather than trusted.

    A second walker inside this module would be invisible in every other test
    here — the numbers would still be right — and would double the cost of
    the screen on the projects where cost matters.
    """
    from spacr import data_manager

    calls = []
    original = data_manager.scan_project

    def _counted(root, **kwargs):
        calls.append(str(root))
        return original(root, **kwargs)

    monkeypatch.setattr(data_manager, "scan_project", _counted)
    plate = _plate(tmp_path, merged=2, db=True)
    projects.browse([tmp_path], with_next_steps=False)
    assert calls == [plate]


def test_a_usage_already_taken_is_reused_rather_than_re_walked(tmp_path, monkeypatch):
    from spacr import data_manager

    plate = _plate(tmp_path, merged=1, db=True)
    usage = data_manager.scan_project(plate)
    monkeypatch.setattr(data_manager, "scan_project", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("scan_project was called with a usage in hand")))
    summary = projects.scan(plate, usage=usage, with_next_steps=False)
    assert summary.size_bytes == usage.total_bytes
    assert summary.usage is usage


def test_format_projects_renders_a_table_with_a_row_per_project(tmp_path):
    _plate(tmp_path, name="alpha", merged=1, db=True)
    _plate(tmp_path, name="beta", merged=0, db=False, raw=True)
    text = projects.format_projects(
        projects.browse([tmp_path], with_next_steps=False))
    assert "Project" in text and "Stage" in text and "Last run" in text
    assert "alpha" in text and "beta" in text
    assert len(text.splitlines()) == 3


def test_format_projects_says_so_when_there_is_nothing(tmp_path):
    assert projects.format_projects(()) == "No projects found."


def test_the_summary_str_is_the_report(tmp_path):
    plate = _plate(tmp_path, merged=1, db=True)
    summary = projects.scan(plate, with_next_steps=False)
    assert str(summary) == projects.format_project(summary)
    assert summary.describe() == str(summary)


def test_a_project_whose_bytes_are_mostly_unaccounted_for_says_so(tmp_path):
    """The Data Manager's category, surfaced where a user first meets it."""
    plate = _plate(tmp_path, merged=1, db=True)
    registry = artifacts.open_registry(plate)
    registry.register(project=plate, kind=ports.MEASUREMENTS_DB, role="db",
                      path=os.path.join(plate, "measurements",
                                        "measurements.db"),
                      module="measure", settings={"src": plate})
    with open(os.path.join(plate, "notes.txt"), "wb") as handle:
        handle.write(b"x" * 500_000)
    summary = projects.scan(plate, with_next_steps=False)
    assert summary.unregistered_bytes > summary.size_bytes // 2
    assert "unaccounted for" in summary.note()


# ---------------------------------------------------------------------------
# The module is importable without a GUI or a scientific stack
# ---------------------------------------------------------------------------

def test_importing_spacr_projects_costs_no_gui_and_no_torch():
    """The browser screen renders this; a notebook and a cron job use it too."""
    import subprocess
    import sys

    code = (
        "import sys, spacr.projects;"
        "bad=[m for m in ('torch','PySide6','tkinter','matplotlib.pyplot',"
        "'spacr.utils') if m in sys.modules];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]", out.stdout


def test_it_is_reachable_as_a_lazy_submodule_of_the_package():
    import spacr

    assert "projects" in spacr._SUBMODULES
    assert spacr.projects is projects
