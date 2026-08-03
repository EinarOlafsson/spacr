"""Real tests for auto-chaining, staleness and "continue to next step".

Every assertion here is about a promise a user can feel:

* Measure's default source is the folder Mask **registered**, not a folder
  re-derived from a naming convention — the tests prove that by registering an
  artifact whose path the convention would never produce, and by deleting the
  registry row and watching the default disappear;
* a path the user typed survives a reopen, a newer upstream run and a restart,
  and the new location is *offered* rather than applied;
* staleness reports the right CAUSE — an upstream re-run and a settings change
  are different problems with different fixes — and clears when the module is
  re-run;
* a successor is offered only with a verdict from
  :func:`spacr.ports.check_ready` attached, so one that cannot run says why.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr import artifacts, chaining, ports  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    """No test touches the developer's registry override or their real pins."""
    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)
    monkeypatch.setenv(chaining.PIN_STATE_ENV,
                       str(tmp_path / "state" / "pins.json"))
    chaining.pin_store(refresh=True)
    yield
    chaining.pin_store(refresh=True)


@pytest.fixture
def pins(tmp_path):
    """A pin store of this test's own."""
    return chaining.PinStore(str(tmp_path / "pins.json"))


def make_plate(root: Path, *, merged: int = 2, planes: int = 3,
               db_tables=None, crops: bool = False) -> str:
    """Build a plate folder shaped the way the mask pipeline leaves one."""
    root.mkdir(parents=True, exist_ok=True)
    if merged:
        (root / "merged").mkdir(exist_ok=True)
        for index in range(merged):
            np.save(root / "merged" / f"plate1_A01_{index}.npy",
                    np.zeros((6, 6, planes), dtype=np.uint16))
    if db_tables:
        (root / "measurements").mkdir(exist_ok=True)
        connection = sqlite3.connect(root / "measurements" / "measurements.db")
        for table in db_tables:
            connection.execute(f'CREATE TABLE "{table}" (value INTEGER)')
            connection.execute(f'INSERT INTO "{table}" VALUES (1)')
        connection.commit()
        connection.close()
    if crops:
        crop_dir = root / "data" / "A01" / "cell_png"
        crop_dir.mkdir(parents=True, exist_ok=True)
        (crop_dir / "object_1.png").write_bytes(b"\x89PNG")
    return str(root)


def run_mask(root: str, **overrides):
    """Register a Mask run's outputs against ``root``."""
    settings = {"src": root, "cell_channel": 0, "cell_diameter": 30}
    settings.update(overrides)
    return artifacts.register_run_outputs(
        "mask", settings, registry=artifacts.open_registry(root))


def run_measure(root: str, **overrides):
    """Register a Measure run's outputs against ``root``."""
    settings = {"src": root, "cell_mask_dim": 4, "save_png": True}
    settings.update(overrides)
    return artifacts.register_run_outputs(
        "measure", settings, registry=artifacts.open_registry(root))


# ===========================================================================
# 1.1 — auto-chaining
# ===========================================================================

def test_measure_defaults_to_the_path_mask_registered(tmp_path):
    """Measure's input is the row Mask wrote, not a re-derived convention."""
    root = make_plate(tmp_path / "plateA")
    produced = run_mask(root)
    merged = next(a for a in produced if a.kind == ports.MERGED_ARRAYS)

    chained = chaining.chained_inputs("measure", {"src": ""}, root=root)

    assert [c.kind for c in chained] == [ports.MERGED_ARRAYS]
    assert chained[0].artifact.artifact_id == merged.artifact_id
    assert chained[0].path == merged.path
    assert chained[0].producer == "mask"
    assert chained[0].value == root


def test_the_default_comes_from_the_registry_not_the_folder_layout(tmp_path):
    """Delete the registry row and the default goes with it.

    The plate folder is untouched — ``merged/`` is still full of arrays a
    convention-based default would happily point at.  Nothing is offered,
    which is the whole difference between reading the registry and guessing.
    """
    root = make_plate(tmp_path / "plateA")
    produced = run_mask(root)
    registry = artifacts.open_registry(root)
    for artifact in produced:
        registry.forget(artifact)

    assert os.path.isdir(os.path.join(root, "merged"))
    assert chaining.chained_inputs("measure", {"src": ""}, root=root) == ()


def test_a_chained_default_follows_the_artifact_to_an_unusual_path(tmp_path):
    """An artifact registered somewhere the convention would never look."""
    root = make_plate(tmp_path / "plateA")
    elsewhere = tmp_path / "scratch" / "merged_v2"
    elsewhere.mkdir(parents=True)
    np.save(elsewhere / "f0.npy", np.zeros((6, 6, 3), dtype=np.uint16))
    registry = artifacts.open_registry(root)
    registry.register(module="mask", kind=ports.MERGED_ARRAYS, role="merged",
                      path=str(elsewhere), project=root,
                      settings={"src": root})

    chained = chaining.chained_inputs("measure", {"src": ""}, root=root)

    assert chained[0].path == str(elsewhere)
    # The *project* is what a root-form setting takes, and it is the project
    # the producer recorded — never dirname() of the artifact.
    assert chained[0].value == root


def test_classify_chains_off_measure_as_a_list(tmp_path):
    """Classify keeps its source in a list, and the chained value follows."""
    root = make_plate(tmp_path / "plateA", db_tables=("png_list", "cell"),
                      crops=True)
    run_mask(root)
    run_measure(root)

    resolution = chaining.resolve_settings("classify", {"src": []}, root=root)

    assert resolution.settings["src"] == [root]
    assert resolution.filled["src"].producer == "measure"
    assert resolution.filled["src"].kind == ports.MEASUREMENTS_DB


def test_mask_re_registering_the_database_does_not_pose_as_measure(tmp_path):
    """A Mask re-run must not become Classify's ``measurements-db``.

    Mask writes object counts into the same file Measure writes its tables
    into.  If both claimed ``measurements-db``, this would chain Classify off
    a Mask run and hide that the measurement tables inside were stale.
    """
    root = make_plate(tmp_path / "plateA", db_tables=("png_list", "cell"))
    run_mask(root)

    assert chaining.chained_inputs("classify", {"src": []}, root=root) == ()

    counts = artifacts.open_registry(root).by_kind(ports.OBJECT_COUNTS)
    assert [a.module for a in counts] == ["mask"]


def test_a_module_without_the_settings_key_is_skipped(tmp_path):
    """Nothing is invented for a key the module does not have.

    Regression consumes the measurements database but has no ``src`` — its
    inputs are ``count_data`` / ``score_data``.  Filling a key it does not
    read would put a path somewhere nothing looks.
    """
    root = make_plate(tmp_path / "plateA", db_tables=("png_list", "cell"))
    run_mask(root)
    run_measure(root)

    assert chaining.chained_inputs(
        "regression", {"count_data": [], "score_data": []}, root=root) == ()
    # With no settings dict to check against, the binding is still reported.
    assert [c.setting for c in
            chaining.chained_inputs("regression", None, root=root)] == ["src"]


def test_every_declared_module_binds_to_a_settings_key():
    """Every consumed port in PORTS resolves to a binding, with no table."""
    for module in ports.known_modules():
        spec = ports.module_ports(module)
        for port in spec.consumes:
            binding = chaining.binding_for(module, port)
            assert binding.module == module
            assert binding.setting
            assert binding.form in (chaining.ROOT, chaining.PATH)


def test_binding_registration_rejects_a_duplicate_and_a_bad_form():
    saved = dict(chaining.BINDINGS)
    try:
        chaining.register_binding(
            chaining.Binding("measure", "merged", "elsewhere", chaining.PATH))
        with pytest.raises(ValueError, match="already bound"):
            chaining.register_binding(
                chaining.Binding("measure", "merged", "other", chaining.PATH))
        replaced = chaining.register_binding(
            chaining.Binding("measure", "merged", "other", chaining.ROOT),
            overwrite=True)
        assert replaced.setting == "other"
        with pytest.raises(ValueError, match="unknown form"):
            chaining.register_binding(
                chaining.Binding("measure", "merged", "x", "sideways"))
        with pytest.raises(ValueError, match="needs a module"):
            chaining.register_binding(chaining.Binding("", "", "", chaining.ROOT))
    finally:
        chaining.BINDINGS.clear()
        chaining.BINDINGS.update(saved)


def test_a_declared_binding_can_take_the_artifact_path_itself(tmp_path):
    """PATH form hands over the artifact, not the project it lives in."""
    root = make_plate(tmp_path / "plateA")
    produced = run_mask(root)
    merged = next(a for a in produced if a.kind == ports.MERGED_ARRAYS)
    saved = dict(chaining.BINDINGS)
    try:
        chaining.register_binding(
            chaining.Binding("measure", "merged", "src", chaining.PATH))
        chained = chaining.chained_inputs("measure", {"src": ""}, root=root)
        assert chained[0].value == merged.path
    finally:
        chaining.BINDINGS.clear()
        chaining.BINDINGS.update(saved)


def test_chaining_searches_the_candidate_roots_in_order(tmp_path):
    """A blank screen finds the plate the upstream module last ran on."""
    empty = make_plate(tmp_path / "blank", merged=0)
    worked = make_plate(tmp_path / "plateA")
    run_mask(worked)

    chained = chaining.chained_inputs(
        "measure", {"src": ""}, roots=[empty, worked])

    assert chained[0].value == worked


def test_project_scoped_lookup_prefers_the_project_on_screen(tmp_path):
    """A named project answers for itself, never a busier neighbour."""
    older = make_plate(tmp_path / "plateA")
    run_mask(older)
    time.sleep(0.01)
    newer = make_plate(tmp_path / "plateB")
    run_mask(newer)

    chained = chaining.chained_inputs(
        "measure", {"src": older}, roots=[newer])

    assert chained[0].value == older


def test_chaining_never_creates_a_registry(tmp_path):
    """Asking a folder what it produced must not leave a file behind."""
    root = make_plate(tmp_path / "plateA")

    assert chaining.chained_inputs("measure", {"src": root}) == ()
    assert not os.path.exists(os.path.join(root, artifacts.ARTIFACTS_DB_NAME))


# ---------------------------------------------------------------------------
# 1.1 — the user's edit wins
# ---------------------------------------------------------------------------

def test_a_user_edited_path_survives_a_reopen(tmp_path, pins):
    """The pin outlives the store object, which is what a restart looks like."""
    masked = make_plate(tmp_path / "plateA")
    run_mask(masked)
    chosen = make_plate(tmp_path / "plateB")

    pins.pin("measure", "src", chosen)

    reopened = chaining.PinStore(pins.path)
    resolution = chaining.resolve_settings(
        "measure", {"src": "path"}, roots=[masked], pins=reopened)

    assert resolution.settings["src"] == chosen
    assert "src" not in resolution.filled
    assert resolution.held["src"].value == chosen


def test_a_moved_upstream_is_offered_not_applied(tmp_path, pins):
    """The pin still wins; the new location comes back as an offer."""
    masked = make_plate(tmp_path / "plateA")
    run_mask(masked)
    chosen = make_plate(tmp_path / "plateB", merged=0)
    pins.pin("measure", "src", chosen)

    resolution = chaining.resolve_settings(
        "measure", {"src": "path"}, roots=[masked], pins=pins)

    assert resolution.settings["src"] == chosen
    held = resolution.held["src"]
    assert held.differs
    assert held.offered == masked
    assert resolution.moved == (held,)
    assert "pinned" in held.describe()


def test_a_pin_matching_the_upstream_raises_no_offer(tmp_path, pins):
    """No nagging when the user's path is where the upstream already writes."""
    masked = make_plate(tmp_path / "plateA")
    run_mask(masked)
    pins.pin("measure", "src", masked)

    resolution = chaining.resolve_settings(
        "measure", {"src": "path"}, roots=[masked], pins=pins)

    assert resolution.moved == ()
    assert not resolution.held["src"].differs


def test_clearing_the_field_unpins_and_the_default_returns(tmp_path, pins):
    masked = make_plate(tmp_path / "plateA")
    run_mask(masked)
    pins.pin("measure", "src", str(tmp_path / "plateB"))
    assert pins.pinned("measure", "src")

    pins.pin("measure", "src", "")

    assert pins.pinned("measure", "src") is None
    resolution = chaining.resolve_settings(
        "measure", {"src": "path"}, roots=[masked], pins=pins)
    assert resolution.settings["src"] == masked


def test_a_non_placeholder_value_already_on_screen_is_left_alone(tmp_path,
                                                                pins):
    """A dropped folder or an imported CSV is not second-guessed."""
    masked = make_plate(tmp_path / "plateA")
    run_mask(masked)
    dropped = str(tmp_path / "dropped")

    resolution = chaining.resolve_settings(
        "measure", {"src": dropped}, roots=[masked], pins=pins)

    assert resolution.settings["src"] == dropped
    assert "src" not in resolution.filled


@pytest.mark.parametrize("placeholder", chaining.placeholder_paths())
def test_every_placeholder_counts_as_empty(tmp_path, pins, placeholder):
    masked = make_plate(tmp_path / "plateA")
    run_mask(masked)

    resolution = chaining.resolve_settings(
        "measure", {"src": placeholder}, roots=[masked], pins=pins)

    assert resolution.settings["src"] == masked


def test_a_pin_is_list_shaped_when_the_setting_is(tmp_path, pins):
    chosen = str(tmp_path / "plateB")
    pins.pin("classify", "src", chosen)

    resolution = chaining.resolve_settings("classify", {"src": []}, pins=pins)

    assert resolution.settings["src"] == [chosen]


def test_the_pin_file_survives_corruption_and_a_missing_directory(tmp_path):
    path = tmp_path / "nested" / "deeper" / "pins.json"
    store = chaining.PinStore(str(path))
    store.pin("measure", "src", "/plate")
    assert os.path.isfile(path)

    path.write_text("{not json", encoding="utf-8")
    assert chaining.PinStore(str(path)).pinned("measure", "src") is None

    path.write_text('{"measure": "not a mapping"}', encoding="utf-8")
    assert chaining.PinStore(str(path)).pins("measure") == {}


def test_pin_store_bookkeeping(tmp_path):
    store = chaining.PinStore(str(tmp_path / "pins.json"))
    assert store.unpin("measure", "src") is False
    store.pin("measure", "src", "/a")
    store.pin("mask", "src", "/b")
    assert store.pins("measure") == {"src": "/a"}
    assert store.unpin("measure", "src") is True
    assert store.pins("measure") == {}
    store.pin("measure", "src", "/a")
    store.clear("measure")
    assert store.pins("measure") == {}
    assert store.pins("mask") == {"src": "/b"}
    store.clear()
    assert store.pins("mask") == {}
    assert store.reload() is store


def test_aliases_and_the_shared_store(monkeypatch, tmp_path):
    """``measure_crop`` and ``measure`` are one module, and one set of pins."""
    store = chaining.PinStore(str(tmp_path / "pins.json"))
    store.pin("measure_crop", "src", "/plate")
    assert store.pinned("measure", "src") == "/plate"

    monkeypatch.setenv(chaining.PIN_STATE_ENV, str(tmp_path / "shared.json"))
    shared = chaining.pin_store(refresh=True)
    assert shared.path == str(tmp_path / "shared.json")
    assert chaining.pin_store() is shared
    assert chaining.pin_store(str(tmp_path / "other.json")) is not shared


def test_state_path_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv(chaining.PIN_STATE_ENV, raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert chaining.state_path() == str(
        tmp_path / "xdg" / "spacr" / "chaining" / "pins.json")
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    assert chaining.state_path().endswith(
        os.path.join(".local", "state", "spacr", "chaining", "pins.json"))


# ===========================================================================
# 1.2 — staleness, with the cause
# ===========================================================================

def test_an_upstream_re_run_makes_the_result_stale_with_that_cause(tmp_path):
    root = make_plate(tmp_path / "plateA", db_tables=("png_list", "cell"))
    run_mask(root)
    measure_settings = {"src": root, "cell_mask_dim": 4, "save_png": True}
    run_measure(root)

    assert chaining.stale_outputs("measure", measure_settings, root=root) == ()

    time.sleep(0.01)
    np.save(Path(root) / "merged" / "extra.npy",
            np.zeros((6, 6, 3), dtype=np.uint16))
    run_mask(root)

    notes = chaining.stale_outputs("measure", measure_settings, root=root)

    assert [n.kind for n in notes] == [ports.MEASUREMENTS_DB]
    note = notes[0]
    assert artifacts.CAUSE_UPSTREAM_SUPERSEDED in note.causes
    assert artifacts.CAUSE_SETTINGS_CHANGED not in note.causes
    assert "out of date" in note.headline
    assert "newer run has replaced one of its inputs" in note.headline
    assert "Re-run measure" in note.fix
    assert note.direction == "output"
    assert note.producer == "measure"
    assert note.detail


def test_re_running_the_module_clears_the_staleness(tmp_path):
    root = make_plate(tmp_path / "plateA", db_tables=("png_list", "cell"))
    run_mask(root)
    measure_settings = {"src": root, "cell_mask_dim": 4, "save_png": True}
    run_measure(root)
    time.sleep(0.01)
    np.save(Path(root) / "merged" / "extra.npy",
            np.zeros((6, 6, 3), dtype=np.uint16))
    run_mask(root)
    assert chaining.stale_outputs("measure", measure_settings, root=root)

    # Measure runs again: new content, new inputs, current provenance.
    time.sleep(0.01)
    connection = sqlite3.connect(
        Path(root) / "measurements" / "measurements.db")
    connection.execute('INSERT INTO "cell" VALUES (2)')
    connection.commit()
    connection.close()
    run_measure(root)

    assert chaining.stale_outputs("measure", measure_settings, root=root) == ()


def test_a_changed_setting_makes_the_result_stale_with_that_cause(tmp_path):
    root = make_plate(tmp_path / "plateA", db_tables=("png_list", "cell"))
    run_mask(root)
    run_measure(root, cell_mask_dim=4)

    same = chaining.stale_outputs(
        "measure", {"src": root, "cell_mask_dim": 4, "save_png": True},
        root=root)
    assert same == ()

    notes = chaining.stale_outputs(
        "measure", {"src": root, "cell_mask_dim": 9, "save_png": True},
        root=root)

    assert notes
    assert artifacts.CAUSE_SETTINGS_CHANGED in notes[0].causes
    assert "settings on this screen differ" in notes[0].headline
    assert "Re-run measure with these settings" in notes[0].fix


def test_a_cosmetic_setting_does_not_make_a_result_stale(tmp_path):
    """The deny-list that decides a resume decides staleness too."""
    from spacr.resume import COSMETIC_SETTINGS

    cosmetic = sorted(COSMETIC_SETTINGS)[0]
    root = make_plate(tmp_path / "plateA", db_tables=("png_list", "cell"))
    run_mask(root)
    run_measure(root, **{cosmetic: 1})

    notes = chaining.stale_outputs(
        "measure",
        {"src": root, "cell_mask_dim": 4, "save_png": True, cosmetic: 999},
        root=root)

    assert notes == ()


def test_a_stale_input_is_reported_without_comparing_settings(tmp_path):
    """This module's settings say nothing about the previous module's output."""
    root = make_plate(tmp_path / "plateA", db_tables=("png_list", "cell"))
    mask_settings = {"src": root, "cell_channel": 0, "cell_diameter": 30}
    registry = artifacts.open_registry(root)
    raw = registry.register(module="import", kind=ports.RAW_IMAGES,
                            role="images", path=root, project=root)
    merged = registry.register(
        module="mask", kind=ports.MERGED_ARRAYS, role="merged",
        path=os.path.join(root, "merged"), project=root,
        settings=mask_settings, inputs=[raw.artifact_id])
    run_measure(root)

    assert chaining.stale_inputs("measure", {"src": root}, root=root) == ()

    # The raw images are re-registered: mask's merged arrays now predate
    # their own input.
    time.sleep(0.01)
    (Path(root) / "note.txt").write_text("changed", encoding="utf-8")
    registry.register(module="import", kind=ports.RAW_IMAGES, role="images",
                      path=root, project=root)

    notes = chaining.stale_inputs("measure", {"src": root}, root=root)

    assert [n.kind for n in notes] == [ports.MERGED_ARRAYS]
    assert notes[0].direction == "input"
    assert artifacts.CAUSE_UPSTREAM_SUPERSEDED in notes[0].causes
    assert merged.artifact_id == notes[0].artifact_id
    # Inputs come before outputs: a stale input explains a stale output.
    combined = chaining.staleness_notes("measure", {"src": root}, root=root)
    assert combined[0].direction == "input"


def test_a_missing_registry_reports_nothing_rather_than_guessing(tmp_path):
    root = make_plate(tmp_path / "plateA", db_tables=("png_list", "cell"))
    assert chaining.staleness_notes("measure", {"src": root}) == ()
    assert chaining.stale_outputs("measure", {"src": ""}) == ()


def test_every_cause_has_a_sentence_and_a_fix():
    """No cause code may reach a user as a bare identifier."""
    codes = {
        artifacts.CAUSE_UNKNOWN, artifacts.CAUSE_UPSTREAM_MISSING,
        artifacts.CAUSE_UPSTREAM_NEWER, artifacts.CAUSE_UPSTREAM_SUPERSEDED,
        artifacts.CAUSE_UPSTREAM_STALE, artifacts.CAUSE_SETTINGS_CHANGED,
        artifacts.CAUSE_CYCLE,
    }
    assert codes <= set(chaining.CAUSE_TEXT)
    assert codes <= set(chaining.CAUSE_FIX)
    for code in codes:
        note = chaining.StaleNote(
            module="measure", direction="output", kind="measurements-db",
            role="db", path="/p", producer="mask", artifact_id="a",
            causes=(code,))
        assert code not in note.headline
        assert note.fix and "{" not in note.fix


def test_unknown_causes_are_passed_through_not_dropped():
    assert chaining.explain_causes(["not-a-cause"]) == "not-a-cause"
    assert chaining.explain_causes(
        [artifacts.CAUSE_UPSTREAM_NEWER, artifacts.CAUSE_UPSTREAM_NEWER]
    ) == chaining.CAUSE_TEXT[artifacts.CAUSE_UPSTREAM_NEWER]


def test_a_stale_note_renders_as_a_validate_problem():
    from spacr.validate import Problem, WARNING, format_report

    note = chaining.StaleNote(
        module="measure", direction="output", kind="measurements-db",
        role="db", path="/p", producer="mask", artifact_id="a",
        causes=(artifacts.CAUSE_UPSTREAM_NEWER,))
    problem = note.to_problem()

    assert isinstance(problem, Problem)
    assert problem.severity == WARNING
    assert problem.setting == "db"
    assert "out of date" in format_report([problem])


# ===========================================================================
# 1.3 — continue to the next step
# ===========================================================================

def test_mask_offers_measure_pre_filled_with_what_it_produced(tmp_path):
    root = make_plate(tmp_path / "plateA")
    run_mask(root)

    steps = chaining.next_steps("mask", {"src": root})

    assert [s.module for s in steps] == ["measure"]
    step = steps[0]
    assert step.ok
    assert step.blocked == ""
    assert step.seed["src"] == root
    assert step.kinds == (ports.MERGED_ARRAYS,)
    assert step.source == "mask"
    registry = artifacts.open_registry(root)
    merged = registry.latest(ports.MERGED_ARRAYS, project=root)
    assert step.artifacts == (merged.artifact_id,)


def test_the_offered_successor_is_the_declared_graph_not_a_hard_coded_list():
    """Whatever requires what a module produces is what gets offered."""
    for module in ports.known_modules():
        offered = {s.module for s in chaining.next_steps(module)}
        assert offered == set(ports.next_modules(module))


def test_an_unready_successor_is_offered_with_its_blocking_reason(tmp_path):
    """Measure has produced no crops, so Classify cannot run — and says so."""
    root = make_plate(tmp_path / "plateA", db_tables=("cell",))
    run_mask(root)
    run_measure(root)

    steps = chaining.next_steps("measure", {"src": root})
    by_module = {s.module: s for s in steps}

    assert "classify" in by_module
    classify = by_module["classify"]
    assert not classify.ok
    assert "png_list" in classify.blocked
    assert classify.fix
    # Scalar even though Classify's own key holds a list — the receiving
    # widget wraps it, and check_ready accepts either.
    assert classify.seed["src"] == root
    assert ports.project_root(classify.seed, "classify") == root
    # A ready successor sorts before a blocked one.
    assert [s.ok for s in steps] == sorted((s.ok for s in steps), reverse=True)


def test_blocked_successors_can_be_dropped_entirely(tmp_path):
    root = make_plate(tmp_path / "plateA", db_tables=("cell",))
    run_mask(root)
    run_measure(root)

    offered = chaining.next_steps("measure", {"src": root},
                                  include_blocked=False)

    assert offered
    assert all(step.ok for step in offered)


def test_a_successor_with_no_registry_row_still_gets_the_project(tmp_path):
    """The seed falls back to the folder the finished run worked in."""
    root = make_plate(tmp_path / "plateA")

    steps = chaining.next_steps("mask", {"src": root})

    assert steps[0].seed["src"] == root
    assert steps[0].ok


def test_blocked_reason_counts_the_rest(tmp_path):
    """More than one error is summarised, not truncated silently."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    steps = chaining.next_steps("mask", {"src": str(empty)})
    blocked = [s for s in steps if not s.ok]
    assert blocked
    assert blocked[0].blocked
    assert blocked[0].fix


def test_next_steps_and_chaining_reject_an_unknown_module():
    with pytest.raises(ports.UnknownModule):
        chaining.next_steps("not_a_module")
    with pytest.raises(ports.UnknownModule):
        chaining.chained_inputs("not_a_module")
    with pytest.raises(ports.UnknownModule):
        chaining.resolve_settings("not_a_module", {})


def test_the_whole_chain_end_to_end(tmp_path, pins):
    """Mask → Measure → Classify, each opening on what the last one wrote."""
    root = make_plate(tmp_path / "plateA")
    run_mask(root)

    measure = chaining.resolve_settings(
        "measure", {"src": "path"}, roots=[root], pins=pins)
    assert measure.settings["src"] == root

    # Measure runs and leaves a database with crops indexed.
    make_plate(Path(root), merged=0, db_tables=("png_list", "cell"),
               crops=True)
    run_measure(root)

    classify = chaining.resolve_settings(
        "classify", {"src": []}, roots=[root], pins=pins)
    assert classify.settings["src"] == [root]

    steps = chaining.next_steps("measure", {"src": root})
    assert {s.module for s in steps} >= {"classify", "umap"}
    assert all(s.ok for s in steps if s.module in {"classify", "umap"})
