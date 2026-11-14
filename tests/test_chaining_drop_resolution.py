"""Table-driven tests of what a dropped path means to a module, ``spacr.chaining``.

``resolve_drop`` decides what happens when a user drags a folder or file onto
a module screen, and ``project_root_of`` is the upward layout walk it leans
on.  The cases are real directory layouts, because what these functions read
IS the directory layout.  Ambiguity is returned as a question, never guessed,
and a folder holding nothing usable answers with ``check_ready``'s own
sentences.  One general test per unit, each walking a case table with
pytest's ``subtests`` fixture and asserting exact targets, choices and
problem sentences.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

from spacr import artifacts, chaining, ports
from spacr.chaining import DropChoice, DropTarget


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    """No test here touches the developer's registry or their real pins."""
    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)
    monkeypatch.setenv(chaining.PIN_STATE_ENV,
                       str(tmp_path / "state" / "pins.json"))
    chaining.pin_store(refresh=True)
    yield
    chaining.pin_store(refresh=True)


def _plate(root, *, merged=True, dbs=(), crops=False, extra_dirs=()):
    """A plate folder shaped the way the pipeline leaves one."""
    root = str(root)
    os.makedirs(root, exist_ok=True)
    if merged:
        os.makedirs(os.path.join(root, "merged"), exist_ok=True)
        np.save(os.path.join(root, "merged", "field1.npy"),
                np.zeros((4, 4, 3), dtype=np.uint16))
    for directory in extra_dirs:
        os.makedirs(os.path.join(root, directory), exist_ok=True)
    for name in dbs:
        os.makedirs(os.path.join(root, "measurements"), exist_ok=True)
        connection = sqlite3.connect(os.path.join(root, "measurements", name))
        for table in ("cell", "png_list"):
            connection.execute(f'CREATE TABLE "{table}" (value INTEGER)')
            connection.execute(f'INSERT INTO "{table}" VALUES (1)')
        connection.commit()
        connection.close()
    if crops:
        folder = os.path.join(root, "data", "A01", "cell_png")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "object_1.png"), "wb") as handle:
            handle.write(b"\x89PNG")
    return root


def _merged_target(plate, *, module="measure", setting="src", value=None):
    """The layout answer for a plate's merged arrays."""
    merged = os.path.join(plate, "merged")
    return DropTarget(module=module, setting=setting, role="merged",
                      kind=ports.MERGED_ARRAYS,
                      value=plate if value is None else value,
                      location=merged, source=chaining.FROM_LAYOUT,
                      required=True,
                      paths=(os.path.join(merged, "field1.npy"),))


def test_a_drop_resolves_the_same_answer_or_says_why_not(tmp_path, subtests):
    """Every drop layout resolves to exact targets, a question, or problem sentences."""
    plate = _plate(tmp_path / "plate1", extra_dirs=("measurements", "settings"))
    single = tmp_path / "single"
    only = _plate(single / "plate1")
    (single / "notes").mkdir()
    (single / "notes" / "readme.txt").write_text("nothing", encoding="utf-8")
    pair = tmp_path / "pair"
    first, second = _plate(pair / "plate1"), _plate(pair / "plate2")
    empty = tmp_path / "not_a_project"
    empty.mkdir()
    empty = str(empty)

    # (case, module, dropped, kwargs, root, targets, choices, problems)
    cases = [
        ("merged_folder_fills_src", "measure", os.path.join(plate, "merged"),
         {}, plate, (_merged_target(plate),), (), ()),
        ("project_root", "measure", plate, {}, plate,
         (_merged_target(plate),), (), ()),
        ("a_file_resolves_from_its_folder", "measure",
         os.path.join(plate, "merged", "field1.npy"), {}, plate,
         (_merged_target(plate),), (), ()),
        ("climbs_out_of_a_layout_subfolder", "measure",
         os.path.join(plate, "settings"), {}, plate,
         (_merged_target(plate),), (), ()),
        ("list_shaped_setting_gets_a_list", "measure", plate,
         {"settings": {"src": ["/old/plate0"]}}, plate,
         (_merged_target(plate, value=[plate]),), (), ()),
        ("one_project_and_a_non_project_sibling", "measure", str(single),
         {}, only, (_merged_target(only),), (), ()),
        ("two_projects_are_a_question", "measure", str(pair), {}, str(pair),
         (), (DropChoice(
             question="2 projects under pair can be used here — which one?",
             kind=ports.MERGED_ARRAYS, options=(first, second),
             setting="src"),), ()),
        ("undeclared_module_uses_its_kinds", "not_a_declared_module", plate,
         {"kinds": (ports.MERGED_ARRAYS,)}, plate,
         (_merged_target(plate, module="not_a_declared_module",
                         setting="merged",
                         value=os.path.join(plate, "merged")),), (), ()),
        ("undeclared_module_taking_the_project", "not_a_declared_module",
         plate, {"kinds": ()}, plate,
         (DropTarget(module="not_a_declared_module", setting="src",
                     role=chaining.PROJECT, kind=chaining.PROJECT,
                     value=plate, location=plate,
                     source=chaining.FROM_LAYOUT),), (), ()),
    ]
    for case_id, module, dropped, kwargs, root, targets, choices, problems in cases:
        with subtests.test(msg=case_id):
            result = chaining.resolve_drop(module, dropped, **kwargs)
            assert result.root == root
            assert result.targets == targets
            assert result.choices == choices
            assert result.problems == problems
            assert result.ambiguous is bool(choices)

    # (case, module, kwargs, expected (message, fix) pairs)
    refusals = [
        ("declared_module_on_an_empty_folder", "measure", {},
         [(f"no merged-arrays at {empty}/merged/*.npy",
           "Run mask or timelapse on this project first.")]),
        ("undeclared_project_screen_on_an_empty_folder",
         "not_a_declared_module", {"kinds": ()},
         [(f"{empty} is not a spaCR project folder",
           "Drop the plate folder itself — the one holding consolidated, "
           "data, masks, measurements and the rest of the layout — or a "
           "folder containing several of them.")]),
    ]
    for case_id, module, kwargs, expected in refusals:
        with subtests.test(msg=case_id):
            result = chaining.resolve_drop(module, empty, **kwargs)
            assert (result.targets, result.choices) == ((), ())
            assert [(p.message, p.fix) for p in result.problems] == expected
            assert result.ok is False
            assert result.reason == f"{expected[0][0]}. {expected[0][1]}"

    with subtests.test(msg="kinds_driven_screen_gets_one_sentence_per_kind"):
        kinds = (ports.MERGED_ARRAYS, ports.MEASUREMENTS_DB)
        result = chaining.resolve_drop("an-unregistered-screen", empty,
                                       kinds=kinds)
        assert (result.targets, result.choices) == ((), ())
        text = "\n".join(str(problem) for problem in result.problems)
        assert ports.MERGED_ARRAYS in text
        assert ports.MEASUREMENTS_DB in text
        assert os.path.join(empty, "merged") in text
        ready = chaining.resolve_drop(
            "an-unregistered-screen",
            _plate(tmp_path / "ready", dbs=("measurements.db",)), kinds=kinds)
        assert ready.problems == ()
        assert {t.kind for t in ready.targets} == set(kinds)


def test_a_classify_drop_asks_the_registry_once_and_asks_about_databases(
        tmp_path, subtests):
    """Classify binds two ports to ``src``: one answer, registry first, databases asked."""
    with subtests.test(msg="two_databases_are_a_question"):
        plate = _plate(tmp_path / "two" / "plate1",
                       dbs=("measurements.db", "spare.db"))
        result = chaining.resolve_drop("classify", plate)
        assert [t.kind for t in result.targets] == [ports.MEASUREMENTS_DB]
        assert result.choices == (DropChoice(
            question="plate1 holds 2 databases — which one?",
            kind=ports.MEASUREMENTS_DB,
            options=(os.path.join(plate, "measurements", "measurements.db"),
                     os.path.join(plate, "measurements", "spare.db")),
            setting="src"),)
        single = _plate(tmp_path / "one" / "plate1", dbs=("measurements.db",))
        assert chaining.resolve_drop("classify", single).choices == ()

    for case_id, settings, value in (("registry_answer_plain", None, None),
                                     ("registry_answer_wrapped_in_a_list",
                                      {"src": ["an/older/plate"]}, "list")):
        with subtests.test(msg=case_id):
            plate = _plate(tmp_path / case_id / "plate1",
                           dbs=("measurements.db",), crops=True)
            registry = artifacts.open_registry(plate)
            artifacts.register_run_outputs(
                "measure", {"src": plate, "cell_mask_dim": 4,
                            "save_png": True}, registry=registry)
            # The crops port has an answer of its own, which must be skipped.
            assert registry.latest(ports.CROPS, project=plate) is not None
            result = chaining.resolve_drop("classify", plate,
                                           registry=registry,
                                           settings=settings)
            assert [t.setting for t in result.targets] == ["src"]
            target = result.targets[0]
            assert target.kind == ports.MEASUREMENTS_DB
            assert target.source == chaining.FROM_REGISTRY
            assert target.location == os.path.join(
                plate, "measurements", "measurements.db")
            assert target.value == ([plate] if value == "list" else plate)


def test_the_project_root_walk_climbs_only_through_layout_folders(
        tmp_path, subtests):
    """``project_root_of`` lands on the project, is bounded, and never raises."""
    project = _plate(tmp_path / "plate1", extra_dirs=("measurements", "data"))
    deep = os.path.join(project, "data", "plate1", "cell_png")
    os.makedirs(deep)
    loose = tmp_path / "somewhere" / "else"
    loose.mkdir(parents=True)

    # (case, path, max_climb, expected)
    cases = [
        ("none_is_not_the_working_directory", None, chaining.MAX_CLIMB, ""),
        ("filesystem_root_stops", os.sep, chaining.MAX_CLIMB,
         os.path.abspath(os.sep)),
        ("layout_folder_costs_no_climb", os.path.join(project, "merged"), 0,
         project),
        ("a_file_answers_its_project",
         os.path.join(project, "merged", "field1.npy"), chaining.MAX_CLIMB,
         project),
        ("deep_drop_reaches_the_project", deep, 8, project),
        ("a_tight_bound_stays_near", deep, 0, deep),
        ("one_climb_meets_no_layout_folder", deep, 1, deep),
        ("two_climbs_reach_data_and_its_project", deep, 2, project),
        ("outside_any_project_is_its_own_folder", str(loose),
         chaining.MAX_CLIMB, str(loose)),
    ]
    for case_id, path, max_climb, expected in cases:
        with subtests.test(msg=case_id):
            assert chaining.project_root_of(path, max_climb=max_climb) == expected
