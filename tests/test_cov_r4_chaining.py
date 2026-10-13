"""``spacr.chaining``: the answers a drop gives when the layout is not tidy.

The paths pinned here are the ones a real project reaches and the happy-path
tests do not:

* :func:`~spacr.chaining.layout_directories` reads folder names off the port
  declarations, and a port that names a *file* at the project root is not one;
* :func:`~spacr.chaining.db_candidates` still finds a database in a project
  that has no ``measurements/`` folder at all;
* a module with two ports bound to the same settings key resolves it once —
  Classify declares both a measurements database and an optional crop folder,
  and both bind to ``src``;
* the registry, not the layout, answers when a run was registered;
* two databases in one project is a question, not a guess;
* a screen with no port declaration of its own — one driven by ``kinds`` —
  still gets its ports' own problem sentences back.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

from spacr import artifacts, chaining, ports


@pytest.fixture
def restore_ports():
    """Snapshot the global port declarations, and the layout cache with them.

    ``layout_directories`` caches against ``len(PORTS)``, so a test that adds
    a declaration must put both back or the next caller reads a cache built
    from ports that no longer exist.
    """
    saved = dict(ports.PORTS)
    yield ports.PORTS
    ports.PORTS.clear()
    ports.PORTS.update(saved)
    chaining._LAYOUT_CACHE.clear()


def _plate(root, *, merged=1, dbs=("measurements.db",), crops=False,
           tables=("cell", "png_list")):
    """A plate folder shaped the way the pipeline leaves one."""
    root = str(root)
    os.makedirs(root, exist_ok=True)
    if merged:
        os.makedirs(os.path.join(root, "merged"), exist_ok=True)
        for index in range(merged):
            np.save(os.path.join(root, "merged", f"plate1_A01_{index}.npy"),
                    np.zeros((4, 4, 3), dtype=np.uint16))
    for name in dbs:
        os.makedirs(os.path.join(root, "measurements"), exist_ok=True)
        connection = sqlite3.connect(os.path.join(root, "measurements", name))
        for table in tables:
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


# --------------------------------------------------------------------------- #
#  The layout is read off the ports, and it is a list of FOLDERS
# --------------------------------------------------------------------------- #

def test_a_port_naming_a_file_at_the_root_is_not_a_layout_directory(
        restore_ports):
    """A plugin's port declaration joins the layout — if it names a folder.

    ``layout_directories`` is the set of names that are never a sub-project
    and that a "this is not a spaCR project" message lists. A port naming a
    single file at the project root — ``summary.csv``, no pattern — is not a
    folder, and putting its filename in that set would exclude a real
    sub-project called ``summary.csv`` from a drop.

    Both ports are declared by the same module so the difference is the
    declaration and nothing else.
    """
    before = chaining.layout_directories()
    assert "qc4" not in before and "summary4.csv" not in before

    ports.register_module_ports(ports.ModulePorts(
        key="r4plugin",
        produces=(
            ports.Port(kind=ports.REGRESSION_RESULTS, role="qc",
                       path="qc4", pattern="*.csv"),
            ports.Port(kind=ports.SETTINGS_CSV, role="summary",
                       path="summary4.csv"),
        )))

    after = chaining.layout_directories()
    assert "qc4" in after, "a port naming a folder joins the layout"
    assert "summary4.csv" not in after
    assert "summary4" not in after


# --------------------------------------------------------------------------- #
#  Finding the databases
# --------------------------------------------------------------------------- #

def test_a_project_with_no_measurements_folder_still_offers_its_database(
        tmp_path):
    """The declared folder is where it looks first, not the only place.

    A project someone assembled by hand — or one whose ``measurements/``
    folder was never created because the database was written beside the
    images — has no declared holder to list, and the shallow listing of the
    root is what saves the drop from answering "no database here".
    """
    laid_out = _plate(tmp_path / "laid_out")
    flat = tmp_path / "flat"
    flat.mkdir()
    sqlite3.connect(str(flat / "loose.db")).close()
    assert not os.path.isdir(str(flat / "measurements"))

    assert chaining.db_candidates(str(flat)) == (str(flat / "loose.db"),)
    # ...and with the declared folder present, the declared file comes first.
    assert chaining.db_candidates(laid_out)[0] == os.path.join(
        laid_out, "measurements", "measurements.db")


def test_two_databases_in_one_project_are_a_question_not_a_guess(tmp_path):
    """Picking the first would be the silent wrong answer.

    A project legitimately holds two: ``measurements.db`` and whatever a
    re-measure into a second file left. The drop resolves the *setting*
    normally and asks which database it means, naming the key the answer
    belongs in so a dialog can write it back.
    """
    project = _plate(tmp_path / "plate1", dbs=("measurements.db", "spare.db"))

    result = chaining.resolve_drop("classify", project)

    assert [t.kind for t in result.targets] == [ports.MEASUREMENTS_DB]
    assert len(result.choices) == 1
    choice = result.choices[0]
    assert choice.kind == ports.MEASUREMENTS_DB
    assert choice.setting == "src"
    assert set(choice.options) == {
        os.path.join(project, "measurements", "measurements.db"),
        os.path.join(project, "measurements", "spare.db")}
    assert "which one" in choice.question.lower()

    # One database in the same layout asks nothing at all.
    single = _plate(tmp_path / "plate2")
    assert chaining.resolve_drop("classify", single).choices == ()


# --------------------------------------------------------------------------- #
#  One settings key, one answer
# --------------------------------------------------------------------------- #

def test_the_second_port_bound_to_src_does_not_resolve_it_again(tmp_path):
    """Classify binds two ports to ``src``; the crop folder must not be walked.

    Resolving the second would recursively glob a folder of crops to arrive
    at the string already in hand, with the mouse button down. The crops are
    really there in this test — the folder the second port declares exists
    and holds a PNG — so "one target" is a decision and not an absence.
    """
    project = _plate(tmp_path / "plate1", crops=True)
    registry = artifacts.open_registry(project)
    artifacts.register_run_outputs(
        "measure", {"src": project, "cell_mask_dim": 4, "save_png": True},
        registry=registry)
    assert registry.latest(ports.CROPS, project=project) is not None, (
        "the crops port has an answer of its own to be skipped")

    result = chaining.resolve_drop("classify", project, registry=registry)

    assert [t.setting for t in result.targets] == ["src"]
    target = result.targets[0]
    assert target.kind == ports.MEASUREMENTS_DB
    assert target.source == chaining.FROM_REGISTRY, (
        "the registry answers before the layout is walked")
    assert target.location == os.path.join(project, "measurements",
                                           "measurements.db")
    assert target.value == project


def test_a_list_valued_source_keeps_its_list_when_the_registry_answers(
        tmp_path):
    """Classify keeps its sources in a list and every other module does not.

    Same drop as above; only the current value of the key differs, which is
    what decides whether the registry's answer is wrapped.
    """
    project = _plate(tmp_path / "plate1")
    registry = artifacts.open_registry(project)
    artifacts.register_run_outputs(
        "measure", {"src": project, "cell_mask_dim": 4, "save_png": True},
        registry=registry)

    plain = chaining.resolve_drop("classify", project, registry=registry)
    wrapped = chaining.resolve_drop("classify", project, registry=registry,
                                    settings={"src": ["an/older/plate"]})

    assert plain.targets[0].value == project
    assert wrapped.targets[0].value == [project]


# --------------------------------------------------------------------------- #
#  A sibling that is not a project
# --------------------------------------------------------------------------- #

def test_a_sibling_folder_that_is_not_a_project_is_not_a_candidate(tmp_path):
    """Dropping the folder that holds the plates is normal; guessing is not.

    With one real plate beside a folder that satisfies nothing, there is
    exactly one candidate, so the drop resolves instead of asking. If the
    non-project were counted the user would get a two-option dialog with one
    wrong answer in it.
    """
    parent = tmp_path / "screen"
    parent.mkdir()
    plate = _plate(parent / "plate1")
    (parent / "notes").mkdir()
    (parent / "notes" / "readme.txt").write_text("nothing to see")

    result = chaining.resolve_drop("measure", str(parent))

    assert result.choices == (), "one candidate is not ambiguous"
    assert [t.value for t in result.targets] == [plate]

    # Two real plates under the same parent *is* the question.
    _plate(parent / "plate2")
    asked = chaining.resolve_drop("measure", str(parent))
    assert len(asked.choices) == 1
    assert set(asked.choices[0].options) == {plate, str(parent / "plate2")}


# --------------------------------------------------------------------------- #
#  A screen with no port declaration of its own
# --------------------------------------------------------------------------- #

def test_a_kinds_driven_screen_still_gets_its_ports_problems(tmp_path):
    """``check_ready`` cannot be asked about a module it has never heard of.

    A screen that declares no ports says what it wants with ``kinds``. Drop
    a folder holding none of it and the answer must still be the port-level
    sentences — one per kind, naming the place it looked — rather than an
    empty ``problems`` tuple that a dialog would render as "nothing wrong".
    """
    empty = tmp_path / "not_a_plate"
    empty.mkdir()

    result = chaining.resolve_drop(
        "an-unregistered-screen", str(empty),
        kinds=(ports.MERGED_ARRAYS, ports.MEASUREMENTS_DB))

    assert result.targets == ()
    assert result.choices == ()
    text = "\n".join(str(problem) for problem in result.problems)
    assert ports.MERGED_ARRAYS in text
    assert ports.MEASUREMENTS_DB in text
    assert os.path.join(str(empty), "merged") in text

    # The same call against a project that has both reports nothing wrong,
    # so the sentences above are a finding and not a constant.
    project = _plate(tmp_path / "plate1")
    ready = chaining.resolve_drop(
        "an-unregistered-screen", project,
        kinds=(ports.MERGED_ARRAYS, ports.MEASUREMENTS_DB))
    assert ready.problems == ()
    assert {t.kind for t in ready.targets} == {ports.MERGED_ARRAYS,
                                               ports.MEASUREMENTS_DB}
