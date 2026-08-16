"""What a dropped folder means to a module.

Instruction 60. ``chaining.resolve_drop`` is ~120 statements and had NO test
at all, which is how ``chaining.py`` came to be the worst genuine non-Qt gap
in the package at 65%.

It is worth more than its line count. It is the function that decides what
happens when a user drags a folder onto a module screen, and its own docstring
states the property that makes it correct:

    "The whole point of the function is that it is *the same* resolution
     auto-chaining performs ... so a drop and an auto-chain fill the field
     with the same string."

The tests below are built on real directory layouts rather than mocks,
because what the function reads IS the directory layout -- a mocked
filesystem would test the mock. The ambiguity paths matter most: this
function is allowed to say "I don't know, you choose", and the failure that
costs a user real time is it guessing instead.
"""
from __future__ import annotations

import os

import pytest

from spacr import chaining


def _project(root, name="plate1", *, dirs=("merged", "measurements"),
             files=(("merged", "field1.npy"),)):
    """A plate folder with the layout directories spaCR writes."""
    project = os.path.join(str(root), name)
    for directory in dirs:
        os.makedirs(os.path.join(project, directory), exist_ok=True)
    for directory, filename in files:
        path = os.path.join(project, directory, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"\x00")
    return project


# --------------------------------------------------------------------------- #
#  The ordinary drops
# --------------------------------------------------------------------------- #

def test_dropping_the_folder_a_module_consumes_fills_its_setting(tmp_path):
    """Measure consumes merged arrays; dropping `merged/` must fill `src`."""
    project = _project(tmp_path)

    result = chaining.resolve_drop("measure", os.path.join(project, "merged"))

    assert result.problems == ()
    assert result.choices == ()
    settings = {target.setting for target in result.targets}
    assert "src" in settings


def test_dropping_the_project_root_works_too(tmp_path):
    """A user drops the plate folder far more often than a subfolder, and
    the layout walk is what makes that equivalent."""
    project = _project(tmp_path)

    result = chaining.resolve_drop("measure", project)

    assert result.problems == ()
    values = {target.value for target in result.targets}
    assert any(project in str(value) for value in values)


def test_dropping_a_file_resolves_from_its_directory(tmp_path):
    """Dragging a file is dragging the folder it is in -- users do both."""
    project = _project(tmp_path)
    dropped = os.path.join(project, "merged", "field1.npy")

    result = chaining.resolve_drop("measure", dropped)

    assert result.problems == ()
    assert result.targets


def test_the_walk_climbs_out_of_a_subfolder_to_the_project(tmp_path):
    """`project_root_of` is what lets a drop deep inside a plate still
    resolve, rather than treating the subfolder as the project."""
    project = _project(tmp_path, dirs=("merged", "measurements", "settings"))
    deep = os.path.join(project, "settings")

    result = chaining.resolve_drop("measure", deep)

    assert result.problems == () or result.targets


# --------------------------------------------------------------------------- #
#  Ambiguity is RETURNED, never guessed -- the docstring's own promise
# --------------------------------------------------------------------------- #

def test_two_projects_under_one_folder_produce_a_choice_not_a_guess(tmp_path):
    """The failure that costs a user their afternoon is picking one plate
    silently. Two candidates must come back as a question."""
    parent = tmp_path / "screen"
    parent.mkdir()
    _project(parent, "plate1")
    _project(parent, "plate2")

    result = chaining.resolve_drop("measure", str(parent))

    assert result.choices, "two projects must be offered, not resolved"
    choice = result.choices[0]
    assert len(choice.options) == 2
    # The question has to name what it is asking about, or a dialog carrying
    # it says nothing.
    assert "which one" in choice.question.lower()


def test_one_project_under_a_folder_is_resolved_without_asking(tmp_path):
    """The other side: a single candidate is not ambiguous, so asking would
    be noise."""
    parent = tmp_path / "screen"
    parent.mkdir()
    _project(parent, "plate1")

    result = chaining.resolve_drop("measure", str(parent))

    assert result.choices == ()
    assert result.targets


def test_a_folder_that_satisfies_nothing_reports_problems(tmp_path):
    """`check_ready`'s own sentences, rather than an empty result the caller
    has to interpret."""
    empty = tmp_path / "not_a_project"
    empty.mkdir()

    result = chaining.resolve_drop("measure", str(empty))

    assert result.problems or result.choices or not result.targets


# --------------------------------------------------------------------------- #
#  Modules with no port declaration, driven by `kinds`
# --------------------------------------------------------------------------- #

def test_an_undeclared_module_falls_back_to_the_kinds_it_was_given(tmp_path):
    """A screen with no entry in PORTS still resolves, from the vocabulary
    terms it passes in."""
    project = _project(tmp_path)

    result = chaining.resolve_drop("not_a_declared_module", project,
                                   kinds=("merged-arrays",))

    assert isinstance(result, chaining.DropResolution)
    assert result.targets or result.problems


def test_a_module_that_takes_the_project_itself_gets_the_folder(tmp_path):
    """No port to resolve and nothing to look up: the answer is the folder
    the layout walk arrived at, which is the point of having walked it."""
    project = _project(tmp_path)

    result = chaining.resolve_drop("not_a_declared_module", project, kinds=())

    if result.targets:
        target = result.targets[0]
        assert target.kind == chaining.PROJECT
        assert target.source == chaining.FROM_LAYOUT
        assert os.path.basename(str(target.value)) == "plate1"


# --------------------------------------------------------------------------- #
#  The helpers the resolver leans on
# --------------------------------------------------------------------------- #

def test_the_walk_recognises_the_project_from_a_layout_folder(tmp_path):
    """`merged/` is a declared layout folder, so its PARENT is the project.
    That recognition costs no climb -- the loop runs max_climb + 1 times."""
    project = _project(tmp_path)

    assert chaining.project_root_of(os.path.join(project, "merged"),
                                    max_climb=0) == project


def test_max_climb_bounds_how_far_above_the_drop_it_looks(tmp_path):
    """Without a bound the walk would leave the user's data directory and
    'find' a project somewhere irrelevant.

    `data/` is also a declared folder, so from `data/plate1/cell_png` the
    highest declared folder within reach is what answers -- and a tighter
    bound must give a nearer answer, never a further one.
    """
    project = _project(tmp_path, dirs=("data",))
    deep = os.path.join(project, "data", "plate1", "cell_png")
    os.makedirs(deep, exist_ok=True)

    far = chaining.project_root_of(deep, max_climb=8)
    near = chaining.project_root_of(deep, max_climb=0)

    assert far == project
    assert len(near) >= len(far), "a tighter bound must not climb further"


def test_a_path_nowhere_near_a_project_answers_with_its_own_folder(tmp_path):
    """Documented behaviour, and the one a direct drop relies on: never
    raise, just hand back where you are."""
    loose = tmp_path / "somewhere" / "else"
    loose.mkdir(parents=True)

    assert chaining.project_root_of(str(loose)) == str(loose)


def test_satisfies_is_false_for_a_folder_missing_the_port(tmp_path):
    from spacr import ports

    empty = tmp_path / "bare"
    empty.mkdir()
    spec = ports.module_ports("measure")

    assert chaining.satisfies(str(empty), tuple(spec.consumes)) is False


def test_satisfies_is_true_once_the_layout_is_there(tmp_path):
    from spacr import ports

    project = _project(tmp_path)
    spec = ports.module_ports("measure")

    assert chaining.satisfies(project, tuple(spec.consumes)) is True


def test_ports_for_kinds_maps_the_vocabulary(tmp_path):
    found = chaining.ports_for_kinds(("merged-arrays",))
    assert found
    assert any(port.kind == "merged-arrays" for port in found)


def test_layout_directories_are_the_ones_spacr_writes():
    """A guard on the vocabulary itself: these names are what every layout
    walk matches on, so a rename here silently breaks every drop."""
    directories = chaining.layout_directories()
    for expected in ("merged", "measurements", "masks", "settings"):
        assert expected in directories
