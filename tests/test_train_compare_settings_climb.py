"""``load_run`` must not read another project's settings off an ancestor folder.

:func:`spacr.train_compare._load_settings` recovers the settings that produced a
training run by walking up from the run folder looking for ``<ancestor>/settings/
*.csv`` — the place :func:`spacr.utils.save_settings` writes a snapshot to.

Walking up *blindly* means the answer depends on where the user happened to put
the project. A run at ``/data/experiments/plate1/model/maxvit_t/rgb/epochs_10``
would, with nothing of its own to find, keep climbing to ``/data/experiments``
and then ``/data`` and report **someone else's** ``settings/`` folder as this
run's provenance. The settings diff then shows differences nobody configured,
which is worse than showing none.

The test suite hit exactly that: ``spacr.deep_spacr.generate_activation_map``
derives ``src`` as ``dirname(dirname(settings['dataset']))``, so a test whose
dataset tar sits directly in ``tmp_path`` makes it write
``<pytest session root>/settings/<cam_type>_settings.csv`` — into the folder
that is an ancestor of *every* test's ``tmp_path``. Two ``train_compare`` tests
that assert "no settings found" then passed alone and failed in a full run,
depending on ordering.

The fix is in the climb, not in the tests: a ``settings/`` folder is only this
run's when the ancestor holding it is either the run folder's own parent, or the
project root the training output tree hangs off — the directory whose child on
the way down to the run is ``model/``, which is how
:func:`spacr.deep_spacr.train_test_model` builds ``dst``.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from spacr.train_compare import _SETTINGS_SEARCH_DEPTH, load_run


def _write_settings(folder, stem, values):
    """Write a ``Key,Value`` CSV where ``spacr.utils.save_settings`` writes one."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.csv"
    pd.DataFrame(list(values.items()), columns=["Key", "Value"]).to_csv(
        path, index=False)
    return path


def _write_curve(folder, rows=3):
    """Both progress CSVs ``spacr.io._save_progress`` writes into a run's dst."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "validation.csv"):
        (folder / name).write_text(
            "epoch,accuracy,loss\n"
            + "".join(f"{e},0.5,1.0\n" for e in range(1, rows + 1)))


def _training_dst(src, model_type="maxvit_t", channels="rgb", epochs=10):
    """The ``dst`` deep_spacr.train_test_model builds: ``<src>/model/<mt>/<ch>/epochs_<N>``."""
    return src / "model" / model_type / channels / f"epochs_{epochs}"


# ---------------------------------------------------------------------------
# The bug: configuration read from an arbitrary ancestor
# ---------------------------------------------------------------------------

def test_settings_are_not_taken_from_an_unrelated_ancestor(tmp_path):
    """A ``settings/`` folder further up the tree is not this run's provenance.

    This is the shape the full-suite failure had: a shared parent directory
    (there, the pytest session root; here, a shared data drive) holding a
    ``settings/`` folder some other pipeline wrote, with the run nested below
    it in a subtree of its own.
    """
    shared = tmp_path / "data"
    _write_settings(shared / "settings", "saliency_image_settings",
                    {"model_type": "somebody_elses_model", "epochs": 99})

    run = shared / "project" / "run"
    _write_curve(run)

    loaded = load_run(str(run), run_id="A")
    assert loaded.settings == {}, (
        f"load_run read {loaded.settings_path} — a settings folder two levels "
        f"up that belongs to no run in this tree")
    assert any("no settings found" in n for n in loaded.notes)
    assert loaded.has_curves


def test_the_climb_never_reaches_past_the_project_root(tmp_path):
    """With BOTH a real and a bogus snapshot on the path, the run's own wins.

    The run's project root is the folder whose child is ``model/``. Anything
    above that belongs to a different run, or to no run at all.
    """
    outer = tmp_path / "screens"
    _write_settings(outer / "settings", "train_test_maxvit_t_10",
                    {"model_type": "wrong", "learning_rate": 999.0})

    src = outer / "screen1"
    _write_settings(src / "settings", "train_test_maxvit_t_10",
                    {"model_type": "maxvit_t", "learning_rate": 0.0001})
    dst = _training_dst(src)
    _write_curve(dst)

    loaded = load_run(str(dst), run_id="A")
    assert loaded.settings["model_type"] == "maxvit_t"
    assert float(loaded.settings["learning_rate"]) == 0.0001
    assert loaded.settings_path.startswith(str(src))


def test_a_run_whose_project_has_no_snapshot_reports_none(tmp_path):
    """The project root exists and simply has no ``settings/`` — say so."""
    outer = tmp_path / "screens"
    _write_settings(outer / "settings", "train_test_maxvit_t_10",
                    {"model_type": "wrong"})
    dst = _training_dst(outer / "screen1")
    _write_curve(dst)

    loaded = load_run(str(dst), run_id="A")
    assert loaded.settings == {}
    assert loaded.settings_path == ""
    assert any("no settings found" in n for n in loaded.notes)


# ---------------------------------------------------------------------------
# …while everything the climb legitimately reaches still works
# ---------------------------------------------------------------------------

def test_the_projects_own_settings_folder_is_still_found(tmp_path):
    """``<src>/settings/train_test_<model_type>_<epochs>.csv``, four levels up."""
    src = tmp_path / "screen1"
    _write_settings(src / "settings", "train_test_maxvit_t_10",
                    {"model_type": "maxvit_t", "epochs": 10})
    dst = _training_dst(src)
    _write_curve(dst)

    loaded = load_run(str(dst), run_id="A")
    assert loaded.settings["model_type"] == "maxvit_t"
    assert loaded.settings_path.endswith("train_test_maxvit_t_10.csv")
    assert loaded.notes == []


def test_a_fold_folder_still_reaches_its_project(tmp_path):
    """A k-fold run is one level deeper: ``<dst>/fold_<i>``."""
    src = tmp_path / "screen1"
    _write_settings(src / "settings", "train_test_maxvit_t_10",
                    {"model_type": "maxvit_t", "epochs": 10})
    fold = _training_dst(src) / "fold_2"
    _write_curve(fold)

    loaded = load_run(str(fold), run_id="A")
    assert loaded.settings["model_type"] == "maxvit_t"


def test_a_settings_folder_beside_the_run_folder_is_still_used(tmp_path):
    """A ``dst`` that is not the layout train_test_model builds still gets its
    sibling snapshot — the immediate parent is always a plausible owner."""
    src = tmp_path / "ds"
    _write_settings(src / "settings", "train_resnet50_25",
                    {"model_type": "resnet50", "learning_rate": 0.001})
    dst = src / "model_out"
    _write_curve(dst)

    loaded = load_run(str(dst), run_id="A")
    assert loaded.settings["model_type"] == "resnet50"


def test_the_search_depth_still_covers_the_deepest_real_layout(tmp_path):
    """The shipped depth has to reach ``<src>`` from a fold folder: five steps."""
    assert _SETTINGS_SEARCH_DEPTH >= 5


# ---------------------------------------------------------------------------
# the backstop underneath the ownership rule
# ---------------------------------------------------------------------------

def test_the_filesystem_root_is_never_a_project():
    """``/`` is its own parent — the climb has to stop there, not loop."""
    from spacr.train_compare import _is_outside_any_project

    assert _is_outside_any_project(Path(os.path.abspath(os.sep))) is True


def test_the_temp_directory_and_the_home_directory_are_never_projects():
    from spacr.train_compare import _is_outside_any_project

    assert _is_outside_any_project(Path(tempfile.gettempdir())) is True
    assert _is_outside_any_project(Path.home()) is True
    assert _is_outside_any_project(Path.home().parent) is True


def test_a_directory_that_cannot_be_resolved_stops_the_climb(monkeypatch):
    """A dead symlink or a stale NFS mount raises on resolve(); the walk must
    stop rather than propagate the error out of load_run."""
    from spacr.train_compare import _is_outside_any_project

    def _blow_up(self, *args, **kwargs):
        raise OSError("Stale file handle")

    monkeypatch.setattr(Path, "resolve", _blow_up)
    assert _is_outside_any_project(Path("/anything")) is True


def test_a_platform_with_no_home_directory_still_stops_at_the_temp_dir(
        monkeypatch):
    """``Path.home()`` raises RuntimeError when HOME is unresolvable — a bare
    container, or a service account. The rest of the stop set still applies."""
    from spacr.train_compare import _is_outside_any_project

    def _no_home():
        raise RuntimeError("Could not determine home directory")

    monkeypatch.setattr(Path, "home", staticmethod(_no_home))
    assert _is_outside_any_project(Path(tempfile.gettempdir())) is True


@pytest.mark.parametrize("relative", [True, False])
def test_a_run_folder_given_relatively_behaves_the_same(tmp_path, monkeypatch,
                                                        relative):
    """A relative path must not turn the climb loose on the current directory."""
    src = tmp_path / "screen1"
    _write_settings(src / "settings", "train_test_maxvit_t_10",
                    {"model_type": "maxvit_t"})
    dst = _training_dst(src)
    _write_curve(dst)

    monkeypatch.chdir(tmp_path)
    target = "screen1/model/maxvit_t/rgb/epochs_10" if relative else str(dst)
    loaded = load_run(target, run_id="A")
    assert loaded.settings["model_type"] == "maxvit_t"
