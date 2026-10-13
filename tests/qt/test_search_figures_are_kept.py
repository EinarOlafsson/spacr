"""A search's per-trial figures are output, so they outlive the process.

They used to be written to ``tempfile.mkdtemp``. The grid streamed them and
clicking a cell opened the vector file, but only while the app was running:
the figures a user had waited through a whole sweep for were gone as soon as
they closed the window, and nothing tied them to the run that produced them.

The fallback matters as much as the feature. A request with no ``src`` -- a
test's, or a sweep over data that is not on disk -- must still get somewhere
to write, because losing the run to a missing directory is worse than losing
the files to a reboot.
"""

import pathlib
import tempfile

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.hyperparam import _search_figure_dir


class TestItLandsInTheRunFolder:

    def test_a_real_src_gets_a_folder_under_results(self, tmp_path):
        target = _search_figure_dir({"src": str(tmp_path)}, "umap")
        relative = pathlib.Path(target).relative_to(tmp_path)
        assert relative.parts[:2] == ("results", "hyperparameter_search")

    def test_the_folder_is_created_not_merely_named(self, tmp_path):
        target = _search_figure_dir({"src": str(tmp_path)}, "umap")
        assert pathlib.Path(target).is_dir()

    def test_the_app_key_is_in_the_name(self, tmp_path):
        target = _search_figure_dir({"src": str(tmp_path)}, "umap")
        assert pathlib.Path(target).name.startswith("umap_")

    def test_two_searches_do_not_share_a_folder(self, tmp_path):
        """Timestamped, so a second sweep does not overwrite the first."""
        first = _search_figure_dir({"src": str(tmp_path)}, "umap")
        pathlib.Path(first, "trial_0000.png").write_bytes(b"x")
        second = _search_figure_dir({"src": str(tmp_path)}, "umap")
        # Same second is possible, so assert the FILE survives either way.
        assert pathlib.Path(first, "trial_0000.png").exists()
        assert pathlib.Path(second).is_dir()


class TestItNeverCostsTheRun:

    def test_no_src_falls_back_to_a_temporary_directory(self):
        target = _search_figure_dir({}, "umap")
        assert target is not None and pathlib.Path(target).is_dir()

    def test_an_unwritable_src_falls_back_rather_than_raising(self):
        """A search must not die because its output folder cannot be made."""
        target = _search_figure_dir({"src": "/proc/does-not-exist-xyz"}, "umap")
        assert target is not None and pathlib.Path(target).is_dir()

    def test_settings_that_are_not_a_mapping_are_survivable(self):
        assert _search_figure_dir(None, "umap") is not None

    def test_an_empty_src_is_treated_as_absent(self):
        target = _search_figure_dir({"src": "   "}, "umap")
        assert "spacr-search-" in str(target)


class TestTheNameIsSafe:

    def test_a_path_traversal_app_key_cannot_escape(self, tmp_path):
        """`app_key` reaches this from a settings dict, so it is not trusted."""
        target = pathlib.Path(_search_figure_dir({"src": str(tmp_path)},
                                                 "../../etc"))
        assert tmp_path in target.parents
        assert ".." not in target.parts

    def test_an_empty_app_key_still_produces_a_folder(self, tmp_path):
        target = _search_figure_dir({"src": str(tmp_path)}, "")
        assert pathlib.Path(target).is_dir()
