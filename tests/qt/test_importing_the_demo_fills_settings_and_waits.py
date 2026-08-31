"""Importing the demo dataset fills the form and stops there.

Asked for on 2026-08-31: "when importing this data the correct settings
should also be filled in... the user should be able to hit import and then
live preview or run".

Two behaviours, and the second is the one that changed. The settings pack
that ships with the dataset is MIGRATED onto this build's defaults rather
than merged over them, and nothing is executed: the screen opens, the form
is filled, and the user presses Live Preview or Run.

The previous version ran Mask, then prompted, then ran Measure, then
prompted again. A demo dataset exists to be looked at, and the first thing
anyone wants is Live Preview on one field.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.settings_pack import (PackReport, read_pack,
                                    settings_from_pack)


def _write_pack(folder, app_key, rows):
    path = folder / f"{app_key}_settings.csv"
    path.write_text("".join(f"{k},{v}\n" for k, v in rows), encoding="utf-8")
    return path


class TestReadingAPack:
    def test_it_reads_key_value_rows(self, tmp_path):
        _write_pack(tmp_path, "mask", [("cell_diameter", "30")])
        values, malformed = read_pack("mask", str(tmp_path))
        assert values == {"cell_diameter": 30}
        assert malformed == 0

    def test_a_missing_pack_is_not_an_error(self, tmp_path):
        """A pack carries settings for some apps and not others."""
        assert read_pack("measure", str(tmp_path)) == ({}, 0)

    def test_comments_are_skipped_and_bad_rows_are_counted(self, tmp_path):
        _write_pack(tmp_path, "mask", [("# a comment", ""),
                                       ("cell_diameter", "30")])
        (tmp_path / "mask_settings.csv").write_text(
            "# a comment\ncell_diameter,30\nlonely\n", encoding="utf-8")
        values, malformed = read_pack("mask", str(tmp_path))
        assert values == {"cell_diameter": 30}
        assert malformed == 1

    @pytest.mark.parametrize("text,expected", [
        ("true", True), ("False", False), ("30", 30), ("1.5", 1.5),
        ("none", None), ("cpsam", "cpsam"),
    ])
    def test_values_get_the_type_the_form_expects(self, tmp_path, text,
                                                  expected):
        """A spin box handed the string "30" shows nothing."""
        _write_pack(tmp_path, "mask", [("k", text)])
        assert read_pack("mask", str(tmp_path))[0]["k"] == expected


class TestMigratingAPack:
    DEFAULTS = {"cell_diameter": 10, "src": "", "nucleus_channel": None}

    def test_a_key_this_build_has_is_applied(self, tmp_path):
        _write_pack(tmp_path, "mask", [("cell_diameter", "30")])
        settings, report = settings_from_pack(
            "mask", str(tmp_path), defaults=self.DEFAULTS)
        assert settings["cell_diameter"] == 30
        assert report.applied == ["cell_diameter"]
        assert report.dropped == []

    def test_a_key_this_build_has_never_heard_of_is_dropped_and_named(
            self, tmp_path):
        """THE BUG THIS MODULE EXISTS FOR.

        The previous loader wrote every row straight over the defaults, so
        a key from an older spaCR travelled into the pipeline to be either
        ignored or to produce an error naming a setting the user never
        typed and cannot find in the form.
        """
        _write_pack(tmp_path, "mask", [("cell_diameter", "30"),
                                       ("gone_in_this_version", "7")])
        settings, report = settings_from_pack(
            "mask", str(tmp_path), defaults=self.DEFAULTS)
        assert "gone_in_this_version" not in settings
        assert report.dropped == ["gone_in_this_version"]
        assert "gone_in_this_version" in report.summary()

    def test_a_renamed_key_lands_under_its_new_name(self, tmp_path,
                                                    monkeypatch):
        monkeypatch.setitem(
            __import__("spacr.qt.settings_pack", fromlist=["x"]).PACK_RENAMES,
            "mask", {"old_diameter": "cell_diameter"})
        _write_pack(tmp_path, "mask", [("old_diameter", "30")])
        settings, report = settings_from_pack(
            "mask", str(tmp_path), defaults=self.DEFAULTS)
        assert settings["cell_diameter"] == 30
        assert report.renamed == [("old_diameter", "cell_diameter")]
        assert report.dropped == []

    def test_src_is_this_machine_not_the_one_that_wrote_the_pack(self,
                                                                 tmp_path):
        """A pack names a path on somebody else's disk.

        Applying it points the run at a folder that is not there, which
        fails much later and blames the dataset rather than the pack.
        """
        _write_pack(tmp_path, "mask", [("src", "/somebody/elses/disk")])
        settings, report = settings_from_pack(
            "mask", str(tmp_path), src="/here", defaults=self.DEFAULTS)
        assert settings["src"] == "/here"
        assert "src" not in report.dropped

    def test_defaults_survive_where_the_pack_says_nothing(self, tmp_path):
        _write_pack(tmp_path, "mask", [("cell_diameter", "30")])
        settings, _report = settings_from_pack(
            "mask", str(tmp_path), defaults=self.DEFAULTS)
        assert settings["nucleus_channel"] is None

    def test_the_summary_names_what_was_lost(self):
        """"dropped 3" tells a user something went missing without telling
        them what, which is the worst of both."""
        report = PackReport(applied=["a"], renamed=[("b", "c")],
                            dropped=["d"])
        text = report.summary()
        assert "2 settings loaded" in text
        assert "b to c" in text and "d" in text


def test_the_import_does_not_start_a_run(monkeypatch, tmp_path):
    """Nothing is executed. The user presses Live Preview or Run.

    Asserted on the SOURCE of the chain rather than by driving a window,
    because what must not happen is a call that no longer exists -- and
    the readable way to pin "this does not run anything" is that the
    method body contains no run invocation.
    """
    import inspect

    from spacr.qt.app import MainWindow

    body = inspect.getsource(MainWindow._run_e2e_chain)
    assert "_on_run()" not in body, (
        "importing the demo starts a pipeline again; the user should press "
        "Live Preview or Run themselves")
    assert "apply_settings_dict" in body, (
        "the demo no longer fills the settings form")
