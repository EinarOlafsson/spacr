"""191 C/D: the button fetches the example screen and fills the right slots.

"add my cound and dependent variable csvs to the datafolder in spacr and add
a button that auto loads them into the correct slots", then "that button
should obviously be in input tables."
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def cached(tmp_path, monkeypatch):
    """A cache already holding the example screen, so nothing downloads.

    The files only have to be the right SIZE and DIGEST to be accepted, and
    fabricating 33 MB in a test would be absurd -- so the manifest is pointed
    at what is written instead.
    """
    import hashlib

    from spacr import example_data, example_data_manifest

    entries = []
    for entry in example_data_manifest.FILES:
        body = f"{entry['name']} stand-in\n".encode()
        (tmp_path / entry["name"]).write_bytes(body)
        entries.append({**entry, "bytes": len(body),
                        "sha256": hashlib.sha256(body).hexdigest()})
    monkeypatch.setattr(example_data, "FILES", entries)
    monkeypatch.setenv("SPACR_EXAMPLE_DATA", str(tmp_path))
    return tmp_path


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    return widget


class TestTheButtonIsWhereItBelongs:

    def test_it_exists_on_the_regression_screen(self, screen):
        assert hasattr(screen, "_example_data_button")

    def test_it_says_what_it_will_cost(self, screen):
        """The figure is DERIVED FROM THE MANIFEST, not pinned as a literal.

        It was pinned at "33 MB" and the manifest had grown to 35, so the
        tooltip either lied or the test failed -- and the test failing is how
        it was noticed. Reading the real total means the tooltip is checked
        against what will actually be downloaded, and adding a file to the
        manifest cannot silently make the promise stale.
        """
        from spacr.example_data import total_bytes

        tip = screen._example_data_button.toolTip()
        megabytes = round(total_bytes() / 1e6)

        assert f"{megabytes} MB" in tip, (
            f"the tooltip does not state the real {megabytes} MB cost: {tip!r}")
        assert "cached" in tip, "and that the second press is free"

    def test_each_half_can_be_fetched_on_its_own(self, screen):
        """Counts are 16 MB and scores 19; a user checking one should not wait
        for the other."""
        from spacr.example_data import entries_of_kind, total_bytes

        for kind in ("counts", "scores"):
            button = getattr(screen, f"_example_{kind}_button", None)
            assert button is not None, f"no button for {kind}"
            megabytes = round(total_bytes(entries_of_kind(kind)) / 1e6)
            assert f"{megabytes} MB" in button.toolTip(), (
                f"the {kind} button does not state its own cost")

    def test_it_is_not_on_a_module_with_no_such_slots(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen

        other = AppScreen("mask")
        qtbot.addWidget(other)

        assert not hasattr(other, "_example_data_button")


class TestItFillsTheRightSlots:

    def test_the_plates_are_paired_score_with_count(self, screen, cached):
        screen.load_the_example_screen(download=False)
        pairs = (screen._settings_model.collect() or {}).get("paired_data")

        assert len(pairs) == 4, pairs
        for row in pairs:
            score = os.path.basename(str(row.get("score") or ""))
            count = os.path.basename(str(row.get("count") or ""))
            assert score.endswith("_dv.csv"), row
            assert "unique_combinations" in count, row
            # THE SAME PLATE ON BOTH SIDES. `add_paths_for_side` re-proposes
            # the table from filename tokens, and a pairing that crossed
            # plates would be a regression fitted on mismatched wells.
            assert score[5] == count[6], f"crossed plates: {row}"

    def test_it_goes_through_paired_data_not_the_legacy_lists(self, screen,
                                                              cached):
        """`count_data`/`score_data` are the legacy flat shape and are not on
        the panel at all -- filling them put the paths where nothing showed
        them."""
        screen.load_the_example_screen(download=False)
        collected = screen._settings_model.collect() or {}

        assert collected.get("paired_data")

    def test_it_says_where_everything_went(self, screen, cached):
        screen.load_the_example_screen(download=False)
        said = screen._console.as_text()

        assert "Paired" in said and "Input Tables" in said, (
            "filling two file fields silently is indistinguishable from a "
            "button that did nothing")

    def test_a_second_press_downloads_nothing(self, screen, cached):
        first = screen.load_the_example_screen(download=False)
        second = screen.load_the_example_screen(download=False)

        assert first["counts"] == second["counts"]


class TestItFailsOutLoud:

    def test_no_cache_and_no_network_says_so_and_changes_nothing(
            self, screen, tmp_path, monkeypatch):
        monkeypatch.setenv("SPACR_EXAMPLE_DATA", str(tmp_path / "empty"))

        out = screen.load_the_example_screen(download=False)

        assert out == {}
        assert "not cached" in screen._console.as_text()
        assert not (screen._settings_model.collect() or {}).get("paired_data")

    def test_the_button_comes_back_after_a_failure(self, screen, tmp_path,
                                                   monkeypatch):
        """A button left disabled and mid-progress is a dead control."""
        monkeypatch.setenv("SPACR_EXAMPLE_DATA", str(tmp_path / "empty"))

        screen.load_the_example_screen(download=False)

        assert screen._example_data_button.isEnabled()
        # RENAMED BY b110a1dfa to "Load test data…". A user with no data of
        # their own was being offered "the example screen", which names a
        # thing in the program rather than the thing they want.
        assert screen._example_data_button.text() == "Load test data…"


class TestTheDownloadLeavesTheWindowFree:
    """The buttons fetch on a worker thread, with coalesced progress.

    Lesson 13 froze for most of a second per file while the example screen
    downloaded, because the download, its digest check and the progress
    pump all ran on the GUI thread.
    """

    @pytest.fixture
    def flooding_fetch(self, monkeypatch, cached):
        import threading

        from spacr import example_data

        real = example_data.fetch
        calls = []

        def fetch(*args, progress=None, **kwargs):
            calls.append(threading.get_ident())
            for seen in range(500):
                progress("plate1_dv.csv", seen, 1000)
            progress("plate1_dv.csv", 1000, 1000)
            return real(*args, **kwargs)

        monkeypatch.setattr(example_data, "fetch", fetch)
        return calls

    def test_the_fetch_runs_off_the_gui_thread(self, screen, flooding_fetch,
                                               qtbot):
        import threading

        button = screen._example_data_button
        button.click()

        assert not button.isEnabled(), "the press did not mark the fetch"
        qtbot.waitUntil(button.isEnabled, timeout=20000)
        assert flooding_fetch and flooding_fetch[0] != threading.get_ident()
        assert button.text() == "Load test data…"
        pairs = (screen._settings_model.collect() or {}).get("paired_data")
        assert len(pairs) == 4, pairs

    def test_progress_is_coalesced_and_keeps_the_last_count(
            self, screen, flooding_fetch, qtbot, monkeypatch):
        shown = []
        monkeypatch.setattr(screen, "EXAMPLE_PROGRESS_INTERVAL_S", 60.0)
        monkeypatch.setattr(
            screen, "_say_the_download_is_moving",
            lambda name, seen, total, pump=True: shown.append(
                (name, seen, total, pump)))

        screen.load_the_example_screen(in_background=True)
        qtbot.waitUntil(screen._example_data_button.isEnabled,
                        timeout=20000)

        assert len(shown) == 2, shown
        assert shown[-1] == ("plate1_dv.csv", 1000, 1000, False)

    def test_a_second_press_while_fetching_starts_nothing(
            self, screen, flooding_fetch, qtbot):
        screen.load_the_example_screen(in_background=True)
        screen.load_the_example_screen(in_background=True)
        qtbot.waitUntil(screen._example_data_button.isEnabled,
                        timeout=20000)

        assert len(flooding_fetch) == 1

    def test_the_results_are_prepared_before_the_run(
            self, screen, flooding_fetch, qtbot, monkeypatch):
        """With the tables in, the results part is readied for the Run.

        Built at Run instead, its steps shared the interpreter lock with
        the pipeline and froze lesson 13 for up to a second.
        """
        import threading

        from spacr.qt.screens import app_screen

        imported_on = []
        real = app_screen.AppScreen._import_the_modules

        def recorded(names):
            imported_on.append(threading.get_ident())
            return real(names)

        monkeypatch.setattr(app_screen.AppScreen, "_import_the_modules",
                            staticmethod(recorded))
        part = app_screen._REGRESSION_RESULTS
        assert screen._part_is_owed(part)

        screen.load_the_example_screen(in_background=True)
        qtbot.waitUntil(lambda: not screen._part_is_owed(part),
                        timeout=30000)

        assert imported_on and imported_on[0] != threading.get_ident()
        assert screen._results_panel_if_built() is not None
