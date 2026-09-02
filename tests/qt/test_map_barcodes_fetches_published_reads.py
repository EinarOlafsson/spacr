"""Map Barcodes offers the paper's own reads, with a limit that really limits.

Asked for on 2026-09-01: a test-data button above ``src``, downloading from
NCBI, "and even better if the user could specify how many sequencing lines
from each file". The runs total 20.4 GB, so the limit is what makes the button
usable at all.

Nothing here touches the network: the picker is handed a file list.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from spacr.qt.screens.app_screen import EXAMPLE_DATA_SECTIONS
from spacr.sra import RunFile

FILES = (
    RunFile(run="SRR33531217", library="hilib_p4",
            url="https://x/SRR33531217_1.fastq.gz", mate=1,
            size_bytes=2_833_522_805, read_count=73_698_595),
    RunFile(run="SRR33531217", library="hilib_p4",
            url="https://x/SRR33531217_2.fastq.gz", mate=2,
            size_bytes=3_222_897_932, read_count=73_698_595),
)


@pytest.fixture
def picker(qtbot, tmp_path):
    from spacr.qt.widgets.sra_picker import SraPicker
    p = SraPicker(tmp_path, files=FILES)
    qtbot.addWidget(p)
    return p


def test_the_control_sits_with_the_sequencing_input():
    assert EXAMPLE_DATA_SECTIONS["map_barcodes"] == "Sequencing Input"


def test_map_barcodes_builds_the_control(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("map_barcodes")
    qtbot.addWidget(screen)
    assert hasattr(screen, "_sequencing_example_button")


def test_every_run_is_listed_and_ticked(picker):
    assert picker._list.count() == len(FILES)
    assert len(picker.chosen_files()) == len(FILES)


def test_the_listing_names_the_plate_not_only_the_run(picker):
    """SRR33531217 does not say which plate it is; hilib_p4 does."""
    assert "hilib_p4" in picker._list.item(0).text()


def test_the_estimate_is_a_small_fraction_of_the_full_size(picker):
    """The number that makes this a decision rather than a surprise."""
    picker._reads.setValue(100_000)
    small = picker._estimate.text()
    picker._whole.setChecked(True)
    whole = picker._estimate.text()
    assert "GB" in whole, whole
    assert "MB" in small, small


def test_unticking_a_run_changes_what_would_be_fetched(picker):
    picker._list.item(0).setCheckState(Qt.Unchecked)
    assert len(picker.chosen_files()) == len(FILES) - 1


def test_the_whole_file_tick_means_no_limit(picker):
    picker._reads.setValue(50_000)
    assert picker.max_reads() == 50_000
    picker._whole.setChecked(True)
    assert picker.max_reads() is None
    assert not picker._reads.isEnabled(), (
        "a live limit box beside a ticked 'whole file' says two things at once")


def test_nothing_selected_disables_the_download(picker):
    for row in range(picker._list.count()):
        picker._list.item(row).setCheckState(Qt.Unchecked)
    assert not picker._download.isEnabled()


def test_the_screen_points_src_at_the_downloaded_folder(qtbot, tmp_path):
    """A download that does not set the source leaves the user to find it."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("map_barcodes")
    qtbot.addWidget(screen)

    class Done:
        written = ["/tmp/a_1.fastq.gz", "/tmp/a_2.fastq.gz"]
        def exec(self): return 1

    result = screen.load_the_sequencing_example(picker=Done())
    assert result["src"].endswith("sequencing")
    assert len(result["files"]) == 2


def test_a_cancelled_picker_changes_nothing(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("map_barcodes")
    qtbot.addWidget(screen)

    class Cancelled:
        written = []
        def exec(self): return 0

    assert screen.load_the_sequencing_example(picker=Cancelled()) == {}


def test_an_unreachable_archive_is_reported_not_raised(qtbot, tmp_path, monkeypatch):
    """A failure here must not take the module down with it."""
    import spacr.qt.widgets.sra_picker as mod

    def boom():
        raise OSError("no route to host")

    monkeypatch.setattr(mod, "runs_for", boom)
    p = mod.SraPicker(tmp_path)
    qtbot.addWidget(p)
    assert not p._download.isEnabled()
    assert "no route to host" in p._blurb.text()
