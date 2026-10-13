"""OPS and Align & Stitch each get a "Load test data…" button that works.

Item 461. The maintainer: "For OPS use a small sample of the ops screen i have
on disk" and "allign and stitch can also use a small segment of the OPS data".

NO NETWORK ANYWHERE IN HERE. `requests.get` is replaced by a server of a local
archive, read in chunks from disk. Two kinds of archive are served:

* a SYNTHETIC one built in ``tmp_path`` -- a few tiny tiles with the real
  names and the real manifest shape -- so the wiring is proved on every
  machine;
* the REAL staged archives, when ``SPACR_OPS_EXAMPLE_UPLOAD`` names the
  folder they were built in (``/mnt/wd4tb/spacr_testdata/ops/upload`` on the
  maintainer's workstation). Those tests are skipped elsewhere, and they are
  the ones that prove the published bytes plan and stitch.
"""
from __future__ import annotations

import csv
import os
import tarfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.example_archives import (EXAMPLE_ARCHIVES, EXAMPLE_SETS,
                                    OPS_EXAMPLE_REPO, STITCH_EXAMPLE_REPO,
                                    example_plate_folder, example_set,
                                    example_set_folder)
from spacr.qt import ops_stitch_demo as demo
from spacr.qt.hf_download import DownloadResult

UPLOAD = Path(os.environ.get("SPACR_OPS_EXAMPLE_UPLOAD",
                             "/mnt/wd4tb/spacr_testdata/ops/upload"))

_STITCH_SITES = (165, 166, 167, 186, 187, 188, 207, 208, 209)


def _stitch_tile_name(site: int) -> str:
    return f"tiles/10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-{site}.tif"


def _write_sample(folder: Path, which: str) -> None:
    """Lay a tiny sample out as its archive unpacks, manifest last."""
    rng = np.random.default_rng(461)
    names = []
    if which == "stitch":
        for site in _STITCH_SITES:
            names.append(_stitch_tile_name(site))
    else:
        for site in (331, 332):
            names.append(
                f"sequencing/c1/10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-{site}.tif")
            for cycle in range(2, 12):
                for channel in ("CY3", "A594", "CY5", "CY7"):
                    names.append(f"sequencing/c{cycle}/10X_c{cycle}_A1_"
                                 f"{channel}_Site-{site}.tif")
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, rng.integers(0, 4000, (8, 8), dtype=np.uint16))
    if which == "ops":
        library = folder / "library" / "pool10_prefixes.csv"
        library.parent.mkdir(parents=True, exist_ok=True)
        library.write_text("prefix\nACGTACGTACG\n", encoding="utf-8")
        names.append("library/pool10_prefixes.csv")
    with (folder / demo.MANIFEST_NAME).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256", "source_path"])
        for name in names:
            writer.writerow([name, (folder / name).stat().st_size, "", ""])


def _synthetic_archive(tmp_path: Path, which: str) -> Path:
    """The sample as a tar, manifest as the LAST member."""
    source = tmp_path / f"source_{which}"
    source.mkdir()
    _write_sample(source, which)
    archive = tmp_path / EXAMPLE_ARCHIVES[
        OPS_EXAMPLE_REPO if which == "ops" else STITCH_EXAMPLE_REPO]
    with tarfile.open(archive, "w") as tar:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.name != demo.MANIFEST_NAME:
                tar.add(path, arcname=path.relative_to(source).as_posix())
        tar.add(source / demo.MANIFEST_NAME, arcname=demo.MANIFEST_NAME)
    return archive


def _real_archive(which: str) -> Path:
    """The staged upload archive, or a skip when it is not on this machine."""
    repo = OPS_EXAMPLE_REPO if which == "ops" else STITCH_EXAMPLE_REPO
    path = UPLOAD / repo.split("/")[1] / EXAMPLE_ARCHIVES[repo]
    if not path.is_file():
        pytest.skip(f"the staged {which} archive is not on this machine")
    return path


class _Hub:
    """`requests.get` serving one archive from disk in chunks."""

    def __init__(self, monkeypatch, archive: Path, *, error=None):
        import requests

        self.requested = []
        size = archive.stat().st_size
        hub = self

        class _Response:
            headers = {"Content-Length": str(size)}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                with archive.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(chunk_size), b""):
                        yield chunk

        def fake_get(url, **_kwargs):
            hub.requested.append(url)
            if error is not None:
                raise error
            return _Response()

        monkeypatch.setattr(requests, "get", fake_get)


def _unpack_with_the_real_worker(monkeypatch, archive: Path, dest: Path,
                                 which: str):
    """Run the set's own tar worker against ``archive``; return the hub."""
    hub = _Hub(monkeypatch, archive)
    worker = demo._WORKERS[which](dest)
    finished = []
    worker.finished.connect(lambda *args: finished.append(args))
    worker.run()
    assert finished and finished[0][0] is True, finished
    return hub


class _FakeDownload:
    """The `ask` seam: unpacks a local archive, or reports a failure."""

    def __init__(self, archive=None, outcome="ok"):
        self.archive = archive
        self.outcome = outcome
        self.calls = 0

    def __call__(self, _parent, dest, on_done):
        from spacr.example_archives import extract_example_archive

        self.calls += 1
        if self.outcome != "ok":
            on_done(None, self.outcome)
            return
        extract_example_archive(self.archive, dest)
        on_done(DownloadResult(dataset_path=Path(dest),
                               settings_path=Path(dest) / "settings"), "")


def test_both_sets_are_published_example_sets_with_their_own_folder():
    for key, repo in (("ops", OPS_EXAMPLE_REPO),
                      ("stitch", STITCH_EXAMPLE_REPO)):
        example = example_set(key)
        assert example in EXAMPLE_SETS
        assert example.repo == repo
        assert example.archive == EXAMPLE_ARCHIVES[repo]
        assert example.archive.endswith(".tar")
        folder = example_set_folder(key)
        assert folder != example_plate_folder()
        assert example_plate_folder() not in folder.parents, (
            f"the {key} sample would unpack into the shared plate")


def test_the_three_plate_sets_still_share_the_plate_folder():
    for key in ("mask", "measure", "annotate"):
        assert example_set_folder(key) == example_plate_folder()


@pytest.mark.parametrize("which,repo", [("ops", OPS_EXAMPLE_REPO),
                                         ("stitch", STITCH_EXAMPLE_REPO)])
def test_the_worker_fetches_its_own_archive_and_unpacks_it(
        tmp_path, monkeypatch, which, repo):
    archive = _synthetic_archive(tmp_path, which)
    dest = tmp_path / "dest"
    hub = _unpack_with_the_real_worker(monkeypatch, archive, dest, which)
    assert hub.requested == [
        f"https://huggingface.co/datasets/{repo}/resolve/main/"
        f"{EXAMPLE_ARCHIVES[repo]}?download=true"]
    assert demo.is_present(dest)
    assert not (dest / archive.name).exists(), "the archive was kept"


def test_a_half_unpacked_sample_is_not_a_cache_hit(tmp_path):
    folder = tmp_path / "s"
    folder.mkdir()
    _write_sample(folder, "stitch")
    assert demo.is_present(folder)
    (folder / _stitch_tile_name(208)).unlink()
    assert not demo.is_present(folder)
    assert not demo.is_present(tmp_path / "nothing")


def test_the_ops_settings_name_the_sample_and_keep_outputs_apart(tmp_path):
    values = demo.ops_settings_for(tmp_path)
    assert values["genotype_source"] == str(tmp_path / "sequencing")
    assert values["ops_library"] == str(
        tmp_path / "library" / "pool10_prefixes.csv")
    assert values["dst_root"] == str(tmp_path / "ops_output")
    assert values["plate"] == demo.OPS_PLATE
    assert values["phenotype_source"] is None


@pytest.fixture
def align_screen(qtbot):
    from spacr.qt.screens.align import AlignScreen

    screen = AlignScreen(threaded=False)
    qtbot.addWidget(screen)
    return screen


def test_align_has_the_button_beside_the_tile_folder(align_screen):
    button = align_screen._btn_test_data
    assert button.text() == "Load test data…"
    assert "nine" in button.toolTip()
    assert button.parentWidget() is align_screen._btn_pick_src.parentWidget()


def test_align_download_fills_the_grid_order_and_overlap(align_screen,
                                                         tmp_path):
    fake = _FakeDownload(_synthetic_archive(tmp_path, "stitch"))
    folder = tmp_path / "cache" / "align_stitch"

    assert demo.load_align_test_data(align_screen, ask=fake,
                                     folder=folder) is False

    assert fake.calls == 1
    settings = align_screen.settings()
    assert settings["src"] == str(folder / "tiles")
    assert settings["dst"] == str(folder / "stitched")
    assert settings["grid"] == demo.STITCH_GRID
    assert settings["order"] == demo.STITCH_ORDER
    assert settings["overlap"] == pytest.approx(demo.STITCH_OVERLAP)
    assert align_screen._btn_test_data.isEnabled()
    assert align_screen._btn_plan.isEnabled()
    assert "Press Plan" in align_screen.status_text()

    assert demo.load_align_test_data(align_screen, ask=fake,
                                     folder=folder) is True
    assert fake.calls == 1, "the cached copy was fetched again"


def test_align_offline_says_why_and_fills_nothing(align_screen, tmp_path):
    fake = _FakeDownload(outcome="Could not reach huggingface.co")
    assert demo.load_align_test_data(align_screen, ask=fake,
                                     folder=tmp_path / "x") is False
    assert "Could not reach huggingface.co" in align_screen.status_text()
    assert align_screen.last_error
    assert align_screen.settings()["src"] is None
    assert align_screen._btn_test_data.isEnabled()


def test_a_download_that_cannot_start_is_reported_not_raised(align_screen,
                                                             tmp_path):
    def broken(_parent, _dest, _on_done):
        raise PermissionError("read-only cache")

    assert demo.load_align_test_data(align_screen, ask=broken,
                                     folder=tmp_path / "x") is False
    assert "read-only cache" in align_screen.status_text()


@pytest.mark.parametrize("contents", [b"\xff\xfe", b"path,bytes\n,10\n",
                                       b"unrecognised\nimage.tif\n"])
def test_unreadable_or_unnamed_manifest_rows_never_count_as_a_cached_sample(tmp_path, contents):
    (tmp_path / demo.MANIFEST_NAME).write_bytes(contents)
    (tmp_path / "image.tif").write_bytes(b"present")
    assert demo.listed_files(tmp_path) == []
    assert not demo.is_present(tmp_path)


@pytest.mark.parametrize("outcome", ["cancelled", "unknown", "incomplete"])
def test_pending_download_preserves_settings_and_failure_allows_retry(align_screen, tmp_path, outcome):
    before = align_screen.settings()
    folder = tmp_path / "cache"
    pending = []

    def delayed(parent, dest, done):
        assert parent is align_screen
        assert dest == folder
        pending.append(done)

    assert not demo.load_align_test_data(align_screen, ask=delayed, folder=folder)
    assert align_screen.settings() == before
    assert not align_screen._btn_test_data.isEnabled()
    assert align_screen._btn_test_data.text() == "Fetching test data…"
    if outcome == "cancelled":
        pending.pop()(None, demo.CANCELLED)
        assert "cancelled" in align_screen.status_text()
        assert not align_screen.last_error
    elif outcome == "unknown":
        pending.pop()(None, "")
        assert "unknown error" in align_screen.status_text()
        assert align_screen.last_error
    else:
        pending.pop()(DownloadResult(dataset_path=folder, settings_path=folder / "settings"), "")
        assert "incomplete" in align_screen.status_text()
        assert align_screen.last_error
    assert align_screen.settings() == before
    assert align_screen._btn_test_data.isEnabled()
    assert align_screen._btn_test_data.text() == "Load test data…"
    retry = _FakeDownload(_synthetic_archive(tmp_path, "stitch"))
    assert not demo.load_align_test_data(align_screen, ask=retry, folder=folder)
    assert retry.calls == 1
    assert demo.is_present(folder)
    assert align_screen.settings()["src"] == str(folder / "tiles")
    assert "Press Plan" in align_screen.status_text()
    assert not align_screen.last_error


@pytest.mark.parametrize("which", ["ops", "stitch"])
def test_default_downloader_uses_the_matching_worker_and_shared_progress_dialog(monkeypatch, tmp_path, which):
    pending, reports, used = [], [], []
    folder = tmp_path / which
    parent = object()
    monkeypatch.setattr(demo, "example_set_folder", lambda key: tmp_path / key)

    def progress_dialog(owner, dest, done, *, worker_factory, title):
        assert owner is parent
        assert dest == folder
        assert worker_factory is demo._WORKERS[which]
        assert title == ("Downloading the OPS test data" if which == "ops"
                         else "Downloading the Align & Stitch test data")
        pending.append(done)

    monkeypatch.setattr(demo, "download_toxo_mito_demo", progress_dialog)
    report = lambda message, error: reports.append((message, error))
    assert not demo.load_the_test_data(which, use=used.append, report=report, parent=parent)
    assert used == []
    assert len(reports) == 1 and reports[0][1] is False
    folder.mkdir()
    _write_sample(folder, which)
    pending.pop()(DownloadResult(dataset_path=folder, settings_path=folder / "settings"), "")
    assert used == [folder]
    assert demo.load_the_test_data(which, use=used.append, report=report, parent=parent)
    assert used == [folder, folder]
    assert pending == [], "a complete cache must not open another download"


@pytest.fixture
def ops_screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("ops")
    qtbot.addWidget(screen)
    return screen


def test_the_ops_panel_has_the_button(ops_screen):
    button = ops_screen._ops_example_button
    assert button.text() == "Load test data…"
    assert "eleven cycles" in button.toolTip()


def test_the_ops_download_fills_the_settings_run_reads(ops_screen, tmp_path):
    fake = _FakeDownload(_synthetic_archive(tmp_path, "ops"))
    folder = tmp_path / "cache" / "ops_screen"

    applied = ops_screen.load_the_ops_example(ask=fake, folder=folder)

    assert fake.calls == 1
    assert applied["genotype_source"] == str(folder / "sequencing")
    collected = ops_screen._settings_model.collect()
    assert collected["genotype_source"] == str(folder / "sequencing")
    assert collected["ops_library"] == str(
        folder / "library" / "pool10_prefixes.csv")
    assert collected["dst_root"] == str(folder / "ops_output")
    assert collected["plate"] == demo.OPS_PLATE
    assert ops_screen._ops_example_button.isEnabled()


def test_the_ops_engine_indexes_the_unpacked_sample(tmp_path, monkeypatch):
    """What Run is handed is a folder the engine recognises: one well, two
    sites, eleven cycles, the nuclear plane in cycle 1."""
    from spacr.ops_engine import _NUCLEAR, _index_tiles, _plane_sources

    dest = tmp_path / "ops"
    _unpack_with_the_real_worker(
        monkeypatch, _synthetic_archive(tmp_path, "ops"), dest, "ops")
    index = _index_tiles(demo.ops_settings_for(dest)["genotype_source"],
                         "cycled")
    assert sorted(index) == ["A1"]
    assert sorted(index["A1"]) == list(range(1, 12))
    assert sorted(index["A1"][1]) == [331, 332]
    assert _NUCLEAR in _plane_sources(index["A1"][1][332])
    for cycle in range(2, 12):
        assert sorted(index["A1"][cycle][331]) == ["A594", "CY3", "CY5",
                                                   "CY7"]


def test_the_real_stitch_archive_plans_every_tile(align_screen, tmp_path,
                                                  monkeypatch):
    """The staged archive's own bytes, through the worker, the button's
    settings and the screen's Plan: every tile registers."""
    archive = _real_archive("stitch")
    dest = tmp_path / "stitch"
    _unpack_with_the_real_worker(monkeypatch, archive, dest, "stitch")
    assert demo.is_present(dest)

    assert demo.load_align_test_data(align_screen, folder=dest) is True
    assert align_screen.build_plan() is True
    plan = align_screen.plan()
    assert len(plan.placements) == 9
    assert plan.n_nominal == 0, align_screen.report_text()
    assert not plan.unplaced


def test_the_real_ops_archive_unpacks_to_what_the_engine_reads(tmp_path,
                                                               monkeypatch):
    from spacr.ops_engine import _index_tiles

    archive = _real_archive("ops")
    dest = tmp_path / "ops"
    _unpack_with_the_real_worker(monkeypatch, archive, dest, "ops")
    assert demo.is_present(dest)
    settings = demo.ops_settings_for(dest)
    index = _index_tiles(settings["genotype_source"], "cycled")
    assert sorted(index["A1"]) == list(range(1, 12))
    assert sorted(index["A1"][1]) == [331, 332]
    with open(settings["ops_library"], newline="") as handle:
        prefixes = [row["prefix"] for row in csv.DictReader(handle)]
    assert len(prefixes) == 20445
    assert {len(p) for p in prefixes} == {11}


def _live_download(which: str, dest: Path) -> None:
    """Fetch a set from the published repository, or skip.

    Only when ``SPACR_OPS_EXAMPLE_LIVE=1``: this goes to huggingface.co for
    about 0.6 GB, which is no test to run by default.
    """
    if os.environ.get("SPACR_OPS_EXAMPLE_LIVE") != "1":
        pytest.skip("set SPACR_OPS_EXAMPLE_LIVE=1 to fetch the published sets")
    worker = demo._WORKERS[which](dest)
    finished = []
    worker.finished.connect(lambda *args: finished.append(args))
    worker.run()
    assert finished and finished[0][0] is True, finished


def test_the_published_stitch_set_plans_every_tile(align_screen, tmp_path):
    dest = tmp_path / "stitch"
    _live_download("stitch", dest)
    assert demo.is_present(dest)
    assert demo.load_align_test_data(align_screen, folder=dest) is True
    assert align_screen.build_plan() is True
    plan = align_screen.plan()
    assert len(plan.placements) == 9
    assert plan.n_nominal == 0, align_screen.report_text()


def test_the_published_ops_set_is_what_the_engine_reads(tmp_path):
    from spacr.ops_engine import _index_tiles

    dest = tmp_path / "ops"
    _live_download("ops", dest)
    assert demo.is_present(dest)
    index = _index_tiles(demo.ops_settings_for(dest)["genotype_source"],
                         "cycled")
    assert sorted(index["A1"]) == list(range(1, 12))
    assert sorted(index["A1"][1]) == [331, 332]
