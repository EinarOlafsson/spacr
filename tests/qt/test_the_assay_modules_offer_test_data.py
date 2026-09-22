"""Replication, Recruitment and Invasion fetch their own test data.

Item 463. Each assay reads a measured plate, ``measurements/measurements.db``.
Replication and Recruitment ship a slice of a real screen's database;
Invasion's is SYNTHETIC, by the maintainer's choice, and has to say so. These tests never reach the network: the download is
replaced, or the HTTP response is.
"""
from __future__ import annotations

import csv
import io
import sqlite3
import tarfile
from pathlib import Path

import pytest

from spacr.example_archives import (DATASET_PLACEHOLDER, EXAMPLE_ARCHIVES,
                                    INVASION_EXAMPLE_REPO,
                                    RECRUITMENT_EXAMPLE_REPO,
                                    REPLICATION_EXAMPLE_REPO, example_set,
                                    example_set_folder, example_plate_folder)
from spacr.qt import assay_examples
from spacr.qt.screens.app_screen import EXAMPLE_DATA_SECTIONS

SHIPPED = {
    "replication": {"src": DATASET_PLACEHOLDER, "vacuole_key": "spatial",
                    "min_parasite_area": "400",
                    "pathogen_types": "['nc', 'pc']"},
    "recruitment": {"src": DATASET_PLACEHOLDER, "channel_of_interest": "1",
                    "pathogen_types": "['nc', 'pc']",
                    "cell_types": "['THP1']"},
    "invasion": {"src": DATASET_PLACEHOLDER, "outside_channel": "3",
                 "total_channel": "2", "stain_baseline_wells": "['c1']",
                 "pathogen_types": "['vehicle', 'inhibitor']"},
}


def _unpack_a_published_copy(folder: Path, key: str, *, src=None) -> Path:
    """Lay out what the archive leaves behind, settings last."""
    (folder / "measurements").mkdir(parents=True, exist_ok=True)
    sqlite3.connect(str(folder / "measurements" / "measurements.db")).close()
    (folder / "settings").mkdir(exist_ok=True)
    with (folder / "settings" / f"{key}_settings.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        for name, value in SHIPPED[key].items():
            if name == "src":
                value = str(folder) if src is None else src
            writer.writerow([name, value])
    return folder


def _field(screen, key):
    return screen._settings_model._widgets.get(key)


@pytest.mark.parametrize("key, repo", [
    ("replication", REPLICATION_EXAMPLE_REPO),
    ("recruitment", RECRUITMENT_EXAMPLE_REPO),
    ("invasion", INVASION_EXAMPLE_REPO)])
def test_each_set_is_published_in_its_own_repo_and_folder(key, repo):
    """Both ship measurements/measurements.db, as the shared plate's Annotate
    set does, so unpacking either into that plate would overwrite it."""
    chosen = example_set(key)
    assert chosen.repo == repo
    assert EXAMPLE_ARCHIVES[repo] == f"spacr-example-{key}.tar"
    assert chosen.folder == key
    assert example_set_folder(key) == example_plate_folder().parent / key
    assert f"settings/{key}_settings.csv" in chosen.markers
    assert 1_000_000 < chosen.bytes < 200_000_000


def test_the_buttons_land_beside_src():
    assert EXAMPLE_DATA_SECTIONS["replication"] == "Assay Inputs"
    assert EXAMPLE_DATA_SECTIONS["recruitment"] == "Data source"


def test_the_invasion_set_says_it_is_synthetic_before_it_is_fetched():
    """No real two-colour acquisition exists; the maintainer chose synthetic
    data, clearly labelled. The label has to be where the choice is made:
    in the button's tooltip and in the set's summary, not only on the card."""
    assert EXAMPLE_DATA_SECTIONS["invasion"] == "Assay Inputs"
    assert "SYNTHETIC" in example_set("invasion").summary
    assert "SYNTHETIC" in assay_examples._tooltip("invasion")
    assert "synthetic" in assay_examples._title("invasion")
    for key in ("replication", "recruitment"):
        assert "SYNTHETIC" not in assay_examples._tooltip(key)


@pytest.mark.parametrize("key", assay_examples.ASSAY_EXAMPLE_KEYS)
def test_the_button_is_built(qtbot, qt_theme_applied, key):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(key)
    qtbot.addWidget(screen)
    button = getattr(screen, "_assay_example_button", None)
    assert button is not None
    assert button.text() == "Load test data…"
    assert "MB" in button.toolTip()
    assert getattr(screen, "_example_images_button", None) is None


@pytest.mark.parametrize("key", assay_examples.ASSAY_EXAMPLE_KEYS)
def test_a_download_fills_src_and_the_shipped_settings(
        qtbot, qt_theme_applied, tmp_path, key):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(key)
    qtbot.addWidget(screen)
    folder = tmp_path / key
    calls = []

    def fake_download(parent, dest, on_done):
        calls.append(Path(dest))
        assert not screen._assay_example_button.isEnabled()
        _unpack_a_published_copy(Path(dest), key)
        on_done(object(), "")

    placed = assay_examples.load_the_assay_example(
        screen, ask=fake_download, folder=folder)

    assert calls == [folder]
    assert placed == {"src": str(folder)}
    assert _field(screen, "src").text() == str(folder)
    settings = screen._settings_model.collect()
    expected_types = (["vehicle", "inhibitor"] if key == "invasion"
                      else ["nc", "pc"])
    assert settings["pathogen_types"] == expected_types
    if key == "replication":
        assert settings["vacuole_key"] == "spatial"
        assert float(settings["min_parasite_area"]) == 400
    elif key == "invasion":
        assert settings["outside_channel"] == 3
        assert settings["total_channel"] == 2
        assert settings["stain_baseline_wells"] == ["c1"]
    else:
        assert settings["channel_of_interest"] == 1
        assert settings["cell_types"] == ["THP1"]
    assert screen._assay_example_button.isEnabled()
    assert screen._assay_example_button.text() == "Load test data…"


def test_a_cached_copy_is_used_without_a_request(qtbot, qt_theme_applied,
                                                 tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("recruitment")
    qtbot.addWidget(screen)
    folder = _unpack_a_published_copy(tmp_path / "recruitment", "recruitment")

    def must_not_download(*_args):
        raise AssertionError("a cached copy was downloaded again")

    placed = assay_examples.load_the_assay_example(
        screen, ask=must_not_download, folder=folder)
    assert placed == {"src": str(folder)}


def test_a_half_unpacked_copy_is_fetched_again(qtbot, qt_theme_applied,
                                               tmp_path):
    """The settings file is the archive's last member, so a copy without it
    is a transfer that died, and reading it as present would never repair
    it."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("replication")
    qtbot.addWidget(screen)
    folder = tmp_path / "replication"
    (folder / "measurements").mkdir(parents=True)
    (folder / "measurements" / "measurements.db").write_bytes(b"")
    calls = []

    assay_examples.load_the_assay_example(
        screen, ask=lambda *a: calls.append(a), folder=folder)
    assert len(calls) == 1


def test_a_failed_download_says_why_and_gives_the_button_back(
        qtbot, qt_theme_applied, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("replication")
    qtbot.addWidget(screen)
    before = _field(screen, "src").text()
    notices = []
    screen._console.append_notice = (
        lambda template, **kw: notices.append(template.format(**kw)))

    placed = assay_examples.load_the_assay_example(
        screen, ask=lambda _p, _d, on_done: on_done(None, "no network"),
        folder=tmp_path / "replication")

    assert placed == {}
    assert _field(screen, "src").text() == before
    assert any("no network" in n for n in notices)
    assert screen._assay_example_button.isEnabled()


def test_a_download_that_cannot_start_is_reported_not_raised(
        qtbot, qt_theme_applied, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("recruitment")
    qtbot.addWidget(screen)
    notices = []
    screen._console.append_notice = (
        lambda template, **kw: notices.append(template.format(**kw)))

    def broken(*_args):
        raise ConnectionError("unreachable")

    assert assay_examples.load_the_assay_example(
        screen, ask=broken, folder=tmp_path / "recruitment") == {}
    assert any("huggingface.co" in n for n in notices)


@pytest.mark.parametrize("key, repo", [
    ("replication", REPLICATION_EXAMPLE_REPO),
    ("recruitment", RECRUITMENT_EXAMPLE_REPO),
    ("invasion", INVASION_EXAMPLE_REPO)])
def test_the_worker_is_pointed_at_the_modules_own_repo(monkeypatch, tmp_path,
                                                       key, repo):
    seen = {}

    def fake_dialog(parent, dest, on_done, *, worker_factory, title):
        seen["worker"] = worker_factory(dest)
        seen["title"] = title

    monkeypatch.setattr(assay_examples, "download_toxo_mito_demo",
                        fake_dialog)
    assay_examples.download_assay_example(None, key, tmp_path, None)
    assert seen["worker"].repo == repo
    assert "test data" in seen["title"]


def test_the_worker_unpacks_the_archive_and_fills_in_the_path(monkeypatch,
                                                              tmp_path):
    """The real worker, with only the HTTP response replaced."""
    staged = _unpack_a_published_copy(tmp_path / "staged", "replication",
                                      src=DATASET_PLACEHOLDER)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for member in ("measurements/measurements.db",
                       "settings/replication_settings.csv"):
            tar.add(staged / member, arcname=member)
    payload = buffer.getvalue()

    class Response:
        headers = {"Content-Length": str(len(payload))}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield payload

    urls = []

    def fake_get(url, **_kw):
        urls.append(url)
        return Response()

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    dest = tmp_path / "replication"
    worker = assay_examples._AssayTarWorker(dest, REPLICATION_EXAMPLE_REPO)
    outcome = []
    worker.finished.connect(lambda ok, *rest: outcome.append((ok, rest)))
    worker.run()

    assert outcome and outcome[0][0] is True, outcome
    assert urls == [f"https://huggingface.co/datasets/{REPLICATION_EXAMPLE_REPO}"
                    "/resolve/main/spacr-example-replication.tar?download=true"]
    assert example_set("replication").is_present(dest)
    text = (dest / "settings" / "replication_settings.csv").read_text()
    assert DATASET_PLACEHOLDER not in text
    assert str(dest) in text
    assert not (dest / "spacr-example-replication.tar").exists()
