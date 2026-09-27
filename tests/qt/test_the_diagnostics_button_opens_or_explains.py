"""The regression Diagnostics button opens the panels, or says why it cannot.

Pinned behaviour of :class:`spacr.qt.screens.regression.DiagnosticsOpener`:

* with no diagnostics on disk it explains that they are written when a
  regression finishes, and opens nothing;
* with a diagnostics folder it opens that folder;
* when the run left a note that the residual panels are not available, the
  note is shown before the folder is opened;
* files in the folder other than the summary are not read for the verdict.
"""
from __future__ import annotations

import csv
import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox, QWidget  # noqa: E402

from spacr.qt.screens import regression as R  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def opened(qtbot, tmp_path, monkeypatch):
    screen = QWidget()
    qtbot.addWidget(screen)
    monkeypatch.setattr(R, "project_path", lambda _screen: str(tmp_path))
    shown, urls = [], []
    monkeypatch.setattr(
        QMessageBox, "information",
        lambda parent, title, text, *a, **k: shown.append((title, text)))
    monkeypatch.setattr(R.QDesktopServices, "openUrl",
                        lambda url: urls.append(url.toLocalFile()) or True)
    return R.DiagnosticsOpener(screen), tmp_path, shown, urls


def _diagnostics(root):
    folder = os.path.join(str(root), R.RESULTS_DIRNAME, "run1",
                          R.DIAGNOSTICS_DIRNAME)
    os.makedirs(folder, exist_ok=True)
    return folder


def test_with_nothing_on_disk_it_explains_and_opens_nothing(opened):
    opener, _root, shown, urls = opened

    opener.open()

    assert len(shown) == 1
    assert "no regression diagnostics yet" in shown[0][1]
    assert urls == []


def test_with_a_diagnostics_folder_it_opens_the_folder(opened):
    opener, root, shown, urls = opened
    folder = _diagnostics(root)

    opener.open()

    assert shown == []
    assert urls == [folder]


def test_a_missing_panels_note_is_shown_before_the_folder_opens(opened):
    opener, root, shown, urls = opened
    folder = _diagnostics(root)
    with open(os.path.join(folder, "residual_panels_not_available.txt"), "w",
              encoding="utf-8") as handle:
        handle.write("  This backend reports no residuals.\n")

    opener.open()

    assert [text for _title, text in shown] == [
        "This backend reports no residuals."]
    assert urls == [folder]


def test_only_the_summary_is_read_for_the_verdict(opened):
    opener, root, _shown, _urls = opened
    folder = _diagnostics(root)
    with open(os.path.join(folder, "a_panel_index.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "metric", "value"])
        writer.writerow(["suite", "verdict_level", "fail"])
    with open(os.path.join(folder, "diagnostic_summary.csv"), "w",
              newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "metric", "value"])
        writer.writerow(["suite", "verdict_level", "check"])
        writer.writerow(["suite", "verdict", "residuals: check"])

    assert opener.verdict() == ("check", "residuals: check")
