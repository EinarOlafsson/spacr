"""The Model Zoo screen's rows: their state words, tooltips and headings.

Pinned here, each as what the user sees:

* a Cellpose-DINO model whose backend is present says whether it is on
  disk, and a bundled model with no file says "bundled";
* a tooltip leads with the scorecard headline when there is one, and is
  the plain card when the headline cannot be read;
* turning the "spaCR community" heading on re-lists with the community
  uploads, and turning another heading re-folds the table without a scan;
* switching Show alpha features re-folds a model registered as alpha;
* rows the table no longer has, and versions picked for models that are not
  listed, change nothing.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("PySide6")

from spacr import model_zoo as zoo
from spacr.qt.screens import model_zoo as mz
from spacr.qt.screens.model_zoo import ModelZooScreen
from spacr.qt.widgets import model_zoo_picker as mzp


@pytest.fixture(autouse=True)
def _isolated_run_journal(monkeypatch, tmp_path):
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


@pytest.fixture
def screen(qtbot, tmp_path):
    widget = ModelZooScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._scan_edit.setText(str(tmp_path))
    return widget


def _local(key, tmp_path):
    path = tmp_path / f"{key}.pth"
    path.write_bytes(b"x")
    return zoo.ModelEntry(key=key, name=key, path=str(path), kind="cellpose",
                          source="local", size_bytes=1)


# ---------------------------------------------------------------------------
# State words
# ---------------------------------------------------------------------------

def test_a_dino_model_with_its_backend_says_whether_it_is_here(tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(mzp, "_cellpose_dino_ready", lambda: True)
    weights = tmp_path / "dino.pth"
    weights.write_bytes(b"x")
    here = types.SimpleNamespace(kind="cellpose_dino", source="local",
                                 path=str(weights), uri="")
    away = types.SimpleNamespace(kind="cellpose_dino", source="declared",
                                 path="", uri="")
    assert mz._status_of(here) == "installed"
    assert mz._status_of(away) == "available"


def test_a_bundled_model_without_a_file_says_bundled():
    entry = types.SimpleNamespace(kind="cellpose", source="bundled", path="",
                                  uri="")
    assert mz._status_of(entry) == "bundled"


# ---------------------------------------------------------------------------
# Tooltips
# ---------------------------------------------------------------------------

class _Card:
    metrics = {"AP50": 0.9}

    def describe(self):
        return "the full card"


def test_a_tooltip_leads_with_the_headline(monkeypatch):
    from spacr import scorecard

    monkeypatch.setattr(scorecard, "headline",
                        lambda metrics, **_k: ["AP50 0.90", "F1 0.80"])
    assert mz._tooltip_prose(_Card()) == "AP50 0.90\nF1 0.80\n\nthe full card"


def test_a_headline_that_cannot_be_read_leaves_the_plain_card(monkeypatch):
    from spacr import scorecard

    def broken(metrics, **_k):
        raise ValueError("not a scorecard")

    monkeypatch.setattr(scorecard, "headline", broken)
    assert mz._tooltip_prose(_Card()) == "the full card"


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------

def test_turning_the_community_heading_on_lists_the_uploads(screen,
                                                            monkeypatch):
    asked = []

    def community(allow_network=False):
        asked.append(allow_network)
        return [zoo.ModelEntry(key="someone_v1", name="someone_v1.CP_model",
                               source="community", sha256="ab" * 32)]

    monkeypatch.setattr(zoo, "community_entries", community)
    screen.sources._headings["spaCR community"].set_on(True)
    screen._sources_changed()

    assert asked == [True]
    assert "someone_v1.CP_model" in [e.name for e in screen._entries]


def test_turning_another_heading_refolds_without_a_scan(screen, tmp_path,
                                                        monkeypatch):
    screen.set_entries([_local("alpha", tmp_path)])
    heading = zoo.source_of(screen._entries[0])
    scans = []
    monkeypatch.setattr(screen, "scan", lambda *a, **k: scans.append(k))

    screen.sources._headings[heading].set_on(False)
    screen._sources_changed()
    assert scans == []
    assert screen._table.isRowHidden(0)

    screen.sources._headings[heading].set_on(True)
    screen._sources_changed()
    assert not screen._table.isRowHidden(0)


def test_the_alpha_switch_refolds_an_alpha_model(screen, tmp_path,
                                                 monkeypatch):
    screen.set_entries([_local("alpha", tmp_path)])
    assert not screen._table.isRowHidden(0)

    monkeypatch.setattr(mz, "_model_is_alpha_hidden", lambda entry: True)
    screen._refresh_alpha_visibility()
    assert screen._table.isRowHidden(0)

    monkeypatch.setattr(mz, "_model_is_alpha_hidden", lambda entry: False)
    screen._refresh_alpha_visibility()
    assert not screen._table.isRowHidden(0)


def test_without_a_heading_strip_nothing_is_folded(screen, tmp_path):
    screen.set_entries([_local("alpha", tmp_path)])
    screen._table.setRowHidden(0, False)
    del screen.sources
    screen._apply_source_filter()
    assert not screen._table.isRowHidden(0)


# ---------------------------------------------------------------------------
# Rows the table no longer has
# ---------------------------------------------------------------------------

def test_a_row_the_table_no_longer_has_is_left_alone(screen, tmp_path):
    screen.set_entries([_local("alpha", tmp_path), _local("beta", tmp_path)])
    lost = screen._table.item(1, 0).data(mz.Qt.UserRole)
    screen._table.removeRow(1)

    assert screen._row_of_group(lost) is None
    screen._fill_row(lost)
    screen._apply_source_filter()
    assert screen._table.rowCount() == 1


def test_a_version_for_a_model_that_is_not_listed_is_ignored(screen,
                                                             tmp_path):
    screen.set_entries([_local("alpha", tmp_path)])
    before = [screen._table.item(0, c).text() for c in range(4)]
    screen._version_picked(7, 0)
    screen._version_picked(-1, 0)
    assert [screen._table.item(0, c).text() for c in range(4)] == before
