"""The Report screen's Deposit on Zenodo button is an alpha feature.

``ReportZenodoDeposit`` is registered in ``spacr.settings.ALPHA_FEATURES``,
so it is hidden until Preferences -> "Show alpha features" is on. Hiding is
display only: a deposit asked for while the button is hidden still reaches
Zenodo -- here a local fake of its API, never Zenodo or its sandbox.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr import report as rep  # noqa: E402
from spacr.settings import ALPHA_FEATURES  # noqa: E402
from tests.test_zenodo_deposit import FORM, TOKEN, make_run  # noqa: E402
from tests.zenodo_fake import FakeZenodo  # noqa: E402


@pytest.fixture
def alpha(monkeypatch):
    from spacr.qt import preferences

    state = {"on": False}
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: state["on"])
    return state


def _screen(qtbot, src=None):
    from spacr.qt.screens.report import ReportScreen

    screen = ReportScreen(threaded=False)
    qtbot.addWidget(screen)
    if src is not None:
        screen.set_source(str(src))
    return screen


def test_registered_as_alpha():
    assert ALPHA_FEATURES[579] == {"widgets": ("ReportZenodoDeposit",)}


def test_hidden_until_alpha_features_are_shown(qtbot, alpha):
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.preferences import _apply_alpha_widgets

    screen = _screen(qtbot)
    button = screen.findChild(QPushButton, "ReportZenodoDeposit")
    assert button is not None and button.isHidden()
    alpha["on"] = True
    _apply_alpha_widgets(screen)
    assert not button.isHidden()
    alpha["on"] = False
    _apply_alpha_widgets(screen)
    assert button.isHidden()


def test_the_form_defaults_to_the_sandbox(qtbot, alpha, tmp_path):
    from PySide6.QtWidgets import QCheckBox, QLineEdit

    src, _run = make_run(tmp_path)
    alpha["on"] = True
    screen = _screen(qtbot, src)
    assert screen._btn_zenodo.isEnabled()
    dialog = screen._zenodo_dialog()
    qtbot.addWidget(dialog)
    assert dialog.findChild(QCheckBox, "ZenodoSandbox").isChecked()
    assert not dialog.findChild(QCheckBox, "ZenodoPublish").isChecked()
    token = dialog.findChild(QLineEdit, "ZenodoToken")
    assert token.echoMode() == QLineEdit.Password
    assert dialog.findChild(QLineEdit, "ArchiveField_title").text() == "toxo"


def test_a_deposit_asked_for_while_hidden_still_reaches_zenodo(
        qtbot, alpha, tmp_path, monkeypatch):
    src, run = make_run(tmp_path)
    monkeypatch.setattr(rep, "_load_journal_runs",
                        lambda *a, **k: ([{"dir": run, "app_key": "measure",
                                           "start_utc": ""}], []))
    screen = _screen(qtbot, src)
    assert screen._btn_zenodo.isHidden()
    assert not screen._deposit_zenodo(str(src), "", FORM, token="")
    assert "token is needed" in screen.status_text()
    with FakeZenodo(TOKEN) as fake:
        monkeypatch.setitem(rep._ZENODO_API, "sandbox", fake.api)
        assert screen._deposit_zenodo(str(src), str(tmp_path / "out"), FORM,
                                      token=TOKEN, include_masks=True)
        assert fake.depositions[1]["metadata"]["title"] == "Toxo screen"
        assert "masks.zip" in fake.depositions[1]["files"]
        assert "10.5072/zenodo.1" in screen.status_text()
        assert rep._load_zenodo_token(True) == TOKEN
        assert screen._deposit_zenodo(str(src), str(tmp_path / "out"), FORM)
        assert 2 in fake.depositions
    with FakeZenodo("other") as fake:
        monkeypatch.setitem(rep._ZENODO_API, "sandbox", fake.api)
        screen._deposit_zenodo(str(src), str(tmp_path / "out"), FORM)
    assert "deposit failed" in screen.status_text()
    assert TOKEN not in screen.status_text()
