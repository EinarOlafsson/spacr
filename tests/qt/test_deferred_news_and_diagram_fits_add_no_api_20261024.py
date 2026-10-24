"""The News panel's late releases and the organism diagram's fit on show.

1ee91f7c1 (News: build the newest two releases with the page, the rest after
its first paint) and 0d09e8022 (organism diagram: fit once when shown after
hidden changes) did it by overriding ``paintEvent`` and ``showEvent``, which
made two new public API symbols with no catalogue records. Both now use a
private event watcher; these tests pin the behaviour those commits shipped.
"""
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent

from spacr.qt.widgets.home import NewsPanel
from spacr.qt.widgets.organism_diagram import OrganismDiagram

pytestmark = pytest.mark.qt


def _release_blocks(panel) -> int:
    column = panel._notes_column
    return sum(1 for index in range(column.count())
               if column.itemAt(index).widget() is not None
               and column.itemAt(index).widget() is not panel._placeholder)


def test_the_news_panel_builds_two_releases_then_the_rest_after_its_first_paint(qtbot):
    panel = NewsPanel("1.5.1.0")
    qtbot.addWidget(panel)
    releases = len(panel._releases)
    if releases <= NewsPanel.FIRST_RELEASES:
        pytest.skip("the bundled resource has too few releases to defer any")
    assert _release_blocks(panel) == NewsPanel.FIRST_RELEASES
    assert len(panel._pending_releases) == releases - NewsPanel.FIRST_RELEASES

    panel.resize(400, 500)
    panel.show()
    qtbot.waitExposed(panel)
    panel.grab()
    qtbot.waitUntil(lambda: not panel._pending_releases, timeout=3000)
    assert _release_blocks(panel) == releases
    assert "paintEvent" not in vars(NewsPanel)


def test_a_hidden_diagram_change_is_fitted_when_the_diagram_is_shown(qtbot,
                                                                     monkeypatch):
    root = Path(__file__).resolve().parents[2]
    diagram = OrganismDiagram(
        "toxoplasma", root / "spacr/resources/images/organism_apicomplexa.svg")
    qtbot.addWidget(diagram)
    diagram.resize(450, 850)
    fits = []
    real = OrganismDiagram._fit_width
    monkeypatch.setattr(OrganismDiagram, "_fit_width",
                        lambda self: (fits.append(self.isVisible()), real(self))[1])

    QCoreApplication.sendEvent(diagram, QEvent(QEvent.StyleChange))
    assert fits == [] and diagram._fit_when_shown is True

    diagram.show()
    qtbot.waitExposed(diagram)
    assert True in fits and diagram._fit_when_shown is False
    assert "showEvent" not in vars(OrganismDiagram)
