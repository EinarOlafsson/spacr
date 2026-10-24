"""Settings-category clicks send subtle field ripples; other waves keep full strength."""
import numpy as np
import pytest

from spacr.qt.widgets import ambient


def test_category_ripples_are_subtle_but_full_strength_waves_are_unchanged():
    assert 0 < ambient.CATEGORY_RIPPLE_STRENGTH <= 0.35
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#191919", seed=4, density=1.0)
    engine._add_column_wave((0.1, 0.4, 0.5, ambient.CATEGORY_RIPPLE_STRENGTH))
    engine._add_ripple(("rings", 0.3, 0.3, ambient.CATEGORY_RIPPLE_STRENGTH))
    assert engine._column_waves[0][2] == ambient.CATEGORY_RIPPLE_STRENGTH
    assert all(wave[2] == ambient.CATEGORY_RIPPLE_STRENGTH for wave in engine._fold_waves)
    distance = np.linspace(0.05, 0.5, 200)
    full, _ = engine._ripple_wave(distance, 0.8)
    subtle, _ = engine._ripple_wave(distance, 0.8, ambient.CATEGORY_RIPPLE_STRENGTH)
    assert np.abs(subtle).max() == pytest.approx(
        ambient.CATEGORY_RIPPLE_STRENGTH * np.abs(full).max())
    engine._add_ripple(("rings", 0.6, 0.6))
    assert engine._fold_waves[-1][2] == 1.0
    for invalid in ((0.1, 0.4, 0.5, 0.0), (0.1, 0.4), ("a", 0.4, 0.5)):
        count = len(engine._column_waves)
        engine._add_column_wave(invalid)
        assert len(engine._column_waves) == count


def test_section_clicks_send_category_strength_to_the_backdrop(qtbot, monkeypatch):
    from spacr.qt.widgets.section import Section
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    sent = []
    monkeypatch.setattr(ambient, "_publish_widget_ripple",
                        lambda widget, edge, rect, strength, **kwargs: sent.append(strength))
    host = QWidget()
    qtbot.addWidget(host)
    section = Section("Advanced", parent=host)
    QVBoxLayout(host).addWidget(section)
    host.show()
    section.set_expanded(True)
    section.set_expanded(False)
    qtbot.waitUntil(lambda: len(sent) == 2)
    assert sent == [ambient.CATEGORY_RIPPLE_STRENGTH] * 2
