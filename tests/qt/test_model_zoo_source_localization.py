"""Source captions translate without changing model-source identities or state."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings, Qt

from spacr import model_zoo
from spacr.qt import i18n
from spacr.qt.widgets import model_zoo_picker as picker


LANGUAGES = ("sv", "de", "es", "fr", "pt", "is", "zh_CN", "hi", "ko")


@pytest.fixture
def localized_sources(tmp_path, monkeypatch):
    """Exercise the real renderer using isolated exact-catalog boundary values."""
    settings = QSettings(str(tmp_path / "sources.ini"), QSettings.IniFormat)
    monkeypatch.setattr(picker, "_store", lambda: settings)
    monkeypatch.setattr(picker, "_third_party_heading_visible", lambda: False)
    raw = dict(picker._SOURCE_TOOLTIPS)
    sources = (*raw.values(), "spaCR community")
    targets = {}
    for language in LANGUAGES:
        targets[language] = {
            source: f"{language} source caption {index}"
            for index, source in enumerate(sources)
        }
        monkeypatch.setitem(i18n.CATALOGS, language, {
            **i18n.CATALOGS.get(language, {}), **targets[language],
        })
    return settings, raw, targets


@pytest.mark.parametrize("language", LANGUAGES)
def test_constructor_translates_all_six_captions_with_canonical_ids(
        qtbot, monkeypatch, localized_sources, language):
    settings, raw, targets = localized_sources
    monkeypatch.setenv("SPACR_LANGUAGE", language)
    strip = picker.SourceStrip()
    qtbot.addWidget(strip)

    for source_name, tooltip in raw.items():
        heading = strip.heading(source_name)
        assert heading.toolTip() == targets[language][tooltip]
        assert heading.source_name == source_name
        assert heading.text() == (targets[language][source_name]
                                  if source_name == "spaCR community"
                                  else source_name)
    assert strip.enabled() == model_zoo.DEFAULT_ZOO_SOURCES
    assert picker._SOURCE_TOOLTIPS == raw
    assert not settings.contains(picker._SOURCES_SETTING)


def test_language_round_trip_preserves_source_keys_and_user_selection(
        qtbot, monkeypatch, localized_sources):
    settings, raw, targets = localized_sources
    monkeypatch.setenv("SPACR_LANGUAGE", "de")
    allowed = []
    permit = [False]

    def guard(source):
        allowed.append(source)
        return permit[0]

    strip = picker.SourceStrip(guard=guard)
    qtbot.addWidget(strip)
    strip.show()
    changed = []
    strip.changed.connect(lambda: changed.append(strip.enabled()))
    community = strip.heading("spaCR community")
    qtbot.mouseClick(community, Qt.LeftButton)
    assert allowed == ["spaCR community"]
    assert not strip.is_on("spaCR community") and not changed
    permit[0] = True
    qtbot.mouseClick(community, Qt.LeftButton)
    assert allowed == ["spaCR community", "spaCR community"]
    assert strip.is_on("spaCR community") and len(changed) == 1
    stored = settings.value(picker._SOURCES_SETTING)
    assert stored == "cellposeSAM,spaCR,spaCR community"

    for language in ("ko", "fr", "en", "de"):
        i18n.retranslate_widget_tree(strip, language)
        for source_name, tooltip in raw.items():
            heading = strip.heading(source_name)
            expected = tooltip if language == "en" else targets[language][tooltip]
            assert heading.toolTip() == expected
            caption = (source_name if language == "en" or source_name != "spaCR community"
                       else targets[language][source_name])
            assert heading.text() == caption
            assert heading.source_name == source_name
        assert settings.value(picker._SOURCES_SETTING) == stored
        assert len(changed) == 1
    assert picker._SOURCE_TOOLTIPS == raw
    reopened = picker.SourceStrip()
    qtbot.addWidget(reopened)
    assert reopened.enabled() == strip.enabled()
    assert reopened.heading("spaCR community").text() == targets["de"]["spaCR community"]
