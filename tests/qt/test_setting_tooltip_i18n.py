"""Presentation-only localization for semantic setting help."""
from __future__ import annotations


def _application():
    """Return the offscreen QApplication without requiring pytest-qt."""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_html_tooltip_localizes_chrome_but_not_partial_scientific_prose():
    from spacr.qt.screens.settings_model import api_docs_url, format_tooltip

    body = "Threshold for class 1 classification."
    swedish_url = api_docs_url("mask", "plot", "sv")

    swedish = format_tooltip(body, "mask", "plot", language="sv")
    assert "<b>Diagram</b> <i>(boolesk)</i>" in swedish
    assert body in swedish
    assert "Klassificering" not in swedish
    assert "Öppna spaCR:s API-dokumentation" in swedish
    assert f'href="{swedish_url}"' in swedish

    korean = format_tooltip(body, "mask", "plot", language="ko")
    assert "<b>플롯</b> <i>(불리언)</i>" in korean
    assert body in korean
    assert "분류" not in korean
    assert "spaCR API 문서 열기" in korean
    korean_url = api_docs_url("mask", "plot", "ko")
    assert f'href="{korean_url}"' in korean


def test_exact_whole_body_translation_is_used_and_url_is_unchanged():
    from spacr.qt.screens.settings_model import api_docs_url, format_tooltip

    url = api_docs_url("umap", "n_trials", "sv")
    tip = format_tooltip(
        "Controls this setting.", "umap", "n_trials", language="sv")

    assert "Styr den här inställningen." in tip
    assert "Controls this setting." not in tip
    assert f'href="{url}"' in tip
    assert tip.count(url) == 1


def test_tooltip_catalog_receives_the_application_identity(monkeypatch):
    """A shared setting name may describe different data in two modules."""
    from spacr.qt import i18n_catalogs
    from spacr.qt.screens.settings_model import format_tooltip

    seen = []

    def translated(key, source, language, app_key=""):
        seen.append((key, source, language, app_key))
        return "Regressionsutdata" if app_key == "regression" else None

    monkeypatch.setattr(i18n_catalogs, "setting_tooltip", translated)

    regression = format_tooltip(
        "Directory for regression outputs.",
        "regression",
        "src",
        language="sv",
    )
    mask = format_tooltip(
        "Directory for segmentation inputs.",
        "mask",
        "src",
        language="sv",
    )

    assert "Regressionsutdata" in regression
    assert "Directory for segmentation inputs." in mask
    assert ("src", "Directory for regression outputs.", "sv", "regression") in seen
    assert ("src", "Directory for segmentation inputs.", "sv", "mask") in seen


def test_plain_tooltip_localizes_chrome_and_retains_canonical_body_and_url():
    from spacr.qt.screens.settings_model import api_docs_url, plain_tooltip

    body = "A deliberately untranslated UMAP scientific explanation."
    url = api_docs_url("umap", "plot", "ko")
    tooltip = plain_tooltip(body, "umap", "plot", language="ko")

    assert tooltip.startswith("플롯 (불리언)")
    assert body in tooltip
    assert url in tooltip
    assert tooltip.count(url) == 1


def test_refresh_regenerates_semantic_help_without_mutating_source():
    from PySide6.QtWidgets import QSpinBox, QVBoxLayout, QWidget
    from spacr.qt.screens.settings_model import (
        attach_api_tooltip,
        refresh_api_tooltips,
    )

    app = _application()
    root = QWidget()
    layout = QVBoxLayout(root)
    field = QSpinBox(root)
    layout.addWidget(field)
    source = "Controls this setting."
    attach_api_tooltip(
        field, "mask", "plot", description=source, _descriptions={})

    assert field.property("apiTooltipDescriptionSource") == source
    refresh_api_tooltips(root, "sv")
    assert "Styr den här inställningen." in field.toolTip()
    assert field.property("apiTooltipDescriptionSource") == source

    from spacr.qt.i18n import retranslate_widget_tree
    retranslate_widget_tree(root, "ko")
    assert "이 설정을 제어합니다." in field.toolTip()
    assert "spaCR API 문서 열기" in field.toolTip()
    assert field.property("apiTooltipDescriptionSource") == source
    root.close()
    app.processEvents()


def test_the_setting_link_follows_the_language_without_a_dot_to_hold_it():
    """The dot is gone; the link it opened is in the label's hover text.

    That is the whole reason the dot was safe to drop, so the link has to
    keep following the language pass on its own -- localised URL included.
    """
    from PySide6.QtWidgets import QFormLayout, QLabel, QSpinBox, QWidget
    from spacr.qt.screens.settings_model import (
        api_docs_url,
        install_api_tooltips,
        refresh_api_tooltips,
    )
    from spacr.qt.widgets.dot_link import DotLink

    app = _application()
    root = QWidget()
    form = QFormLayout(root)
    label = QLabel("N trials", root)
    field = QSpinBox(root)
    form.addRow(label, field)
    install_api_tooltips(root, "umap", {field: "n_trials"})

    assert root.findChildren(DotLink) == []
    assert api_docs_url("umap", "n_trials") in label.toolTip()

    refresh_api_tooltips(root, "sv")
    assert api_docs_url("umap", "n_trials", "sv") in label.toolTip()
    assert field.toolTip() == ""

    refresh_api_tooltips(root, "ko")
    assert api_docs_url("umap", "n_trials", "ko") in label.toolTip()
    assert label.property("apiTooltipDescriptionSource") == ""
    assert "이 설정을 제어합니다." in label.toolTip()
    root.close()
    app.processEvents()


def test_attached_unknown_setting_uses_localized_generic_fallback():
    from PySide6.QtWidgets import QSpinBox
    from spacr.qt.screens.settings_model import (
        attach_api_tooltip,
        refresh_api_tooltips,
    )

    app = _application()
    field = QSpinBox()
    attach_api_tooltip(
        field, "mask", "made_up_setting", _descriptions={})
    assert field.property("apiTooltipDescriptionSource") == ""

    refresh_api_tooltips(field, "sv")
    assert "Styr den här inställningen." in field.toolTip()
    assert "Controls made up setting." not in field.toolTip()
    field.close()
    app.processEvents()


def test_dictionary_type_name_is_localized():
    from spacr.qt.screens.settings_model import format_tooltip
    from spacr.settings import expected_types

    # Any dict-typed setting exercises the "dictionary" type name. The key is
    # looked up instead of hard-coded because this test used to name
    # `plate_dict`, which was deleted along with four other unread settings --
    # a removed setting is absent from `expected_types`, so `_type_hint`
    # correctly renders no type at all and the check failed for a reason that
    # had nothing to do with localization.
    dict_keys = sorted(k for k, t in expected_types.items() if t is dict)
    assert dict_keys, "no dict-typed setting left to exercise the type name"
    key = dict_keys[0]

    swedish = format_tooltip("Controls this setting.", "mask", key, "sv")
    korean = format_tooltip("Controls this setting.", "mask", key, "ko")
    assert "ordbok" in swedish
    assert "사전" in korean
