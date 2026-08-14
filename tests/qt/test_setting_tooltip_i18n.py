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


def test_installed_api_dot_refreshes_accessibility_and_uses_setting_url():
    from PySide6.QtWidgets import QFormLayout, QLabel, QSpinBox, QWidget
    from spacr.qt.screens.settings_model import (
        api_docs_url,
        install_api_tooltips,
        refresh_api_tooltips,
    )
    from spacr.qt.widgets.info_link import InfoLink

    app = _application()
    root = QWidget()
    form = QFormLayout(root)
    label = QLabel("N trials", root)
    field = QSpinBox(root)
    form.addRow(label, field)
    install_api_tooltips(root, "umap", {field: "n_trials"})

    dots = root.findChildren(InfoLink)
    assert len(dots) == 1
    dot = dots[0]
    assert dot.url() == api_docs_url("umap", "n_trials")

    refresh_api_tooltips(root, "sv")
    assert dot.toolTip().startswith("Öppna API-referens")
    assert dot.url().endswith("?lang=sv")
    assert dot.accessibleName() == dot.toolTip()
    assert field.toolTip() == ""

    refresh_api_tooltips(root, "ko")
    assert "API 참조 열기" in dot.toolTip()
    assert dot.url().endswith("?lang=ko")
    assert dot.accessibleName() == dot.toolTip()
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
