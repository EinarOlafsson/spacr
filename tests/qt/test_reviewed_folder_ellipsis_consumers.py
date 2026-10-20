"""Corrected shared captions reach real catalogs through persisted language choices."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("de", "Ordner auswählen…"),
        ("es", "Elegir carpeta…"),
        ("fr", "Choisir un dossier…"),
        ("hi", "फ़ोल्डर चुनें…"),
        ("pt", "Escolha a pasta…"),
        ("sv", "Välj katalog…"),
    ],
)
def test_reviewed_folder_ellipsis_uses_real_persisted_language(
    language, expected, qapp, tmp_path, monkeypatch,
):
    from PySide6.QtCore import QSettings

    from spacr.qt import i18n, i18n_catalogs, preferences

    root = Path(__file__).resolve().parents[2]
    for module in (i18n, i18n_catalogs, preferences):
        expected_file = root / (module.__name__.replace(".", "/") + ".py")
        if module is i18n_catalogs:
            expected_file = root / "spacr/qt/i18n_catalogs/__init__.py"
        assert Path(module.__file__).resolve() == expected_file

    settings_path = tmp_path / "language.ini"
    store = QSettings(str(settings_path), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.delenv(i18n.ENV_LANGUAGE, raising=False)
    assert Path(store.fileName()).resolve() == settings_path
    preferences.set_language(language)
    store.sync()
    assert store.status() == QSettings.NoError
    assert settings_path.is_file()
    persisted = QSettings(str(settings_path), QSettings.IniFormat)
    assert persisted.value(preferences._KEY_LANGUAGE) == language
    assert preferences.get_language() == i18n.current_language() == language

    source = "Choose folder…"
    module = importlib.import_module(f"spacr.qt.i18n_catalogs.{language}")
    assert Path(module.__file__).resolve() == (
        root / "spacr/qt/i18n_catalogs" / f"{language}.py"
    )
    assert module.SOURCE_HASHES[("UI", source)] == hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()
    assert module.UI[source] == expected
    assert i18n_catalogs.ui_text(source, language) == expected
    assert source not in i18n.CATALOGS.get(language, {})
    assert source not in i18n.TERM_CATALOGS.get(language, {})

    def refuse_fallback(*_args, **_kwargs):
        pytest.fail("This caption must use its actual compiled UI row")

    monkeypatch.setattr(i18n, "_composite_translation", refuse_fallback)
    monkeypatch.setattr(i18n, "_term_translation", refuse_fallback)
    actual = i18n.tr(source)
    assert actual == expected
    assert actual.endswith("…") and not actual.endswith("...")
    (tmp_path / "consumer.json").write_text(json.dumps({
        "language": language, "persisted_language": persisted.value(
            preferences._KEY_LANGUAGE),
        "source": source, "actual": actual,
        "catalog_path": str(Path(module.__file__).resolve()),
        "settings_path": str(settings_path),
        "source_sha256": module.SOURCE_HASHES[("UI", source)],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
