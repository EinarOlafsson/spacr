"""Installed Debian catalogs reach actual backend and shutdown-prompt callbacks."""
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from PySide6.QtCore import QSettings

from spacr import install_cleanup
from spacr.qt import app as app_module
from spacr.qt import i18n, preferences


METADATA_SOURCE = "invalid Debian package metadata"
PROMPT_SOURCE = (
    "spaCR {version} is ready. Restart to use the update. "
    "Your data and caches will be kept."
)


@pytest.mark.parametrize("language", ["sv", "de", "es", "zh_CN", "pt", "ko", "hi", "is", "fr"])
def test_debian_localized_refusal_and_cancelled_authorization_prompt(monkeypatch, tmp_path, language):
    reviewed = Path(__file__).parents[1] / "docs/i18n/reviewed/runtime" / language / "2026-10-15-debian-update-transaction.json"
    rows = json.loads(reviewed.read_text())["records"]
    targets = {row["source"]: row["translation"] for row in rows}
    native = Path(__file__).parents[1] / "docs/i18n/reviewed/runtime" / language / "2026-10-16-n703-native-operations.json"
    targets.update({row["source"]: row["translation"]
                    for row in json.loads(native.read_text())["records"]})
    assert targets[METADATA_SOURCE] != METADATA_SOURCE and targets[PROMPT_SOURCE] != PROMPT_SOURCE
    store = QSettings(str(tmp_path / "isolated-language.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.delenv(i18n.ENV_LANGUAGE, raising=False)
    preferences.set_language(language)
    assert i18n.current_language() == language
    archive = tmp_path / "untouched.deb"
    original = b"private immutable archive bytes; no package process is run"
    archive.write_bytes(original)
    package_commands = []

    def package_metadata(argv):
        package_commands.append(argv)
        return 0, "malformed metadata without a field delimiter"

    with pytest.raises(ValueError) as error:
        install_cleanup._debian_package_metadata(
            str(archive), "1.5.1.5", "amd64", run=package_metadata)
    assert str(error.value) == targets[METADATA_SOURCE]
    assert package_commands == [["/usr/bin/dpkg-deb", "--field", str(archive),
                                 "Package", "Version", "Architecture"]]
    assert archive.read_bytes() == original
    events = []
    prompts = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Cancelled authorization must not approve, save, close or run a package command")

    handshake = SimpleNamespace(status=lambda: {"state": "ready"},
                                cancel=lambda: events.append("cancel"), check=forbidden)
    window = SimpleNamespace(_closing=False, _frozen_update_handshake=handshake,
                             _frozen_update_plan={"adapter": "linux-deb-v1", "version": "1.5.1.5"},
                             _frozen_update_timer=SimpleNamespace(stop=lambda: events.append("stop")),
                             _save_pip_update_restart=forbidden, close=forbidden)
    window._cancel_frozen_update = MethodType(app_module.MainWindow._cancel_frozen_update, window)
    window._hide_update_progress = MethodType(app_module.MainWindow._hide_update_progress, window)

    def restart(message):
        prompts.append(message)
        return 2

    window._confirm_update_restart = restart
    monkeypatch.setattr(app_module, "QMessageBox", SimpleNamespace(
        Ok=1, Cancel=2, information=forbidden, warning=forbidden))
    app_module.MainWindow._poll_frozen_update(window)
    assert prompts == [targets[PROMPT_SOURCE].format(version="1.5.1.5")]
    assert events == ["stop", "stop", "cancel"]
    assert window._frozen_update_handshake is None and window._frozen_update_plan is None
    assert archive.read_bytes() == original
    assert preferences.get_language() == language
