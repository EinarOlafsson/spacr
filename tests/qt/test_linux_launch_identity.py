"""Linux launch surfaces have an identity and an opaque first frame."""

from __future__ import annotations

import ast
import os
import shlex
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt


ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "packaging/linux/io.github.olafssonlab.spacr.desktop"
APP_ID = "io.github.olafssonlab.spacr"


def test_desktop_entry_matches_qt_application_id():
    text = DESKTOP.read_text(encoding="utf-8")
    assert "Icon=io.github.olafssonlab.spacr" in text
    assert "StartupWMClass=spaCR" in text
    source = (ROOT / "spacr/qt/app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "setDesktopFileName"]
    assert len(calls) == 1
    assert ast.literal_eval(calls[0].args[0]) == APP_ID


def test_main_window_and_loading_cover_are_opaque(qapp, qtbot, monkeypatch):
    import spacr.qt.app as app_module
    from spacr.qt.widgets.loading_screen import LoadingScreen

    monkeypatch.setattr(app_module.MainWindow, "_install_loading_screen",
                        lambda self: None)
    window = app_module.MainWindow()
    qtbot.addWidget(window)
    # `autoFillBackground` ALONE, NOT `WA_OpaquePaintEvent`. That attribute
    # is a PROMISE that the widget paints every pixel of its own rect, and
    # `MainWindow` does not keep it: applying the stylesheet clears
    # `autoFillBackground` again, so by the time the window is shown it read
    # `autoFill=False, opaquePaint=True` -- claiming to fill while filling
    # nothing. Qt then skips the backdrop repaint it would otherwise have
    # done, which is what produced the overlapping text in the bottom-left
    # corner and the flicker on the top bar (2fbf1de11).
    #
    # So the opacity being asserted is the one actually kept.
    assert window.autoFillBackground()

    cover = LoadingScreen(parent=window)
    assert cover.testAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
    assert cover.autoFillBackground()


def test_frozen_desktop_launch_ignores_an_older_path_installation(tmp_path):
    root = tmp_path / "installed package"
    packaged = root / "opt" / "spacr" / "spacr"
    shadow = tmp_path / "old-bin" / "spacr"
    proof = tmp_path / "launched.txt"
    for executable, identity in ((packaged, "packaged"), (shadow, "old")):
        executable.parent.mkdir(parents=True)
        executable.write_text(
            '#!/bin/sh\nprintf "%s\n" ' + shlex.quote(identity)
            + ' "$@" > "$SPACR_LAUNCH_PROOF"\n'
        )
        executable.chmod(0o755)
    environment = dict(os.environ, PATH=str(shadow.parent), HOME=str(tmp_path),
                       SPACR_LAUNCH_PROOF=str(proof))
    arguments = ["--source", "field with spaces.tif"]
    subprocess.run([str(packaged), *arguments], env=environment, check=True,
                   capture_output=True, text=True, timeout=5)
    assert proof.read_text(encoding="utf-8").splitlines() == ["packaged", *arguments]
    proof.unlink()
    command = shlex.split(next(line.split("=", 1)[1]
                              for line in DESKTOP.read_text(encoding="utf-8").splitlines()
                              if line.startswith("Exec=")))
    assert command[0] in {"spacr", "/opt/spacr/spacr"}
    if command[0] == "/opt/spacr/spacr":
        command[0] = str(packaged)
    subprocess.run([*command, *arguments], env=environment, check=True,
                   capture_output=True, text=True, timeout=5)
    assert proof.read_text(encoding="utf-8").splitlines() == ["packaged", *arguments]
