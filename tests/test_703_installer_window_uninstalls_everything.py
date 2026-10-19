"""The installer window can remove all spaCR software, with the Preferences cleanup categories.

Maintainer requirement 2026-10-18. The user-level step removes every detected
installation and the ticked data; protected app parts are only reported, for
the installer's single administrator prompt.
"""
import os
import pathlib

import pytest

from spacr import install_cleanup as ic


def _record(root, layout="macos-runtime", needs_admin=False):
    return ic.InstallRecord(kind="installer", root=root, version="1.5.0.6", layout=layout,
                            platform="macos", needs_admin=needs_admin)


@pytest.fixture
def user_runtime(tmp_path):
    return str(tmp_path / "Library" / "Application Support" / "spaCR")


@pytest.fixture
def preview(user_runtime):
    return {"operation_id": "op", "snapshot_id": "snap", "categories": list(ic._UNINSTALL_CATEGORIES),
            "cleanup": [], "installations": [
                {"key": key, "current": False, "selectable": True,
                 "record": {"root": root, "version": "1", "kind": "installer"}, "paths": [root]}
                for key, root in (("app", "/Library/Application Support/spaCR"),
                                  ("runtime", user_runtime))]}


def test_protected_app_parts_are_reported_not_removed(preview, user_runtime, monkeypatch):
    app = _record("/Library/Application Support/spaCR", layout="macos-app", needs_admin=True)
    runtime = _record(user_runtime)
    seen = {}
    monkeypatch.setattr(ic, "_validate_uninstall_selection",
                        lambda p, s, m, after_shutdown: ([app, runtime], []))

    def apply(p, s, records, c, m):
        seen["removed"] = records
        return {"ok": True, "reports": []}

    monkeypatch.setattr(ic, "_apply_uninstall_records", apply)
    monkeypatch.setattr(ic.os.path, "lexists", lambda path: path in ic._INSTALLER_ADMIN_PATHS)
    outcome = ic._installer_uninstall_apply(preview, ["caches"], system=ic._Machine(running_prefix=""))
    assert [record.root for record in seen["removed"]] == [runtime.root]
    assert outcome["admin"] == list(ic._INSTALLER_ADMIN_PATHS)


def test_unknown_categories_are_refused(preview):
    with pytest.raises(ValueError):
        ic._installer_uninstall_apply(preview, ["everything"], system=ic._Machine(running_prefix=""))


def test_a_running_spacr_blocks_the_installer_uninstall(preview, monkeypatch):
    monkeypatch.setattr(ic, "_desktop_processes", lambda: [4242])
    with pytest.raises(ValueError, match="Quit spaCR"):
        ic._installer_uninstall_apply(preview, ["caches"])


def test_the_installer_window_offers_it_with_every_category():
    source = (pathlib.Path(ic.__file__).resolve().parents[1]
              / "packaging" / "online" / "macos_installer.m").read_text(encoding="utf-8")
    assert '@selector(uninstall:)' in source and '"uninstall-preview"' in source and '"uninstall-apply"' in source
    for key in ic._UNINSTALL_CATEGORIES:
        assert f'@"{key}"' in source
    assert "with administrator privileges" in source


def test_the_admin_step_removes_the_command_only_when_it_is_the_spacr_launcher():
    source = (pathlib.Path(ic.__file__).resolve().parents[1]
              / "packaging" / "online" / "macos_installer.m").read_text(encoding="utf-8")
    script = source[source.index("do shell script"):source.index("with administrator privileges")]
    assert ("case $(/usr/bin/readlink /usr/local/bin/spacr) in /Applications/spaCR.app/*) "
            "/bin/rm -f /usr/local/bin/spacr;; esac;") in script
    assert script.count("rm -f /usr/local/bin/spacr") == 1


def test_nothing_is_imported_after_the_first_removal(tmp_path, monkeypatch):
    """The installer window runs this helper on the runtime's own Python and deletes that Python while it runs.

    Verified natively on an Intel Mac 2026-10-18: it completes because every import happens before the first
    removal. Any later import would raise ModuleNotFoundError halfway through an uninstall.
    """
    import shutil
    import sys

    from tests.test_install_cleanup import Box, linux_online

    box = Box(tmp_path)
    linux_online(box, "1.5.1.0")
    machine = box.machine("linux")
    preview = ic._uninstall_preview("no-late-imports", system=machine)
    snapshot = {}

    bookkeeping = (".spacr-installer-lock", ".spacr-update-lock", ".installing", ".takeover")

    def first_removal(original):
        def removing(path, *args, **kwargs):
            text = os.fspath(path)
            if not (text.endswith("/owner") or any(text.endswith(name) for name in bookkeeping)):
                snapshot.setdefault("modules", set(sys.modules))
                snapshot.setdefault("first", text)
            return original(path, *args, **kwargs)
        return removing

    for owner, name in ((ic.os, "unlink"), (ic.os, "rmdir"), (ic.os, "remove"), (ic.shutil, "rmtree"),
                        (shutil, "rmtree")):
        monkeypatch.setattr(owner, name, first_removal(getattr(owner, name)))
    outcome = ic._installer_uninstall_apply(preview, list(ic._UNINSTALL_CATEGORIES), system=machine)
    assert outcome["ok"], outcome
    removed = [path for report in outcome["reports"] for path in report["removed"]]
    assert any("venv" in path for path in removed), outcome
    assert "modules" in snapshot, "nothing was removed"
    assert set(sys.modules) - snapshot["modules"] == set()


def test_an_update_error_with_a_traceback_is_shown_as_one_sentence(tmp_path, monkeypatch):
    """Seen natively 2026-10-19: the dialog showed a whole Python traceback from the staged identity probe."""
    from spacr.qt.app import MainWindow

    monkeypatch.setenv("HOME", str(tmp_path))
    why = ("The staged application identity could not be verified: Traceback (most recent call last):\n"
           '  File "<string>", line 40, in <module>\nAssertionError: Requested spaCR version differs\n')
    shown = MainWindow._removal_reason_text(why)
    assert "Traceback" not in shown and "File" not in shown
    assert shown.startswith("The staged application identity could not be verified: Requested spaCR version differs")
    assert str(tmp_path / ".spacr" / "logs" / "update.log") in shown
    assert MainWindow._removal_reason_text("Plain reason.") == "Plain reason."
