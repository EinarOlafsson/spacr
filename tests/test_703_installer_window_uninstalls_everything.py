"""The installer window can remove all spaCR software, with the Preferences cleanup categories.

Maintainer requirement 2026-10-18. The user-level step removes every detected
installation and the ticked data; protected app parts are only reported, for
the installer's single administrator prompt.
"""
import pathlib

import pytest

from spacr import install_cleanup as ic


def _record(root, layout="macos-runtime", needs_admin=False):
    return ic.InstallRecord(kind="installer", root=root, version="1.5.0.6", layout=layout,
                            platform="macos", needs_admin=needs_admin)


@pytest.fixture
def preview():
    return {"operation_id": "op", "snapshot_id": "snap", "categories": list(ic._UNINSTALL_CATEGORIES),
            "cleanup": [], "installations": [
                {"key": key, "current": False, "selectable": True,
                 "record": {"root": root, "version": "1", "kind": "installer"}, "paths": [root]}
                for key, root in (("app", "/Library/Application Support/spaCR"),
                                  ("runtime", "/Users/u/Library/Application Support/spaCR"))]}


def test_protected_app_parts_are_reported_not_removed(preview, monkeypatch):
    app = _record("/Library/Application Support/spaCR", layout="macos-app", needs_admin=True)
    runtime = _record("/Users/u/Library/Application Support/spaCR")
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
