"""A checkout reports its own version; an installed wheel reports its metadata.

The workstation ran spaCR from a 1.5.1.4 checkout whose environment still held
``spacr-1.5.1.3.dist-info``, and the app showed 1.5.1.3.  ``get_version`` now
asks which files were imported before it trusts the installed metadata.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path

import spacr.version as version


class _Dist:
    """A stand-in distribution whose files live under ``root``."""

    def __init__(self, root: Path):
        self.root = root

    def locate_file(self, relative):
        return self.root / relative


def _wheel_tree(tmp_path: Path) -> Path:
    package = tmp_path / "site-packages" / "spacr"
    package.mkdir(parents=True)
    (package / "version.py").write_text("", encoding="utf-8")
    return package


def test_a_source_checkout_reports_its_own_literal_over_stale_metadata(monkeypatch):
    """The literal in spacr/_version.py wins when the checkout is imported."""
    monkeypatch.setattr(version, "package_version", lambda name: "1.5.1.3")
    monkeypatch.setattr(version, "_runs_from_source", lambda *a: True)
    monkeypatch.setattr(version, "_source_version", lambda: "1.5.1.4")

    assert version.get_version() == "1.5.1.4"


def test_a_tree_with_setup_py_beside_it_is_a_checkout(tmp_path):
    """``setup.py`` next to the package marks a source checkout."""
    package = tmp_path / "spacr"
    package.mkdir()
    (tmp_path / "setup.py").write_text("", encoding="utf-8")

    assert version._runs_from_source(package) is True


def test_an_editable_install_is_not_owned_by_its_metadata(monkeypatch, tmp_path):
    """An editable dist-info locates files in site-packages, not the import."""
    checkout = tmp_path / "checkout" / "spacr"
    checkout.mkdir(parents=True)
    monkeypatch.setattr(version, "distribution",
                        lambda name: _Dist(tmp_path / "site-packages"))

    assert version._runs_from_source(checkout) is True


def test_an_installed_wheel_reports_its_metadata(monkeypatch, tmp_path):
    """A wheel is imported from the files its own dist-info lists."""
    package = _wheel_tree(tmp_path)
    monkeypatch.setattr(version, "distribution",
                        lambda name: _Dist(package.parent))
    monkeypatch.setattr(version, "_PACKAGE_DIR", package)
    monkeypatch.setattr(version, "package_version", lambda name: "2.0.0")
    monkeypatch.setattr(version, "_source_version", lambda: "1.9.9")

    assert version._runs_from_source(package) is False
    assert version.get_version() == "2.0.0"


def test_a_nightly_wheel_is_owned_through_the_second_candidate(monkeypatch, tmp_path):
    """``spacr-nightly`` owning the files counts as an installed wheel."""
    package = _wheel_tree(tmp_path)

    def _only_nightly(name):
        if name == "spacr-nightly":
            return _Dist(package.parent)
        raise PackageNotFoundError(name)

    monkeypatch.setattr(version, "distribution", _only_nightly)

    assert version._runs_from_source(package) is False


def test_a_frozen_installer_without_metadata_reports_the_bundled_literal(
        monkeypatch, tmp_path):
    """A frozen bundle with no dist-info falls back to spacr/_version.py."""
    package = tmp_path / "_internal" / "spacr"
    package.mkdir(parents=True)

    def _absent(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(version, "distribution", _absent)
    monkeypatch.setattr(version, "package_version", _absent)
    monkeypatch.setattr(version, "_PACKAGE_DIR", package)
    monkeypatch.setattr(version, "_source_version", lambda: "1.5.1.4")

    assert version.get_version() == "1.5.1.4"


def test_a_frozen_installer_with_metadata_reports_its_metadata(monkeypatch, tmp_path):
    """A frozen bundle that copied its dist-info owns its files like a wheel."""
    package = tmp_path / "_internal" / "spacr"
    package.mkdir(parents=True)
    (package / "version.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(version, "distribution",
                        lambda name: _Dist(package.parent))
    monkeypatch.setattr(version, "package_version", lambda name: "1.5.1.4")
    monkeypatch.setattr(version, "_PACKAGE_DIR", package)

    assert version.get_version() == "1.5.1.4"


def test_this_checkout_reports_the_literal_it_ships():
    """Run from this tree, the version is the one in spacr/_version.py."""
    from spacr._version import __version__ as literal

    assert version.get_version() == literal


def test_the_updater_and_the_run_journal_agree_with_get_version():
    """Every reader of the running version goes through ``get_version``."""
    from spacr import run_journal, updater

    assert updater._installed_version() == version.get_version()
    assert run_journal._spacr_version() == version.get_version()
