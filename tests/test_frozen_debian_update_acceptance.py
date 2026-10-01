"""The Debian frozen-update driver checks what dpkg really did (item 53).

These build tiny real ``.deb`` packages and drive real ``fakeroot dpkg
--root`` transactions in a private root, so the driver's replace, rollback
and interruption verdicts are shown to detect what they claim to detect.
The full-size run on the CI-built frozen packages is the receipt
``features/data/53_frozen_debian_replacement_2026-09-30.json``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "accept_frozen_debian_update.py"
pytestmark = pytest.mark.skipif(
    not all(shutil.which(name) for name in ("fakeroot", "dpkg", "dpkg-deb", "dpkg-query")),
    reason="needs Debian packaging tools")


def _driver():
    """Import the driver from ``tools/`` without putting ``tools`` on sys.path."""
    spec = importlib.util.spec_from_file_location("accept_frozen_debian_update", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deb(tmp_path: Path, version: str, files: dict, bulk: int = 0) -> Path:
    """Build a minimal ``spacr`` package.

    :param tmp_path: the test's scratch folder.
    :param version: the package version.
    :param files: bundle-relative file names to contents.
    :param bulk: extra small files, so an unpack takes long enough to interrupt.
    """
    stage = tmp_path / f"stage-{version}"
    (stage / "DEBIAN").mkdir(parents=True)
    (stage / "DEBIAN" / "control").write_text(
        f"Package: spacr\nVersion: {version}\nArchitecture: all\n"
        "Maintainer: test <t@example.org>\nDescription: fixture\n")
    bundle = stage / "opt" / "spacr"
    bundle.mkdir(parents=True)
    for name, content in files.items():
        (bundle / name).parent.mkdir(parents=True, exist_ok=True)
        (bundle / name).write_text(content)
    for index in range(bulk):
        folder = bundle / "bulk" / f"{index // 500:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{index}.txt").write_text(f"{version}-{index}\n")
    (bundle / "link").symlink_to("spacr")
    out = tmp_path / f"spacr_{version}_all.deb"
    subprocess.run(["dpkg-deb", "--root-owner-group", "-Znone", "--build", str(stage), str(out)],
                   check=True, capture_output=True)
    return out


def test_replace_and_rollback_leave_exactly_one_payload(tmp_path):
    """An upgrade removes old-only files; a downgrade restores them exactly."""
    driver = _driver()
    old = _deb(tmp_path, "1.0", {"spacr": "old\n", "only-old.txt": "x\n"})
    new = _deb(tmp_path, "2.0", {"spacr": "new\n", "only-new.txt": "y\n"})
    expected = {"old": driver.package_inventory(old), "new": driver.package_inventory(new)}
    assert expected["old"]["opt/spacr/link"] == "link:spacr"
    dpkg = driver.PrivateDpkg(tmp_path / "root")
    assert dpkg.run("-i", str(old))["exit"] == 0
    assert driver.compare(expected["old"], dpkg.root)["exact"]
    assert dpkg.run("-i", str(new))["exit"] == 0
    assert dpkg.status() == {"abbrev": "ii", "status": "install ok installed", "version": "2.0"}
    assert driver.compare(expected["new"], dpkg.root)["exact"]
    assert not (dpkg.root / "opt/spacr/only-old.txt").exists()
    assert dpkg.run("-i", str(old))["exit"] == 0
    assert dpkg.status()["version"] == "1.0"
    assert driver.compare(expected["old"], dpkg.root)["exact"]


def test_compare_reports_a_changed_a_missing_and_a_stray_file(tmp_path):
    """A tampered tree is never called exact, and each fault is named."""
    driver = _driver()
    old = _deb(tmp_path, "1.0", {"spacr": "old\n", "data.txt": "d\n"})
    dpkg = driver.PrivateDpkg(tmp_path / "root")
    assert dpkg.run("-i", str(old))["exit"] == 0
    (dpkg.root / "opt/spacr/spacr").write_text("tampered\n")
    (dpkg.root / "opt/spacr/data.txt").unlink()
    (dpkg.root / "opt/spacr/stray.txt").write_text("left over\n")
    verdict = driver.compare(driver.package_inventory(old), dpkg.root)
    assert not verdict["exact"]
    assert verdict["examples"] == {"missing": ["opt/spacr/data.txt"],
                                   "changed": ["opt/spacr/spacr"],
                                   "extra": ["opt/spacr/stray.txt"]}


def test_a_killed_unpack_is_flagged_and_both_recoveries_are_exact(tmp_path):
    """SIGKILL mid-unpack leaves a reinstall-required package that dpkg repairs."""
    driver = _driver()
    old = _deb(tmp_path, "1.0", {"spacr": "old\n"}, bulk=4000)
    new = _deb(tmp_path, "2.0", {"spacr": "new\n"}, bulk=4000)
    dpkg = driver.PrivateDpkg(tmp_path / "root")
    assert dpkg.run("-i", str(old))["exit"] == 0
    for recovery in (old, new):
        kill = dpkg.interrupt_install(new, after_new_files=20)
        if not kill["killed"]:
            pytest.skip("the unpack finished before it could be interrupted")
        assert kill["orphaned_fakeroot_daemons_stopped"] >= 1
        assert "reinstreq" in dpkg.status()["status"]
        assert any("spacr" in line for line in dpkg.audit())
        assert dpkg.run("-i", str(recovery))["exit"] == 0
        assert dpkg.status()["abbrev"] == "ii"
        assert driver.compare(driver.package_inventory(recovery), dpkg.root)["exact"]
