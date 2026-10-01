"""Replace, roll back and interrupt a frozen Debian spaCR without root.

Item 53 (2026-09-30). The Debian frozen family updates through dpkg, not
through spaCR's in-app helper: the helper deliberately refuses a frozen
``linux-deb`` plan before removing anything. This driver exercises the real
dpkg transaction on two real frozen packages inside a private dpkg root
(``fakeroot dpkg --root``), so no system package database, ``/opt`` or user
installation is touched.

Scenarios, each recorded in one JSON receipt:

* ``install``      the older package installs and matches its own payload.
* ``helper``       the installed frozen update helper refuses a frozen
                   ``linux-deb`` plan with exit 2 and changes nothing.
* ``upgrade``      the newer package replaces the older one in place; the
                   tree equals the new payload, no old-only file survives,
                   and the user's profile and earlier analysis are kept.
* ``interrupt``    dpkg is SIGKILLed mid-unpack; the database must then flag
                   the package as needing reinstallation, and both a
                   forward repair and a rollback to the older package must
                   restore an exact payload tree.
* ``rollback``     the older package is reinstalled over the newer one.
* ``gui``          after an update, the installed frozen executable is
                   restarted under a private Xvfb display and its real
                   Measure screen and Run action complete one CPU field
                   (``SPACR_DISTRIBUTION_SMOKE=1``); cells are recounted
                   independently from the read-only SQLite database.

Caveats recorded in every receipt: the payload is installed below a private
root, so ``/usr/bin/spacr`` (which execs ``/opt/spacr/spacr``) is not used;
the frozen executable is launched directly. Package dependencies are not
resolved (``--force-depends``) because the private database is empty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sqlite3
import subprocess
import tarfile
import time
from typing import Dict, Iterable, List, Optional

PACKAGE = "spacr"
BUNDLE = Path("opt") / "spacr"


def sha256_file(path: Path) -> str:
    """Return the SHA256 of a file, streamed.

    :param path: the file to hash.
    :returns: the lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deb_field(deb: Path, field: str) -> str:
    """Read one control field from a binary package.

    :param deb: the ``.deb`` file.
    :param field: the control field name, for example ``Version``.
    :returns: the field value.
    """
    return subprocess.run(["dpkg-deb", "-f", str(deb), field], check=True,
                          capture_output=True, text=True).stdout.strip()


def package_inventory(deb: Path) -> Dict[str, str]:
    """Hash every file and symlink the package payload would install.

    :param deb: the ``.deb`` file.
    :returns: relative path to ``sha256:<hex>`` or ``link:<target>``;
        directories are omitted because they may be shared with other packages.
    """
    process = subprocess.Popen(["dpkg-deb", "--fsys-tarfile", str(deb)],
                               stdout=subprocess.PIPE)
    inventory: Dict[str, str] = {}
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = os.path.normpath(member.name).lstrip("./")
                if member.issym():
                    inventory[name] = "link:" + member.linkname
                elif member.isfile():
                    digest = hashlib.sha256()
                    handle = archive.extractfile(member)
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(block)
                    inventory[name] = "sha256:" + digest.hexdigest()
    finally:
        process.stdout.close()
        if process.wait() != 0:
            raise RuntimeError(f"dpkg-deb could not read {deb}")
    return inventory


def tree_inventory(root: Path, prefixes: Iterable[Path]) -> Dict[str, str]:
    """Hash every file and symlink below the given prefixes of a root.

    :param root: the private installation root.
    :param prefixes: relative directories to walk, for example ``opt/spacr``.
    :returns: the same mapping shape as :func:`package_inventory`.
    """
    inventory: Dict[str, str] = {}
    for prefix in prefixes:
        base = root / prefix
        if not base.exists():
            continue
        for folder, directories, files in os.walk(base):
            for name in list(files) + [d for d in directories
                                       if (Path(folder) / d).is_symlink()]:
                path = Path(folder) / name
                relative = str(path.relative_to(root))
                if path.is_symlink():
                    inventory[relative] = "link:" + os.readlink(path)
                elif path.is_file():
                    inventory[relative] = "sha256:" + sha256_file(path)
    return inventory


def compare(expected: Dict[str, str], root: Path) -> Dict:
    """Compare a root's installed package paths with a package payload.

    :param expected: the payload inventory from :func:`package_inventory`.
    :param root: the private installation root.
    :returns: counts plus up to ten example paths of each kind of mismatch;
        ``exact`` is true only when nothing is missing, changed or extra
        below the bundle directory.
    """
    actual = tree_inventory(root, [BUNDLE])
    for path in expected:
        if not path.startswith(str(BUNDLE) + os.sep):
            candidate = root / path
            if candidate.is_symlink():
                actual[path] = "link:" + os.readlink(candidate)
            elif candidate.is_file():
                actual[path] = "sha256:" + sha256_file(candidate)
    missing = sorted(p for p in expected if p not in actual)
    changed = sorted(p for p in expected if p in actual and actual[p] != expected[p])
    extra = sorted(p for p in actual if p not in expected)
    return {"expected": len(expected), "actual": len(actual),
            "missing": len(missing), "changed": len(changed), "extra": len(extra),
            "examples": {"missing": missing[:10], "changed": changed[:10],
                         "extra": extra[:10]},
            "exact": not (missing or changed or extra)}


class PrivateDpkg:
    """A dpkg database and installation root owned by the current user."""

    def __init__(self, root: Path):
        """Create an empty dpkg administration directory below ``root``.

        :param root: the private installation root; it must not exist yet.
        """
        self.root = Path(root)
        self.admin = self.root / "var" / "lib" / "dpkg"
        for name in ("info", "updates", "triggers"):
            (self.admin / name).mkdir(parents=True)
        for name in ("status", "available"):
            (self.admin / name).touch()

    def command(self, *arguments: str) -> List[str]:
        """Return the unprivileged dpkg command for this root.

        :param arguments: the dpkg action and its operands.
        :returns: an argv run under ``fakeroot``.
        """
        return ["fakeroot", "dpkg", f"--root={self.root}", "--force-not-root",
                "--force-depends", "--force-script-chrootless", *arguments]

    def run(self, *arguments: str, timeout: float = 1800) -> Dict:
        """Run dpkg to completion and retain its outcome.

        :param arguments: the dpkg action and its operands.
        :param timeout: seconds before the run is treated as hung.
        :returns: exit code, elapsed seconds and the output tail.
        """
        started = time.monotonic()
        result = subprocess.run(self.command(*arguments), capture_output=True,
                                text=True, timeout=timeout)
        return {"argv": ["dpkg", *arguments], "exit": result.returncode,
                "seconds": round(time.monotonic() - started, 2),
                "output_tail": (result.stdout + result.stderr).strip().splitlines()[-6:]}

    def status(self) -> Dict[str, str]:
        """Read the package's recorded state, including the update journal.

        :returns: ``abbrev``, ``status`` and ``version`` as dpkg-query reports
            them; empty strings when the package is unknown.
        """
        result = subprocess.run(
            ["dpkg-query", f"--admindir={self.admin}", "-W",
             "-f=${db:Status-Abbrev}|${Status}|${Version}", PACKAGE],
            capture_output=True, text=True)
        if result.returncode != 0:
            return {"abbrev": "", "status": "", "version": ""}
        abbrev, status, version = (result.stdout.split("|") + ["", "", ""])[:3]
        return {"abbrev": abbrev.strip(), "status": status, "version": version}

    def audit(self) -> List[str]:
        """Return dpkg's audit of broken packages in this database.

        :returns: the audit's non-empty output lines.
        """
        result = subprocess.run(["dpkg", f"--admindir={self.admin}", "--audit"],
                                capture_output=True, text=True)
        return [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]

    def interrupt_install(self, deb: Path, *, after_new_files: int = 200,
                          timeout: float = 900) -> Dict:
        """Start ``dpkg -i`` and SIGKILL it once the unpack is under way.

        :param deb: the package being installed.
        :param after_new_files: how many ``.dpkg-new`` files must exist in the
            bundle before the kill, so the interruption is genuinely mid-unpack.
        :param timeout: seconds to wait for that point.
        :returns: what was observed at the kill and the process's end.
        """
        marker = os.urandom(16).hex()
        environment = dict(os.environ, SPACR_ACCEPTANCE_FAKEROOT=marker)
        process = subprocess.Popen(self.command("-i", str(deb)), stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True,
                                   env=environment)
        started = time.monotonic()
        seen = 0
        try:
            while time.monotonic() - started < timeout:
                if process.poll() is not None:
                    break
                seen = sum(1 for _folder, _dirs, files in os.walk(self.root / BUNDLE)
                           for name in files if name.endswith(".dpkg-new"))
                if seen >= after_new_files:
                    break
                time.sleep(0.02)
        finally:
            finished_first = process.poll() is not None
            if not finished_first:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            orphans = _stop_marked_processes(marker)
        return {"killed": not finished_first, "orphaned_fakeroot_daemons_stopped": orphans, "dpkg_new_files_at_kill": seen,
                "seconds_to_kill": round(time.monotonic() - started, 2),
                "returncode": process.returncode}


def _stop_marked_processes(marker: str) -> int:
    """Stop this run's daemonized ``faked`` left behind by a SIGKILLed fakeroot.

    Only processes of this user whose environment carries ``marker`` are
    signalled, so other sessions' fakeroot daemons are never touched.

    :param marker: the random value this run put in the child environment.
    :returns: how many processes were stopped.
    """
    needle = f"SPACR_ACCEPTANCE_FAKEROOT={marker}".encode()
    stopped = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            if needle in (entry / "environ").read_bytes().split(b"\0"):
                os.kill(int(entry.name), signal.SIGTERM)
                stopped += 1
        except (OSError, ProcessLookupError):
            continue
    return stopped


def launch_version(executable: Path, environment: Dict[str, str]) -> Dict:
    """Run ``spacr --version`` from the installed frozen executable.

    :param executable: the frozen ``spacr`` binary.
    :param environment: the private launch environment.
    :returns: exit code and the first output line, or the failure.
    """
    try:
        result = subprocess.run([str(executable), "--version"], env=environment,
                                capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"exit": None, "error": type(error).__name__}
    lines = (result.stdout + result.stderr).strip().splitlines()
    return {"exit": result.returncode, "output": lines[-1] if lines else ""}


def base_environment(home: Path, display: Optional[str]) -> Dict[str, str]:
    """Return a CPU-only launch environment with a private profile.

    :param home: the private HOME that stands in for the user's profile.
    :param display: the private X display, or ``None`` for version checks.
    :returns: the environment mapping.
    """
    environment = {
        "HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
        "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2", "SPACR_DEVICE": "cpu", "SPACR_NO_SETUP": "1",
    }
    if display:
        environment.update(DISPLAY=display, QT_QPA_PLATFORM="xcb")
    return environment


def gui_measure(executable: Path, home: Path, display: str, receipt_dir: Path,
                version: str) -> Dict:
    """Restart the installed GUI and complete one real CPU Measure run.

    :param executable: the frozen ``spacr`` binary.
    :param home: the private profile kept across updates.
    :param display: the private X display.
    :param receipt_dir: a new folder for this launch's smoke output.
    :param version: the package version the launch must report.
    :returns: the smoke verdict plus an independent database recount.
    """
    receipt_dir.mkdir(parents=True)
    smoke = receipt_dir / "smoke.json"
    environment = base_environment(home, display)
    environment.update(SPACR_BENCHMARK_JSON=str(smoke), SPACR_DISTRIBUTION_SMOKE="1",
                       SPACR_DISTRIBUTION_KIND="debian-frozen")
    started = time.monotonic()
    with open(receipt_dir / "application.log", "w") as log:
        try:
            code = subprocess.run([str(executable), "--no-setup"], env=environment,
                                  cwd=receipt_dir, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=720).returncode
        except subprocess.TimeoutExpired:
            code = None
    record = json.loads(smoke.read_text()) if smoke.is_file() else {}
    verdict = {"exit": code, "seconds": round(time.monotonic() - started, 1),
               "status": record.get("status"), "error": record.get("error"),
               "frozen": record.get("frozen"), "kind": record.get("kind"),
               "version": record.get("version"), "qt_platform": record.get("qt_platform"),
               "torch": record.get("torch"), "bundle_root": record.get("bundle_root"),
               "run_status": record.get("run_status"), "cells": record.get("cells"),
               "real_run_clicked": record.get("real_run_clicked"),
               "smoke_sha256": sha256_file(smoke) if smoke.is_file() else None}
    database = record.get("database")
    if database and Path(database).is_file():
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            verdict["integrity"] = connection.execute("PRAGMA integrity_check").fetchone()[0]
            verdict["independent_cells"] = connection.execute(
                "SELECT COUNT(*) FROM cell").fetchone()[0]
        finally:
            connection.close()
        verdict["database"] = database
        verdict["database_sha256"] = sha256_file(Path(database))
    verdict["passed"] = bool(
        code == 0 and verdict["status"] == "passed" and verdict["frozen"] is True
        and verdict["version"] == version and verdict.get("integrity") == "ok"
        and (verdict.get("independent_cells") or 0) >= 1
        and verdict["independent_cells"] == verdict["cells"])
    return verdict


def helper_refusal(bundle: Path, work: Path, home: Path, version: str) -> Dict:
    """Give the installed frozen helper a frozen ``linux-deb`` plan.

    :param bundle: the installed ``opt/spacr`` directory.
    :param work: a scratch folder outside the bundle.
    :param home: the private profile.
    :param version: the version the plan asks for.
    :returns: the helper's exit code, its message and whether the bundle is
        byte-identical afterwards.
    """
    helper = bundle / "_internal" / "spacr-update-helper"
    copy_root = work / "helper"
    copy_root.mkdir(mode=0o700, parents=True)
    extraction = copy_root / "extraction"
    extraction.mkdir(mode=0o700)
    copied = copy_root / helper.name
    shutil.copy2(helper, copied)
    plan = copy_root / "plan.json"
    plan.write_text(json.dumps({
        "schema": 1, "pid": 1, "version": version, "ticked": [],
        "records": [{"kind": "installer", "layout": "linux-deb", "platform": "linux",
                     "root": str(bundle), "running": True}],
        "log": str(copy_root / "update.log"), "workdir": str(copy_root),
        "fetch": None, "install": None, "relaunch": None,
        "frozen_application": True}))
    before = tree_inventory(bundle.parent.parent, [BUNDLE])
    environment = {"HOME": str(home), "PATH": "", "PYINSTALLER_RESET_ENVIRONMENT": "1",
                   "TMPDIR": str(extraction), "TMP": str(extraction), "TEMP": str(extraction)}
    result = subprocess.run([str(copied), "run-plan", str(plan)], env=environment,
                            capture_output=True, text=True, timeout=120)
    after = tree_inventory(bundle.parent.parent, [BUNDLE])
    return {"helper_sha256": sha256_file(helper), "exit": result.returncode,
            "message": (result.stdout + result.stderr).strip().splitlines()[-1:],
            "bundle_unchanged": before == after, "passed": result.returncode == 2
            and before == after and "nothing was changed" in result.stdout}


class Xvfb:
    """A private virtual X display for native (xcb) Qt acceptance."""

    def __init__(self, binary: Path, number: int = 97):
        """Start ``Xvfb`` on ``:number`` without TCP listening.

        :param binary: the Xvfb executable.
        :param number: the display number to claim.
        """
        self.display = f":{number}"
        self.process = subprocess.Popen(
            [str(binary), self.display, "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        socket = Path(f"/tmp/.X11-unix/X{number}")
        for _ in range(100):
            if socket.exists():
                return
            if self.process.poll() is not None:
                break
            time.sleep(0.1)
        self.close()
        raise RuntimeError(f"Xvfb did not start on {self.display}")

    def close(self) -> None:
        """Stop the display server."""
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=30)


def main(argv: Optional[List[str]] = None) -> int:
    """Run every scenario and write the receipt.

    :param argv: command-line arguments; :data:`sys.argv` when ``None``.
    :returns: 0 when every scenario passed, otherwise 1.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old", type=Path, required=True, help="older .deb")
    parser.add_argument("--new", type=Path, required=True, help="newer .deb")
    parser.add_argument("--work", type=Path, required=True, help="new scratch folder")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--xvfb", type=Path, help="Xvfb binary; GUI scenarios skipped without it")
    parser.add_argument("--display", type=int, default=97)
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True)
    home = args.work / "profile"
    (home / ".config").mkdir(parents=True)
    sentinel = home / ".spacr" / "preferences-sentinel.json"
    sentinel.parent.mkdir()
    sentinel.write_text('{"kept": true}\n')
    old_version, new_version = deb_field(args.old, "Version"), deb_field(args.new, "Version")
    receipt: Dict = {
        "schema": 1, "item": 53, "date": time.strftime("%Y-%m-%d"),
        "scope": "Debian frozen family: real dpkg replace, interrupted update, rollback and "
                 "post-update GUI restart/CPU Measure in a private unprivileged dpkg root",
        "host": {"platform": platform.platform(), "os_release": Path("/etc/os-release")
                 .read_text().splitlines()[:4]},
        "artifacts": {label: {"file": deb.name, "sha256": sha256_file(deb),
                              "version": deb_field(deb, "Version")}
                      for label, deb in (("old", args.old), ("new", args.new))},
        "caveats": [
            "Installed below a private root with fakeroot dpkg --root --force-depends; "
            "system dependency resolution and /usr/bin/spacr (which execs /opt/spacr/spacr) "
            "are not exercised; the frozen executable is launched directly.",
            "Xvfb is a private virtual display, not a desktop session.",
        ],
        "scenarios": {},
    }
    inventories = {"old": package_inventory(args.old), "new": package_inventory(args.new)}
    receipt["payload_files"] = {k: len(v) for k, v in inventories.items()}
    receipt["old_only_paths"] = len(set(inventories["old"]) - set(inventories["new"]))
    scenarios = receipt["scenarios"]
    xvfb = Xvfb(args.xvfb, args.display) if args.xvfb else None
    try:
        root = args.work / "root"
        dpkg = PrivateDpkg(root)
        executable = root / BUNDLE / "spacr"
        run = dpkg.run("-i", str(args.old))
        scenarios["install"] = {"dpkg": run, "status": dpkg.status(),
                                "tree": compare(inventories["old"], root),
                                "launch": launch_version(executable, base_environment(home, None))}
        scenarios["install"]["passed"] = (run["exit"] == 0 and scenarios["install"]["tree"]["exact"]
                                          and scenarios["install"]["status"]["abbrev"] == "ii")
        scenarios["helper"] = helper_refusal(root / BUNDLE, args.work, home, new_version)
        if xvfb:
            scenarios["gui_old"] = gui_measure(executable, home, xvfb.display,
                                               args.work / "launch-old", old_version)
        earlier = {str(p): sha256_file(p) for p in sorted(home.rglob("*")) if p.is_file()}
        run = dpkg.run("-i", str(args.new))
        tree = compare(inventories["new"], root)
        kept = {p: sha256_file(Path(p)) if Path(p).is_file() else None for p in earlier}
        scenarios["upgrade"] = {
            "dpkg": run, "status": dpkg.status(), "tree": tree,
            "profile_files_checked": len(earlier), "profile_unchanged": kept == earlier,
            "launch": launch_version(executable, base_environment(home, None))}
        scenarios["upgrade"]["passed"] = (
            run["exit"] == 0 and tree["exact"] and kept == earlier
            and scenarios["upgrade"]["status"] == {"abbrev": "ii", "status": "install ok installed",
                                                   "version": new_version})
        if xvfb:
            scenarios["gui_after_upgrade"] = gui_measure(executable, home, xvfb.display,
                                                         args.work / "launch-upgraded", new_version)
        run = dpkg.run("-i", str(args.old))
        scenarios["rollback"] = {"dpkg": run, "status": dpkg.status(),
                                 "tree": compare(inventories["old"], root),
                                 "launch": launch_version(executable, base_environment(home, None))}
        scenarios["rollback"]["passed"] = (run["exit"] == 0 and scenarios["rollback"]["tree"]["exact"]
                                           and scenarios["rollback"]["status"]["version"] == old_version)
        for label, recovery, expected in (("interrupt_then_rollback", args.old, "old"),
                                          ("interrupt_then_repair", args.new, "new")):
            kill = dpkg.interrupt_install(args.new)
            broken = {"status": dpkg.status(), "audit": dpkg.audit()[:6],
                      "launch": launch_version(executable, base_environment(home, None))}
            run = dpkg.run("-i", str(recovery))
            tree = compare(inventories[expected], root)
            status = dpkg.status()
            scenarios[label] = {"kill": kill, "after_kill": broken, "recovery": run,
                                "status": status, "tree": tree}
            scenarios[label]["passed"] = (
                kill["killed"] and "reinstreq" in broken["status"]["status"]
                and run["exit"] == 0 and tree["exact"] and status["abbrev"] == "ii"
                and status["version"] == deb_field(recovery, "Version"))
        if xvfb:
            scenarios["gui_after_interrupted_repair"] = gui_measure(
                executable, home, xvfb.display, args.work / "launch-repaired", new_version)
    finally:
        if xvfb:
            xvfb.close()
    receipt["all_passed"] = all(s.get("passed") for s in scenarios.values())
    receipt["gui_exercised"] = bool(args.xvfb)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return 0 if receipt["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
