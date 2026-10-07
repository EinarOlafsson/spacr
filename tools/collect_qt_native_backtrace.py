"""Collect a bounded native backtrace after a failed serial Qt run, if possible.

Read existing ELF cores in place. A matching systemd zstd core may be
temporarily extracted with disk, size and time bounds, then removed after
gdb. Only the text backtrace is uploaded; Apport reports are not extracted.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import resource
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MAX_BACKTRACE_BYTES = 4 * 1024 * 1024
MAX_CORE_BYTES = 16 * 1024 * 1024 * 1024
GDB_TIMEOUT_SECONDS = 90
CORE_EXTRACT_TIMEOUT_SECONDS = 60
CORE_DISK_RESERVE_BYTES = 512 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _session(evidence: Path) -> tuple[int | None, int]:
    journal = evidence / "file-rss.jsonl"
    try:
        with journal.open(encoding="utf-8") as source:
            first = json.loads(source.readline())
        if first.get("event") == "session_start":
            return int(first["pid"]), int(first["time_ns"])
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        pass
    return None, time.time_ns() - 6 * 60 * 60 * 1_000_000_000


def _core_type(path: Path) -> bool:
    try:
        with path.open("rb") as source:
            header = source.read(18)
    except OSError:
        return False
    if len(header) < 18 or header[:4] != b"\x7fELF":
        return False
    order = "<" if header[5] == 1 else ">" if header[5] == 2 else None
    return order is not None and struct.unpack(order + "H", header[16:18])[0] == 4


def _pid_in_name(name: str, pid: int | None) -> bool:
    if pid is None:
        return False
    digits = str(pid)
    return any(part == digits for part in re.split(r"\D+", name))


def _candidate_files(workspace: Path, evidence: Path, pid: int | None,
                     started_ns: int):
    locations = (
        (workspace, False),
        (evidence, False),
        (evidence.parent, False),
        (Path("/tmp"), True),
        (Path("/var/crash"), True),
        (Path("/var/lib/apport/coredump"), True),
        (Path("/var/lib/systemd/coredump"), True),
    )
    seen: set[Path] = set()
    for directory, shared in locations:
        try:
            entries = itertools.islice(directory.glob("core*"), 2000)
            for path in entries:
                if path in seen or path.is_symlink():
                    continue
                seen.add(path)
                if shared and not _pid_in_name(path.name, pid):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if (not path.is_file() or stat.st_mtime_ns < started_ns - 60_000_000_000
                        or stat.st_size > MAX_CORE_BYTES or not _core_type(path)):
                    continue
                yield path, stat.st_size
        except OSError:
            continue


def _apport_reports(started_ns: int) -> list[str]:
    reports = []
    for directory in (Path("/var/crash"), Path("/var/lib/systemd/coredump")):
        try:
            for path in itertools.islice(directory.iterdir(), 2000):
                if path.is_symlink() or path.suffix not in (".crash", ".zst"):
                    continue
                stat = path.stat()
                if stat.st_mtime_ns >= started_ns - 60_000_000_000:
                    reports.append(f"{path} ({stat.st_size} compressed bytes)")
        except OSError:
            continue
    return reports


def _cap_backtrace_file() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE,
                       (MAX_BACKTRACE_BYTES, MAX_BACKTRACE_BYTES))


def _extract_systemd_core(directory: Path, scratch: Path, pid: int | None,
                          started_ns: int, lines: list[str]) -> Path | None:
    """Expand only this process's recent zstd core into a bounded private file."""
    decoder = shutil.which("zstd")
    if pid is None or decoder is None:
        lines.append("systemd extraction unavailable: missing process identity or zstd")
        return None
    candidates = []
    try:
        for path in itertools.islice(directory.glob("core*.zst"), 2000):
            parts = path.name.split(".")
            if path.is_symlink() or len(parts) < 6 or parts[-3] != str(pid):
                continue
            stat = path.stat()
            if (path.is_file() and stat.st_mtime_ns >= started_ns - 60_000_000_000
                    and 0 < stat.st_size <= MAX_CORE_BYTES):
                candidates.append(path)
    except OSError as error:
        lines.append(f"systemd extraction discovery failed: {error}")
        return None
    if not candidates:
        return None
    source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    available = shutil.disk_usage(scratch).free - CORE_DISK_RESERVE_BYTES
    limit = min(MAX_CORE_BYTES, available)
    if limit < source.stat().st_size:
        lines.append("systemd extraction skipped: insufficient scratch disk")
        return None
    temporary = None
    accepted = False
    try:
        with tempfile.NamedTemporaryFile(prefix="qt-native-", suffix=".elf",
                                         dir=scratch, delete=False) as output:
            temporary = Path(output.name)

            def cap_extraction() -> None:
                resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))

            result = subprocess.run(
                [decoder, "--decompress", "--stdout", "--quiet", "--", str(source)],
                stdout=output, stderr=subprocess.PIPE, check=False,
                timeout=CORE_EXTRACT_TIMEOUT_SECONDS, preexec_fn=cap_extraction,
            )
        size = temporary.stat().st_size
        lines.append(f"systemd extraction source={source} exit={result.returncode} "
                     f"bytes={size} limit={limit}")
        if result.returncode == 0 and 0 < size <= limit and _core_type(temporary):
            accepted = True
            return temporary
        lines.append("systemd extraction rejected: failed decoder or non-core ELF")
    except (OSError, subprocess.TimeoutExpired) as error:
        lines.append(f"systemd extraction failed: {error}")
    finally:
        if temporary is not None and not accepted:
            temporary.unlink(missing_ok=True)
    return None


def main() -> int:
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    evidence = Path(os.environ["RUNNER_TEMP"]).resolve() / "spacr-qt-serial"
    evidence.mkdir(parents=True, exist_ok=True)
    report = evidence / "native-core-backtrace.txt"
    pid, started_ns = _session(evidence)
    executable = Path(sys.executable).resolve()
    source_sha = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=5, check=False,
    ).stdout.strip()
    expected_sha = os.environ.get("GITHUB_SHA", "")
    pattern = Path("/proc/sys/kernel/core_pattern").read_text(
        encoding="utf-8", errors="replace").strip()
    core_limit = resource.getrlimit(resource.RLIMIT_CORE)
    cores = list(_candidate_files(workspace, evidence, pid, started_ns))
    lines = [
        f"source_sha={source_sha}",
        f"expected_sha={expected_sha}",
        f"python_executable={executable}",
        f"python_sha256={_sha256(executable)}",
        f"serial_pid={pid}",
        f"core_pattern={pattern}",
        f"rlimit_core={core_limit}",
        f"regular_elf_cores={len(cores)}",
    ]
    lines.extend(f"candidate={path} size={size}" for path, size in cores)
    lines.extend(f"compressed_report={entry}"
                 for entry in _apport_reports(started_ns))
    extracted = None
    if not cores and source_sha == expected_sha and shutil.which("gdb"):
        extracted = _extract_systemd_core(
            Path("/var/lib/systemd/coredump"), evidence.parent, pid, started_ns, lines)
        if extracted is not None:
            cores.append((extracted, extracted.stat().st_size))
    if not cores:
        lines.append("No regular ELF core was available; no native backtrace can be recovered.")
    elif source_sha != expected_sha:
        lines.append("Checkout does not match the dispatched SHA; native backtrace skipped.")
    elif not shutil.which("gdb"):
        lines.append("gdb is unavailable; native backtrace skipped.")
    else:
        lines.append("gdb reads the core locally; the core is never uploaded.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if cores and source_sha == expected_sha and shutil.which("gdb"):
        core = max(cores, key=lambda candidate: candidate[0].stat().st_mtime_ns)[0]
        command = [
            "gdb", "-q", "-nx", "-nh", "-batch",
            "-iex", "set auto-load safe-path /dev/null",
            "-iex", "set debuginfod enabled off",
            "-ex", "set pagination off",
            "-ex", "info threads",
            "-ex", "thread apply all bt 24",
            str(executable), str(core),
        ]
        with report.open("ab") as output:
            output.write(f"\ncommand={' '.join(command)}\n".encode())
            output.flush()
            try:
                result = subprocess.run(
                    command, stdout=output, stderr=subprocess.STDOUT,
                    timeout=GDB_TIMEOUT_SECONDS, check=False,
                    preexec_fn=_cap_backtrace_file,
                )
                status = f"gdb_exit={result.returncode}"
            except subprocess.TimeoutExpired:
                status = f"gdb_timeout={GDB_TIMEOUT_SECONDS}s"
            finally:
                if extracted is not None:
                    extracted.unlink(missing_ok=True)
            with report.open("a", encoding="utf-8") as output:
                output.write(status + "\n")
            print(status)
    print(f"Native diagnostic report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
