"""Collect a bounded native backtrace after a failed Qt run, if possible.

Read existing ELF cores in place. A matching systemd zstd core may be
temporarily extracted with disk, size and time bounds, then removed after
gdb. Only the text backtrace is uploaded; Apport reports are not extracted.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import resource
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MAX_BACKTRACE_BYTES = 4 * 1024 * 1024
MAX_CORE_BYTES = 16 * 1024 * 1024 * 1024
MAX_PROGRAM_HEADER_TABLE_BYTES = 1024 * 256
GDB_TIMEOUT_SECONDS = 90
CORE_EXTRACT_TIMEOUT_SECONDS = 60
CORE_DISK_RESERVE_BYTES = 512 * 1024 * 1024

_ORDINARY_FAULT_SCRIPT = r"""
import gdb
import hashlib
import re
from pathlib import Path

_SHIBOKEN_612_SHA256 = "2b3d9767d69da0afabe4a383dc241a110fac4c0bb63cdeb3d01314e9c702241e"

def wrapper_type_at_fault(frame):
    '''Read only the class of an exact-version Shiboken faulting wrapper.'''
    if frame.name() != "Shiboken::BindingManager::unregisterWrapper(SbkObject*)":
        return
    try:
        library = gdb.solib_name(frame.pc())
        if not library or not library.endswith("/libshiboken6.abi3.so.6.12"):
            gdb.write("wrapper_type_diagnostic=unsupported_library\n")
            return
        library_path = Path(library)
        if not 0 < library_path.stat().st_size <= 1024 * 1024:
            gdb.write("wrapper_type_diagnostic=unsupported_library_size\n")
            return
        if hashlib.sha256(library_path.read_bytes()).hexdigest() != _SHIBOKEN_612_SHA256:
            gdb.write("wrapper_type_diagnostic=unsupported_library_hash\n")
            return
        version = int(gdb.parse_and_eval("Py_Version"))
        if (version >> 24) != 3 or ((version >> 16) & 255) != 12:
            gdb.write("wrapper_type_diagnostic=unsupported_python_abi\n")
            return
        wrapper = int(gdb.parse_and_eval("$r15"))
        if wrapper < 4096:
            gdb.write("wrapper_type_diagnostic=unreadable_wrapper\n")
            return
        memory = gdb.selected_inferior().read_memory
        type_pointer = int.from_bytes(memory(wrapper + 8, 8), "little")
        name_pointer = int.from_bytes(memory(type_pointer + 24, 8), "little")
        name = bytes(memory(name_pointer, 96)).split(b"\0", 1)[0]
        if not re.fullmatch(rb"[A-Za-z_][A-Za-z0-9_.]{0,95}", name):
            gdb.write("wrapper_type_diagnostic=unreadable_type_name\n")
            return
        gdb.write("wrapper_type_diagnostic=" + name.decode("ascii") +
                  " wrapper_ptr=" + hex(wrapper) + "\n")
    except Exception:
        gdb.write("wrapper_type_diagnostic=unavailable\n")

def diagnostic(command):
    try:
        gdb.execute(command)
    except Exception as error:
        gdb.write("ordinary fault diagnostic unsupported: " + command + ": " + str(error) + "\n")

diagnostic("set print elements 32")
diagnostic("set print max-depth 4")
diagnostic("p $_siginfo")
diagnostic("bt full 24")
try:
    newest = gdb.newest_frame()
    frame = newest
    original = newest
    selection = "current_fault_frame"
    for depth in range(24):
        if frame is None:
            break
        if frame.type() == gdb.SIGTRAMP_FRAME:
            original = frame.older()
            selection = "after_sigtramp"
            break
        frame = frame.older()
    if original is None:
        gdb.write("ordinary fault diagnostic unsupported: no original frame\n")
    else:
        original.select()
        gdb.write("ordinary_fault_frame selection=" + selection + " level=" + str(original.level())
                  + " name=" + str(original.name()) + "\n")
        diagnostic("info registers")
        diagnostic("x/16i $pc")
        wrapper_type_at_fault(original)
    if newest is not None:
        newest.select()
except Exception as error:
    gdb.write("ordinary fault diagnostic unsupported: original frame: " + str(error) + "\n")
"""


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


def _pid_in_name(name: str, pid: int | set[int] | None) -> bool:
    if pid is None:
        return False
    identities = {str(value) for value in pid} if isinstance(pid, set) else {str(pid)}
    return bool(identities.intersection(re.split(r"\D+", name)))


def _candidate_files(workspace: Path, evidence: Path, pid: int | set[int] | None,
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
                          started_ns: int, lines: list[str], *,
                          boot_id: str | None = None, strict_start: bool = False) -> Path | None:
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
            if strict_start and (
                parts[-4] != str(boot_id).replace("-", "")
                or not parts[-2].isdigit()
                or int(parts[-2]) * 1000 < started_ns
            ):
                continue
            stat = path.stat()
            minimum_ns = started_ns if strict_start else started_ns - 60_000_000_000
            if (path.is_file() and stat.st_mtime_ns >= minimum_ns
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


def _core_process_ids(path: Path, executable: Path,
                      diagnostics: list[str] | None = None) -> set[int]:
    """Read the unique Linux x86 process leader, never an LWP, for this executable."""
    observed: dict[str, object] = {}

    def result(reason: str, process_ids: set[int] | None = None) -> set[int]:
        """Record bounded ELF identity facts while retaining strict refusal."""
        if diagnostics is not None:
            details = {**observed, "reason": reason}
            diagnostics.append("elf_identity=" + json.dumps(details, sort_keys=True))
        return process_ids or set()

    try:
        with path.open("rb") as source:
            header = source.read(64)
            if len(header) != 64 or header[:4] != b"\x7fELF" or header[4] not in (1, 2):
                return result("invalid_elf_header")
            observed["elf_class"] = 64 if header[4] == 2 else 32
            order = "<" if header[5] == 1 else ">" if header[5] == 2 else None
            if order is None:
                return result("invalid_byte_order")
            if struct.unpack_from(order + "H", header, 16)[0] != 4:
                return result("not_core_elf")
            wide = header[4] == 2
            machine = struct.unpack_from(order + "H", header, 18)[0]
            observed["machine"] = machine
            if machine != (62 if wide else 3):
                return result("unsupported_machine")
            word = "Q" if wide else "I"
            offset = struct.unpack_from(order + word, header, 32 if wide else 28)[0]
            stride, count = struct.unpack_from(order + "HH", header, 54 if wide else 42)
            observed.update(program_headers=count, program_header_stride=stride)
            minimum = 56 if wide else 32
            if stride < minimum or stride > 256:
                return result("invalid_program_header_stride")
            table_bytes = stride * count
            observed["program_header_table_bytes"] = table_bytes
            if table_bytes > MAX_PROGRAM_HEADER_TABLE_BYTES:
                return result("program_header_table_byte_ceiling")
            source.seek(0, os.SEEK_END)
            file_bytes = source.tell()
            if offset > file_bytes or table_bytes > file_bytes - offset:
                return result("truncated_program_headers")
            source.seek(offset)
            entries = source.read(table_bytes)
            if len(entries) != table_bytes:
                return result("truncated_program_headers")
            process_ids = set()
            found_executable = False
            remaining = 4 * 1024 * 1024
            for index in range(count):
                entry = entries[index * stride: (index + 1) * stride]
                if struct.unpack_from(order + "I", entry)[0] != 4:
                    continue
                position = struct.unpack_from(order + word, entry, 8 if wide else 4)[0]
                size = struct.unpack_from(order + word, entry, 32 if wide else 16)[0]
                if size > remaining:
                    observed.update(note_bytes=size, note_bytes_remaining=remaining)
                    return result("note_byte_ceiling")
                remaining -= size
                source.seek(position)
                notes = source.read(size)
                if len(notes) != size:
                    return result("truncated_notes")
                cursor = 0
                while cursor + 12 <= len(notes):
                    names, length, kind = struct.unpack_from(order + "III", notes, cursor)
                    name_end = cursor + 12 + names
                    begin = cursor + 12 + ((names + 3) & ~3)
                    end = begin + length
                    cursor = begin + ((length + 3) & ~3)
                    if name_end > len(notes) or end > len(notes) or cursor > len(notes):
                        return result("malformed_note")
                    if notes[name_end - names:name_end].rstrip(b"\0") != b"CORE":
                        continue
                    payload = notes[begin:end]
                    if kind == 3 and len(payload) >= (28 if wide else 16):
                        process_ids.add(struct.unpack_from(order + "i", payload,
                                                           24 if wide else 12)[0])
                    elif kind == 0x46494C45 and len(payload) >= (16 if wide else 8):
                        maps = struct.unpack_from(order + word, payload)[0]
                        start = (2 + 3 * maps) * (8 if wide else 4)
                        if start > len(payload):
                            return result("malformed_nt_file")
                        names = payload[start:].split(b"\0")
                        found_executable |= os.fsencode(executable) in names[:maps]
            observed.update(process_pid_count=len(process_ids),
                            process_pids=sorted(process_ids)[:8],
                            executable_path_present=found_executable)
            if not process_ids:
                return result("missing_process_pid")
            if len(process_ids) != 1:
                return result("conflicting_process_pids")
            if not found_executable:
                return result("absent_executable_path")
            return result("accepted", process_ids)
    except (OSError, ValueError, struct.error, OverflowError):
        return result("unreadable_or_invalid_elf")


def _ordinary_sessions(evidence: Path, workspace: Path, source_sha: str,
                       expected_sha: str, executable: Path, lines: list[str]) -> list[dict]:
    """Accept only unfinished, current-source process journals from this runner."""
    sessions = []
    counts = {}
    if source_sha != expected_sha or not expected_sha:
        lines.append("ordinary identity refused: checkout differs from dispatched SHA")
        return sessions
    digest = _sha256(executable)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    for path in itertools.islice(evidence.glob("process-*.jsonl"), 2000):
        try:
            if path.is_symlink() or path.stat().st_size > 16384:
                continue
            records = [json.loads(row) for row in path.read_text().splitlines()]
            if not records:
                continue
            first = records[0]
            if (first.get("event") != "session_start" or first.get("mode") != "identity_only"
                    or first.get("source_sha") != source_sha
                    or first.get("root") != str(workspace)
                    or first.get("executable") != str(executable)
                    or first.get("executable_sha256") != digest
                    or first.get("boot_id") != boot
                    or int(first["pid"]) <= 0 or int(first["time_ns"]) <= 0
                    or int(first["process_start_ticks"]) <= 0):
                continue
            pid = int(first["pid"])
            counts[pid] = counts.get(pid, 0) + 1
            if not any(row.get("event") == "session_finish" for row in records):
                sessions.append(first)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    ambiguous = {pid for pid, count in counts.items() if count > 1}
    sessions = [row for row in sessions if int(row["pid"]) not in ambiguous]
    lines.append(f"ambiguous_process_pids_refused={sorted(ambiguous)}")
    lines.append(f"unfinished_owned_processes={len(sessions)}")
    return sorted(sessions, key=lambda row: int(row["time_ns"]), reverse=True)


def _ordinary_core(workspace: Path, evidence: Path, sessions: list[dict],
                   executable: Path, lines: list[str]) -> tuple[Path | None, Path | None]:
    """Recover at most one proven owned worker core within existing bounds."""
    if not sessions:
        return None, None
    owned = {int(row["pid"]): row for row in sessions}
    earliest = min(int(row["time_ns"]) for row in sessions)
    for path, _ in _candidate_files(workspace, evidence, set(owned), earliest):
        identities = _core_process_ids(path, executable)
        matches = [owned[pid] for pid in identities.intersection(owned)
                   if path.stat().st_mtime_ns >= int(owned[pid]["time_ns"])]
        if matches:
            session = max(matches, key=lambda row: int(row["time_ns"]))
            lines.append(f"matched_process={json.dumps(session, sort_keys=True)}")
            return path, None
    directory = Path("/var/lib/systemd/coredump")
    matches = []
    try:
        for path in itertools.islice(directory.glob("core*.zst"), 2000):
            parts = path.name.split(".")
            if len(parts) < 6 or not parts[-3].isdigit() or path.is_symlink():
                continue
            session = owned.get(int(parts[-3]))
            if session is None or not parts[-2].isdigit():
                continue
            if (parts[-4] == session["boot_id"].replace("-", "")
                    and int(parts[-2]) * 1000 >= int(session["time_ns"])
                    and path.stat().st_mtime_ns >= int(session["time_ns"])):
                matches.append(session)
    except OSError as error:
        lines.append(f"ordinary systemd discovery failed: {error}")
    if not matches:
        return None, None
    session = max(matches, key=lambda row: int(row["time_ns"]))
    extracted = _extract_systemd_core(
        directory, evidence.parent, int(session["pid"]), int(session["time_ns"]),
        lines, boot_id=session["boot_id"], strict_start=True,
    )
    if extracted is None:
        return None, None
    diagnostics: list[str] = []
    identities = _core_process_ids(extracted, executable, diagnostics)
    lines.extend(diagnostics)
    lines.append(f"elf_identity_expected_pid={int(session['pid'])}")
    if int(session["pid"]) in identities:
        lines.append(f"matched_process={json.dumps(session, sort_keys=True)}")
        return extracted, extracted
    extracted.unlink(missing_ok=True)
    lines.append("ordinary extraction rejected: ELF PID or executable differs")
    return None, None


def _record_core_route(workspace: Path, evidence: Path) -> None:
    """Record inherited core limits and the actual writable hosted capture route."""
    pattern = Path("/proc/sys/kernel/core_pattern").read_text().strip()
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    details = {"core_pattern": pattern, "core_limit_soft_bytes": soft,
               "core_limit_hard_bytes": hard, "core_limit_ceiling_bytes": MAX_CORE_BYTES}
    if pattern.startswith("|"):
        arguments = shlex.split(pattern[1:])
        handler = Path(arguments[0]) if arguments else Path("/nonexistent")
        storage = Path("/var/lib/systemd/coredump")
        supported = handler.name == "systemd-coredump" and os.access(handler, os.X_OK)
        details.update(route="systemd" if supported else "unsupported_pipe",
                       handler=str(handler), handler_executable=os.access(handler, os.X_OK),
                       storage=str(storage), storage_readable=os.access(storage, os.R_OK),
                       kernel_pipe_ignores_rlimit=True,
                       route_verified=supported and storage.is_dir())
    else:
        destination = Path(pattern)
        directory = destination.parent if destination.is_absolute() else workspace / destination.parent
        writable = directory.is_dir() and os.access(directory, os.W_OK)
        details.update(route="regular_file", directory=str(directory),
                       directory_writable=writable,
                       route_verified=writable and (soft == resource.RLIM_INFINITY or soft > 0))
    details["capture_guaranteed"] = False
    details["scope"] = "route preflight; core existence and ELF identity are checked after failure"
    (evidence / "core-route.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--ordinary", action="store_true")
    parser.add_argument("--record-core-route", action="store_true")
    options = parser.parse_args(argv)
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    evidence = (options.evidence_dir or
                Path(os.environ["RUNNER_TEMP"]) / "spacr-qt-serial").resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    if options.record_core_route:
        _record_core_route(workspace, evidence)
        return 0
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
    cores = [] if options.ordinary else list(
        _candidate_files(workspace, evidence, pid, started_ns))
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
    if not options.ordinary:
        lines.extend(f"compressed_report={entry}" for entry in _apport_reports(started_ns))
    extracted = None
    if options.ordinary:
        sessions = _ordinary_sessions(evidence, workspace, source_sha, expected_sha,
                                      executable, lines)
        if shutil.which("gdb"):
            core, extracted = _ordinary_core(workspace, evidence, sessions, executable, lines)
            if core is not None:
                cores.append((core, core.stat().st_size))
        lines.append("ordinary mode samples no RSS or Qt objects; unfinished does not prove SIGSEGV")
    elif not cores and source_sha == expected_sha and shutil.which("gdb"):
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
            "-ex", "info proc",
        ]
        if options.ordinary:
            command.extend(["-ex", "python exec(" + repr(_ORDINARY_FAULT_SCRIPT) + ")"])
        command.extend([
            "-ex", "info threads",
            "-ex", "thread apply all bt 24",
            str(executable), str(core),
        ])
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
            tail = (status + "\n").encode()
            with report.open("r+b") as output:
                output.seek(0, os.SEEK_END)
                if output.tell() + len(tail) > MAX_BACKTRACE_BYTES:
                    output.truncate(MAX_BACKTRACE_BYTES - len(tail))
                output.seek(0, os.SEEK_END)
                output.write(tail)
            print(status)
    print(f"Native diagnostic report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
