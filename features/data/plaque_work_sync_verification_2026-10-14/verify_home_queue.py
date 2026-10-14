#!/usr/bin/env python3
"""Read-only Home saved-queue byte comparator; run under existing run_capped.

Example (replace every path with the actual Home paths):
  /path/to/worktree/tools/run_capped.sh 4G /path/to/python verify_home_queue.py \
    --worktree /path/to/worktree --queue /mapped/af3/queue \
    --manifest /path/to/committed/queue-content-set-identity.tsv \
    --output /new/private/scratch/home-queue-comparison.json

The output parent must already exist. This script never writes queue files,
decodes images, imports the app, transfers files, or merges annotation ledgers.
Exit 0 means the saved byte set matched during this observation, not GUI-save,
global quiescence, path rebinding, annotation correctness, or sync acceptance.
Exit 1 means comparison failed; exit 2 means invalid arguments/input or failure
to create a NEW evidence output. All symlinks (even contained ones) are refused.
"""

import argparse
import sys


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True,
                        help="Actual Home caller worktree; placed first on sys.path")
    parser.add_argument("--queue", required=True, help="Exact absolute Home queue root")
    parser.add_argument("--manifest", required=True, help="Committed baseline TSV")
    parser.add_argument("--output", required=True, help="NEW JSON outside the queue")
    return parser.parse_args()


ARGS = arguments()
# No app imports. Parse the caller's worktree before further stdlib imports.
sys.path.insert(0, ARGS.worktree)

import datetime
import hashlib
import json
import os
import re
import stat
import time

BASELINE_SHA256 = "b48bb00dfe46d9433b2dafc15229fd322e971885e67d8230876e9682941f8d52"
EXPECTED_COUNT = 16981
CHUNK_BYTES = 1024 * 1024
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def absolute_parts(value):
    if not value.startswith("/") or "\x00" in value:
        raise ValueError("Paths must be absolute and contain no NUL")
    parts = value.split("/")[1:]
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError("Paths must have canonical components; no traversal or empty parts")
    return parts


def relative_parts(value):
    parts = value.split("/")
    if (value.startswith("/") or "\x00" in value or "\n" in value
            or "\r" in value or "\t" in value
            or any(part in ("", ".", "..") for part in parts)):
        raise ValueError("Unsafe manifest relative path")
    return parts


def signature(info):
    # Exclude atime: this read-only content scan may update filesystem atime.
    return {"device": info.st_dev, "inode": info.st_ino, "mode": info.st_mode,
            "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns}


def directory(value):
    """Open every absolute component without following symlinks."""
    fd = os.open("/", DIR_FLAGS)
    try:
        for component in absolute_parts(value):
            next_fd = os.open(component, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_relative(root_fd, relative):
    """Open a regular file beneath an already pinned directory descriptor."""
    parts = relative_parts(relative)
    fd = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        result = os.open(parts[-1], READ_FLAGS, dir_fd=fd)
        try:
            if not stat.S_ISREG(os.fstat(result).st_mode):
                raise ValueError("Not a regular file")
        except BaseException:
            os.close(result)
            raise
        return result
    finally:
        os.close(fd)


def read_file(fd, collect=False):
    before = signature(os.fstat(fd))
    digest = hashlib.sha256()
    chunks = [] if collect else None
    count = 0
    while True:
        chunk = os.read(fd, CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        count += len(chunk)
        if collect:
            chunks.append(chunk)
    after = signature(os.fstat(fd))
    return digest.hexdigest(), before, after, count, b"".join(chunks) if collect else None


def inventory(root_fd):
    files, directories, rejected, errors = {}, {}, [], []

    def visit(fd, prefix):
        directories[prefix or "."] = signature(os.fstat(fd))
        try:
            with os.scandir(fd) as entries:
                names = sorted(entry.name for entry in entries)
        except OSError as error:
            errors.append({"relative_path": prefix or ".", "error_type": type(error).__name__})
            return
        for name in names:
            relative = prefix + "/" + name if prefix else name
            try:
                relative_parts(relative)
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    rejected.append({"relative_path": relative, "reason": "symlink_refused"})
                elif stat.S_ISDIR(info.st_mode):
                    child = os.open(name, DIR_FLAGS, dir_fd=fd)
                    try:
                        if signature(os.fstat(child)) != signature(info):
                            raise ValueError("Directory changed before opening")
                        visit(child, relative)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    files[relative] = signature(info)
                else:
                    rejected.append({"relative_path": relative, "reason": "nonregular_file_refused"})
            except (OSError, ValueError) as error:
                errors.append({"relative_path": relative, "error_type": type(error).__name__})

    visit(root_fd, "")
    return {"files": files, "directories": directories,
            "rejected_entries": rejected, "errors": errors}


def difference(before, after):
    return {"added": sorted(after.keys() - before.keys()),
            "removed": sorted(before.keys() - after.keys()),
            "changed_stat": sorted(path for path in before.keys() & after.keys()
                                   if before[path] != after[path])}


def manifest():
    parent, name = os.path.split(ARGS.manifest)
    fd = directory(parent)
    try:
        handle = open_relative(fd, name)
        try:
            digest, before, after, count, raw = read_file(handle, collect=True)
        finally:
            os.close(handle)
    finally:
        os.close(fd)
    if before != after or count != before["bytes"]:
        raise ValueError("Manifest changed during reading")
    if digest != BASELINE_SHA256:
        raise ValueError("Manifest does not match the pinned Work baseline")
    rows = raw.decode("utf-8").splitlines()
    expected = {}
    for row in rows:
        fields = row.split("\t")
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[1]):
            raise ValueError("Malformed manifest row")
        relative_parts(fields[0])
        if fields[0] in expected:
            raise ValueError("Duplicate manifest relative path")
        expected[fields[0]] = fields[1]
    if len(expected) != EXPECTED_COUNT or list(expected) != sorted(expected):
        raise ValueError("Manifest count/order differs from the Work baseline")
    return expected, {"path": ARGS.manifest, "sha256": digest, "bytes": count,
                      "stat_before": before, "stat_after": after}


def output_parent(queue_fd):
    parts = absolute_parts(ARGS.output)
    if ARGS.output == ARGS.queue or ARGS.output.startswith(ARGS.queue + "/"):
        raise ValueError("Evidence output must be outside the queue")
    parent = "/" + "/".join(parts[:-1])
    if parent == "/":
        raise ValueError("Use an existing private scratch output directory")
    # Reject a bind-mount alias of the queue, including aliases in any ancestor.
    queue_identity = (os.fstat(queue_fd).st_dev, os.fstat(queue_fd).st_ino)
    fd = os.open("/", DIR_FLAGS)
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) == queue_identity:
                raise ValueError("Output parent aliases the queue")
        try:
            os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return fd, parts[-1]
        raise ValueError("Preserve existing evidence: output already exists")
    except BaseException:
        os.close(fd)
        raise


def main():
    started = utc()
    clock = time.monotonic()
    for value in (ARGS.worktree, ARGS.queue, ARGS.manifest, ARGS.output):
        absolute_parts(value)
    worktree_fd = directory(ARGS.worktree)
    os.close(worktree_fd)
    expected, bound_manifest = manifest()
    queue_fd = directory(ARGS.queue)
    out_fd = None
    try:
        out_fd, out_name = output_parent(queue_fd)
        before = inventory(queue_fd)
        results = []
        for relative, required_hash in expected.items():
            row = {"relative_path": relative, "expected_sha256": required_hash}
            try:
                handle = open_relative(queue_fd, relative)
                try:
                    digest, start, end, count, _ = read_file(handle)
                finally:
                    os.close(handle)
                row.update({"observed_sha256": digest, "bytes_read": count,
                            "stat_before_read": start, "stat_after_read": end,
                            "hash_matches": digest == required_hash,
                            "stable_read_and_initial_stat": start == end == before["files"].get(relative)
                            and count == start["bytes"]})
            except (OSError, ValueError) as error:
                row.update({"error_type": type(error).__name__, "hash_matches": False,
                            "stable_read_and_initial_stat": False})
            results.append(row)
        after = inventory(queue_fd)
        # Also refuse a separately mounted alias of a nested queue directory.
        # The scratch directory must be trusted/private and not concurrently moved.
        parent_info = os.fstat(out_fd)
        parent_identity = (parent_info.st_dev, parent_info.st_ino)
        queue_directory_identities = {
            (row["device"], row["inode"])
            for view in (before, after) for row in view["directories"].values()
        }
        if parent_identity in queue_directory_identities:
            raise ValueError("Output parent aliases a nested queue directory")
        reopened = directory(ARGS.queue)
        try:
            root_path_matches_descriptor = signature(os.fstat(reopened)) == signature(os.fstat(queue_fd))
        finally:
            os.close(reopened)
        file_changes = difference(before["files"], after["files"])
        directory_changes = difference(before["directories"], after["directories"])
        missing = sorted(expected.keys() - after["files"].keys())
        extra = sorted(after["files"].keys() - expected.keys())
        conflicts = [row for row in results if not row["hash_matches"]]
        unstable = [row["relative_path"] for row in results if not row["stable_read_and_initial_stat"]]
        rejected = before["rejected_entries"] + after["rejected_entries"]
        inventory_errors = before["errors"] + after["errors"]
        matched = not (missing or extra or conflicts or unstable or rejected or inventory_errors
                       or any(file_changes.values()) or any(directory_changes.values()))
        matched = matched and root_path_matches_descriptor
        report = {"schema_version": 1, "started_utc": started, "finished_utc": utc(),
                  "elapsed_seconds": round(time.monotonic() - clock, 3),
                  "worktree_first_sys_path": ARGS.worktree,
                  "queue_argument": ARGS.queue, "baseline_manifest": bound_manifest,
                  "expected_file_count": EXPECTED_COUNT, "expected_files_attempted": len(results),
                  "hashes_matching": sum(row["hash_matches"] for row in results),
                  "saved_byte_set_matches_baseline_during_observation": matched,
                  "missing_files": missing, "extra_files": extra,
                  "content_conflict_or_unreadable_files": conflicts,
                  "sync_conflict_named_paths": sorted(path for path in after["files"]
                                                      if ".sync-conflict-" in path),
                  "unstable_or_unreadable_files": unstable,
                  "file_inventory_changes": file_changes,
                  "directory_inventory_changes": directory_changes,
                  "root_path_matches_pinned_descriptor": root_path_matches_descriptor,
                  "initial_inventory": before, "final_inventory": after,
                  "file_results": results,
                  "limitations": ["Saved encoded bytes only; no image/mask decoding or GUI inspection.",
                                  "Stat checks cannot prove global quiescence or detect every transient rewrite.",
                                  "No unsaved GUI-buffer, biological/annotation-quality, verdict merge, path-rebinding or sync-readiness acceptance.",
                                  "Content conflicts mean byte differences or read failures; expert judgment conflicts are not parsed.",
                                  "All symlinks and nonregular files are refused; no queue writes or transfers.",
                                  "Output requires a trusted private scratch directory that is not concurrently moved into the queue.",
                                  "Read access may update filesystem atime; atime is excluded from stability comparisons."]}
        encoded = (json.dumps(report, ensure_ascii=True, indent=2) + "\n").encode("utf-8")
        output = os.open(out_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                         | os.O_CLOEXEC, 0o600, dir_fd=out_fd)
        with os.fdopen(output, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"output": ARGS.output, "output_sha256": hashlib.sha256(encoded).hexdigest(),
                          "expected_files_attempted": len(results), "matching": matched,
                          "missing_count": len(missing), "extra_count": len(extra),
                          "content_conflict_or_unreadable_count": len(conflicts),
                          "unstable_or_unreadable_count": len(unstable)}), flush=True)
        return 0 if matched else 1
    finally:
        if out_fd is not None:
            os.close(out_fd)
        os.close(queue_fd)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, UnicodeError) as error:
        print(json.dumps({"status": "invalid_input_or_incomplete_audit",
                          "error_type": type(error).__name__, "message": str(error)}), file=sys.stderr)
        sys.exit(2)
