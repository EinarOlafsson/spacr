#!/usr/bin/env python3
"""Read ``canonical_sources()`` from an arbitrary revision, provably in isolation.

Item 306 pins the external caption inventory as a digest rather than a list, so
reviewing what a ratchet move admits means recomputing the PREVIOUS identity set
from an older tree and diffing it against the current one.  Doing that from a
live interpreter is what instruction 310 entry 30 warns about: an editable
install registers ``_EditableFinder`` on ``sys.meta_path``, and on this machine
that finder maps ``spacr`` to a DIFFERENT checkout than the one being read.
Nothing announces the substitution -- the import simply succeeds against the
wrong lineage.

So this tool never imports the target tree into the calling interpreter.  It
runs the builder in a subprocess whose ``sys.path`` and ``sys.meta_path`` cannot
reach any editable install, and then PROVES the isolation held by checking that
every loaded ``spacr*`` module resolves inside the target tree.  An unproven
isolation is the failure mode entry 30 describes; a claim of isolation that is
not checked would reproduce it.

Usage
-----

    # identity set for one revision
    python tools/i18n_reference_sources.py capture --rev d0c1a633c --out ref.json

    # identity set for the working tree
    python tools/i18n_reference_sources.py capture --tree . --out head.json

    # what a ratchet move admits, by name
    python tools/i18n_reference_sources.py diff ref.json head.json

    # report editable-install hazards in the CURRENT interpreter (entry 30)
    python tools/i18n_reference_sources.py doctor

The captured digest is computed exactly as
``tests/qt/test_i18n_caption_ratchet.py`` computes ``EXTERNAL_SOURCE_KEY_SHA256``
so a captured file can be compared against the pinned constant directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Tables of the external caption layer, and the ``canonical_sources()`` key
#: each is read from.  The names on the left are the ratchet's names; keeping
#: them identical is what lets a capture be compared to the pinned constant
#: without a translation step in between.
EXTERNAL_TABLES = {
    "SETTING_LABELS": "setting_labels",
    "SETTING_TOOLTIPS": "setting_tooltips",
    "CATEGORY_HELP": "categories",
    "UI": "ui",
    "MODULE_SUMMARIES": "module_summaries",
}

_BEGIN = "<<<SPACR_I18N_REF_BEGIN>>>"
_END = "<<<SPACR_I18N_REF_END>>>"

# The child program.  It is written to a temporary file OUTSIDE the target tree
# and outside this repository, so the tree under test contributes nothing but
# the modules being read -- not even the runner.
_CHILD = r'''
import json, os, sys, traceback

TREE = os.path.abspath(sys.argv[1])
BEGIN, END = sys.argv[2], sys.argv[3]

# 1. Remove every editable-install finder.  These sit on sys.meta_path, so they
#    answer imports that sys.path never sees, which is precisely how a foreign
#    checkout gets in without the operator noticing.
removed_finders = []
kept = []
for finder in sys.meta_path:
    name = getattr(finder, "__name__", type(finder).__name__)
    module = getattr(finder, "__module__", "") or ""
    if "_Editable" in name or module.startswith("__editable__"):
        removed_finders.append(f"{module}.{name}" if module else name)
        continue
    kept.append(finder)
sys.meta_path[:] = kept

# 2. Remove every sys.path entry that carries its own spacr package, and the
#    editable path placeholders.  Anything left cannot shadow the target.
removed_paths = []
survivors = []
for entry in sys.path:
    if not entry or entry.endswith(".__path_hook__"):
        if entry:
            removed_paths.append(entry)
        continue
    try:
        candidate = os.path.join(entry, "spacr")
        is_pkg = os.path.isdir(candidate) or os.path.isfile(candidate + ".py")
    except OSError:
        is_pkg = False
    if is_pkg and os.path.abspath(entry) != TREE:
        removed_paths.append(entry)
        continue
    survivors.append(entry)
sys.path[:] = survivors

# 3. Put the target tree first, and its tools/ directory with it -- the builder
#    is imported as a top-level module, the way the ratchet test imports it.
sys.path.insert(0, os.path.join(TREE, "tools"))
sys.path.insert(0, TREE)

result = {"tree": TREE, "removed_finders": removed_finders,
          "removed_paths": removed_paths}

try:
    import build_i18n_catalogs as builder
    canonical = builder.canonical_sources()
except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
    result["ok"] = False
    result["error_type"] = type(exc).__name__
    result["error"] = str(exc)
    result["traceback"] = traceback.format_exc()
    # Where did the two sides of a failed "cannot import name" actually live?
    # If both are inside TREE the tree is internally inconsistent at this
    # revision, which is a DIFFERENT fault from the module mixing this runner
    # exists to prevent, and must not be reported as if it were the same.
    runner = os.path.abspath(__file__)
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        path = tb.tb_frame.f_code.co_filename
        # This runner lives outside TREE by design; counting it would make
        # every failure look like a resolution fault.
        if path not in frames and os.path.abspath(path) != runner:
            frames.append(path)
        tb = tb.tb_next
    result["frame_files"] = frames
    considered = [
        f for f in frames if os.path.isabs(f) and os.path.exists(f)
    ]
    result["frames_all_inside_tree"] = bool(considered) and all(
        os.path.abspath(f).startswith(TREE + os.sep) for f in considered
    )
else:
    # 4. PROVE the isolation.  Every spacr module and the builder itself must
    #    resolve inside TREE.  Without this the runner would only be asserting
    #    its own good intentions.
    foreign = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == "spacr" or name.startswith("spacr.")
                or name == "build_i18n_catalogs"):
            continue
        path = getattr(module, "__file__", None)
        if path is None:
            paths = list(getattr(module, "__path__", []) or [])
            path = paths[0] if paths else None
        if path is None:
            continue
        if not os.path.abspath(path).startswith(TREE + os.sep):
            foreign[name] = path
    result["foreign_modules"] = foreign
    result["module_count"] = sum(
        1 for n in sys.modules if n == "spacr" or n.startswith("spacr."))
    result["ok"] = not foreign
    if not foreign:
        result["sources"] = {
            key: sorted(str(k) for k in canonical[key])
            for key in ("setting_labels", "setting_tooltips", "categories",
                        "ui", "module_summaries")
        }

sys.stdout.write("\n" + BEGIN + "\n")
sys.stdout.write(json.dumps(result))
sys.stdout.write("\n" + END + "\n")
'''


def _identities(sources: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Table/key pairs in the ratchet's own order."""
    return sorted(
        (table, key)
        for table, source_key in EXTERNAL_TABLES.items()
        for key in sources[source_key]
    )


def _digest(identities: list[tuple[str, str]]) -> str:
    """The fingerprint ``test_i18n_caption_ratchet`` pins, computed identically."""
    return hashlib.sha256(
        "\0".join(f"{table}\0{key}" for table, key in identities).encode("utf-8")
    ).hexdigest()


def _run_child(tree: Path, python: str) -> dict:
    """Read canonical_sources() from ``tree`` in a hermetic subprocess."""
    env = dict(os.environ)
    # PYTHONPATH is the other way a foreign checkout arrives, and it is
    # inherited silently.  Drop it rather than trying to filter it.
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("MPLBACKEND", "Agg")

    with tempfile.TemporaryDirectory(prefix="spacr-i18n-ref-") as scratch:
        child = Path(scratch) / "read_canonical_sources.py"
        child.write_text(_CHILD, encoding="utf-8")
        completed = subprocess.run(
            [python, str(child), str(tree), _BEGIN, _END],
            capture_output=True, text=True, env=env,
            # cwd is deliberately NOT the tree: an accidental '' on sys.path
            # must not be what makes this work.
            cwd=scratch,
        )

    out = completed.stdout
    if _BEGIN not in out or _END not in out:
        raise SystemExit(
            "the reader produced no result block; it likely died before it "
            f"could report.\nexit={completed.returncode}\n"
            f"--- stdout ---\n{out[-4000:]}\n--- stderr ---\n"
            f"{completed.stderr[-4000:]}"
        )
    payload = out.split(_BEGIN, 1)[1].split(_END, 1)[0].strip()
    return json.loads(payload)


def _worktree_for(rev: str, workdir: Path) -> Path:
    """Materialise ``rev`` as a detached worktree, reusing one if it is current."""
    sha = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", rev],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    target = workdir / sha[:12]
    if target.is_dir():
        return target
    workdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(ROOT), "worktree", "add", "--detach",
         str(target), sha],
        check=True, capture_output=True, text=True,
    )
    return target


def _report_failure(result: dict, label: str) -> None:
    """Say which of the two faults this is, because they have different fixes."""
    print(f"FAILED to read canonical_sources() from {label}", file=sys.stderr)
    print(f"  {result['error_type']}: {result['error']}", file=sys.stderr)
    print(file=sys.stderr)
    if result.get("frames_all_inside_tree"):
        print(
            "  DIAGNOSIS: every frame in that traceback is inside the target\n"
            "  tree, so no foreign module took part.  This revision does not\n"
            "  import ON ITS OWN -- it is internally inconsistent.  sys.path\n"
            "  isolation cannot fix it and did not cause it.  Pick a revision\n"
            "  that imports, or repair the tree in a scratch worktree.",
            file=sys.stderr,
        )
    elif result.get("foreign_modules"):
        print("  DIAGNOSIS: modules resolved OUTSIDE the target tree:",
              file=sys.stderr)
        for name, path in result["foreign_modules"].items():
            print(f"    {name} -> {path}", file=sys.stderr)
    else:
        print("  DIAGNOSIS: not a module-resolution fault; read the traceback.",
              file=sys.stderr)
    print(file=sys.stderr)
    print("  --- traceback ---", file=sys.stderr)
    print("  " + result.get("traceback", "").replace("\n", "\n  "),
          file=sys.stderr)


def _apply_patch(tree: Path, patch: Path) -> str:
    """Apply ``patch`` inside ``tree``, returning its sha256.

    Some historical revisions do not import on their own (see the DIAGNOSIS
    branch below).  Reading their caption inventory means repairing the import
    in a SCRATCH checkout -- never in the repository -- and saying exactly what
    was repaired.  The digest goes into the capture so a reviewer can tell a
    patched reading from an unpatched one without re-deriving it.
    """
    body = patch.read_bytes()
    completed = subprocess.run(
        ["git", "-C", str(tree), "apply", "--3way", str(patch.resolve())],
        capture_output=True, text=True,
    )
    if completed.returncode != 0:
        # Already applied is fine; a genuine conflict is not.
        check = subprocess.run(
            ["git", "-C", str(tree), "apply", "--reverse", "--check",
             str(patch.resolve())],
            capture_output=True, text=True,
        )
        if check.returncode != 0:
            raise SystemExit(
                f"could not apply {patch}:\n{completed.stderr}")
    return hashlib.sha256(body).hexdigest()


def cmd_capture(args: argparse.Namespace) -> int:
    if bool(args.rev) == bool(args.tree):
        raise SystemExit("give exactly one of --rev or --tree")
    if args.rev:
        tree = _worktree_for(args.rev, Path(args.workdir))
        label = f"{args.rev} ({tree})"
    else:
        tree = Path(args.tree).resolve()
        label = str(tree)

    patch_digest = None
    if args.patch:
        if tree == ROOT:
            raise SystemExit(
                "refusing to patch the repository itself; use --rev so the "
                "patch lands in a scratch worktree")
        patch_digest = _apply_patch(tree, Path(args.patch))
        print(f"applied {args.patch} (sha256 {patch_digest[:16]}…)",
              file=sys.stderr)

    result = _run_child(tree, args.python)

    if result.get("removed_finders"):
        print(f"isolated: removed meta_path finders "
              f"{result['removed_finders']}", file=sys.stderr)
    if result.get("foreign_modules"):
        print("ISOLATION FAILED -- foreign modules were loaded:",
              file=sys.stderr)
        for name, path in result["foreign_modules"].items():
            print(f"  {name} -> {path}", file=sys.stderr)
        return 2
    if not result.get("ok"):
        _report_failure(result, label)
        return 1

    sources = result["sources"]
    identities = _identities(sources)
    counts = {
        table: len(sources[key]) for table, key in EXTERNAL_TABLES.items()
    }
    payload = {
        "tree": str(tree),
        "rev": args.rev or None,
        "counts": counts,
        "total": sum(counts.values()),
        "digest": _digest(identities),
        "patch_sha256": patch_digest,
        "identities": [list(pair) for pair in identities],
        "spacr_modules_loaded": result.get("module_count"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"captured {payload['total']} identities from {label}")
    for table, count in counts.items():
        print(f"  {table:<18} {count}")
    print(f"  digest {payload['digest']}")
    print(f"-> {args.out}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    old = {tuple(pair) for pair in before["identities"]}
    new = {tuple(pair) for pair in after["identities"]}

    added, removed = sorted(new - old), sorted(old - new)
    print(f"before {before['total']} identities  digest {before['digest']}")
    print(f"after  {after['total']} identities  digest {after['digest']}")
    print(f"added {len(added)}, removed {len(removed)}")
    for heading, rows in (("ADDED", added), ("REMOVED", removed)):
        if not rows:
            continue
        print(f"\n--- {heading} ({len(rows)}) ---")
        for table, key in rows:
            print(f"  {table:<18} {key}")
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Report the entry-30 hazards present in THIS interpreter."""
    problems = 0
    print(f"interpreter: {sys.executable}")

    finders = [
        f for f in sys.meta_path
        if "_Editable" in getattr(f, "__name__", type(f).__name__)
        or (getattr(f, "__module__", "") or "").startswith("__editable__")
    ]
    for finder in finders:
        mapping = getattr(sys.modules.get(finder.__module__, None), "MAPPING", {})
        for name, path in sorted(mapping.items()):
            if name != "spacr" and not name.startswith("spacr"):
                continue
            exists = Path(path).is_dir()
            same = Path(path).resolve() == (ROOT / "spacr").resolve()
            status = "OK" if (exists and same) else "HAZARD"
            if status == "HAZARD":
                problems += 1
            print(f"  [{status}] editable {name} -> {path}"
                  f"{'' if exists else '   (MISSING)'}"
                  f"{'' if same else '   (NOT this checkout: ' + str(ROOT / 'spacr') + ')'}")

    for entry in sys.path:
        pkg = Path(entry) / "spacr" if entry else None
        if pkg and pkg.is_dir() and not (pkg / "__init__.py").exists():
            problems += 1
            print(f"  [HAZARD] {pkg} has no __init__.py; 'spacr' will resolve "
                  f"as a NAMESPACE package and submodules will appear missing")

    print(f"\n{problems} hazard(s)."
          + ("" if problems else "  This interpreter resolves spacr cleanly."))
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    default_workdir = Path(
        os.environ.get("TMPDIR", "/tmp")) / "spacr-i18n-reference-trees"

    capture = sub.add_parser(
        "capture", help="read one revision's caption identity set")
    capture.add_argument("--rev", help="git revision to read")
    capture.add_argument("--tree", help="existing checkout to read")
    capture.add_argument("--out", required=True, help="JSON output path")
    capture.add_argument("--python", default=sys.executable,
                         help="interpreter for the reader subprocess")
    capture.add_argument("--workdir", default=str(default_workdir),
                         help="where --rev worktrees are materialised")
    capture.add_argument("--patch", help="patch applied to the scratch "
                         "worktree before reading, for revisions that do not "
                         "import on their own; its sha256 is recorded")
    capture.set_defaults(func=cmd_capture)

    diff = sub.add_parser("diff", help="name what changed between two captures")
    diff.add_argument("before")
    diff.add_argument("after")
    diff.set_defaults(func=cmd_diff)

    doctor = sub.add_parser(
        "doctor", help="report editable-install hazards in this interpreter")
    doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
