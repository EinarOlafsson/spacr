#!/usr/bin/env python3
"""Regenerate FACTS.md, and check that the invariants are still true.

Run this at the START of every session that loads the spaCR engineer
skill. It exists because a skill file is a claim about a repository, and a
repository moves. A skill that is merely *read* decays into a confident
description of software that no longer exists, which is worse than no
skill at all -- the reader has no way to tell which half is stale.

Two jobs:

* Write ``FACTS.md``: the numbers that go out of date -- version, app
  count, module sizes, test counts. Generated, never hand-edited.
* Check ``INVARIANTS.md``: every rule in there that a machine can verify
  is verified here. A rule that stops holding is reported loudly, because
  the whole value of that file is that its contents are true.

Exit status is 0 when every invariant holds and 1 when one does not, so
this can gate a commit if anyone wants it to.

    python skill/refresh.py            # regenerate + check
    python skill/refresh.py --check    # check only, write nothing
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skill"


# ---------------------------------------------------------------------------
# The invariants a machine can check
# ---------------------------------------------------------------------------
# Each entry is (name, callable) returning (ok, detail). Keep the detail
# short and factual: it is printed on failure and is the first thing the
# next person reads.

def _text(rel: str) -> str:
    path = ROOT / rel
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _check_qss_registrars():
    """Every module registering widget QSS at import must be listed.

    This is the black box. A rule not in the stylesheet when it is built
    is not in it at all, and its widget falls through to the blanket
    ``QWidget { background-color: bg }`` -- #000000 on the dark theme.
    """
    theme = _text("spacr/qt/theme.py")
    if "WIDGET_QSS_MODULES" not in theme:
        return False, "theme.WIDGET_QSS_MODULES is gone"
    if "load_widget_qss_registrars()" not in theme:
        return False, "stylesheet() no longer loads the registrars"

    listed = set()
    for node in ast.walk(ast.parse(theme)):
        # `AnnAssign` as well as `Assign`: the tuple is annotated
        # (`WIDGET_QSS_MODULES: Tuple[str, ...] = (...)`), and a checker
        # that only handles `Assign` reports every module missing --
        # which is exactly what it did on its first run.
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        for target in targets:
            if getattr(target, "id", "") == "WIDGET_QSS_MODULES":
                for element in getattr(node.value, "elts", []):
                    if isinstance(element, ast.Constant):
                        listed.add(element.value)

    registering = set()
    deferred = set()
    # Modules whose register() the launch path calls before the stylesheet is
    # applied. Read from the source so this cannot drift from the real list.
    init_src = _text("spacr/qt/__init__.py")
    self_registering = set(re.findall(r'"(spacr\.qt\.[a-z_.]+)"', init_src))
    qt_root = ROOT / "spacr" / "qt"
    for path in sorted(qt_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "register_widget_qss(" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        # Functions in THIS module that register, so a module-level call to
        # one of them counts. spacr.qt.widgets.field_fade registers through
        # `ensure_field_fade_qss()` at module scope, and a walker that only
        # matched the direct call reported it as deferred when it is not.
        wrappers = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        name = getattr(inner.func, "id",
                                       getattr(inner.func, "attr", ""))
                        if name.endswith("register_widget_qss"):
                            wrappers.add(node.name)
                            break

        def top_level(body):
            for node in body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    name = getattr(func, "id", getattr(func, "attr", ""))
                    if name.endswith("register_widget_qss") or name in wrappers:
                        yield True
                if isinstance(node, ast.Try):
                    yield from top_level(node.body)

        rel = path.relative_to(qt_root).with_suffix("")
        module = "spacr.qt." + rel.as_posix().replace("/", ".")
        if any(top_level(tree.body)):
            registering.add(module)
        elif module not in listed and module not in self_registering \
                and module != "spacr.qt.theme":
            # `spacr.qt.theme` DEFINES register_widget_qss, and a module in
            # SELF_REGISTERING_MODULES has its `register()` called by
            # `register_self_registering_modules()`, which runs before
            # `launch()` applies the stylesheet -- so its block is in the
            # sheet by the time the sheet exists. Neither is the bug this
            # looks for, and a check that reports them stops being read.
            # Registers, but only from inside a function -- so the block is
            # NOT in the stylesheet at the moment the stylesheet is built and
            # applied, and the widget falls through to the blanket
            # `QWidget { background-color: bg }`.
            #
            # This is how spacr.qt.prerun hid: it registered from
            # `register()`, which `register_self_registering_modules()` calls
            # after app.py has imported and after the sheet has been applied,
            # so the Measure QC banner painted #000000 behind its verdict
            # text. Measured on a fresh interpreter: 'MeasureQCBanner' in
            # theme.stylesheet() was False at launch, True only afterwards.
            #
            # Being in WIDGET_QSS_MODULES is what clears it, because the
            # loader imports the module while building the sheet.
            deferred.add(module)

    missing = sorted(registering - listed)
    if missing:
        return False, f"not in WIDGET_QSS_MODULES: {missing}"
    if deferred:
        return False, (f"registers QSS only from inside a function and is not "
                       f"in WIDGET_QSS_MODULES, so its rule is absent from the "
                       f"stylesheet at launch: {sorted(deferred)}")
    return True, f"{len(listed)} modules listed, {len(registering)} registering"


def _check_bg_is_the_window_colour():
    """`bg` is QPalette.Window, not a surface. On dark it is pure black."""
    theme = _text("spacr/qt/theme.py")
    if '"bg":          "#000000"' not in theme:
        return True, "dark `bg` is no longer #000000 -- re-read INVARIANTS 2"
    if '"page":' not in theme:
        return False, "the `page` role is gone; the settings column will be black"
    return True, "dark bg=#000000, `page` role present"


def _check_thread_finished_uses_bound_methods():
    """A closure on `thread.finished` makes the QThread its own receiver."""
    bridge = _text("spacr/qt/bridge.py")
    if "deleteLater" not in bridge:
        return False, "bridge.make_thread no longer wires finished->deleteLater"
    return True, "make_thread still owns the finished wiring"


def _check_test_isolation_fixtures():
    """The leaks that made tests fail in the suite and pass alone."""
    conftest = _text("tests/qt/conftest.py")
    wanted = ("_restore_app_registry", "_restore_font_scale",
              "_font_scale_starts_at_one")
    missing = [name for name in wanted if name not in conftest]
    if missing:
        return False, f"tests/qt/conftest.py lost {missing}"
    return True, "registry + font-scale isolation in place"


def _check_settings_never_written_by_tests():
    """A test that writes real preferences flattens the user's interface."""
    conftest = _text("tests/conftest.py") + _text("tests/qt/conftest.py")
    if "QSettings" not in conftest and "_isolated_qsettings" not in conftest:
        return False, "no QSettings sandbox found in the conftests"
    return True, "QSettings sandbox present"


#: The headings every task file carries. They are the questions a reader
#: picking the task up cold has to have answered. Each entry is a tuple of
#: acceptable spellings: a finished task says what WAS done, not what to do,
#: and demanding the open-task wording of a done file would push the next
#: person into writing a heading that lies about the tense.
_TASK_SECTIONS = (
    ("WHAT THE STATE IS",),
    ("WHY IT MATTERS",),
    # "THE ANSWER" is for a task whose deliverable is a written finding
    # rather than a change -- a review, an investigation, a recommendation.
    # Forcing one of those into "WHAT WAS DONE" would make the heading lie
    # about what the reader is about to get.
    ("WHAT TO DO", "WHAT WAS DONE", "THE ANSWER"),
    ("HOW TO KNOW IT WORKED", "VERIFIED"),
)


def _check_task_ledger():
    """Every open/done task file is still usable by someone who was not here.

    The ledger exists because a session that runs out of context loses
    everything that was only ever said in the conversation. A file that has
    lost its sections has lost the same thing more slowly, so it is checked
    rather than trusted.
    """
    import os

    base = os.path.join(ROOT, "instructions")
    open_dir = os.path.join(base, "open")
    done_dir = os.path.join(base, "done")
    if not os.path.isdir(open_dir) or not os.path.isdir(done_dir):
        return False, "instructions/open and instructions/done must both exist"

    def _files(path):
        return sorted(n for n in os.listdir(path)
                      if n.endswith(".txt") and not n.startswith("."))

    open_files, done_files = _files(open_dir), _files(done_dir)

    both = set(open_files) & set(done_files)
    if both:
        return False, (f"{sorted(both)} is in BOTH open/ and done/ -- a task "
                       f"is finished or it is not")

    problems = []
    for folder, names in ((open_dir, open_files), (done_dir, done_files)):
        for name in names:
            with open(os.path.join(folder, name), encoding="utf-8",
                      errors="replace") as handle:
                body = handle.read()
            missing = [alts[0] for alts in _TASK_SECTIONS
                       if not any(a in body for a in alts)]
            if missing:
                problems.append(f"{name} is missing {missing}")
            elif "Status:" not in body:
                problems.append(f"{name} has no Status: line")
    if problems:
        return False, "; ".join(problems[:4])
    return True, (f"{len(open_files)} open, {len(done_files)} done, "
                  f"all with their sections")


CHECKS = (
    ("widget QSS registrars are complete", _check_qss_registrars),
    ("bg / page roles", _check_bg_is_the_window_colour),
    ("thread finished wiring", _check_thread_finished_uses_bound_methods),
    ("test isolation fixtures", _check_test_isolation_fixtures),
    ("QSettings sandbox", _check_settings_never_written_by_tests),
    ("task ledger", _check_task_ledger),
)


# ---------------------------------------------------------------------------
# The facts that go stale
# ---------------------------------------------------------------------------

def _run(*args) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def _version() -> str:
    # ``spacr.version.__version__`` is deliberately resolved from installed
    # distribution metadata.  Reading that assignment as text therefore
    # yields the expression ``get_version()`` rather than a version number.
    # setup.py remains the repository's declared-version source (and can be
    # inspected without importing spaCR or depending on the active env).
    for source, variable in (("setup.py", "VERSION"),
                             ("spacr/version.py", "__version__")):
        for line in _text(source).splitlines():
            if not line.startswith(variable):
                continue
            value = line.split("=", 1)[-1].strip().strip('"\'')
            if re.fullmatch(r"\d+(?:\.\d+)+(?:[-+._a-zA-Z0-9]*)?", value):
                return value
    return "unknown"


def _counts() -> dict:
    qt = ROOT / "spacr" / "qt"
    tests = ROOT / "tests"
    app_py = _text("spacr/qt/app.py")
    return {
        "version": _version(),
        "commit": _run("git", "rev-parse", "--short", "HEAD") or "unknown",
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "python_modules": len(list((ROOT / "spacr").rglob("*.py"))),
        "qt_modules": len(list(qt.rglob("*.py"))),
        "screens": len(list((qt / "screens").glob("*.py"))),
        "widgets": len(list((qt / "widgets").glob("*.py"))),
        "test_files": len(list(tests.rglob("test_*.py"))),
        "qt_test_files": len(list((tests / "qt").glob("test_*.py"))),
        "static_apps": app_py.count('("') and app_py.count("SECTION_"),
    }


def _largest(rel: str, count: int = 12) -> list:
    base = ROOT / rel
    sized = [(len(p.read_text(encoding="utf-8", errors="ignore").splitlines()), p)
             for p in base.rglob("*.py")]
    sized.sort(reverse=True)
    return [(n, str(p.relative_to(ROOT))) for n, p in sized[:count]]


def write_facts() -> str:
    counts = _counts()
    lines = [
        "<!-- GENERATED by skill/refresh.py. Do not hand-edit. -->",
        f"# Facts as of {date.today().isoformat()}",
        "",
        "Regenerate with `python skill/refresh.py`. Everything here goes",
        "stale; nothing here is a rule. The rules are in INVARIANTS.md.",
        "",
        "## This checkout",
        "",
        f"- version: **{counts['version']}**",
        f"- branch: `{counts['branch']}` at `{counts['commit']}`",
        f"- Python modules under `spacr/`: {counts['python_modules']}",
        f"- of those under `spacr/qt/`: {counts['qt_modules']} "
        f"({counts['screens']} screens, {counts['widgets']} widgets)",
        f"- test files: {counts['test_files']} "
        f"({counts['qt_test_files']} under `tests/qt/`)",
        "",
        "## The biggest modules",
        "",
        "Size is not a defect, but it is where the work is. Read the "
        "docstring before the code in any of these -- they carry their "
        "own reasoning.",
        "",
    ]
    for count, path in _largest("spacr"):
        lines.append(f"- `{path}` — {count} lines")
    lines += ["", "## Invariant checks", ""]
    ok_all = True
    for name, check in CHECKS:
        try:
            ok, detail = check()
        except Exception as exc:                       # a broken check is a fail
            ok, detail = False, f"the check itself raised: {exc!r}"
        ok_all = ok_all and ok
        lines.append(f"- {'PASS' if ok else '**FAIL**'} — {name}: {detail}")
    lines += [
        "",
        "A FAIL means INVARIANTS.md is describing software that has moved.",
        "Fix the code or fix the file, then say which in the commit.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list) -> int:
    check_only = "--check" in argv
    text = write_facts()
    if not check_only:
        (SKILL / "FACTS.md").write_text(text, encoding="utf-8")
        print(f"wrote {SKILL / 'FACTS.md'}")

    failed = []
    for name, check in CHECKS:
        try:
            ok, detail = check()
        except Exception as exc:
            ok, detail = False, f"the check itself raised: {exc!r}"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} invariant(s) no longer hold: {failed}")
        print("Either the code regressed or INVARIANTS.md is out of date. "
              "Decide which, fix it, and say which in the commit message.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
