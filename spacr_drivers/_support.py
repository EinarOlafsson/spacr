"""Preconditions and scratch staging shared by the real-data drivers.

A driver names the dataset root it reads and the files inside it the run
needs. This module turns that declaration into three guarantees, each of
which a script written without them gets wrong quietly:

* **A run that cannot succeed does not start.** :func:`require` reports
  *every* missing input at once, names the root they were looked for under,
  and says how to point the driver somewhere else. A driver that discovers
  the third of five inputs missing halfway through has already written
  output, burned GPU time and left a tree that looks like a result.
* **The dataset is never written to.** :func:`stage` copies the declared
  inputs into a scratch tree and refuses outright when the destination lies
  inside the dataset root, so "copy, never write in place" is enforced
  rather than remembered.
* **Settings are checked before the work starts.** :func:`preflight` runs
  spaCR's own pre-flight validator and stops on any error it reports, which
  is the same check the GUI makes and the one a bare script skips.
* **A wrong answer is an exit code, not a printed line.** :func:`check`
  raises :class:`WrongAnswer`, which leaves the process with status 1 and the
  reason on stderr. A driver that prints "0 of 4,800 reads mapped" and exits
  0 has reported the failure to nobody: the caller sees success, and the
  number scrolls away. Status 2 stays reserved for "the data is not here",
  so a caller can still tell an absent disk from a broken run.

Nothing here imports spaCR, numpy or torch at module scope: the refusal path
must cost a bare interpreter start, so pointing a driver at an unmounted
disk answers immediately instead of after a multi-second import.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path


class MissingData(Exception):
    """A declared input is absent, so the run must not start.

    Carries the whole list rather than the first offender: fixing one missing
    path at a time turns a mount that is not there into five runs.
    """


class WrongAnswer(Exception):
    """The run finished and produced the wrong result.

    Distinct from :class:`MissingData`, which means it never started. A
    driver exists to ask a question the test suite does not, and a question
    whose wrong answer costs nothing has not been asked.
    """


def check(condition, message):
    """Refuse unless ``condition`` holds, naming what the answer should be.

    :param condition: the property the run had to have.
    :param message: what was expected and what came out instead. It is the
        whole report a caller gets, so it names both numbers rather than
        saying a check failed.
    :raises WrongAnswer: when ``condition`` is false.
    """
    if not condition:
        raise WrongAnswer(message)
    return True


def dataset_root(argv, default):
    """The dataset root a driver was pointed at.

    ``argv[1]`` when given, otherwise the recorded default, with ``~``
    expanded either way. Existence is :func:`require`'s job, so a driver can
    print the root it is about to use before it checks it.
    """
    chosen = argv[1] if len(argv) > 1 and argv[1] else default
    return Path(str(chosen)).expanduser()


def _matches(root, pattern):
    """Every path under ``root`` matching one declared input.

    A plain relative path matches itself; anything containing ``*`` or ``?``
    is globbed, which is how a driver declares "at least one field" without
    naming the fields.
    """
    if any(ch in pattern for ch in "*?["):
        return sorted(root.glob(pattern))
    candidate = root / pattern
    return [candidate] if candidate.exists() else []


def require(root, required, what=""):
    """Refuse unless every declared input exists under ``root``.

    :param root: the dataset root, as returned by :func:`dataset_root`.
    :param required: relative paths the run reads; ``*`` and ``?`` are
        globbed and must match at least once.
    :param what: what the dataset is, named in the refusal so the reader
        knows which disk to go and find.
    :returns: the resolved root.
    :raises MissingData: naming the root and every missing entry.
    """
    root = Path(root)
    subject = what or "the dataset"
    if not root.is_dir():
        raise MissingData(
            f"{subject} is not on this machine: {root} does not exist "
            f"(or is not a directory).\n"
            f"Pass the dataset root as the first argument to run this driver "
            f"against a copy somewhere else.")
    missing = [pattern for pattern in required if not _matches(root, pattern)]
    if missing:
        listed = "\n".join(f"  {root / pattern}" for pattern in missing)
        raise MissingData(
            f"{subject} at {root} is missing {len(missing)} of "
            f"{len(required)} inputs this run reads:\n{listed}\n"
            f"Pass the dataset root as the first argument to run this driver "
            f"against a complete copy somewhere else.")
    return root.resolve()


def scratch(name):
    """A writable working directory for one driver, emptied first.

    Lives under ``$SPACR_DRIVER_SCRATCH`` when that is set, otherwise under
    the system temporary directory. Emptying it is what makes a rerun mean
    the same thing as a first run -- spaCR pipelines skip work whose output
    is already present, so a half-deleted tree silently measures nothing.
    """
    base = os.environ.get("SPACR_DRIVER_SCRATCH") or os.path.join(
        tempfile.gettempdir(), "spacr_drivers")
    path = Path(base).expanduser() / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _inside(child, parent):
    """Whether ``child`` is ``parent`` or lies beneath it, symlinks resolved."""
    child = Path(child).expanduser().resolve()
    parent = Path(parent).expanduser().resolve()
    return child == parent or parent in child.parents


def _make_writable(path):
    """Give the owner write permission on a staged copy.

    Datasets are often read-only on disk and :func:`shutil.copy2` preserves
    the mode, so a staged tree can be copied successfully and then refuse the
    write the run exists to make.
    """
    for target in [path] if path.is_file() else path.rglob("*"):
        target.chmod(os.stat(target).st_mode | stat.S_IWUSR)


def stage(root, items, into, flatten=False):
    """Copy declared inputs out of the dataset into a scratch tree.

    :param root: the dataset root; never written to.
    :param items: relative paths or globs to copy.
    :param into: the scratch directory to copy them into.
    :param flatten: drop the relative layout and copy every matched file
        straight into ``into`` -- what a pipeline whose ``src`` must hold the
        images directly needs.
    :returns: ``into``.
    :raises ValueError: if ``into`` lies inside ``root``. Writing a staged
        copy back into the dataset is the one mistake these drivers must not
        be able to make.
    """
    root, into = Path(root), Path(into)
    if _inside(into, root):
        raise ValueError(
            f"refusing to stage into {into}: it is inside the dataset root "
            f"{root}. Drivers copy out of a dataset and never write into it.")
    into.mkdir(parents=True, exist_ok=True)
    for pattern in items:
        for source in _matches(root, pattern):
            relative = source.relative_to(root)
            destination = into / (source.name if flatten else relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
            _make_writable(destination)
    return into


def read_settings(path):
    """Load a saved spaCR settings CSV or JSON exactly as spaCR loads one.

    This delegates to :func:`spacr.cli.load_settings_file` rather than
    re-reading the two columns, because the obvious re-implementation --
    ``pandas.read_csv`` plus ``ast.literal_eval`` -- turns every blank cell
    into ``nan``. A blank cell is what a saved settings file looks like for a
    box the user left empty, so the naive loader hands the pipeline
    ``custom_regex=nan`` and the run is refused over a value nobody set.

    :param path: the settings file to load.
    :returns: the settings dict.
    """
    from spacr.cli import load_settings_file

    return load_settings_file(str(path))


def settings_file(root, candidates, what=""):
    """The first recorded settings file that exists, or a refusal.

    Recorded runs keep their settings next to the data, but not always in the
    same place, so a driver names every location it knows. The refusal lists
    all of them: "look in one of these" is actionable, "not found" is not.

    :raises MissingData: when none of the candidates exists.
    """
    root = Path(root)
    for candidate in candidates:
        path = (root / candidate).resolve()
        if path.is_file():
            return path
    listed = "\n".join(f"  {(root / c)}" for c in candidates)
    raise MissingData(
        f"no settings file for {what or 'this run'} under {root}; looked for:"
        f"\n{listed}\nPass one as the second argument to use a settings file "
        f"kept somewhere else.")


def undeclared(settings, app_key):
    """Settings keys the installed spaCR does not declare.

    A key spaCR no longer declares is a key nothing reads: the value in an
    older settings file is silently ignored and the default is used instead.
    The names are returned rather than removed, so a driver reports the drift
    without changing what a recorded run was given.
    """
    from spacr.validate import _APP_EXTRA_KEYS, _known_setting_keys, _normalize_app

    known = _known_setting_keys() | _APP_EXTRA_KEYS.get(_normalize_app(app_key),
                                                        frozenset())
    return sorted(key for key in settings if key not in known)


def preflight(settings, app_key, known_false_positives=None):
    """Run spaCR's own pre-flight check and refuse on any error.

    The GUI makes this check before it starts a run and a bare script does
    not, which is how a settings file with a typo in it reaches a pipeline.

    :param known_false_positives: ``{setting: reason}`` for errors that are
        defects in the CHECK rather than in the settings. They are printed
        under their own heading, with the reason, and do not stop the run --
        so a driver can get past a bad check without the check quietly
        becoming optional for everything else.
    :raises MissingData: if any other error is reported.
    """
    from spacr.validate import format_report, validate_settings

    waived = dict(known_false_positives or {})
    problems = validate_settings(settings, app_key)
    print(format_report(problems, settings, app_key))
    errors = [problem for problem in problems if problem.is_error]
    excused = [problem for problem in errors if problem.setting in waived]
    for problem in excused:
        print(f"pre-flight error overridden -- the check is wrong, not the "
              f"settings: [{problem.setting}] {waived[problem.setting]}")
    remaining = [problem for problem in errors if problem not in excused]
    if remaining:
        raise MissingData(
            f"pre-flight found {len(remaining)} error(s) in the settings for "
            f"'{app_key}'; the run was not started.")
    return settings


def cap_gpu(fraction=0.80):
    """Leave the desktop room on the shared card.

    The one GPU also drives the display, and a batch job that takes the whole
    card kills the session. Returns whether CUDA is usable at all, so a
    driver can pass ``gpu=`` honestly instead of assuming.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return True


def run(main):
    """Call a driver's ``main`` and turn a refusal into a clear exit.

    Three outcomes, three exit codes, because a caller has to be able to tell
    them apart without reading the log:

    * ``0`` the run finished and every check it makes held;
    * ``1`` the run finished and produced a wrong answer -- the reason is on
      stderr under ``WRONG ANSWER``;
    * ``2`` the data is not on this machine, so the run never started.
    """
    try:
        main(sys.argv)
    except MissingData as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        raise SystemExit(2)
    except WrongAnswer as wrong:
        print(f"WRONG ANSWER: {wrong}", file=sys.stderr)
        raise SystemExit(1)
    return 0
