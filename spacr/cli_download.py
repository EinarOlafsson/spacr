"""``spacr-download`` — fetch spaCR's published example data, with no GUI.

Every example dataset spaCR publishes has, until now, been reachable only by
pressing a button inside the application: "Load test data", "Load example
data", the screen-data picker. That is fine on a laptop and useless on a
cluster, where the data has to be on disk BEFORE a batch job starts and there
is no display to press a button on. This is that download as a command.

WHAT IT WILL AND WILL NOT DO WITHOUT BEING ASKED. With no arguments it fetches
the three example sets -- Mask, Measure, Annotate/Classify -- which come to
about 1.1 GB. It does NOT fetch the published TSG101 screen, which is 33 GB.
A command that spent 33 GB of somebody's quota because they typed its name
with no arguments would be a bug however well documented, so the screen is
opt-in, is asked for in pieces, and is confirmed before it starts. The pieces
are :mod:`spacr.screen_data`'s, unchanged: four measurement databases of about
0.5 GB and four crop folders of about 8 GB, any subset of which can be named.

NOTHING IS DOWNLOADED THAT CANNOT FIRST BE PRICED. ``--list`` prints every
piece with its size and whether it is already on disk, and the total the
current selection would cost, without opening a socket. ``--dry-run`` is the
same listing for the same reason: a user deciding what to spend an hour of
network on should be able to see the bill first.

Importing this module must stay light -- no Qt, no torch, no matplotlib -- so
``spacr-download --help`` answers instantly on a login node with a cold NFS
cache. That is why the download primitives live in
:mod:`spacr.example_archives` rather than in :mod:`spacr.qt.hf_download`, which
imports PySide6 at module scope. ``tests/test_cli_download.py`` pins it.

Usage::

    spacr-download                            # every example set (~1.1 GB)
    spacr-download --list                     # what exists, what is here
    spacr-download measure annotate           # two of the three
    spacr-download --screen measurements      # the four databases (~2.1 GB)
    spacr-download --screen crops --plate 1   # one plate of crops (~8.9 GB)
    spacr-download all --yes                  # everything, screen included

Exit codes (a job that exits 0 having downloaded nothing is the classic
footgun, so these are exact):

  0  everything asked for is on disk, or a confirmation was declined
  1  a download failed; the pieces that succeeded are still on disk
  2  bad arguments, not enough disk space, or a large download that could not
     be confirmed because nothing was there to confirm it
"""
from __future__ import annotations

import argparse
import difflib
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .example_archives import (EXAMPLE_SETS, ExampleSet, download_archive,
                               example_plate_folder, expand_measure_arrays,
                               explain_download_failure,
                               extract_example_archive,
                               make_the_example_paths_absolute)
from .screen_data import (SCREEN_ASSETS, SCREEN_REPO, ScreenAsset, assets_for,
                          human_size, total_size)

__all__ = [
    "EXIT_OK",
    "EXIT_RUNTIME",
    "EXIT_USAGE",
    "CONFIRM_ABOVE_BYTES",
    "Piece",
    "SelectionError",
    "default_destination",
    "example_folder",
    "screen_folder",
    "resolve_selection",
    "build_plan",
    "render_listing",
    "build_parser",
    "main",
]

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2

#: Above this, the download is confirmed before a byte moves.
#:
#: TWO GIGABYTES, which is a little more than all three example sets together
#: and a little less than the four measurement databases. So the default run
#: never asks -- being asked to confirm the thing the command does when you
#: give it no arguments teaches people to type ``--yes`` reflexively, and a
#: reflex is exactly what must not be in front of the 33 GB -- and every
#: request that reaches into the screen does.
CONFIRM_ABOVE_BYTES = 2_000_000_000

#: What the user may type, beyond the example-set keys themselves.
#:
#: ``classify`` is here because Annotate and Classify share one published set:
#: a user following the Classify tutorial has no reason to know that the data
#: for it is filed under the other module's name.
ALIASES = {"classify": "annotate"}

#: The groups.
GROUPS = ("examples", "screen", "all")

#: Which plates exist, taken from the assets rather than written out, so a
#: fifth plate becomes selectable by being published.
PLATES: Tuple[int, ...] = tuple(sorted({a.plate for a in SCREEN_ASSETS}))

#: The screen kinds, in the order the listing shows them.
KINDS: Tuple[str, ...] = ("measurements", "crops")


class SelectionError(Exception):
    """A name, kind or plate number the user got wrong.

    Always exit code 2: nothing was attempted, so it is an argument problem
    rather than a failed download.
    """


# ---------------------------------------------------------------------------
# where things land
# ---------------------------------------------------------------------------


def default_destination() -> Path:
    """``~/.cache/spacr/example_data`` -- where the GUI already looks.

    Chosen so that a plate fetched by this command is the plate the
    application's own "Load example data" buttons point at:
    :func:`spacr.example_archives.example_plate_folder` is
    ``<this>/plate1``. Someone who ran ``spacr-download`` before opening the
    GUI should find the data already there, not download it twice.
    """
    return example_plate_folder().parent


def example_folder(dest) -> Path:
    """The one plate folder all three example sets unpack into.

    ONE FOLDER because the sets compose: the Measure example's ``merged/``,
    the Annotate example's ``data/`` and ``measurements/``, and the Mask
    demo's raw images are three stages of the same plate, and spaCR expects to
    be pointed at a plate.

    :param dest: the root everything unpacks under.
    """
    return Path(dest) / "plate1"


def screen_folder(dest, plate: int) -> Path:
    """One folder per screen plate, and it has to be.

    Every plate's measurements archive unpacks to
    ``measurements/measurements.db`` and every plate's crop archive unpacks to
    ``data/`` -- the same two paths, four times over. Unpacked into one folder
    the fourth plate would silently overwrite the third, and
    :meth:`spacr.screen_data.ScreenAsset.is_present` would report a plate as
    downloaded because a DIFFERENT plate's file is sitting where its own would
    go. So the plate number is in the path, and a selection of all four is
    four plate folders that can each be opened as itself.

    :param dest: the root everything unpacks under.
    :param plate: which screen plate.
    """
    return Path(dest) / "screen" / f"plate{int(plate)}"


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    """One archive this command can fetch, priced and located.

    Examples and screen pieces are described by two different dataclasses --
    :class:`~spacr.example_archives.ExampleSet` and
    :class:`~spacr.screen_data.ScreenAsset` -- because they answer to two
    different publishers. This is the one shape the listing, the size
    arithmetic and the download loop all work in, so none of them has to know
    which kind of thing it is holding.

    :param key: how a selection names it, unique across both publishers.
    :param name: what the listing's first column shows.
    :param detail: the rest of the row -- a summary for an example set, the
        archive name for a screen piece.
    :param repo: the dataset repository it is published in.
    :param archive: the ``.tar`` to fetch from it.
    :param folder: where it unpacks.
    :param bytes: the archive's size, so a selection can be priced before it
        is paid.
    :param present: whether it is already unpacked where it would go. Computed
        when the plan is built rather than looked up later, so the listing and
        the download agree about what will happen.
    :param expands_npz: whether the unpacked arrays have to be rewritten as
        ``.npy`` afterwards.
    """

    key: str
    name: str
    detail: str
    repo: str
    archive: str
    folder: Path
    bytes: int
    present: bool
    expands_npz: bool = False


def resolve_selection(what: Sequence[str] = (), *,
                      screen: Optional[str] = None,
                      plates: Sequence[int] = ()
                      ) -> Tuple[List[ExampleSet], List[ScreenAsset]]:
    """Turn what the user typed into the exact list of things to fetch.

    Kept apart from argparse so the rules can be read -- and tested -- as
    rules rather than as a parser's side effects.

    :param what: names and groups: an example key, ``examples``, ``screen`` or
        ``all``. Empty means the default.
    :param screen: ``measurements``, ``crops``, ``all`` or ``None``. Naming a
        kind is itself a request for the screen, so ``--screen crops`` needs no
        positional argument to go with it.
    :param plates: which screen plates; empty means all of them.
    :returns: ``(example sets, screen assets)`` in listing order.
    :raises SelectionError: on a name, kind or plate that does not exist. A
        typo must not quietly select nothing and then report success.
    """
    if screen is not None and screen not in KINDS + ("all",):
        raise SelectionError(
            f"unknown screen kind {screen!r}; there is "
            f"{', '.join(KINDS)} and all")
    for plate in plates:
        if int(plate) not in PLATES:
            raise SelectionError(
                f"there is no plate {plate}; the screen has plates "
                f"{', '.join(str(p) for p in PLATES)}")

    # NAMING A SCREEN FILTER IS NAMING THE SCREEN, read from both ends.
    # `--screen crops` on its own must not fall through to the default and
    # download the examples instead, and `mask --screen crops` must not
    # silently drop the filter -- both would give a user who typed the word
    # "screen" the one thing they cannot have meant.
    take_screen = screen is not None or bool(plates)

    asked = [str(name).strip().lower() for name in what if str(name).strip()]
    asked = [ALIASES.get(name, name) for name in asked]
    if not asked and not take_screen:
        asked = ["examples"]

    known = {s.key for s in EXAMPLE_SETS} | set(GROUPS) | set(ALIASES)
    wanted_examples: List[ExampleSet] = []
    wanted_screen: List[ScreenAsset] = []
    for name in asked:
        if name not in known:
            raise SelectionError(_unknown_name_message(name))
        if name in ("examples", "all"):
            wanted_examples = list(EXAMPLE_SETS)
        if name in ("screen", "all"):
            take_screen = True
        for candidate in EXAMPLE_SETS:
            if candidate.key == name and candidate not in wanted_examples:
                wanted_examples.append(candidate)

    if take_screen:
        kind = None if screen in (None, "all") else screen
        for plate in (list(plates) or [None]):
            for asset in assets_for(kind, plate):
                if asset not in wanted_screen:
                    wanted_screen.append(asset)

    # Listing order, not typing order: the sets read as a pipeline and the
    # screen reads as a table, and neither should be shuffled by the order
    # somebody happened to type two names in.
    wanted_examples.sort(key=EXAMPLE_SETS.index)
    return wanted_examples, wanted_screen


def _unknown_name_message(name: str) -> str:
    """What to say about a name that is not a thing to download."""
    every = sorted({s.key for s in EXAMPLE_SETS} | set(GROUPS) | set(ALIASES))
    near = difflib.get_close_matches(name, every, n=1)
    hint = f" Did you mean {near[0]!r}?" if near else ""
    return (f"there is nothing called {name!r} to download.{hint} "
            f"There is: {', '.join(every)}. Use --list to see every piece.")


def build_plan(examples: Sequence[ExampleSet], assets: Sequence[ScreenAsset],
               dest) -> List[Piece]:
    """One :class:`Piece` per thing to fetch, with its size and its folder.

    :param examples: the example sets to include.
    :param assets: the screen pieces to include.
    :param dest: the root everything unpacks under.
    """
    dest = Path(dest)
    plan: List[Piece] = []
    folder = example_folder(dest)
    for example in examples:
        plan.append(Piece(
            key=example.key,
            name=example.key,
            detail=example.summary,
            repo=example.repo,
            archive=example.archive,
            folder=folder,
            bytes=example.bytes,
            present=example.is_present(folder),
            expands_npz=example.expands_npz,
        ))
    for asset in assets:
        where = screen_folder(dest, asset.plate)
        plan.append(Piece(
            key=f"plate{asset.plate}-{asset.kind}",
            name=f"plate {asset.plate} {asset.kind}",
            detail=asset.archive,
            repo=SCREEN_REPO,
            archive=asset.archive,
            folder=where,
            bytes=asset.bytes,
            present=asset.is_present(where),
        ))
    return plan


def pieces_to_fetch(plan: Sequence[Piece], *, force: bool = False
                    ) -> List[Piece]:
    """The plan minus what is already on disk.

    ``--force`` keeps everything, which is how a truncated or edited copy gets
    repaired: the archive is re-fetched and unpacked over what is there.

    :param plan: the pieces a selection resolved to.
    :param force: fetch even what is already on disk.
    """
    return [piece for piece in plan if force or not piece.present]


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _rows(pieces: Sequence[Piece], chosen: Sequence[str],
          width: int = 79) -> List[str]:
    """The table body, aligned and wrapped.

    :param pieces: the rows.
    :param chosen: keys to mark with a ``*``.
    :param width: the column to wrap the description at.
    """
    if not pieces:
        return []
    name_width = max(len(p.name) for p in pieces)
    size_width = max(len(human_size(p.bytes)) for p in pieces)
    lead = 2 + 2 + name_width + 2 + size_width + 2 + 7 + 2
    rows = []
    for piece in pieces:
        mark = "*" if piece.key in chosen else " "
        head = (f"  {mark} {piece.name:<{name_width}}  "
                f"{human_size(piece.bytes):>{size_width}}  "
                f"{'present' if piece.present else 'missing':<7}  ")
        body = textwrap.wrap(piece.detail, width=max(24, width - lead)) or [""]
        rows.append(head + body[0])
        rows.extend(" " * lead + line for line in body[1:])
    return rows


def render_listing(plan: Sequence[Piece], chosen: Sequence[Piece], dest,
                   *, force: bool = False) -> str:
    """Every piece, its size, whether it is here, and what the total would be.

    ``plan`` is the whole inventory and ``chosen`` is the selection, so the
    listing answers both "what is there?" and "what would this command do?" at
    once. A ``*`` marks the selected rows.

    :param plan: every piece there is.
    :param chosen: the pieces this run would fetch.
    :param dest: the root everything unpacks under.
    :param force: whether pieces already on disk count towards the total.
    """
    dest = Path(dest)
    keys = [piece.key for piece in chosen]
    examples = [p for p in plan if p.folder == example_folder(dest)]
    screen = [p for p in plan if p not in examples]

    out: List[str] = []
    if examples:
        out.append(f"Example data — unpacked into {example_folder(dest)}")
        out.append("")
        out.extend(_rows(examples, keys))
        out.append("")
    if screen:
        out.append(f"Published TSG101 screen — unpacked into "
                   f"{dest / 'screen'}/plate<N>")
        out.append("")
        out.extend(_rows(screen, keys))
        out.append("")

    wanted = list(chosen)
    queue = pieces_to_fetch(wanted, force=force)
    skipped = len(wanted) - len(queue)
    out.append(f"Selected: {len(wanted)} of {len(plan)} pieces, "
               f"about {human_size(total_size(queue))} to download.")
    if skipped:
        out.append(f"          {skipped} already on disk and skipped; "
                   f"--force fetches them again.")
    return "\n".join(out)


def screen_notice() -> str:
    """How to ask for the screen, for a run that did not.

    Printed on the default run because the screen is the thing this command
    deliberately does NOT do by itself, and a user who never learns it is
    there will conclude spaCR does not publish it.
    """
    return "\n".join((
        "The published TSG101 screen (33 GB) was not included. Ask for it in "
        "pieces:",
        "  spacr-download --screen measurements        "
        "the four databases (~2.1 GB)",
        "  spacr-download --screen crops --plate 1     "
        "one plate of crops (~8.9 GB)",
        "  spacr-download --list                       "
        "every piece, with its size",
    ))


# ---------------------------------------------------------------------------
# room, and permission
# ---------------------------------------------------------------------------


def room_for(pieces: Sequence[Piece], dest) -> Optional[str]:
    """Complaint about free disk space, or ``None`` when there is enough.

    The peak is not the total: each archive is unpacked and then deleted, so
    what has to fit at once is everything that will be kept plus the largest
    single archive still on disk beside its own unpacked copy.

    Returns ``None`` when the free space cannot be read at all. A filesystem
    that will not answer is not evidence of a full one, and refusing a
    download over a failed ``statvfs`` would break the command on exactly the
    network filesystems a cluster user has.

    :param pieces: what would be downloaded.
    :param dest: where it would go; the nearest existing parent is measured,
        because the destination itself may not have been made yet.
    """
    if not pieces:
        return None
    needed = total_size(pieces) + max(p.bytes for p in pieces)
    where = Path(dest)
    while not where.exists() and where != where.parent:
        where = where.parent
    try:
        free = shutil.disk_usage(str(where)).free
    except OSError:
        return None
    if free >= needed:
        return None
    return (f"not enough room in {dest}: the selection needs about "
            f"{human_size(needed)} at its peak (the archives are unpacked "
            f"and then deleted) and there is {human_size(free)} free. "
            f"Choose fewer pieces, or pass --dest to a disk that has room.")


def _is_interactive() -> bool:
    """Whether there is somebody there to answer a question."""
    try:
        return bool(sys.stdin is not None and sys.stdin.isatty())
    except (AttributeError, ValueError):        # a closed or exotic stdin
        return False


def _yes_at_the_prompt(question: str) -> bool:
    """Ask, and treat everything except an explicit yes as no.

    EOF is a no: a pipe that closed is not consent.
    """
    try:
        answer = input(question)
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


# ---------------------------------------------------------------------------
# doing it
# ---------------------------------------------------------------------------


def _progress(out) -> object:
    """A callback that redraws one line of percentage, or ``None``.

    Only for a terminal. Into a log file the same callback would write a
    hundred thousand carriage returns, so a redirected run gets the one line
    the download loop already prints per piece.

    :param out: the stream the download is reporting to.
    """
    try:
        if not out.isatty():
            return None
    except (AttributeError, ValueError):
        return None

    state = {"percent": -1}

    def report(written: int, expected: Optional[int]) -> None:
        """Redraw the terminal line when the integer percentage changes.

        An unknown or zero total cannot yield a useful percentage and is
        ignored; every emitted update is flushed so it is immediately visible.
        """
        if not expected:
            return
        percent = int(written * 100 / expected)
        if percent == state["percent"]:
            return
        state["percent"] = percent
        out.write(f"\r      {human_size(written)} of {human_size(expected)}"
                  f"  ({percent}%)   ")
        out.flush()

    return report


def fetch_piece(piece: Piece, *, out=None, progress: bool = True) -> int:
    """Download one archive, unpack it, and throw the archive away.

    The archive is removed as soon as it is unpacked: it is a second copy of
    everything just written, and for a crop plate that is another 8 GB sitting
    on the disk for no reason.

    :param piece: what to fetch.
    :param out: where progress is drawn; ``sys.stdout`` when None.
    :param progress: draw a percentage while it arrives.
    :returns: how many members were unpacked.
    """
    out = sys.stdout if out is None else out
    piece.folder.mkdir(parents=True, exist_ok=True)
    archive = download_archive(
        piece.repo, piece.archive, piece.folder,
        progress=_progress(out) if progress else None,
        # A MEGABYTE, not the 32 KB the per-file demo pull uses: these are
        # archives of gigabytes, and the small chunk buys nothing on a body
        # that is never displayed as it arrives.
        chunk_size=1 << 20)
    members = extract_example_archive(archive, piece.folder)
    Path(archive).unlink(missing_ok=True)
    if piece.expands_npz:
        # The .npz compression is a transport detail; Measure reads .npy.
        expand_measure_arrays(piece.folder / "merged")
    return members


def download(pieces: Sequence[Piece], *, out=None, err=None,
             quiet: bool = False
             ) -> Tuple[List[Piece], List[Tuple[Piece, BaseException]]]:
    """Fetch every piece, and keep going after one fails.

    ONE FAILURE DOES NOT ABANDON THE REST. A crop plate is an hour of network;
    losing the three that would have succeeded because the second one's
    connection dropped means starting the hour again. Each failure is reported
    as it happens and repeated in the summary, and the caller turns a non-empty
    result into exit code 1.

    :param pieces: what to fetch, in order.
    :param out: stream for progress; ``sys.stdout`` when None.
    :param err: stream for failures; ``sys.stderr`` when None.
    :param quiet: print neither the per-piece line nor the percentage.
    :returns: ``(finished, failed)``. Both are needed and neither can be
        derived from the other: an interrupt stops the loop, so the pieces
        that were never attempted are in neither list, and a summary that
        subtracted the failures from the selection would report them as
        downloaded.
    """
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    finished: List[Piece] = []
    failures: List[Tuple[Piece, BaseException]] = []
    touched: List[Path] = []
    for position, piece in enumerate(pieces, start=1):
        if not quiet:
            print(f"[{position}/{len(pieces)}] {piece.name}  "
                  f"{human_size(piece.bytes)}  -> {piece.folder}", file=out)
        try:
            members = fetch_piece(piece, out=out, progress=not quiet)
        except KeyboardInterrupt as exc:
            # STOP, rather than skip to the next piece. Ctrl-C on the second
            # of eight plates means all eight, not "abandon this 8 GB and
            # start the next 8 GB". Nothing partial is kept: the archive is
            # still a `.part` file, which `download_archive` removes on its
            # way out.
            failures.append((piece, exc))
            print(f"\nerror: interrupted; {piece.name} was not downloaded.",
                  file=err)
            break
        except Exception as exc:                              # noqa: BLE001
            failures.append((piece, exc))
            print(f"\nerror: {piece.name} was not downloaded.", file=err)
            print(explain_download_failure(exc), file=err)
            continue
        finished.append(piece)
        if piece.folder not in touched:
            touched.append(piece.folder)
        if not quiet:
            print(f"\r      unpacked {members} files into {piece.folder}"
                  f"        ", file=out)

    for folder in touched:
        # LAST, AND ONCE PER FOLDER. A measurements database stores absolute
        # paths to its crops; the published copy stores them relative to the
        # dataset root so it is portable. This is what turns them back into
        # paths that open -- and it has to run after the crops are there, not
        # between two pieces of the same plate.
        try:
            make_the_example_paths_absolute(folder)
        except Exception as exc:                              # noqa: BLE001
            print(f"warning: could not rewrite the paths in {folder}: {exc}",
                  file=err)
    return finished, failures


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------


def cmd_download(args: argparse.Namespace, *, out=None, err=None) -> int:
    """``spacr-download`` proper. See :func:`main` for the exit codes.

    :param args: the parsed command line.
    :param out: stream for ordinary output; ``sys.stdout`` when None.
    :param err: stream for errors; ``sys.stderr`` when None.
    :returns: the process exit code.
    """
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    dest = Path(args.dest).expanduser() if args.dest else default_destination()

    try:
        examples, assets = resolve_selection(
            args.what, screen=args.screen, plates=args.plate or ())
    except SelectionError as exc:
        print(f"error: {exc}", file=err)
        return EXIT_USAGE

    inventory = build_plan(EXAMPLE_SETS, SCREEN_ASSETS, dest)
    chosen = build_plan(examples, assets, dest)

    if args.list or args.dry_run:
        print(render_listing(inventory, chosen, dest, force=args.force),
              file=out)
        print("", file=out)
        if not assets:
            print(screen_notice(), file=out)
            print("", file=out)
        print("Nothing was downloaded." if args.list
              else "Dry run: nothing was downloaded.", file=out)
        return EXIT_OK

    queue = pieces_to_fetch(chosen, force=args.force)
    if not queue:
        print(f"Everything selected is already in {dest} "
              f"({len(chosen)} piece(s)). --force downloads it again.",
              file=out)
        return EXIT_OK

    total = total_size(queue)
    if not args.quiet:
        print(render_listing(inventory, chosen, dest, force=args.force),
              file=out)
        print("", file=out)

    complaint = room_for(queue, dest)
    if complaint is not None:
        print(f"error: {complaint}", file=err)
        return EXIT_USAGE

    if total > CONFIRM_ABOVE_BYTES and not args.yes:
        if not _is_interactive():
            print(f"error: refusing to download about {human_size(total)} "
                  f"without confirmation. Re-run with --yes, or with --list "
                  f"to see what it is first.", file=err)
            return EXIT_USAGE
        if not _yes_at_the_prompt(
                f"Download about {human_size(total)} into {dest}? [y/N] "):
            print("Nothing was downloaded.", file=out)
            return EXIT_OK

    done, failures = download(queue, out=out, err=err, quiet=args.quiet)
    print(f"\nDownloaded {len(done)} of {len(queue)} piece(s), "
          f"about {human_size(total_size(done))}, into {dest}.", file=out)
    # THE PATHS, NAMED. What a user does next is put one of these in `src`,
    # and a summary that said only how many gigabytes arrived would leave
    # them to work out where from the flags they typed.
    folders: List[Path] = []
    for piece in done:
        if piece.folder not in folders:
            folders.append(piece.folder)
    for folder in folders:
        print(f"  src: {folder}", file=out)
    if failures:
        untried = len(queue) - len(done) - len(failures)
        print(f"{len(failures)} failed: "
              f"{', '.join(p.name for p, _ in failures)}"
              + (f", and {untried} were not attempted" if untried else "")
              + ". Nothing partial was kept, so running the same command "
                "again resumes at what is missing.", file=err)
        return EXIT_RUNTIME
    if not assets and not args.quiet:
        print("", file=out)
        print(screen_notice(), file=out)
    return EXIT_OK


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors exit 2 through the same path as ours."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def build_parser() -> argparse.ArgumentParser:
    """Return the ``spacr-download`` argument parser.

    Building it imports nothing beyond the standard library and two
    dependency-light spaCR modules, so ``--help`` is instant.
    """
    keys = ", ".join(s.key for s in EXAMPLE_SETS)
    parser = _Parser(
        prog="spacr-download",
        description="Download spaCR's published example data. With no "
                    "arguments: every example set, about 1.1 GB. The 33 GB "
                    "TSG101 screen is never downloaded unless it is asked "
                    "for by name.",
        epilog=textwrap.dedent("""\
            examples:
              spacr-download                            every example set
              spacr-download --list                     what exists, and what is already here
              spacr-download measure annotate           two of the three
              spacr-download --screen measurements      the four screen databases
              spacr-download --screen crops --plate 1   one plate of object crops
              spacr-download all --yes                  everything, screen included

            exit codes: 0 done (or a confirmation declined), 1 a download
            failed, 2 bad arguments, no room on disk, or a large download that
            could not be confirmed."""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "what", nargs="*", metavar="WHAT",
        help=f"What to download: {keys}, classify (an alias for annotate), "
             f"examples (all three), screen, or all. Default: examples.")
    parser.add_argument(
        "--dest", "-d", metavar="DIR",
        help="Where to unpack it. Default: ~/.cache/spacr/example_data, "
             "which is where the GUI's own example buttons look.")
    parser.add_argument(
        "--screen", metavar="KIND", default=None,
        choices=list(KINDS) + ["all"],
        help="Which part of the published screen: measurements (the "
             "databases, about 0.5 GB a plate), crops (the object images, "
             "about 8 GB a plate), or all. Naming a kind asks for the screen, "
             "so no other argument is needed.")
    parser.add_argument(
        "--plate", nargs="+", type=int, metavar="N", default=[],
        help=f"Which screen plates: {', '.join(str(p) for p in PLATES)}. "
             f"Default: all of them.")
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="Print every piece with its size and whether it is already on "
             "disk, plus what the selection would cost, and download nothing.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="The same listing as --list. There for the habit: every other "
             "spaCR command takes it.")
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help=f"Do not ask before a download over "
             f"{human_size(CONFIRM_ABOVE_BYTES)}. Required for a large "
             f"download from a script, where there is nobody to ask.")
    parser.add_argument(
        "--force", action="store_true",
        help="Download pieces that are already on disk, replacing them. This "
             "is how a truncated or edited copy is repaired.")
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Print only what failed and the closing summary.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``spacr-download`` entry point.

    :param argv: argument list; ``sys.argv[1:]`` when None.
    :returns: 0 done, 1 a download failed, 2 bad arguments or no room.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    return cmd_download(args)


if __name__ == "__main__":
    raise SystemExit(main())
