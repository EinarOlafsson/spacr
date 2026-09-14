"""``spacr-make-masks`` — open the mask editor on a folder, as a queue.

Ledger item 396 asks for a curation session that can be started from a
terminal, pointed at a folder, bounded, and resumed. Curation is done over
SSH and on more than one machine, and the answer until now was
``spacr/cli.py``'s flat refusal — *"Make Masks is a manual mask editor; run
it in the GUI"*. That sentence is still true about the brush and false about
the session. This module is the session::

    spacr-make-masks --folder <dir>
    spacr-make-masks --folder <dir> --order value --limit 25
    spacr-make-masks --folder <dir> --dry-run      # no display needed

:mod:`spacr.curation_queue` does the thinking — which fields are waiting,
in what order, and what was already decided about each one. This module is
the thin part: parse four arguments, build the queue, say what the session
is, and only then start Qt.

Nothing is imported from :mod:`spacr.qt` until the folder has been read and
accepted. That order is the point of the module rather than a detail of it:
a folder that does not exist, or that holds no layout spaCR recognises, is
answered with a sentence on a login node with no display, not with a Qt
crash after a ten-second import.

What is printed before the editor opens
---------------------------------------

The session summary, in the ledger's own terms::

    nested layout at /data/pv: 500 bundles, 366 done, 29 skip, 105 remaining;
    25 this session, sorted by easy

so a curator sees what they are resuming into, and sees it in the shell they
started from rather than only in a window.

One layout opens the editor, three open the queue
-------------------------------------------------

:func:`spacr.curation_queue.detect_layout` reads three layouts. The editor
edits ONE of them: Make Masks reads a draft from ``<folder>/masks/<stem>.tif``
and writes the curated mask back to the same place, which is the ``nested``
layout and nothing else. So a ``sibling`` or ``seg`` folder is summarised,
ordered and listed by ``--dry-run`` — and the editor is refused, by name,
with what it would have done wrong. Opening a sibling set on its ``images``
folder would read and write ``images/masks``, silently orphaning every draft
mask the set already has; that is worse than a refusal, and it is the failure
mode the ledger's "loaded as empty" warning is about.

Exit codes::

    0   the editor ran, or --dry-run printed the queue, or there was
        nothing left to curate
    1   the Qt interface could not start
    2   bad arguments, a folder that is not there, a folder holding no
        recognisable layout, or a layout the editor cannot edit in place
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .curation_queue import (
    DEFAULT_ORDER,
    LAYOUT_NESTED,
    LAYOUT_SEG,
    LAYOUT_SIBLING,
    ORDERS,
    STATUS_FILENAME,
    CurationQueue,
    CurationQueueError,
    build_queue,
)

__all__ = [
    "EXIT_OK",
    "EXIT_NO_GUI",
    "EXIT_USAGE",
    "build_parser",
    "has_display",
    "hand_over",
    "take_handover",
    "editor_refusal",
    "open_editor",
    "main",
]

#: The session ran, or said truthfully that there was nothing to run.
EXIT_OK = 0

#: Qt could not start. The same code :func:`spacr.qt.run` returns for it.
EXIT_NO_GUI = 1

#: Arguments, or a folder, that the session could not be built from. Matches
#: :data:`spacr.cli.EXIT_USAGE`, so the two commands agree about what a 2
#: from a spaCR CLI means: nothing started, so nothing was half-done.
EXIT_USAGE = 2

#: The queue a terminal built, waiting for the screen that will show it.
#:
#: A GLOBAL, and deliberately the smallest one that works. The Qt main window
#: is built by :func:`spacr.qt.app.launch`, which takes an app key and
#: nothing else; there is no argument to thread a folder through, and adding
#: one would mean changing a 5,000-line launch path to carry a parameter that
#: exactly one command sets. So the queue is left here, and the first
#: ``MakeMasksScreen`` built takes it — once, through :func:`take_handover`,
#: which empties the slot so a screen built later opens on nothing rather
#: than on somebody's finished session.
_HANDOVER: Optional[CurationQueue] = None


def build_parser() -> argparse.ArgumentParser:
    """Build the ``spacr-make-masks`` argument parser.

    :returns: the parser, with ``--folder``, ``--order``, ``--limit`` and
        ``--dry-run`` on it.
    """
    parser = argparse.ArgumentParser(
        prog="spacr-make-masks",
        description="Open Make Masks on a folder as a resumable curation "
                    "queue.",
        epilog=f"Progress is kept in <folder>/{STATUS_FILENAME}: done and "
               f"skip stay distinct, so a field that cannot be curated is "
               f"not offered again, and the record syncs with the images.")
    parser.add_argument(
        "--folder", required=True, metavar="DIR",
        help="the folder to curate. Accepted layouts: images with a masks/ "
             "folder beneath them, images/ beside masks/, or Cellpose "
             "*_seg.npy bundles.")
    parser.add_argument(
        "--order", choices=list(ORDERS), default=DEFAULT_ORDER,
        help=f"what to offer first (default: {DEFAULT_ORDER}). easy: "
             f"populated drafts before empty ones. prob: least certain "
             f"first. value: the fields worth a curator's judgement. name: "
             f"by stem, ignoring everything else.")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="end the session after N fields. Applied AFTER ordering, so it "
             "is the N most worthwhile still to do, not the first N found.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the session and the fields it would offer, in order, "
             "and exit. Needs no display, and works for every layout.")
    return parser


def hand_over(queue: Optional[CurationQueue]) -> None:
    """Leave ``queue`` for the next Make Masks screen that is built.

    :param queue: the session to hand over, or ``None`` to clear the slot.
    """
    global _HANDOVER
    _HANDOVER = queue


def take_handover() -> Optional[CurationQueue]:
    """Take the queue the terminal handed over, if there is one.

    Emptying the slot is half of what this call is for: a screen rebuilt
    later in the same process — the user navigating back to Make Masks —
    must open on nothing rather than on a session that has already been
    worked through.

    :returns: the handed-over :class:`~spacr.curation_queue.CurationQueue`,
        or ``None`` when the screen was opened the ordinary way.
    """
    global _HANDOVER
    queue, _HANDOVER = _HANDOVER, None
    return queue


def editor_refusal(queue: CurationQueue) -> str:
    """Say why the editor will not open this layout in place.

    :param queue: the queue that was built, in a layout Make Masks cannot
        edit.
    :returns: the refusal, naming what the editor would have read and
        written, and what still works on this folder.
    """
    folder = queue.folder
    if queue.layout.kind == LAYOUT_SIBLING:
        specific = (
            f"  Make Masks reads a draft from <folder>/masks/<stem>.tif and "
            f"writes the curated mask back there. Opened on {folder}/images "
            f"it would read and write {folder}/images/masks — not the "
            f"{folder}/masks this set already has — so every draft in it "
            f"would be ignored and every save would land somewhere new.\n"
            f"  To edit this set now, give the editor the nested layout: a "
            f"folder of images with masks/ beneath it.")
    elif queue.layout.kind == LAYOUT_SEG:
        specific = (
            "  Make Masks edits image files and TIFF masks. A Cellpose "
            "*_seg.npy bundle carries its image and its labels inside one "
            "pickle, which this editor does not open.\n"
            "  To edit these now, write each bundle out as an image with "
            "its mask in masks/ beneath it.")
    else:                                                # pragma: no cover
        specific = (
            "  Make Masks edits the nested layout: images with masks/ "
            "beneath them.")
    return (f"{queue.layout.kind} layout at {folder}: spaCR can read this "
            f"queue but cannot yet edit it in place.\n"
            f"{specific}\n"
            f"  The queue itself works on this folder: 'spacr-make-masks "
            f"--folder {folder} --dry-run' prints it, in order, with no "
            f"display.")


def _session_lines(queue: CurationQueue) -> List[str]:
    """Number the fields this session offers, in the order it offers them.

    :param queue: the built session.
    :returns: one line per field, ready to print.
    """
    width = len(str(len(queue.items)))
    return [f"{position:>{width}}  {item.stem}"
            for position, item in enumerate(queue.items, start=1)]


def has_display() -> bool:
    """Whether there is a windowing system for the editor to open on.

    The same three-line question :func:`spacr.cli.use_agg_if_headless` asks
    of matplotlib, asked of Qt and asked BEFORE Qt is imported: an SSH
    session with no X forwarding otherwise gets "could not load the Qt
    platform plugin xcb" and an abort, which says nothing about the queue
    that was perfectly readable a moment earlier.

    An explicit ``QT_QPA_PLATFORM`` counts as a display. It is how offscreen
    rendering, VNC and the embedded platforms are asked for, and somebody
    who set it has already said which surface Qt is to use.

    :returns: whether the editor can be opened here.
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    if os.environ.get("QT_QPA_PLATFORM"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def open_editor(queue: CurationQueue) -> int:
    """Start the GUI on ``queue`` and return the process exit code.

    Qt is imported HERE and nowhere above: everything that can be refused
    has been refused by the time this is called.

    :param queue: the session the editor opens on.
    :returns: the exit code :func:`spacr.qt.run` returns, or
        :data:`EXIT_NO_GUI` when there is no display to open on, or when the
        Qt interface is not installed.
    """
    if not has_display():
        print(f"{queue.folder} reads as a curation queue "
              f"({queue.summary.describe()}), but this session has no "
              f"display to open the editor on: no DISPLAY, no "
              f"WAYLAND_DISPLAY and no QT_QPA_PLATFORM.\n"
              f"  Reconnect with 'ssh -X', or run the same command where the "
              f"screen is -- the resume record travels with the images, so "
              f"the session continues there.\n"
              f"  Without a display, --dry-run prints the queue and its "
              f"order.", file=sys.stderr)
        return EXIT_NO_GUI
    hand_over(queue)
    try:
        from .qt import run
    except ImportError as exc:                           # pragma: no cover
        hand_over(None)
        print(f"the Qt interface could not be imported ({exc}); install it "
              f"with: pip install 'spacr[qt]'", file=sys.stderr)
        return EXIT_NO_GUI
    return int(run(["make_masks"]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Open a folder as a curation session, or say why it cannot be opened.

    :param argv: command-line arguments without the program name. ``None``
        reads :data:`sys.argv`.
    :returns: a process exit code; see the module docstring for what each
        one means.
    :raises SystemExit: with status ``2`` when the arguments themselves are
        invalid, which is argparse's own refusal.
    """
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.limit is not None and args.limit < 1:
        parser.error("--limit takes a positive number of fields; a session "
                     "of none is the same as not starting one")

    folder = Path(args.folder).expanduser()
    if not folder.exists():
        print(f"no such folder: {folder}", file=sys.stderr)
        return EXIT_USAGE
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return EXIT_USAGE

    try:
        queue = build_queue(folder, order=args.order, limit=args.limit)
    except CurationQueueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE

    print(queue.describe())

    if not queue.items:
        print(f"nothing to open: every field in {queue.folder} already has a "
              f"state in {STATUS_FILENAME}. Delete a row from that file to "
              f"offer its field again.")
        return EXIT_OK

    if args.dry_run:
        for line in _session_lines(queue):
            print(line)
        return EXIT_OK

    if queue.layout.kind != LAYOUT_NESTED:
        print(editor_refusal(queue), file=sys.stderr)
        return EXIT_USAGE

    return open_editor(queue)


if __name__ == "__main__":
    raise SystemExit(main())
