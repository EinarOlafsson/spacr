"""``spacr-tutorial``: list the spaCR tutorial lessons and open them.

Usage::

    spacr-tutorial                 list every lesson, then open the library
    spacr-tutorial mask            open one lesson: number, slug or screen
    spacr-tutorial --list          list the lessons and open nothing
    spacr-tutorial 7 --print-url   print the lesson's address instead

The lessons are the online tutorial library the Help menu opens; the list
is the lesson catalog the documentation and the in-app tutorial links read,
bundled with spaCR (:func:`spacr.qt.tutorials.lessons`).
"""
from __future__ import annotations

import argparse
import sys
import webbrowser

from ..i18n import tr
from ..tutorials import TUTORIALS_URL, find_lesson, lesson_url, lessons


def _print_lessons(rows: list) -> None:
    """Print ``rows`` as a table of number, title and the screen covered.

    :param rows: lessons as :func:`spacr.qt.tutorials.lessons` gives them.
    """
    if not rows:
        print(tr("The lesson list is not available in this installation."))
        return
    width = max(len(row["title"]) for row in rows)
    print(f"{tr('No.'):>4}  {tr('Lesson'):<{width}}  {tr('Screen')}")
    for row in rows:
        print(f"{row['number']:>4}  {row['title']:<{width}}  "
              f"{row['app_key'] or '-'}")


def main(argv=None) -> int:
    """List the tutorial lessons, or open one, in the default web browser.

    :param argv: optional argv list; defaults to ``sys.argv[1:]``.
    :returns: process exit code: 0 on success, 1 when no web browser could
        be opened, 2 for a lesson name that matches no lesson.
    """
    parser = argparse.ArgumentParser(
        prog="spacr-tutorial",
        description=tr("List the spaCR tutorial lessons and open them in "
                       "your web browser."))
    parser.add_argument(
        "lesson", nargs="?", default="",
        help=tr("A lesson number, slug or screen name, such as 7, make_masks "
                "or measure. Without one, the whole tutorial library opens."))
    parser.add_argument(
        "--list", action="store_true",
        help=tr("Print the lessons and open nothing."))
    parser.add_argument(
        "--print-url", action="store_true",
        help=tr("Print the web address instead of opening a browser."))
    args = parser.parse_args(argv)

    rows, url = lessons(), TUTORIALS_URL
    if args.lesson:
        found, suggestions = find_lesson(args.lesson)
        if found is None:
            print(tr("No lesson matches '{name}'.", name=args.lesson),
                  file=sys.stderr)
            if suggestions:
                print(tr("Did you mean: {names}?",
                         names=", ".join(suggestions)), file=sys.stderr)
            print(tr("Run 'spacr-tutorial --list' to see every lesson."),
                  file=sys.stderr)
            return 2
        rows, url = [found], lesson_url(found["lesson"])

    if args.print_url:
        print(url)
        return 0
    _print_lessons(rows)
    if args.list:
        return 0
    if not args.lesson:
        print(tr("Open one lesson with 'spacr-tutorial <number, slug or "
                 "screen>', for example 'spacr-tutorial mask'."))
    if not webbrowser.open(url):
        print(tr("No web browser could be opened. Open this address "
                 "yourself: {url}", url=url), file=sys.stderr)
        return 1
    print(tr("Opened {url}", url=url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
