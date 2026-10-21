"""Resolve recorded crop paths after a dataset is moved or remounted.

``png_list.png_path`` is absolute when written. If the dataset later moves to
another computer or mount point, the prefix changes while the directory
structure below the screen remains the same::

    <recorded root>/plate1/data/single_nucleus/single_pathogen/plate1_H19/...
    <current  root>/plate1/data/single_nucleus/single_pathogen/plate1_H19/...

Resolution selects the deepest recorded suffix that exists below the current
root. The root may be a database file, measurements folder, plate folder, or
screen folder. Paths are changed only when the reconstructed file exists, and
the database itself is never modified.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: The folder exported crops live under. Only used to PREFER a suffix that
#: starts at it, never to require one.
DATA_FOLDER = "data"

#: Folder names that sit BESIDE ``data/`` rather than above it, so a root
#: pointing at one of them means the screen is one level up.
_SIBLINGS: Tuple[str, ...] = ("measurements", "results", "settings", "merged",
                              "orig", "stack")

#: How far above a given root to look for the screen. Two is enough for
#: ``<plate>/measurements/measurements.db``; more would start matching
#: unrelated folders that happen to share a name.
_MAX_CLIMB = 2


#: Paths at least this long take the extended-length spelling on Windows.
#: MAX_PATH is 260 for a file and 248 for a folder that is created; staying
#: under 240 leaves room for the temporary siblings atomic writes add.
_WINDOWS_PATH_ROOM = 240


def _long_path(path, *, always: bool = False, windows: Optional[bool] = None):
    """``path`` as Windows opens it past MAX_PATH; ``path`` itself elsewhere.

    With long paths disabled (the Windows default) a file below a deep folder,
    such as a Leica Matrix Screener ``slide--/chamber--/field--/image--``
    tree, cannot be opened, listed or created. The ``\\\\?\\`` form (``\\\\?\\UNC\\``
    for a share) lifts that limit. Only paths that need it are respelled, so
    ordinary Windows paths and every POSIX path reach the file call unchanged.

    :param path: a str or path-like; anything else is returned as it is.
    :param always: respell even a short path, for a walk whose entries may
        be deeper than the root.
    :param windows: whether to act as on Windows; this system by default.
    :returns: the extended-length string, or ``path`` unchanged.
    """
    if not (os.name == "nt" if windows is None else windows):
        return path
    if not isinstance(path, (str, os.PathLike)):
        return path
    text = os.fspath(path)
    if not isinstance(text, str) or text.startswith("\\\\?\\"):
        return path
    import ntpath

    absolute = ntpath.abspath(text) if os.name == "nt" else ntpath.normpath(text)
    if not always and len(absolute) < _WINDOWS_PATH_ROOM:
        return path
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def _plain_path(path, *, windows: Optional[bool] = None):
    """:func:`_long_path` undone, for paths that are recorded or shown.

    :param path: a path a walk under a :func:`_long_path` root produced.
    :param windows: whether to act as on Windows; this system by default.
    :returns: the ordinary spelling, or ``path`` unchanged.
    """
    if not (os.name == "nt" if windows is None else windows) or not isinstance(path, str):
        return path
    if path.startswith("\\\\?\\UNC\\"):
        return "\\\\" + path[8:]
    if path.startswith("\\\\?\\"):
        return path[4:]
    return path


def _parts(path: str) -> List[str]:
    """``path`` split into components, separator-agnostic."""
    return [p for p in str(path).replace("\\", "/").split("/") if p]


def candidate_roots(root: Optional[str]) -> Tuple[str, ...]:
    """Return the candidate root folders derived from ``root``, nearest first.

    :param root: plate, screen, measurements folder, or database path to
        normalize into candidate roots; ``None`` yields no candidates.

    Accepts the plate folder, the screen folder, the ``measurements/`` folder,
    or the ``measurements.db`` file itself -- callers hold different ones and
    should not each have to normalise.
    """
    if not root:
        return ()
    here = os.path.abspath(os.path.expanduser(os.fspath(root)))
    if os.path.isfile(here):
        here = os.path.dirname(here)
    out: List[str] = []
    for _ in range(_MAX_CLIMB + 1):
        out.append(here)
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return tuple(out)


def _suffixes(path: str) -> List[Tuple[str, ...]]:
    """Suffixes of ``path``, deepest structure first.

    A suffix beginning at a ``data`` component is tried before the others of
    the same length, because that is the layout ``measure`` writes and the one
    a match is most likely to be meaningful in.
    """
    parts = _parts(path)
    if len(parts) < 2:
        return []
    ordered: List[Tuple[str, ...]] = []
    for start in range(len(parts) - 1):
        ordered.append(tuple(parts[start:]))
    ordered.sort(key=lambda s: (0 if s[0] == DATA_FOLDER else 1, -len(s)))
    return ordered


def reroot_crop_path(path: Optional[str], src_root: Optional[str]
                     ) -> Optional[str]:
    """``path`` as it exists under ``src_root``, or ``path`` unchanged.

    :param path: the recorded absolute path, or anything falsy.
    :param src_root: the plate folder, the screen folder, the ``measurements``
        folder, or the database file. All four resolve.
    :returns: a path that EXISTS when one could be built; otherwise the input
        untouched, so a caller's error still names what was recorded.
    """
    mapped, _prefixes = _reroot_with_prefix(path, src_root)
    return mapped


def _reroot_with_prefix(path: Optional[str], src_root: Optional[str]
                        ) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
    """:func:`reroot_crop_path`, also returning the (old, new) prefix used.

    The prefix is what makes a 60,000-row frame cheap: it is discovered once
    and then applied as a string replacement, instead of asking the filesystem
    about every row.
    """
    if not path or not isinstance(path, str) or not path.strip():
        return path, None
    if os.path.exists(path):
        return path, None
    roots = candidate_roots(src_root)
    if not roots:
        return path, None
    normalised = str(path).replace("\\", "/")
    for suffix in _suffixes(path):
        tail = "/".join(suffix)
        if not normalised.endswith(tail):
            continue
        head = normalised[: len(normalised) - len(tail)]
        for root in roots:
            candidate = os.path.join(root, *suffix)
            if os.path.exists(candidate):
                new_head = candidate[: len(candidate) - len(os.path.join(*suffix))]
                return candidate, (head, new_head)
    return path, None


@dataclass(frozen=True)
class RerootReport:
    """Result of one re-rooting pass: how many paths moved and how many did not.

    The last two fields are the reason this is a record and not a bare count.
    A path with no recognisable structure under the root is returned unchanged
    and otherwise fails later as a missing file with less context. The report
    counts unresolved paths and includes an example so callers can explain the
    problem where re-rooting was attempted.

    :param column: DataFrame column inspected and, when at least one value
        moves, rewritten in place.
    :param moved: number of nonblank path values replaced by existing paths
        discovered below a candidate root.
    :param unresolved: number of nonblank string paths that remained missing
        after re-rooting; existing, blank, and non-string values do not count.
    :param first_unresolved: first unresolved source path encountered, or an
        empty string when none remained.
    :param root: first normalized candidate derived from ``src_root``, used in
        report messages; it need not be the ancestor where a match was found.
    """

    column: str = ""
    moved: int = 0
    unresolved: int = 0
    first_unresolved: str = ""
    root: str = ""

    def __bool__(self) -> bool:
        """Return whether at least one path was moved to a new root."""
        return bool(self.moved)

    def __int__(self) -> int:
        """Return the number of paths moved to a new root."""
        return int(self.moved)

    @property
    def partial(self) -> bool:
        """True when some paths in this column moved and some stayed unresolved.

        This separates a partial failure from a route whose files are not on
        this machine. A column where nothing moved usually belongs to such a
        route -- a screen with PNG crops and no ``merged/`` folder has 60,816
        unplaceable ``path_name`` values and is healthy (see :attr:`absent`).
        A column where most paths resolved and a few did not is the partial
        failure that a caller should report.
        """
        return bool(self.moved) and bool(self.unresolved)

    @property
    def absent(self) -> bool:
        """True when no path in this column moved and at least one stayed unresolved."""
        return not self.moved and bool(self.unresolved)

    def describe(self) -> str:
        """Return one line for a caller to print, or ``""`` when nothing moved or failed."""
        parts: List[str] = []
        if self.moved:
            parts.append(f"re-rooted {self.moved:,} {self.column} value(s) "
                         f"under {self.root}")
        if self.absent:
            return (f"none of the {self.unresolved:,} {self.column} value(s) "
                    f"are under {self.root} — that route's files are not on "
                    f"this machine")
        if self.unresolved:
            parts.append(
                f"{self.unresolved:,} could not be placed under {self.root}; "
                f"the first is {self.first_unresolved}")
        return "; ".join(parts)


def reroot_column(frame, column: str, src_root: Optional[str]):
    """Re-root one path column of ``frame`` in place; files on disk are not changed.

    :param frame: any DataFrame; a missing column is not an error, because the
        PNG route and the merged route carry different ones and a caller
        should be able to ask for both.
    :param column: name of the path-bearing column to resolve and, when any
        paths move, replace in place.
    :param src_root: anything :func:`candidate_roots` accepts.
    :returns: a :class:`RerootReport`, which counts as its own ``moved`` in a
        boolean or integer context, so a caller that only wants the number
        still gets it.

    Resolves the first dead path against the filesystem, then applies the
    prefix that worked to the rest -- 60,816 rows cost one search plus one
    string replacement each, rather than 60,816 searches. Any row the prefix
    does not fix is still resolved on its own, so a frame holding crops from
    two different screens is not half-abandoned.
    """
    root_label = (candidate_roots(src_root) or ("",))[0]
    if frame is None or column not in getattr(frame, "columns", ()):
        return RerootReport(column=column, root=root_label)
    values = frame[column].tolist()
    unresolved = 0
    first_unresolved = ""
    prefix: Optional[Tuple[str, str]] = None
    unresolvable: set = set()
    out: List[object] = []
    moved = 0
    for value in values:
        if not isinstance(value, str) or not value.strip():
            out.append(value)
            continue
        if prefix is not None:
            was, now = prefix
            forward = value.replace("\\", "/")
            if forward.startswith(was):
                candidate = now + forward[len(was):]
                if os.path.exists(candidate):
                    out.append(candidate)
                    moved += 1
                    continue
        folder = os.path.dirname(value.replace("\\", "/"))
        if folder in unresolvable:
            unresolved += 1
            first_unresolved = first_unresolved or value
            out.append(value)
            continue
        mapped, found = _reroot_with_prefix(value, src_root)
        if found is not None:
            prefix = found
        elif mapped == value and not os.path.exists(value):
            unresolvable.add(folder)
        if mapped != value:
            moved += 1
        elif not os.path.exists(value):
            unresolved += 1
            first_unresolved = first_unresolved or value
        out.append(mapped)
    if moved:
        frame[column] = out
    return RerootReport(column=column, moved=moved, unresolved=unresolved,
                        first_unresolved=first_unresolved, root=root_label)


def source_root_for_database(db_path: str) -> str:
    """The plate folder a ``measurements.db`` belongs to.

    :param db_path: path to the measurements database; an empty path yields an
        empty result.

    ``<plate>/measurements/measurements.db`` -> ``<plate>``, which is the
    folder that holds ``data/``. Derived rather than passed so a reader gains
    portability without new plumbing through every caller;
    :func:`candidate_roots` then covers the cases where the layout differs.
    """
    if not db_path:
        return ""
    return os.path.dirname(os.path.dirname(os.path.abspath(db_path)))


_QUOTE_FOLDS = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2033": '"', "\u00ab": '"', "\u00bb": '"', "\uff02": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u2032": "'", "\u2039": "'", "\u203a": "'", "\uff07": "'",
    "\u00a0": " ", "\u2007": " ", "\u202f": " ",
    "\u200b": "", "\u200c": "", "\u200d": "", "\u2060": "", "\ufeff": "",
}

_PATH_START = None

_QUOTED_ITEMS = None


def _fold_path_quotes(text: str) -> str:
    """``text`` with typographic quotes made straight and invisible spaces removed.

    Word, Pages, Notes and Excel turn ``"`` into ``“ ”`` and ``'`` into
    ``‘ ’`` as they are typed; a path list copied from them carries those
    characters, and no-break or zero-width spaces besides.

    :param text: the pasted text.
    :returns: the same text with straight ASCII quotes and plain spaces.
    """
    return "".join(_QUOTE_FOLDS.get(char, char) for char in str(text))


def _path_start_pattern():
    """The compiled pattern for text that begins an absolute or home path."""
    global _PATH_START
    if _PATH_START is None:
        import re
        _PATH_START = re.compile(
            r"""^\s*["']?\s*(?:~|/|\\\\|[A-Za-z]:[\\/]|\.{1,2}[\\/]"""
            r"""|[A-Za-z][A-Za-z0-9+.-]*://)""")
    return _PATH_START


def _quoted_items_pattern():
    """The compiled patterns for a line made only of quoted items."""
    global _QUOTED_ITEMS
    if _QUOTED_ITEMS is None:
        import re
        item = r"""\s*(?:"[^"]*"|'[^']*')\s*"""
        _QUOTED_ITEMS = (
            re.compile(rf"^{item}(?:[,;]?{item})*[,;]?\s*$"),
            re.compile(r'''"([^"]*)"|'([^']*)\''''))
    return _QUOTED_ITEMS


def _clean_path_item(text: str) -> str:
    """One pasted path with its wrapping removed and ``~`` expanded.

    Strips surrounding whitespace, a trailing comma or semicolon, a stray
    opening or closing bracket that has no partner in the item, and the
    quotes around it, straight or curly. Brackets and quotes inside the path
    are kept.

    :param text: one path as it was pasted.
    :returns: the bare path, or ``""`` when nothing but wrapping was pasted.
    """
    item = _fold_path_quotes(text).strip()
    previous = None
    while item != previous:
        previous = item
        item = item.strip().rstrip(",;").strip()
        inner = item[1:-1].strip()
        if (item[:1] in "[(" and item[-1:] in "])" and len(inner) > 1
                and inner[0] in "'\"" and inner[-1] in "'\""):
            item = inner
        if item.startswith(("[", "(")) and not any(c in item for c in "])"):
            item = item[1:]
        if item.endswith(("]", ")")) and not any(c in item for c in "[("):
            item = item[:-1]
        if item[:1] in ("'", '"'):
            item = item[1:]
        if item[-1:] in ("'", '"'):
            item = item[:-1]
    return os.path.expanduser(item) if item else ""


def _literal_paths(text: str) -> Optional[List[str]]:
    """The strings of a JSON or Python list literal, or ``None``.

    Only text that opens with a bracket or a quote is read. A Python literal
    that turns a Windows separator into a control character (``"C:\\new"``)
    is refused, so the quoted items are then read as written.
    """
    head = text.lstrip("[( \t\r\n")
    if not head.startswith(("'", '"')):
        return None
    import ast
    import json
    import warnings

    attempts = [(json.loads, text), (json.loads, f"[{text}]")]
    attempts += [(ast.literal_eval, text), (ast.literal_eval, f"[{text}]")]
    for reader, candidate in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = reader(candidate)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if isinstance(value, str):
            value = [value]
        if not (isinstance(value, (list, tuple))
                and all(isinstance(item, str) for item in value)):
            continue
        if any(ord(char) < 32 for item in value for char in item):
            continue
        if len(_quoted_items_pattern()[1].findall(text)) != len(value):
            continue
        return list(value)
    return None


def _unescape(item: str) -> str:
    """A quoted item with doubled backslashes halved when it was escaped.

    A doubled backslash at the very start is a UNC share as typed; one
    anywhere else means the whole item was written with Python or JSON
    escapes.
    """
    if "\\\\" in item[1:]:
        return item.replace("\\\\", "\\")
    return item


def _split_line(line: str, exists) -> List[str]:
    """One pasted line as the paths it names, split at commas only between paths.

    A comma splits the line when the text after it starts an absolute, home,
    UNC, drive-letter, relative ``./`` or URL path, or when every piece exists
    on disk. Any other comma belongs to the path it sits in.
    """
    whole = _clean_path_item(line)
    if not whole:
        return []
    if exists(whole):
        return [whole]
    exact, quoted = _quoted_items_pattern()
    if exact.match(line):
        return [_unescape(_clean_path_item(a or b)) for a, b in quoted.findall(line)]
    if "," not in line:
        return [whole]
    pieces = line.split(",")
    cleaned = [_clean_path_item(piece) for piece in pieces]
    if all(item and exists(item) for item in cleaned):
        return cleaned
    starts = _path_start_pattern()
    out: List[str] = []
    current = pieces[0]
    for piece in pieces[1:]:
        if starts.match(piece) and not exists(_clean_path_item(f"{current},{piece}")):
            out.append(current)
            current = piece
        else:
            current = f"{current},{piece}"
    out.append(current)
    return [item for item in (_clean_path_item(piece) for piece in out) if item]


def _parse_path_list(value, exists=None) -> List:
    """Every path a pasted ``src`` names, as a list.

    One reader for every setting that takes one or more folders or files, so
    the GUI, the command line and the API agree on what a pasted value means.
    It accepts:

    * one path, with spaces, commas or brackets in it;
    * a JSON or Python list, with straight or curly quotes, single or double,
      and a trailing comma;
    * paths one per line, or separated by commas;
    * stray brackets, surrounding whitespace and quotes around each path.

    Text that exists on disk exactly as pasted is always one path. A comma
    only separates two paths when what follows it starts a path (``/``,
    ``~``, ``C:\\``, ``\\\\server``, ``./`` or ``s3://``) or every piece
    exists. Curly quotes become straight ones and ``~`` is expanded.

    :param value: a string, a path-like object, a list or tuple of them, or
        ``None``. List entries that are not strings are kept as they are, so
        a validator can name them.
    :param exists: the existence test, :func:`os.path.exists` by default.
    :returns: the paths, in order, without repeats; empty for ``None`` or
        blank text.
    """
    exists = exists or os.path.exists
    if value is None:
        return []
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if isinstance(value, (list, tuple)):
        items: List = []
        for entry in value:
            if isinstance(entry, os.PathLike):
                entry = os.fspath(entry)
            if isinstance(entry, str):
                items.extend(_parse_path_list(entry, exists))
            else:
                items.append(entry)
        return _unique(items)
    if not isinstance(value, str):
        return [value]
    raw = value.strip()
    if not raw:
        return []
    as_typed = os.path.expanduser(raw)
    if exists(as_typed):
        return [as_typed]
    text = _fold_path_quotes(raw).strip()
    whole = _clean_path_item(text)
    if whole and exists(whole):
        return [whole]
    literal = _literal_paths(text)
    if literal is not None:
        return _unique([item for item in map(_clean_path_item, literal) if item])
    if _literal_of_other_values(text):
        return [raw]
    body = text
    wrapped = body.startswith(("[", "(")) and body.endswith(("]", ")"))
    if wrapped:
        body = body[1:-1]
    items = []
    for line in body.replace("\r", "\n").split("\n"):
        items.extend(_split_line(line, exists))
    if (wrapped and len(items) == 1
            and not _path_start_pattern().match(items[0])):
        return [whole]
    return _unique(items)


def _literal_of_other_values(text: str) -> bool:
    """True when ``text`` is a list literal holding something besides strings.

    ``[1, 2, 3]`` names no folders; it is passed on as typed so the
    validator can say so.
    """
    if not text.startswith(("[", "(")):
        return False
    import ast

    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False
    return (isinstance(value, (list, tuple))
            and any(not isinstance(item, str) for item in value))


def _unique(items: Sequence) -> List:
    """``items`` in order with repeated strings dropped."""
    seen = set()
    out: List = []
    for item in items:
        if isinstance(item, str):
            if item in seen:
                continue
            seen.add(item)
        out.append(item)
    return out


def _describe_missing_paths(paths: Iterable[str], exists=None) -> List[str]:
    """One line per missing path, saying whether its parent folder exists.

    A missing parent usually means an unmounted drive or share; a present
    parent usually means a typo in the last folder name.

    :param paths: the paths to check.
    :param exists: the existence test, :func:`os.path.exists` by default.
    :returns: one ``"<path> (...)"`` line per path that does not exist.
    """
    from .qt.i18n import tr

    exists = exists or os.path.exists
    lines = []
    for path in paths:
        if not isinstance(path, str) or exists(path):
            continue
        parent = os.path.dirname(os.path.normpath(path))
        if os.sep == "\\" and "/" in path and "\\" not in path:
            parent = parent.replace("\\", "/")
        if parent and exists(parent):
            note = tr("its parent folder {parent} exists, so check the last "
                      "folder name for a typo", parent=parent)
        else:
            note = tr("its parent folder {parent} does not exist either; "
                      "is the drive or share mounted?", parent=parent or path)
        lines.append(f"  {path} ({note})")
    return lines
