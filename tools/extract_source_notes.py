#!/usr/bin/env python3
"""Move the prose out of the modules and into ``docs/notes/``, leaving the code.

There are 49,857 non-directive comments in ``spacr/`` -- 7.3% of every line
the package ships. Some of them carry a measured number, a rejected
alternative, or the reason a thing is done the slower way; those are the only
record of a decision anyone made. Most of them restate the line below, or draw
a row of dashes. Both kinds are in the way of the code, and only one kind is
worth keeping, so this tool sorts them and moves the keepers out.

THE DESTINATION IS ONE FILE PER MODULE, at the same path under
``docs/notes/``: ``spacr/ml.py`` becomes ``docs/notes/spacr/ml.md``. That
convention is the whole discoverability mechanism -- NO POINTER COMMENT IS
LEFT IN THE CODE, because a pointer comment is the thing being removed. A
reader who wants the notes for a module computes the path.

DOCSTRINGS ARE NEVER TOUCHED. They are the public API surface and they are
translated into nine languages; they are also, being strings, invisible to
this tool, which only ever looks at COMMENT tokens.

DRY RUN IS THE DEFAULT. Nothing is written -- not a notes file, not a stripped
module -- and the full report is printed instead. A real run needs
``--execute``.

The safety property
-------------------
Stripping a comment must not change the code. That is not asserted by care,
it is checked, per file, by parsing both versions::

    ast.dump(ast.parse(before)) == ast.dump(ast.parse(after))

Comments do not appear in the AST, so an identical dump means the edit removed
comments AND NOTHING ELSE. A file whose dump differs is refused: it is
reported and left exactly as it was. THE CHECK HAS NO OPT-OUT FLAG. A flag to
skip it would be used on the one file where it fired, which is the one file
where it mattered.

The check also covers the failure mode that motivates the next rule. A ``#``
inside a string literal is not a comment; a regex that thinks it is truncates
the string, and the truncated string is a different constant in the AST. So a
regex-based version of this tool would be caught here -- but it would be
caught after damaging a file, which is why there is no regex. EVERY comment
is located with :mod:`tokenize`, which knows what a string is.

What is not a comment
---------------------
Four things look like comments and are not prose. They are left in the source:

* the shebang and the PEP 263 coding declaration -- executable and decoded,
  respectively, before Python ever sees a token;
* ``# noqa``, ``# type:``, ``# pragma`` and the other tool directives, which
  are read by linters, type checkers and coverage;
* ``#:`` -- SPHINX ATTRIBUTE DOCSTRINGS. These render into the published API
  documentation. Removing one is not tidying, it is deleting a paragraph of
  the manual, silently. There are 14,180 of them in ``spacr/``. They are
  treated as code.

The report counts each kind it preserved, so the maintainer can see them
survive rather than take it on trust.

Classification
--------------
Every standalone block (and every trailing comment) lands in one of three
buckets:

DISCARD
    carries nothing the code does not: a divider banner, a restatement of the
    line below it, commented-out code, a bare ``TODO``.
RETAIN
    carries something not recoverable from the code: a measured number, a
    rejected alternative, a warning, a reference to a real failure.
UNSURE
    the classifier could not tell, and says so rather than guessing.

THE CLASSIFIER IS CONSERVATIVE ON PURPOSE. Deleting a measurement nobody can
re-derive is far worse than carrying a dull line in a notes file, so every
ambiguous signal resolves to RETAIN, and UNSURE is written to the notes by
default (``--unsure``). Discarded blocks can be logged to a file with
``--discard-log`` so that even a real run throws nothing away irrecoverably.
"""
from __future__ import annotations

import argparse
import ast
import io
import random
import re
import sys
import textwrap
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DISCARD = "DISCARD"
RETAIN = "RETAIN"
UNSURE = "UNSURE"

#: The three buckets, in report order.
VERDICTS: Sequence[str] = (DISCARD, RETAIN, UNSURE)

#: Where a module's notes land, relative to the repository root. A module at
#: ``spacr/ml.py`` gets ``docs/notes/spacr/ml.md``; the path IS the pointer,
#: since no pointer is left in the code.
DEFAULT_OUT_DIR = "docs/notes"

#: Comments that are not prose, matched against the comment token's own text.
#: Order matters only in that ``sphinx`` is asked first, so a ``#:`` line is
#: never reported under a lesser name. Anything matching here is left in the
#: source untouched and counted in the preserved tally.
DIRECTIVE_PATTERNS: Sequence[Tuple[str, "re.Pattern[str]"]] = (
    ("sphinx", re.compile(r"^#:")),
    ("noqa", re.compile(r"#\s*noqa\b", re.IGNORECASE)),
    ("type", re.compile(r"^#\s*type:\s*\S")),
    ("pragma", re.compile(r"#\s*pragma[:\s]", re.IGNORECASE)),
    ("fmt", re.compile(r"^#\s*fmt:\s*(on|off|skip)\b", re.IGNORECASE)),
    ("isort", re.compile(r"^#\s*isort:\s*\S", re.IGNORECASE)),
    ("lint", re.compile(
        r"#\s*(pylint|mypy|ruff|flake8|nosec|bandit|pyright)\s*:", re.IGNORECASE)),
)

#: PEP 263. Only honoured on the first two lines, which is where Python looks.
CODING_RE = re.compile(r"^#.*?coding[:=]\s*[-\w.]+")

#: A line of pure rule: dashes, equals, tildes. The house banner style.
RULE_RE = re.compile(r"^[\s\-=~*_#+.<>/\\|]*$")

#: A banner with a short label in it -- ``# -- shutdown ------------``. The
#: label names a region of the file, which the file already shows.
BANNER_EDGE_RE = re.compile(r"^[\s\-=~*_#+]{2,}|[\s\-=~*_#+]{2,}$")

#: A bare marker with nothing after it. ``# TODO: make this faster`` says
#: something; ``# TODO`` says only that someone was unhappy once.
BARE_MARKER_RE = re.compile(
    r"^(TODO|FIXME|XXX|HACK|NOTE|WIP)\b\W{0,3}(?P<rest>.*)$", re.IGNORECASE)

#: Lines that read as Python rather than as English. Used together with a
#: successful :func:`ast.parse` of the whole block, never on its own.
CODE_SIGNAL_RES: Sequence["re.Pattern[str]"] = (
    re.compile(r"^\s*(from|import)\s+[\w.]+"),
    re.compile(r"^\s*(def|class)\s+\w+"),
    re.compile(r"^\s*(return|yield|raise|assert|del|global|nonlocal)\b"),
    re.compile(r"^\s*(if|elif|else|for|while|try|except|finally|with)\b.*:\s*$"),
    re.compile(r"^\s*(pass|break|continue)\s*$"),
    re.compile(r"^\s*[\w.\[\]'\"]+\s*(\+|-|\*|/|//|%|\|)?=[^=]"),
    re.compile(r"^\s*[\w.]+\(.*\)[\s,)]*$"),
    re.compile(r"^\s*(print|logger|log|self)\.\w+\("),
    re.compile(r"^\s*#"),
    re.compile(r"^\s*[\)\]\}],?\s*$"),
)

#: A number with a unit attached is a measurement, and a measurement is the
#: canonical thing nobody can re-derive from the code.
MEASUREMENT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:%|(?:ms|msec|sec|secs|seconds?|minutes?|mins?|"
    r"hours?|hrs?|days?|[kKMGT]i?B|bytes?|px|pixels?|fps|Hz|[kKMG]Hz|rows?|"
    r"files?|columns?|iterations?|frames?|wells?|plates?|images?|cells?|"
    r"calls?|times|threads?|workers?|entries|items?|lines?)\b)")

#: Any two-digit-or-longer number. Deliberately broad: a bare ``4096`` or
#: ``0.05`` in a comment is nearly always a value someone chose, and pulling a
#: few dull lines into the notes is the cheap side of this trade.
NUMBER_RE = re.compile(r"(?<![\w.])\d{2,}(?![\w])|\b\d+\.\d+\b")

#: Two or more quoted alternatives -- ``'std' or 'sem'``, ``"ln" | "log10"``.
#: A comment that lists the values a parameter accepts is documenting an
#: interface, and the code beside it shows only the one value it was given.
ENUM_RE = re.compile(
    r"""(['"])[^'"]{1,30}\1\s*(?:\||,|/|\bor\b)\s*(['"])""")

#: Dates, issue references, links -- a pointer at something outside the file.
REFERENCE_RE = re.compile(
    r"https?://|\b(19|20)\d{2}-\d{2}-\d{2}\b|\b(19|20)\d{2}\b|"
    r"\bCVE-\d{4}-\d+\b|#\d{2,}\b|\b(issue|ticket|PR|pull request)\b",
    re.IGNORECASE)

#: Words that mark a claim the code cannot make: a reason, a rejected road, a
#: warning, a failure that actually happened.
REASON_WORDS: Sequence[str] = (
    "because", "otherwise", "instead", "rather than", "so that", "the reason",
    "on purpose", "deliberate", "deliberately", "intentional", "intentionally",
    "must not", "cannot", "can't", "do not", "don't", "never", "avoid",
    "beware", "careful", "carefully", "trap", "gotcha", "pitfall",
    "workaround", "work around", "fragile", "silently", "silent", "subtle",
    "surprising", "upstream", "crash", "crashes", "crashed", "segfault",
    "hang", "hangs", "deadlock", "race", "leak", "regression", "regressed",
    "broke", "broken", "breaks", "fails", "failed", "failure", "wrong",
    "incorrect", "used to", "previously", "formerly", "tried", "reverted",
    "revert", "warning", "caution", "important", "does not work",
    "doesn't work", "thread-safe", "not safe", "order matters", "has to be",
    "slower", "faster", "expensive", "cheaper", "costs", "cost of",
    "measured", "benchmark", "profiled", "profiling", "profiler", "oom",
    "out of memory", "caught by",
    "found by", "reported", "traceback", "exception", "silently drops",
    "would", "which is why", "not enough", "only works", "at import",
    "side effect", "circular import", "monkeypatch", "history", "legacy",
    "thread", "threads", "event loop", "blocking", "reentrant", "concurrent",
    "concurrency", "atomic",
)
#: Matched on WHOLE WORDS. Without the boundaries ``race`` fires inside
#: "braces", ``hang`` inside "change" and ``oom`` inside "room", and a
#: classifier that retains everything is not a classifier.
REASON_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in REASON_WORDS) + r")\b",
    re.IGNORECASE)

#: The house voice puts the load-bearing claim in capitals. Two or more
#: capitalised words in a row is a shout, and a shout is never a restatement.
SHOUT_RE = re.compile(r"\b[A-Z]{3,}(?:\s+[A-Z]{2,}){1,}\b")

#: Words carried by almost every sentence; ignored when asking whether a
#: comment says anything the line below it does not.
STOPWORDS = frozenset("""
a an the this that these those it its is are was were be been being of to in
on at for with from by and or not no if then else when while as into over
under out up down we you they he she i do does did done has have had here
there each any all one two both same other another new old more most less
than so but very just only also which who whom what why how per via using use
used uses make makes made get gets got set sets setting give gives given take
takes taken keep keeps kept let lets need needs about after before now still
""".split())

#: Below this many words, a comment is short enough that overlap with the code
#: line under it really does mean it is restating that line.
RESTATE_MAX_WORDS = 14

#: How much of a short comment's vocabulary has to appear in the code below
#: before it counts as a restatement of that code.
RESTATE_OVERLAP = 0.7

#: How many code lines below a comment count as "the code below it". One is
#: too few: a statement spanning three lines would make a label naming its
#: last argument look like new information.
RESTATE_CONTEXT_LINES = 3

#: The shortest prefix two words may share and still be the same word.
#: ``min``/``minimum`` and ``column``/``columns`` are the cases that matter;
#: three characters is short enough to catch them and long enough that
#: unrelated words do not collide.
STEM_PREFIX = 3

#: Prose this long has said something. Three lines of English is not a label.
RETAIN_MIN_LINES = 3

#: ...and so is one long line, whatever it is made of.
RETAIN_MIN_WORDS = 26


class NoteExtractionError(RuntimeError):
    """An extraction that must not continue, with the reason a user can act on."""


@dataclass
class Block:
    """One standalone run of comment lines, or one trailing comment.

    :param path: module the block came from, relative to the repository root.
    :param start_line: 1-based line of the block's first comment.
    :param end_line: 1-based line of its last -- equal to ``start_line`` for
        a one-liner and for every trailing comment.
    :param texts: the raw comment token text of each line, ``#`` included.
    :param column: 0-based column the ``#`` sits at, taken from the token.
        A trailing comment is cut at this column rather than searched for, so
        a ``#`` earlier in the line -- inside a string, where the tokenizer
        already proved it is not a comment -- is never mistaken for it.
    :param trailing: the comment sat after code on its line, so stripping it
        edits that line rather than deleting it.
    :param scope: the qualified name of the function or class the block sits
        in, or ``""`` at module level. This is what the notes file groups on.
    :param anchor: the first code line at or below the block -- for a
        trailing comment, the code it sits behind. It is what the notes print
        so a reader arriving from the code can recognise the place.
    :param context: the first few code lines below the block, which is what a
        suspected restatement is actually compared against.
    """

    path: str
    start_line: int
    end_line: int
    texts: List[str]
    column: int = 0
    trailing: bool = False
    scope: str = ""
    anchor: str = ""
    context: str = ""
    verdict: str = UNSURE
    reason: str = ""

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def bodies(self) -> List[str]:
        """The comment text of each line with the ``#`` and one space gone."""
        out = []
        for text in self.texts:
            stripped = text.lstrip("#")
            if stripped.startswith(" "):
                stripped = stripped[1:]
            out.append(stripped.rstrip())
        return out

    def prose(self) -> List[str]:
        """The lines that are not rules, banners or blanks."""
        kept = []
        for body in self.bodies():
            core = banner_core(body)
            if core:
                kept.append(core)
        return kept

    def prose_text(self) -> str:
        return " ".join(self.prose())


@dataclass
class FileReport:
    """What one module's pass produced, whether or not anything was written."""

    path: str
    total_lines: int
    counts: Counter = field(default_factory=Counter)
    preserved: Counter = field(default_factory=Counter)
    lines_removed: int = 0
    note_lines: int = 0
    ast_verified: bool = False
    ast_failed: bool = False
    error: str = ""
    blocks: List[Block] = field(default_factory=list)

    @property
    def touched(self) -> bool:
        return self.lines_removed > 0


# ---------------------------------------------------------------------------
# reading the source
# ---------------------------------------------------------------------------

def read_source(path: Path) -> Tuple[str, str]:
    """Return ``(text, encoding)``, honouring a PEP 263 coding cookie.

    The cookie is read rather than assumed so that a module declaring a
    non-UTF-8 encoding is decoded and re-encoded the way Python itself would,
    and so a coding line stays meaningful after the pass.
    """
    with open(path, "rb") as handle:
        encoding, _ = tokenize.detect_encoding(handle.readline)
    return path.read_text(encoding=encoding), encoding


def classify_directive(text: str, line_no: int) -> Optional[str]:
    """Name the directive ``text`` is, or None when it is prose.

    :param text: the comment token's own text, ``#`` included.
    :param line_no: 1-based, because the shebang and the coding declaration
        are only themselves at the top of the file.
    """
    if line_no == 1 and text.startswith("#!"):
        return "shebang"
    if line_no <= 2 and CODING_RE.match(text):
        return "coding"
    for name, pattern in DIRECTIVE_PATTERNS:
        if pattern.search(text):
            return name
    return None


def code_lines(tokens: Iterable[tokenize.TokenInfo]) -> set:
    """The 1-based lines carrying a token that is not a comment or a newline.

    Used to find the code a comment sits above without guessing from the raw
    text -- a line inside a triple-quoted string can start with ``#`` and is
    not a comment, and this keeps that distinction with the tokenizer where
    it belongs.
    """
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
            tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    lines = set()
    for token in tokens:
        if token.type in skip:
            continue
        for line in range(token.start[0], token.end[0] + 1):
            lines.add(line)
    return lines


def scope_ranges(tree: ast.AST) -> List[Tuple[int, int, str]]:
    """Every def and class as ``(first_line, last_line, qualified_name)``.

    A decorated definition starts at its first decorator, so a comment sitting
    among the decorators is attributed to the thing being decorated rather
    than to the module.
    """
    ranges: List[Tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                start = child.lineno
                for decorator in child.decorator_list:
                    start = min(start, decorator.lineno)
                end = getattr(child, "end_lineno", child.lineno) or child.lineno
                name = f"{prefix}.{child.name}" if prefix else child.name
                ranges.append((start, end, name))
                walk(child, name)
            else:
                walk(child, prefix)

    walk(tree, "")
    return ranges


def scope_at(ranges: Sequence[Tuple[int, int, str]], line: int) -> str:
    """The innermost definition containing ``line``, or ``""`` for module level."""
    best = ""
    best_span = None
    for start, end, name in ranges:
        if start <= line <= end:
            span = end - start
            if best_span is None or span < best_span:
                best, best_span = name, span
    return best


def definition_starting_at(
    ranges: Sequence[Tuple[int, int, str]], line: int,
) -> Optional[str]:
    """The definition whose own first line is ``line``, if there is one.

    A comment immediately above ``def load()`` is about ``load``, not about
    the module it happens to sit in, and grouping it under ``load`` is the
    difference between a notes file a reader can navigate and a flat dump.
    """
    for start, _end, name in ranges:
        if start == line:
            return name
    return None


# ---------------------------------------------------------------------------
# finding the blocks
# ---------------------------------------------------------------------------

def collect_blocks(
    rel_path: str, source: str,
) -> Tuple[List[Block], Counter]:
    """Split ``source`` into comment blocks and count what was left alone.

    :returns: ``(blocks, preserved)`` -- the prose blocks to be classified,
        and a tally of the directives that were recognised and skipped.
    :raises NoteExtractionError: when the source cannot be tokenized or
        parsed. Nothing is guessed at from a file Python itself would reject.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        raise NoteExtractionError(f"{rel_path}: cannot tokenize: {exc}") from exc
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise NoteExtractionError(f"{rel_path}: cannot parse: {exc}") from exc

    lines = source.splitlines()
    ranges = scope_ranges(tree)
    code = code_lines(tokens)
    preserved: Counter = Counter()

    #: (line, column, text, standalone) for every comment that is prose.
    prose_comments: List[Tuple[int, int, str, bool]] = []
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        line_no, column = token.start
        directive = classify_directive(token.string, line_no)
        if directive is not None:
            preserved[directive] += 1
            continue
        standalone = lines[line_no - 1][:column].strip() == ""
        prose_comments.append((line_no, column, token.string, standalone))

    blocks: List[Block] = []
    current: Optional[Block] = None
    current_col = -1
    for line_no, column, text, standalone in prose_comments:
        if not standalone:
            current = None
            blocks.append(Block(
                path=rel_path, start_line=line_no, end_line=line_no,
                texts=[text], column=column, trailing=True))
            continue
        contiguous = (
            current is not None
            and not current.trailing
            and current.end_line == line_no - 1
            and current_col == column
        )
        if contiguous and current is not None:
            current.texts.append(text)
            current.end_line = line_no
        else:
            current = Block(
                path=rel_path, start_line=line_no, end_line=line_no,
                texts=[text], column=column)
            current_col = column
            blocks.append(current)

    for block in blocks:
        if block.trailing:
            own = lines[block.start_line - 1][:block.column].strip()
            block.anchor = own
            block.context = own
            anchor_line = None
        else:
            block.anchor = _anchor(lines, code, block.end_line)
            block.context = _context(lines, code, block.end_line)
            anchor_line = _anchor_line(code, block.end_line)
        owned = (definition_starting_at(ranges, anchor_line)
                 if anchor_line is not None else None)
        block.scope = owned or scope_at(ranges, block.start_line)
    return blocks, preserved


def _anchor_line(code: set, after: int) -> Optional[int]:
    """The first code line at or below ``after + 1``."""
    if not code:
        return None
    candidates = [line for line in code if line > after]
    return min(candidates) if candidates else None


def _anchor(lines: Sequence[str], code: set, after: int) -> str:
    line_no = _anchor_line(code, after)
    if line_no is None or line_no > len(lines):
        return ""
    return lines[line_no - 1].strip()


def _context(lines: Sequence[str], code: set, after: int,
             span: int = RESTATE_CONTEXT_LINES) -> str:
    """The next few code lines, joined.

    A comment above a statement is above ALL of it, and a statement here is
    routinely three lines long. Comparing a one-line label against only the
    first of them makes the label look like it said something new.
    """
    following = sorted(line for line in code if line > after)[:span]
    return " ".join(lines[line - 1].strip()
                    for line in following if line <= len(lines))


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def banner_core(body: str) -> str:
    """The words in a comment line, with any surrounding rule stripped off.

    ``"-- shutdown ------------"`` yields ``"shutdown"``; a pure rule and a
    blank ``#`` yield ``""``, which is how the caller tells prose from
    decoration.
    """
    if RULE_RE.match(body):
        return ""
    core = BANNER_EDGE_RE.sub("", body).strip()
    return core


def is_banner_block(block: Block) -> bool:
    """Whether the block is only decoration, or decoration plus a short label.

    A banner carrying real prose is not a banner -- the house style opens a
    long note with a ruled title, and that note is exactly the kind worth
    keeping.
    """
    prose = block.prose()
    if not prose:
        return True
    if len(prose) > 1:
        return False
    only = prose[0]
    ruled = any(RULE_RE.match(body) or BANNER_EDGE_RE.search(body)
                for body in block.bodies())
    return ruled and len(only.split()) <= 5 and not only.endswith(".")


def is_commented_out_code(block: Block) -> bool:
    """Whether the block is Python that was commented out rather than deleted.

    Two things have to agree: the text parses as Python, and most of its lines
    read as Python. Either alone is not enough -- a one-word English comment
    parses fine as an expression, and a sentence mentioning ``foo(bar)`` looks
    code-ish without being code.
    """
    bodies = [body for body in block.bodies() if body.strip()]
    if not bodies:
        return False
    text = textwrap.dedent("\n".join(bodies))
    try:
        parsed = ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    if not parsed.body:
        return False
    signals = sum(
        1 for body in bodies
        if any(pattern.match(body) for pattern in CODE_SIGNAL_RES))
    if signals == 0:
        return False
    if len(bodies) == 1:
        return True
    return signals / len(bodies) >= 0.5


def is_bare_marker(block: Block) -> bool:
    """A ``TODO`` with nothing after it but a shrug."""
    prose = block.prose()
    if len(prose) != 1:
        return False
    match = BARE_MARKER_RE.match(prose[0])
    if not match:
        return False
    return len(match.group("rest").split()) <= 3


def content_words(text: str) -> List[str]:
    """Lower-cased words worth comparing, with the filler dropped."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", text.lower())
    return [word for word in words if word not in STOPWORDS and len(word) > 2]


def code_vocabulary(code_line: str) -> set:
    """Every word a line of code mentions, snake_case and camelCase split."""
    vocabulary = set()
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code_line):
        vocabulary.add(raw.lower())
        for part in raw.split("_"):
            if part:
                vocabulary.add(part.lower())
        for part in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", raw):
            vocabulary.add(part.lower())
    return vocabulary


def _same_word(word: str, vocabulary: set) -> bool:
    """Whether ``word`` names something in ``vocabulary``, allowing for
    English's plurals and the abbreviations code prefers -- ``minimum`` for
    ``min``, ``columns`` for ``column``."""
    if word in vocabulary:
        return True
    for other in vocabulary:
        shortest = min(len(word), len(other))
        if shortest >= STEM_PREFIX and (word.startswith(other)
                                        or other.startswith(word)):
            return True
    return False


def restates_the_code(block: Block) -> bool:
    """Whether a short comment says only what the code beside it says.

    Asked only of short blocks, and only after every RETAIN signal has already
    failed: a long note that happens to share vocabulary with the code is
    elaborating on it, not repeating it.
    """
    prose = block.prose()
    if not prose or len(prose) > 2:
        return False
    code = block.context or block.anchor
    if not code:
        return False
    words = content_words(" ".join(prose))
    if not words or len(words) > RESTATE_MAX_WORDS:
        return False
    vocabulary = code_vocabulary(code)
    if not vocabulary:
        return False
    hits = sum(1 for word in words if _same_word(word, vocabulary))
    return hits / len(words) >= RESTATE_OVERLAP


def strong_retain_signal(block: Block) -> str:
    """Name a thing in ``block`` that the code cannot be carrying, or ``""``.

    Checked before every DISCARD rule but one, because the cost of the two
    mistakes is not symmetric: a dull line in a notes file is noise, a
    deleted measurement is gone.
    """
    text = block.prose_text()
    if not text:
        return ""
    if MEASUREMENT_RE.search(text):
        return "a measured quantity"
    if REFERENCE_RE.search(text):
        return "a date, link or issue reference"
    if ENUM_RE.search(text):
        return "the set of values something accepts"
    if SHOUT_RE.search(text):
        return "an emphasised claim"
    match = REASON_RE.search(text)
    if match:
        return f"a reason or warning ({match.group(0).lower()!r})"
    return ""


def weak_retain_signal(block: Block) -> str:
    """Name a weaker reason to keep ``block``, or ``""``.

    These fire on shape rather than on content -- a bare number, or simply
    enough prose that someone was explaining something. They are asked AFTER
    the banner test, because ``# --- 19 ---`` is a numbered section heading
    and not a chosen constant, and there are a lot of those.
    """
    text = block.prose_text()
    if not text:
        return ""
    if NUMBER_RE.search(text):
        return "a chosen number"
    prose = block.prose()
    if len(prose) >= RETAIN_MIN_LINES:
        return f"{len(prose)} lines of prose"
    if len(text.split()) >= RETAIN_MIN_WORDS:
        return f"{len(text.split())} words of prose"
    return ""


def classify(block: Block) -> Block:
    """Set ``block.verdict`` and ``block.reason``, and return the block."""
    if not block.prose():
        block.verdict, block.reason = DISCARD, "a divider or a blank"
        return block
    if is_commented_out_code(block):
        block.verdict, block.reason = DISCARD, "commented-out code"
        return block
    if is_bare_marker(block):
        block.verdict, block.reason = DISCARD, "a marker with no content"
        return block
    signal = strong_retain_signal(block)
    if signal:
        block.verdict, block.reason = RETAIN, signal
        return block
    if is_banner_block(block):
        block.verdict, block.reason = DISCARD, "a section-divider banner"
        return block
    signal = weak_retain_signal(block)
    if signal:
        block.verdict, block.reason = RETAIN, signal
        return block
    if restates_the_code(block):
        block.verdict, block.reason = DISCARD, "restates the line below it"
        return block
    block.verdict, block.reason = UNSURE, "short, and says nothing measurable"
    return block


# ---------------------------------------------------------------------------
# the edit, and the proof that it was only an edit
# ---------------------------------------------------------------------------

def strip_source(source: str, blocks: Sequence[Block]) -> str:
    """Remove ``blocks`` from ``source`` and return the result.

    Line-based on purpose: the token positions say exactly which characters
    are comment, so every other byte of the file is carried across untouched
    rather than round-tripped through :func:`tokenize.untokenize`, which
    rewrites whitespace it was not asked to rewrite.

    This function is the seam the AST guard defends. It is deliberately the
    only thing that edits source text, so a test can replace it with a
    damaged version and watch the guard refuse the write.
    """
    lines = source.splitlines(keepends=True)
    drop = set()
    trim: Dict[int, int] = {}
    for block in blocks:
        if block.trailing:
            trim[block.start_line] = block.column
        else:
            for line_no in range(block.start_line, block.end_line + 1):
                drop.add(line_no)
    out = []
    for index, line in enumerate(lines, start=1):
        if index in drop:
            continue
        if index in trim:
            column = trim[index]
            ending = "\n" if line.endswith("\n") else ""
            out.append(line[:column].rstrip() + ending)
            continue
        out.append(line)
    return "".join(out)


def verify_ast(before: str, after: str) -> bool:
    """Whether ``after`` is ``before`` with comments and nothing else removed.

    Comments never reach the AST, so two identical dumps mean the edit was
    confined to them. THIS IS NOT OPTIONAL and there is no flag that skips
    it: a flag would be reached for on the one file where it fired.
    """
    try:
        return ast.dump(ast.parse(before)) == ast.dump(ast.parse(after))
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# the notes file
# ---------------------------------------------------------------------------

#: A comment's opening line is often a title rather than the first clause of
#: a sentence -- ``# log10 read fraction`` above four lines explaining it.
#: Reflowing the block would run the title into the prose, so a short opening
#: line that does not read as the start of a sentence is kept on its own.
TITLE_MAX_CHARS = 60
TITLE_NEXT_RE = re.compile(r"^[A-Z\"'`(\[]")


def _paragraphs(block: Block) -> List[str]:
    """The block's prose as paragraphs, split on its blank ``#`` lines.

    Hand-wrapped lines are reflowed, because a 66-column wrap chosen for a
    source file is not a wrap for a rendered page -- except where the first
    line is a title, which reflowing would bury.
    """
    runs: List[List[str]] = []
    current: List[str] = []
    for body in block.bodies():
        core = banner_core(body)
        if not core:
            if current:
                runs.append(current)
                current = []
            continue
        current.append(core)
    if current:
        runs.append(current)

    paragraphs: List[str] = []
    for run in runs:
        head = run[0]
        if (len(run) > 1
                and len(head) <= TITLE_MAX_CHARS
                and (head.endswith(":")
                     or (head[-1] not in ".,;-\u2014"
                         and TITLE_NEXT_RE.match(run[1])))):
            paragraphs.append(head)
            run = run[1:]
        paragraphs.append(" ".join(run))
    return paragraphs


def _slug(label: str) -> str:
    """The in-page anchor a Markdown renderer gives a heading of this text.

    Underscores survive and dots do not, which is what GitHub and most
    renderers do -- and it matters here because nearly every heading in these
    files is a Python name.
    """
    return re.sub(r"[^a-z0-9 _-]", "", label.lower()).replace(" ", "-")


def render_notes(rel_path: str, blocks: Sequence[Block]) -> str:
    """The Markdown for one module's notes.

    Grouped by the function or class the block sat in and carrying the source
    line number and the code line it sat above, so a reader going from a line
    of code to its note lands on the right entry instead of scanning a flat
    dump in file order.
    """
    kept = [block for block in blocks if block.verdict != DISCARD]
    header = [
        f"# Notes from `{rel_path}`",
        "",
        f"Prose lifted out of `{rel_path}` by `tools/extract_source_notes.py`.",
        "The module itself carries no comments now, so this file is where its "
        "reasons live; the path mirrors the source path, which is how it is "
        "found.",
        "",
        "Entries are grouped by the function or class they sat in and carry "
        "the line they came from. Line numbers are from the state of the "
        "module when the notes were taken, so they drift; the quoted code "
        "line is the durable anchor.",
        "",
    ]
    if not kept:
        header.append("_Nothing in this module was worth keeping out of "
                      "line: every comment it carried restated its code._")
        header.append("")
        return "\n".join(header)

    by_scope: Dict[str, List[Block]] = {}
    for block in kept:
        by_scope.setdefault(block.scope, []).append(block)
    order = sorted(by_scope, key=lambda scope: by_scope[scope][0].start_line)

    if len(order) > 1:
        header.append("## Contents")
        header.append("")
        for scope in order:
            label = scope or "Module level"
            count = len(by_scope[scope])
            header.append(f"- [{label}](#{_slug(label)}) "
                          f"({count} entr{'y' if count == 1 else 'ies'})")
        header.append("")

    out = list(header)
    for scope in order:
        out.append(f"## {scope or 'Module level'}")
        out.append("")
        for block in by_scope[scope]:
            span = (f"line {block.start_line}" if block.line_count == 1
                    else f"lines {block.start_line}-{block.end_line}")
            tag = "" if block.verdict == RETAIN else "  _(unsure)_"
            kind = ", trailing" if block.trailing else ""
            out.append(f"### {span}{kind}{tag}")
            out.append("")
            if block.anchor:
                anchor = block.anchor
                if len(anchor) > 100:
                    anchor = anchor[:97] + "..."
                out.append("```python")
                out.append(anchor)
                out.append("```")
                out.append("")
            for paragraph in _paragraphs(block):
                out.append(paragraph)
                out.append("")
    return "\n".join(out)


def render_discards(blocks: Sequence[Block]) -> str:
    """Every discarded block, verbatim, for the log that makes a run reversible."""
    out: List[str] = []
    for block in blocks:
        if block.verdict != DISCARD:
            continue
        out.append(f"--- {block.path}:{block.start_line}"
                   f"{'' if block.line_count == 1 else '-' + str(block.end_line)}"
                   f"  [{block.reason}]")
        out.extend(block.texts)
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# one file, then all of them
# ---------------------------------------------------------------------------

def process_file(
    path: Path,
    root: Path,
    out_dir: Path,
    execute: bool = False,
    unsure: str = "notes",
) -> FileReport:
    """Classify, strip and (optionally) write one module.

    :param path: the module.
    :param root: repository root, so the report and the notes path are
        relative and stable.
    :param out_dir: where ``docs/notes`` lives.
    :param execute: write the stripped module and its notes. False -- the
        default -- computes everything, verifies everything, and writes
        nothing at all.
    :param unsure: what happens to UNSURE blocks. ``notes`` (the default)
        moves them to the notes file with the retained ones; ``keep`` leaves
        them in the source; ``discard`` drops them.
    :raises NoteExtractionError: when the AST guard refuses the result. The
        file is left as it was, in a dry run and in a real one alike.
    """
    rel_path = str(path.relative_to(root))
    source, encoding = read_source(path)
    report = FileReport(path=rel_path, total_lines=len(source.splitlines()))
    blocks, preserved = collect_blocks(rel_path, source)
    report.preserved = preserved
    for block in blocks:
        classify(block)
        report.counts[block.verdict] += 1
    report.blocks = blocks

    if unsure == "keep":
        removable = [b for b in blocks if b.verdict != UNSURE]
    else:
        removable = list(blocks)
    to_notes = [b for b in blocks
                if b.verdict == RETAIN
                or (b.verdict == UNSURE and unsure == "notes")]

    if not removable:
        report.ast_verified = True
        return report

    stripped = strip_source(source, removable)
    if not verify_ast(source, stripped):
        report.ast_failed = True
        raise NoteExtractionError(
            f"{rel_path}: the stripped source does not have the same AST as "
            "the original. The file was NOT written and NOT changed. This "
            "means the strip removed something that was not a comment.")
    report.ast_verified = True
    report.lines_removed = (len(source.splitlines())
                            - len(stripped.splitlines()))

    notes = render_notes(rel_path, to_notes) if to_notes else ""
    report.note_lines = len(notes.splitlines()) if notes else 0

    if execute:
        path.write_text(stripped, encoding=encoding)
        if notes:
            note_path = out_dir / rel_path
            note_path = note_path.with_suffix(".md")
            note_path.parent.mkdir(parents=True, exist_ok=True)
            note_path.write_text(notes, encoding="utf-8")
    return report


def iter_modules(targets: Sequence[Path]) -> List[Path]:
    """Every ``.py`` file under ``targets``, sorted, de-duplicated."""
    found: List[Path] = []
    seen = set()
    for target in targets:
        candidates = ([target] if target.is_file()
                      else sorted(target.rglob("*.py")))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(candidate)
    return found


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def _sample(blocks: Sequence[Block], verdict: str, count: int) -> List[Block]:
    """A deterministic spread of ``count`` blocks with ``verdict``.

    Seeded, so two runs over the same tree show the same examples and a
    maintainer comparing reports is comparing like with like.
    """
    pool = [block for block in blocks if block.verdict == verdict]
    if len(pool) <= count:
        return pool
    return random.Random(0).sample(pool, count)


def _show(block: Block, out, limit: int = 4) -> None:
    span = (str(block.start_line) if block.line_count == 1
            else f"{block.start_line}-{block.end_line}")
    where = f" in {block.scope}" if block.scope else ""
    kind = " (trailing)" if block.trailing else ""
    print(f"  {block.path}:{span}{where}{kind}", file=out)
    print(f"    why: {block.reason}", file=out)
    for text in block.texts[:limit]:
        print(f"    | {text[:100]}", file=out)
    if block.line_count > limit:
        print(f"    | ... {block.line_count - limit} more line(s)", file=out)
    print("", file=out)


def report(
    reports: Sequence[FileReport],
    execute: bool,
    unsure: str,
    out,
    samples: int = 20,
    top: Optional[int] = None,
) -> None:
    """Print the per-file table, the totals, and a sample of each bucket."""
    totals: Counter = Counter()
    preserved: Counter = Counter()
    lines_removed = 0
    note_lines = 0
    verified = 0
    failed: List[FileReport] = []
    errored: List[FileReport] = []
    all_blocks: List[Block] = []

    for item in reports:
        if item.error:
            errored.append(item)
            continue
        totals.update(item.counts)
        preserved.update(item.preserved)
        lines_removed += item.lines_removed
        note_lines += item.note_lines
        verified += 1 if item.ast_verified else 0
        if item.ast_failed:
            failed.append(item)
        all_blocks.extend(item.blocks)

    touched = [item for item in reports
               if not item.error and sum(item.counts.values())]
    touched.sort(key=lambda item: -sum(item.counts.values()))
    shown = touched if top is None else touched[:top]

    print("", file=out)
    print("PER FILE" + ("" if top is None else f" (heaviest {len(shown)})"),
          file=out)
    print(f"{'module':<58}{'DISC':>6}{'RET':>6}{'UNSR':>6}"
          f"{'-lines':>8}{'notes':>7}{'kept':>6}  ast", file=out)
    print("-" * 105, file=out)
    for item in shown:
        print(f"{item.path[:57]:<58}"
              f"{item.counts[DISCARD]:>6}{item.counts[RETAIN]:>6}"
              f"{item.counts[UNSURE]:>6}{item.lines_removed:>8}"
              f"{item.note_lines:>7}{sum(item.preserved.values()):>6}"
              f"  {'ok' if item.ast_verified else 'FAILED'}", file=out)
    if top is not None and len(touched) > len(shown):
        print(f"... and {len(touched) - len(shown)} more modules", file=out)

    total_blocks = sum(totals.values())
    print("", file=out)
    print("TOTALS", file=out)
    print(f"  modules read            : {len(reports)}", file=out)
    print(f"  modules with comments   : {len(touched)}", file=out)
    print(f"  blocks classified       : {total_blocks}", file=out)
    for verdict in VERDICTS:
        share = (100.0 * totals[verdict] / total_blocks) if total_blocks else 0
        print(f"    {verdict:<20}: {totals[verdict]:>6}  ({share:4.1f}%)",
              file=out)
    print(f"  comment lines removed   : {lines_removed}", file=out)
    print(f"  lines written to notes  : {note_lines}", file=out)
    print("", file=out)
    print("PRESERVED, NOT TOUCHED (these are not prose)", file=out)
    if preserved:
        for name, count in sorted(preserved.items(), key=lambda kv: -kv[1]):
            label = {
                "sphinx": "#:  Sphinx attribute docstrings (PUBLISHED DOCS)",
                "noqa": "#  noqa",
                "type": "#  type:",
                "pragma": "#  pragma",
                "fmt": "#  fmt: on/off/skip",
                "isort": "#  isort:",
                "lint": "#  pylint/mypy/ruff/flake8/nosec",
                "shebang": "#! shebang",
                "coding": "#  coding declaration (PEP 263)",
            }.get(name, name)
            print(f"  {label:<52}: {count:>6}", file=out)
    else:
        print("  (none found)", file=out)
    print("", file=out)
    print("AST VERIFICATION", file=out)
    print(f"  files verified identical: {verified} / "
          f"{len([r for r in reports if not r.error])}", file=out)
    if failed:
        print(f"  FILES REFUSED           : {len(failed)}", file=out)
        for item in failed:
            print(f"    {item.path}", file=out)
    else:
        print("  files refused           : 0", file=out)
    if errored:
        print(f"  files that could not be read: {len(errored)}", file=out)
        for item in errored:
            print(f"    {item.path}: {item.error}", file=out)

    for verdict in VERDICTS:
        picked = _sample(all_blocks, verdict, samples)
        print("", file=out)
        print(f"SAMPLE -- {verdict} ({len(picked)} of {totals[verdict]})",
              file=out)
        print("", file=out)
        for block in picked:
            _show(block, out)

    print("", file=out)
    if execute:
        print("WRITTEN. The modules above were stripped and their notes "
              "written.", file=out)
    else:
        print("DRY RUN -- nothing was written. No module was changed, no "
              "notes file was created.", file=out)
        print(f"UNSURE blocks would go to: {unsure}.", file=out)
        print("Re-run with --execute to apply it.", file=out)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def run(
    targets: Sequence[Path],
    root: Path,
    out_dir: Path,
    execute: bool = False,
    unsure: str = "notes",
    samples: int = 20,
    top: Optional[int] = None,
    discard_log: Optional[Path] = None,
    out=None,
) -> int:
    """Classify every module under ``targets`` and report.

    :returns: 0 when every file was verified, 1 when any file was refused by
        the AST guard or could not be read.
    """
    if out is None:
        out = sys.stdout
    modules = iter_modules(targets)
    print(f"repository : {root}", file=out)
    print("scanning   : "
          + ", ".join(str(t.relative_to(root)) for t in targets), file=out)
    print(f"notes to   : {out_dir.relative_to(root)}/<same path>.md", file=out)
    print(f"mode       : {'EXECUTE' if execute else 'DRY RUN'}", file=out)
    print(f"modules    : {len(modules)}", file=out)

    reports: List[FileReport] = []
    for module in modules:
        rel = str(module.relative_to(root))
        try:
            reports.append(process_file(
                module, root, out_dir, execute=execute, unsure=unsure))
        except NoteExtractionError as exc:
            item = FileReport(path=rel, total_lines=0)
            item.error = str(exc)
            item.ast_failed = "same AST" in str(exc)
            reports.append(item)

    if discard_log is not None:
        blocks = [b for item in reports for b in item.blocks]
        text = render_discards(blocks)
        if execute:
            discard_log.parent.mkdir(parents=True, exist_ok=True)
            discard_log.write_text(text, encoding="utf-8")
            print(f"discard log: {discard_log}", file=out)
        else:
            print(f"discard log: {discard_log} "
                  f"({len(text.splitlines())} lines; not written in a dry run)",
                  file=out)

    report(reports, execute=execute, unsure=unsure, out=out,
           samples=samples, top=top)
    bad = [item for item in reports if item.error or item.ast_failed]
    return 1 if bad else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify the comments in a package, move the ones worth keeping "
            "to docs/notes/<same path>.md, and strip them from the source. "
            "Dry run unless --execute is given. Every file is checked with "
            "ast.dump before it is written; a file whose AST changed is "
            "refused."),
    )
    parser.add_argument(
        "targets", nargs="*", default=["spacr"], metavar="PATH",
        help="files or directories to scan (default: spacr)")
    parser.add_argument(
        "--root", default=".",
        help="repository root, used for relative paths (default: cwd)")
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"where the notes go (default: {DEFAULT_OUT_DIR})")
    parser.add_argument(
        "--execute", action="store_true",
        help="write the stripped modules and the notes (default is a dry run)")
    parser.add_argument(
        "--unsure", choices=("notes", "keep", "discard"), default="notes",
        help=("what to do with blocks the classifier could not call: move "
              "them to the notes (default), keep them in the source, or "
              "drop them"))
    parser.add_argument(
        "--samples", type=int, default=20, metavar="N",
        help="how many example blocks to print per class (default: 20)")
    parser.add_argument(
        "--top", type=int, default=None, metavar="N",
        help="show only the N heaviest modules in the per-file table")
    parser.add_argument(
        "--discard-log", default=None, metavar="PATH",
        help=("write every discarded block verbatim to PATH, so a real run "
              "throws nothing away irrecoverably"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    targets = [Path(t) if Path(t).is_absolute() else root / t
               for t in args.targets]
    missing = [t for t in targets if not t.exists()]
    if missing:
        print("extract_source_notes: no such path: "
              + ", ".join(str(m) for m in missing), file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    discard_log = None
    if args.discard_log:
        discard_log = Path(args.discard_log)
        if not discard_log.is_absolute():
            discard_log = root / discard_log
    try:
        return run(targets, root, out_dir, execute=args.execute,
                   unsure=args.unsure, samples=args.samples, top=args.top,
                   discard_log=discard_log)
    except NoteExtractionError as exc:
        print(f"extract_source_notes: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
