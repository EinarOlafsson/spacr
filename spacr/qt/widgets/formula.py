"""Computed columns — a small expression language, parsed rather than ``eval``-ed.

``ratio = cell_area / cell_perimeter ** 2`` is the sort of thing a user wants
five seconds after seeing a measurement table, and until this module the answer
was "export it and open pandas". The obvious implementation is one line::

    frame[name] = eval(expression, {}, frame)          # never

and that line hands anyone who can type into the box — or anyone who can put a
saved chart spec in front of the user — the whole interpreter. A spaCR settings
file, a shared ``.spacr`` project, a macro pasted from a colleague: all of them
would become executable. So the expression is **tokenised, parsed into an AST of
six node types, and walked**. There is no ``eval``, no ``exec``, no ``compile``
and no attribute access anywhere in the grammar; a hostile string does not fail
a blacklist, it fails to *parse*.

The grammar
-----------

::

    expression := or_expr
    or_expr    := and_expr ('or' and_expr)*
    and_expr   := not_expr ('and' not_expr)*
    not_expr   := 'not' not_expr | comparison
    comparison := sum (('<' | '<=' | '>' | '>=' | '==' | '!=') sum)?
    sum        := product (('+' | '-') product)*
    product    := unary (('*' | '/' | '//' | '%') unary)*
    unary      := ('-' | '+') unary | power
    power      := atom ('**' unary)?
    atom       := NUMBER | COLUMN | FUNCTION '(' [expression {',' expression}] ')'
                | '(' expression ')'

    COLUMN     := [A-Za-z_][A-Za-z_0-9]*  |  '`' <anything but a backtick> '`'
    NUMBER     := digits ['.' digits] [('e'|'E') ['+'|'-'] digits]

Five deliberate absences, each of which is a class of attack or a class of
confusion rather than a feature nobody got round to:

* **no string literals** — so no payload can be smuggled through as data, and
  the language has nothing to feed to a function that might want a name;
* **no attribute access, no indexing, no assignment** — ``.``, ``[`` and a bare
  ``=`` are rejected by the tokeniser with the reason, so the
  ``().__class__.__bases__`` ladder does not get as far as the parser;
* **no lambdas, no comprehensions, no statements** — an expression evaluates to
  one column and nothing else;
* **no arbitrary names** — a name is either a column of the frame or one of
  :data:`FUNCTIONS`. ``__import__``, ``open`` and ``eval`` are not blocked as
  special cases; they are simply not columns, and the error says so;
* **comparisons do not chain** — ``0 < area < 5`` is refused rather than
  silently parsed as something numpy would evaluate elementwise into a shape
  nobody meant. The message says to write the conjunction.

Numbers are read as **floats**, always. That is not cosmetic: ``9 ** 9 ** 9``
with Python integers is a multi-second allocation of a number with 300 million
digits — a denial of service typed in eleven characters — and with floats it is
``inf`` in a nanosecond. The node count and nesting depth are capped as well
(:data:`MAX_NODES`, :data:`MAX_DEPTH`), so a pathological expression is refused
at parse time rather than during a redraw.

What a formula computes over
----------------------------

**The whole table, never the filtered view.** ``zscore(area)`` computed over
whatever the Local Data Filter currently shows would change every time a slider
moved, which means the column would not be a column: two charts drawn a second
apart would disagree, and an exported CSV would record one arbitrary moment. So
:func:`compute` is given the loaded frame and the aggregates
(:data:`AGGREGATE_FUNCTIONS`) reduce over all of it. Filtering happens
afterwards, to the computed column like any other.

Formulas are evaluated **in list order**, and each one sees the columns added
before it — so ``density = count / area`` then ``log_density = log(density)``
works, and referring to a formula defined further down is an error that says so
rather than a NaN column.

What comes out
--------------

Arithmetic gives ``float64``; a comparison or an ``and``/``or``/``not`` gives
``bool``. The bool case is on purpose: a boolean column lands in
:func:`spacr.qt.widgets.data_filter_panel.classify_columns` as a *category*, so
``infected = pathogen_count > 0`` immediately becomes a tick box in the filter
panel and a colour channel in the Graph Builder, which is what anyone writing
that expression wanted next.

Division by zero produces ``inf`` rather than an exception — a ratio column with
a few infinities is a real answer about a few objects — but the count of
non-finite results is carried in :attr:`ColumnResult.n_nonfinite` and said out
loud in :attr:`ColumnResult.notice`. A column that is 90% ``inf`` is a mistake,
and the only way to notice is to be told.

Text columns are not addressable. ``gene`` cannot be used in a formula, and the
error says to use the Local Data Filter for it. Arithmetic on a gene name has no
meaning, and coercing it to NaN silently would produce an all-NaN column with no
explanation.

No Qt in here — pure numpy and pandas, like
:mod:`spacr.qt.widgets.graph_spec` and :mod:`spacr.selection`, so the grammar
can be tested without a display and used from a notebook.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "FormulaError",
    "MAX_LENGTH", "MAX_NODES", "MAX_DEPTH",
    "FUNCTIONS", "FUNCTION_HELP", "AGGREGATE_FUNCTIONS",
    "TABLE_DEPENDENT_FUNCTIONS", "KEYWORDS",
    "Node", "Number", "Column", "Unary", "Binary", "Call",
    "tokenize", "parse", "unparse", "referenced_columns",
    "ColumnFormula", "ColumnResult", "FormulaSet", "compute", "evaluate",
]


class FormulaError(ValueError):
    """A formula that cannot be computed, with the reason and the place.

    Every message names the thing that is wrong — the column, the function, the
    character, its position in the string — because the user is looking at a
    one-line text box and "invalid syntax" tells them nothing about which of
    the forty characters to change.
    """


#: Characters in an expression. Long enough for anything readable, short
#: enough that a pasted blob is refused before it is parsed.
MAX_LENGTH = 2_000

#: AST nodes in one expression. A legitimate formula is a dozen.
MAX_NODES = 500

#: Nesting depth. Deep enough for any human-written expression, shallow
#: enough that the recursive walk cannot exhaust the Python stack — a
#: ``RecursionError`` out of a GUI callback takes the window with it.
MAX_DEPTH = 40

#: The three words that are not column names. A column genuinely called
#: ``and`` is still reachable, as `` `and` ``.
KEYWORDS = frozenset({"and", "or", "not"})


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

_NUMBER = "number"
_NAME = "name"
_OP = "op"
_LPAREN = "("
_RPAREN = ")"
_COMMA = ","
_END = "end"

#: Longest first, so ``**`` is never read as two ``*`` and ``<=`` never as
#: ``<`` followed by a stray ``=``.
_OPERATORS = ("**", "//", "<=", ">=", "==", "!=", "+", "-", "*", "/", "%",
              "<", ">")

_TOKEN_RE = re.compile(r"""
    (?P<space>\s+)
  | (?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<quoted>`[^`]*`)
  | (?P<op>\*\*|//|<=|>=|==|!=|[+\-*/%<>])
  | (?P<lparen>\()
  | (?P<rparen>\))
  | (?P<comma>,)
""", re.VERBOSE)

#: Characters worth a better message than "unexpected". Each of these is
#: something a user could reasonably type, and each one means the language
#: they think they are writing is not this one.
_REJECTED = {
    ".": ("attribute access is not part of this language. A decimal point is "
          "fine ({example}); a dot after a name is not"),
    "[": "indexing is not part of this language",
    "]": "indexing is not part of this language",
    "{": "braces are not part of this language",
    "}": "braces are not part of this language",
    '"': ("text is not part of this language — formulas compute over numeric "
          "columns. Use the Local Data Filter to select on text"),
    "'": ("text is not part of this language — formulas compute over numeric "
          "columns. Use the Local Data Filter to select on text"),
    ";": "one expression per formula; there are no statements",
    ":": "there are no lambdas, slices or annotations in this language",
    "=": ("assignment is not part of the expression — the new column's name "
          "goes in the name box. Did you mean '==' ?"),
    "&": "write 'and' rather than '&'",
    "|": "write 'or' rather than '|'",
    "!": "write 'not' rather than '!'",
    "~": "write 'not' rather than '~'",
    "@": "there is no matrix product in this language",
    "\\": "there are no escapes in this language",
    "#": "there are no comments in this language",
    "$": "'$' is not part of this language",
    "?": "there is no conditional operator; use where(test, a, b)",
}


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    at: int

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.kind}:{self.text}@{self.at}"


def tokenize(expression: str) -> Tuple[_Token, ...]:
    """Split ``expression`` into tokens, or say exactly where it stopped making
    sense.

    :raises FormulaError: on an over-long expression, an unterminated backtick
        or a character the language does not contain — with the character, its
        position, and what to write instead where there is an alternative.
    """
    text = str(expression)
    if len(text) > MAX_LENGTH:
        raise FormulaError(
            f"this formula is {len(text):,} characters; the limit is "
            f"{MAX_LENGTH:,}. A formula that long is a script, and this is an "
            f"expression language")
    tokens: List[_Token] = []
    position = 0
    while position < len(text):
        match = _TOKEN_RE.match(text, position)
        if match is None:
            character = text[position]
            hint = _REJECTED.get(character)
            if character == "`":
                raise FormulaError(
                    f"unterminated ` at position {position + 1}; a column name "
                    f"in backticks needs a closing one, as `cell area`")
            if isinstance(hint, str) and "{example}" in hint:
                hint = hint.format(example="0.5")
            detail = f": {hint}" if hint else " is not part of this language"
            raise FormulaError(
                f"{character!r} at position {position + 1}{detail}")
        position = match.end()
        kind = match.lastgroup
        body = match.group()
        if kind == "space":
            continue
        if kind == "quoted":
            name = body[1:-1]
            if not name.strip():
                raise FormulaError(
                    f"empty column name in backticks at position "
                    f"{match.start() + 1}")
            tokens.append(_Token(_NAME, name, match.start()))
            continue
        mapping = {"number": _NUMBER, "name": _NAME, "op": _OP,
                   "lparen": _LPAREN, "rparen": _RPAREN, "comma": _COMMA}
        tokens.append(_Token(mapping[kind], body, match.start()))
    tokens.append(_Token(_END, "", len(text)))
    return tuple(tokens)


# ---------------------------------------------------------------------------
# The AST — five node types and a base, all frozen
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """Base of the five node types. Frozen: a parsed formula is a value."""


@dataclass(frozen=True)
class Number(Node):
    """A numeric literal. Always a ``float`` — see the module docstring."""

    value: float


@dataclass(frozen=True)
class Column(Node):
    """A reference to a column of the frame."""

    name: str


@dataclass(frozen=True)
class Unary(Node):
    """``-x``, ``+x`` or ``not x``."""

    op: str
    operand: Node


@dataclass(frozen=True)
class Binary(Node):
    """Every infix operator, arithmetic, comparison and boolean alike."""

    op: str
    left: Node
    right: Node


@dataclass(frozen=True)
class Call(Node):
    """One of :data:`FUNCTIONS`, applied to its arguments."""

    func: str
    args: Tuple[Node, ...]


# ---------------------------------------------------------------------------
# The functions
# ---------------------------------------------------------------------------

def _elementwise(fn):
    """Wrap a numpy ufunc so a divide/overflow warning is not printed.

    ``log(0)`` and ``x / 0`` are answers (``-inf``, ``inf``) that
    :class:`ColumnResult` counts and reports. A ``RuntimeWarning`` per redraw
    is noise that hides the warnings worth reading.
    """
    def call(*args):
        with np.errstate(all="ignore"):
            return fn(*args)
    return call


def _sample_std(values: np.ndarray) -> float:
    """Sample SD (ddof 1), NaN-aware. NaN at n < 2.

    ``ddof=1`` to agree with :mod:`spacr.qt.widgets.pivot_spec`: one object has
    no spread, and a 0 there reads as "perfectly reproducible".
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan")
    return float(np.std(finite, ddof=1))


def _aggregate(fn):
    def call(values):
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return float("nan")
        with np.errstate(all="ignore"):
            return float(fn(finite))
    return call


def _zscore(values):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size < 2:
        return np.full(array.shape, np.nan)
    with np.errstate(all="ignore"):
        return (array - float(np.mean(finite))) / _sample_std(array)


def _rank(values):
    """Average ranks, 1-based, NaN left as NaN.

    The rank of an object among the objects that *have* a value, which is what
    makes ``rank(area) / count(area)`` a percentile.
    """
    return pd.Series(np.asarray(values, dtype=float)).rank(
        method="average", na_option="keep").to_numpy(dtype=float)


def _quantile(values, q):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    fraction = float(np.asarray(q, dtype=float).reshape(-1)[0])
    if not 0.0 <= fraction <= 1.0:
        raise FormulaError(
            f"quantile() takes a fraction in [0, 1]; {fraction:g} is outside "
            f"it. quantile(area, 0.75) is the upper quartile of area")
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, fraction))


#: ``name -> (implementation, min_args, max_args, is_aggregate)``.
#:
#: Aggregates reduce the whole column to one number and broadcast it, so
#: ``area / mean(area)`` is each object relative to the table. Elementwise
#: functions map value to value.
FUNCTIONS: Dict[str, Tuple[Any, int, int, bool]] = {
    # -- elementwise --------------------------------------------------
    "abs": (_elementwise(np.abs), 1, 1, False),
    "sqrt": (_elementwise(np.sqrt), 1, 1, False),
    "exp": (_elementwise(np.exp), 1, 1, False),
    "log": (_elementwise(np.log), 1, 1, False),
    "log2": (_elementwise(np.log2), 1, 1, False),
    "log10": (_elementwise(np.log10), 1, 1, False),
    "log1p": (_elementwise(np.log1p), 1, 1, False),
    "sign": (_elementwise(np.sign), 1, 1, False),
    "floor": (_elementwise(np.floor), 1, 1, False),
    "ceil": (_elementwise(np.ceil), 1, 1, False),
    "round": (_elementwise(lambda x, n=0.0: np.round(
        x, int(np.asarray(n, dtype=float).reshape(-1)[0]))), 1, 2, False),
    "clip": (_elementwise(np.clip), 3, 3, False),
    "where": (_elementwise(
        lambda c, a, b: np.where(np.asarray(c, dtype=bool), a, b)), 3, 3, False),
    "minimum": (_elementwise(np.minimum), 2, 2, False),
    "maximum": (_elementwise(np.maximum), 2, 2, False),
    "isnan": (_elementwise(lambda x: ~np.isfinite(np.asarray(x, dtype=float))),
              1, 1, False),
    "isfinite": (_elementwise(lambda x: np.isfinite(np.asarray(x, dtype=float))),
                 1, 1, False),
    "zscore": (_zscore, 1, 1, False),
    "rank": (_rank, 1, 1, False),
    # -- aggregates ---------------------------------------------------
    "mean": (_aggregate(np.mean), 1, 1, True),
    "median": (_aggregate(np.median), 1, 1, True),
    "std": (lambda v: _sample_std(np.asarray(v, dtype=float)), 1, 1, True),
    "var": (lambda v: _sample_std(np.asarray(v, dtype=float)) ** 2, 1, 1, True),
    "sum": (_aggregate(np.sum), 1, 1, True),
    "count": (lambda v: float(np.isfinite(
        np.asarray(v, dtype=float)).sum()), 1, 1, True),
    "min": (_aggregate(np.min), 1, 1, True),
    "max": (_aggregate(np.max), 1, 1, True),
    "quantile": (_quantile, 2, 2, True),
}

#: The ones that reduce. Named separately because the difference matters to
#: anyone reading a formula: ``min(area)`` is one number for the whole table,
#: and ``minimum(area, 100)`` is a value per object.
AGGREGATE_FUNCTIONS: Tuple[str, ...] = tuple(
    sorted(name for name, (_f, _lo, _hi, agg) in FUNCTIONS.items() if agg))

#: Functions whose answer for one object depends on the other objects.
#:
#: Every aggregate, plus the two that do not *reduce* but still read the whole
#: column: ``zscore`` needs the mean and SD, ``rank`` needs the order. Kept
#: apart from :data:`AGGREGATE_FUNCTIONS` because the two facts are used for
#: different things — the evaluator cares which calls return a scalar, and the
#: user cares which columns change when the table does.
TABLE_DEPENDENT_FUNCTIONS = frozenset(AGGREGATE_FUNCTIONS) | {"zscore", "rank"}

#: One line each, for the function list beside the box.
FUNCTION_HELP: Dict[str, str] = {
    "abs": "abs(x) — magnitude",
    "sqrt": "sqrt(x) — square root; negative input gives NaN",
    "exp": "exp(x) — e to the x",
    "log": "log(x) — natural log; log(0) is -inf, log(negative) is NaN",
    "log2": "log2(x)",
    "log10": "log10(x)",
    "log1p": "log1p(x) — log(1 + x), accurate for small x",
    "sign": "sign(x) — -1, 0 or 1",
    "floor": "floor(x)",
    "ceil": "ceil(x)",
    "round": "round(x[, digits])",
    "clip": "clip(x, low, high) — pin x into a range",
    "where": "where(test, a, b) — a where test holds, b elsewhere",
    "minimum": "minimum(a, b) — the smaller of the two, per object",
    "maximum": "maximum(a, b) — the larger of the two, per object",
    "isnan": "isnan(x) — true where x has no finite value",
    "isfinite": "isfinite(x) — true where it does",
    "zscore": "zscore(x) — (x - mean) / sd over the whole table",
    "rank": "rank(x) — 1-based average rank; ties share their mean rank",
    "mean": "mean(x) — one number, over the whole table",
    "median": "median(x) — one number, over the whole table",
    "std": "std(x) — sample SD (ddof 1); NaN when fewer than two values",
    "var": "var(x) — sample variance (ddof 1)",
    "sum": "sum(x) — one number, over the whole table",
    "count": "count(x) — how many objects have a finite value",
    "min": "min(x) — the smallest value in the table (see minimum)",
    "max": "max(x) — the largest value in the table (see maximum)",
    "quantile": "quantile(x, q) — the q-fraction of x, q in [0, 1]",
}

_COMPARISONS = {"<", "<=", ">", ">=", "==", "!="}
_ARITHMETIC = {"+", "-", "*", "/", "//", "%", "**"}


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------

class _Parser:
    """Recursive descent over the grammar in the module docstring.

    One class rather than a pile of closures so the node budget and the depth
    are counters rather than globals, and so a parse cannot leak state into the
    next one.
    """

    def __init__(self, tokens: Sequence[_Token], source: str):
        self._tokens = list(tokens)
        self._at = 0
        self._source = source
        self._nodes = 0
        self._depth = 0

    # -- token helpers -------------------------------------------------
    @property
    def _current(self) -> _Token:
        return self._tokens[self._at]

    def _advance(self) -> _Token:
        token = self._tokens[self._at]
        self._at += 1
        return token

    def _accept_op(self, *ops: str) -> Optional[_Token]:
        token = self._current
        if token.kind == _OP and token.text in ops:
            return self._advance()
        return None

    def _accept_word(self, word: str) -> Optional[_Token]:
        token = self._current
        if token.kind == _NAME and token.text == word:
            return self._advance()
        return None

    def _where(self, token: _Token) -> str:
        return f" at position {token.at + 1}"

    def _count(self, node: Node) -> Node:
        self._nodes += 1
        if self._nodes > MAX_NODES:
            raise FormulaError(
                f"this formula has more than {MAX_NODES} parts; a formula that "
                f"big should be built as several columns, each with a name")
        return node

    # -- the grammar ---------------------------------------------------
    def parse(self) -> Node:
        node = self._or()
        if self._current.kind != _END:
            token = self._current
            extra = ""
            if token.kind == _NAME and token.text in KEYWORDS:
                extra = f" — {token.text!r} joins two comparisons"
            raise FormulaError(
                f"unexpected {token.text!r}{self._where(token)}{extra}; the "
                f"expression already ended before it")
        return node

    def _nest(self, method):
        self._depth += 1
        if self._depth > MAX_DEPTH:
            raise FormulaError(
                f"this formula nests more than {MAX_DEPTH} levels deep; "
                f"split it into several named columns")
        try:
            return method()
        finally:
            self._depth -= 1

    def _or(self) -> Node:
        node = self._and()
        while self._accept_word("or"):
            node = self._count(Binary("or", node, self._and()))
        return node

    def _and(self) -> Node:
        node = self._not()
        while self._accept_word("and"):
            node = self._count(Binary("and", node, self._not()))
        return node

    def _not(self) -> Node:
        if self._accept_word("not"):
            return self._count(Unary("not", self._nest(self._not)))
        return self._comparison()

    def _comparison(self) -> Node:
        node = self._sum()
        token = self._accept_op(*_COMPARISONS)
        if token is None:
            return node
        node = self._count(Binary(token.text, node, self._sum()))
        second = self._current
        if second.kind == _OP and second.text in _COMPARISONS:
            raise FormulaError(
                f"chained comparison{self._where(second)}: "
                f"'a {token.text} b {second.text} c' is ambiguous here. Write "
                f"it as two, joined with 'and'")
        return node

    def _sum(self) -> Node:
        node = self._product()
        while True:
            token = self._accept_op("+", "-")
            if token is None:
                return node
            node = self._count(Binary(token.text, node, self._product()))

    def _product(self) -> Node:
        node = self._unary()
        while True:
            token = self._accept_op("*", "/", "//", "%")
            if token is None:
                return node
            node = self._count(Binary(token.text, node, self._unary()))

    def _unary(self) -> Node:
        token = self._accept_op("+", "-")
        if token is not None:
            return self._count(Unary(token.text, self._nest(self._unary)))
        return self._power()

    def _power(self) -> Node:
        node = self._atom()
        token = self._accept_op("**")
        if token is None:
            return node
        # Right-associative, and the exponent goes through `_unary` so
        # `2 ** -1` parses. `-a ** 2` is `-(a ** 2)`, as in Python and as in
        # every maths textbook.
        return self._count(Binary("**", node, self._nest(self._unary)))

    def _atom(self) -> Node:
        token = self._advance()
        if token.kind == _NUMBER:
            return self._count(Number(float(token.text)))
        if token.kind == _LPAREN:
            node = self._nest(self._or)
            closing = self._advance()
            if closing.kind != _RPAREN:
                raise FormulaError(
                    f"missing ')' — the '(' at position {token.at + 1} is "
                    f"never closed")
            return node
        if token.kind == _NAME:
            if token.text in KEYWORDS:
                raise FormulaError(
                    f"{token.text!r}{self._where(token)} needs something "
                    f"before it; it joins two expressions")
            if self._current.kind == _LPAREN:
                return self._call(token)
            return self._count(Column(token.text))
        if token.kind == _END:
            raise FormulaError(
                "the formula ends early — something is missing after "
                f"{self._previous_text()!r}"
                if self._at > 1 else "the formula is empty")
        raise FormulaError(
            f"unexpected {token.text!r}{self._where(token)}")

    def _previous_text(self) -> str:
        return self._tokens[max(0, self._at - 2)].text

    def _call(self, name_token: _Token) -> Node:
        name = name_token.text
        if name not in FUNCTIONS:
            close = get_close_matches(name, sorted(FUNCTIONS), n=1, cutoff=0.6)
            suggestion = f". Did you mean {close[0]}()?" if close else ""
            raise FormulaError(
                f"there is no function called {name}(){self._where(name_token)}"
                f"{suggestion} The functions are: "
                f"{', '.join(sorted(FUNCTIONS))}")
        self._advance()                      # the '('
        args: List[Node] = []
        if self._current.kind != _RPAREN:
            args.append(self._nest(self._or))
            while self._current.kind == _COMMA:
                self._advance()
                args.append(self._nest(self._or))
        closing = self._advance()
        if closing.kind != _RPAREN:
            raise FormulaError(
                f"missing ')' after the arguments of {name}()")
        _fn, low, high, _agg = FUNCTIONS[name]
        if not low <= len(args) <= high:
            wanted = (f"{low}" if low == high else f"{low} to {high}")
            raise FormulaError(
                f"{name}() takes {wanted} argument(s), not {len(args)} — "
                f"{FUNCTION_HELP.get(name, name)}")
        return self._count(Call(name, tuple(args)))


def parse(expression: str) -> Node:
    """Parse ``expression`` into an AST.

    :raises FormulaError: for anything that is not a valid expression in the
        grammar above, with the position and what to write instead.
    """
    text = str(expression).strip()
    if not text:
        raise FormulaError(
            "the formula is empty. Write something like "
            "cell_area / cell_perimeter ** 2")
    return _Parser(tokenize(text), text).parse()


def referenced_columns(node: Node) -> Tuple[str, ...]:
    """Every column ``node`` reads, in first-appearance order, de-duplicated."""
    found: Dict[str, None] = {}

    def walk(item: Node) -> None:
        if isinstance(item, Column):
            found.setdefault(item.name, None)
        elif isinstance(item, Unary):
            walk(item.operand)
        elif isinstance(item, Binary):
            walk(item.left)
            walk(item.right)
        elif isinstance(item, Call):
            for arg in item.args:
                walk(arg)

    walk(node)
    return tuple(found)


_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


def unparse(node: Node) -> str:
    """Print ``node`` back as an expression, fully parenthesised.

    Not for display — for the round trip. ``parse(unparse(parse(text)))``
    equalling ``parse(text)`` is what says the parser and the tree agree about
    precedence, which is the one property of a hand-written parser that is
    hard to eyeball and easy to get wrong.
    """
    if isinstance(node, Number):
        return repr(node.value)
    if isinstance(node, Column):
        return (node.name if _SAFE_NAME.match(node.name)
                and node.name not in KEYWORDS else f"`{node.name}`")
    if isinstance(node, Unary):
        space = " " if node.op == "not" else ""
        return f"({node.op}{space}{unparse(node.operand)})"
    if isinstance(node, Binary):
        return f"({unparse(node.left)} {node.op} {unparse(node.right)})"
    if isinstance(node, Call):
        return f"{node.func}({', '.join(unparse(a) for a in node.args)})"
    raise FormulaError(f"cannot print a {type(node).__name__}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """``frame[name]`` as float, or a message naming the column and the problem."""
    if name not in frame.columns:
        close = get_close_matches(name, [str(c) for c in frame.columns],
                                  n=3, cutoff=0.6)
        suggestion = (f". The closest columns are: {', '.join(close)}"
                      if close else "")
        raise FormulaError(
            f"there is no column called {name!r} in this table (it has "
            f"{len(frame.columns)} columns){suggestion}")
    series = frame[name]
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.notna().any():
        # Numbers stored as text — a CSV column read as object. Usable, and
        # the values that are not numbers become NaN rather than an error.
        return coerced.to_numpy(dtype=float)
    raise FormulaError(
        f"column {name!r} is text, not a number, so there is nothing to "
        f"compute with it. Use the Local Data Filter to select on {name!r}")


def _as_array(value: Any, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(length, array.item() if array.size else np.nan)
    return array


def evaluate(node: Node, frame: pd.DataFrame) -> Any:
    """Evaluate ``node`` over ``frame``.

    :returns: an ``ndarray`` the length of the frame, or a python ``float``
        when the whole expression reduces (``mean(area)``). The caller
        broadcasts — keeping scalars scalar is what lets ``area / mean(area)``
        cost one division rather than two array allocations.
    :raises FormulaError: naming the column or the function that failed.
    """
    length = len(frame)

    def walk(item: Node) -> Any:
        if isinstance(item, Number):
            return item.value
        if isinstance(item, Column):
            return _numeric_column(frame, item.name)
        if isinstance(item, Unary):
            operand = walk(item.operand)
            with np.errstate(all="ignore"):
                if item.op == "-":
                    return -np.asarray(operand, dtype=float) \
                        if not np.isscalar(operand) else -float(operand)
                if item.op == "+":
                    return operand
                return ~np.asarray(operand, dtype=bool)
        if isinstance(item, Binary):
            return _binary(item, walk(item.left), walk(item.right))
        if isinstance(item, Call):
            return _call(item, [walk(arg) for arg in item.args], length)
        raise FormulaError(  # pragma: no cover - every node type is above
            f"cannot evaluate a {type(item).__name__}")

    return walk(node)


def _binary(item: Binary, left: Any, right: Any) -> Any:
    op = item.op
    with np.errstate(all="ignore"):
        if op in ("and", "or"):
            a = np.asarray(left, dtype=bool)
            b = np.asarray(right, dtype=bool)
            return (a & b) if op == "and" else (a | b)
        if op in _COMPARISONS:
            a = np.asarray(left, dtype=float)
            b = np.asarray(right, dtype=float)
            return {"<": np.less, "<=": np.less_equal, ">": np.greater,
                    ">=": np.greater_equal, "==": np.equal,
                    "!=": np.not_equal}[op](a, b)
        a = np.asarray(left, dtype=float)
        b = np.asarray(right, dtype=float)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        if op == "//":
            return a // b
        if op == "%":
            return a % b
        # `**`. Floats throughout, so a huge exponent is `inf` rather than a
        # bignum allocation that never returns — see the module docstring.
        return a ** b


def _call(item: Call, args: List[Any], length: int) -> Any:
    fn, _low, _high, aggregate = FUNCTIONS[item.func]
    try:
        if aggregate:
            first = _as_array(args[0], length).astype(float, copy=False)
            return fn(first, *args[1:]) if len(args) > 1 else fn(first)
        return fn(*args)
    except FormulaError:
        raise
    except Exception as exc:
        raise FormulaError(
            f"{item.func}() could not be computed: {exc}") from None


# ---------------------------------------------------------------------------
# The user-facing objects
# ---------------------------------------------------------------------------

def _valid_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        raise FormulaError("a computed column needs a name")
    if not _SAFE_NAME.match(text):
        raise FormulaError(
            f"{text!r} is not a usable column name — letters, digits and "
            f"underscores, not starting with a digit. Anything else has to be "
            f"quoted everywhere it is used")
    if text in FUNCTIONS or text in KEYWORDS:
        raise FormulaError(
            f"{text!r} is already the name of a function in this language; a "
            f"column called that could never be referred to")
    return text


@dataclass(frozen=True)
class ColumnFormula:
    """One computed column: a name and an expression.

    Frozen and JSON round-tripping like
    :class:`~spacr.qt.widgets.graph_spec.GraphSpec`, for the same reason — a
    derived column is part of what a saved analysis *is*, and a chart of
    ``ratio`` that cannot say what ``ratio`` was is not reproducible.

    :param name: the new column's name. A plain identifier, so it can be typed
        into another formula without backticks.
    :param expression: the source text.
    :param replace: allow overwriting an existing column of the same name. Off
        by default: silently replacing ``cell_area`` with something derived
        would make every earlier chart of that column unreproducible.
    :raises FormulaError: on an unusable name, or an expression that does not
        parse — at construction, so a bad formula never reaches a render.
    """

    name: str
    expression: str
    replace: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _valid_name(self.name))
        object.__setattr__(self, "expression", str(self.expression).strip())
        object.__setattr__(self, "replace", bool(self.replace))
        # Parse now, so an unparseable formula cannot be stored, serialised,
        # or reach a redraw.
        object.__setattr__(self, "_ast", parse(self.expression))

    @property
    def ast(self) -> Node:
        """The parsed expression.

        Held on an attribute that is not a dataclass *field*, so two formulas
        compare equal on their name and text — which is what a saved formula
        is — rather than on two structurally identical trees.
        """
        return self._ast          # set in __post_init__; not a field

    def inputs(self) -> Tuple[str, ...]:
        """The columns this formula reads."""
        return referenced_columns(self.ast)

    def uses_whole_table(self) -> bool:
        """Whether one object's value depends on the other objects.

        True for every :data:`TABLE_DEPENDENT_FUNCTIONS` call anywhere in the
        expression. Worth showing beside the formula, because such a column is
        not a property of the object: the same formula over two plates gives
        two different columns, and re-running it after a re-segmentation moves
        every value.
        """
        def walk(node: Node) -> bool:
            if isinstance(node, Call):
                if node.func in TABLE_DEPENDENT_FUNCTIONS:
                    return True
                return any(walk(a) for a in node.args)
            if isinstance(node, Unary):
                return walk(node.operand)
            if isinstance(node, Binary):
                return walk(node.left) or walk(node.right)
            return False
        return walk(self.ast)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "expression": self.expression,
                "replace": self.replace}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ColumnFormula":
        known = {k: v for k, v in dict(payload).items()
                 if k in {"name", "expression", "replace"}}
        return cls(**known)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ColumnFormula":
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        note = " (uses the whole table)" if self.uses_whole_table() else ""
        return f"{self.name} = {self.expression}{note}"


@dataclass(frozen=True)
class ColumnResult:
    """One computed column's values and what computing them cost.

    :param values: the column, aligned to the frame it was computed over.
    :param n_nonfinite: NaN and ±inf in the result. Reported rather than
        hidden — a ratio column that is a third infinities is a division by a
        zero the user did not know was there, and it looks identical to a good
        column on a chart that drops non-finite points.
    :param n_input_missing: rows where at least one input was already missing,
        so a NaN in the output can be attributed rather than guessed at.
    """

    formula: ColumnFormula
    values: np.ndarray
    n_rows: int
    n_nonfinite: int = 0
    n_input_missing: int = 0
    is_boolean: bool = False

    @property
    def notice(self) -> str:
        """One line for the panel: what came out, and what did not."""
        if self.is_boolean:
            true = int(np.asarray(self.values, dtype=bool).sum())
            return (f"{self.formula.name}: {true:,} of {self.n_rows:,} rows "
                    f"true")
        parts = [f"{self.formula.name}: {self.n_rows - self.n_nonfinite:,} of "
                 f"{self.n_rows:,} rows have a finite value"]
        if self.n_nonfinite:
            unattributed = self.n_nonfinite - self.n_input_missing
            if unattributed > 0:
                parts.append(
                    f"{unattributed:,} became NaN or infinite in the "
                    f"calculation (a division by zero, a log of a "
                    f"non-positive number)")
            if self.n_input_missing:
                parts.append(f"{self.n_input_missing:,} had a missing input")
        return " · ".join(parts)


def _apply_one(frame: pd.DataFrame, formula: ColumnFormula) -> ColumnResult:
    if formula.name in frame.columns and not formula.replace:
        raise FormulaError(
            f"this table already has a column called {formula.name!r}. Pick "
            f"another name, or tick 'replace' if you mean to shadow it")
    if formula.name in formula.inputs() and formula.name not in frame.columns:
        # `area = area * 2` with ``replace`` on is a rescale, and legitimate:
        # `compute` always starts from a fresh copy of the loaded table, so it
        # reads the measured column and is idempotent however often it is
        # re-applied. `x = x + 1` where no `x` exists is a genuine circle.
        raise FormulaError(
            f"{formula.name!r} refers to itself and there is no column called "
            f"that to read; a computed column cannot be defined in terms of "
            f"its own previous value")
    raw = evaluate(formula.ast, frame)
    values = _as_array(raw, len(frame))
    is_boolean = values.dtype == bool
    if is_boolean:
        return ColumnResult(formula=formula, values=values,
                            n_rows=len(frame), is_boolean=True)
    values = values.astype(float, copy=False)
    finite = np.isfinite(values)
    missing = np.zeros(len(frame), dtype=bool)
    for column in formula.inputs():
        missing |= ~np.isfinite(_numeric_column(frame, column))
    return ColumnResult(
        formula=formula, values=values, n_rows=len(frame),
        n_nonfinite=int((~finite).sum()),
        n_input_missing=int((~finite & missing).sum()))


@dataclass
class FormulaSet:
    """An ordered list of :class:`ColumnFormula`, and the frame they make.

    Ordered, not a set: each formula sees the columns the ones before it added,
    which is what makes ``density`` then ``log_density`` work. A formula that
    refers to one defined below it gets an error saying to move it up, rather
    than a column of NaN.

    Mutable, unlike the specs elsewhere in this package, because it is a *list
    the user edits* rather than a value that is diffed — but every entry in it
    is frozen, and :meth:`to_dict` round-trips.
    """

    formulas: List[ColumnFormula] = field(default_factory=list)

    def add(self, formula: ColumnFormula) -> "FormulaSet":
        """Append, replacing any formula of the same name."""
        self.formulas = [f for f in self.formulas
                         if f.name != formula.name] + [formula]
        return self

    def remove(self, name: str) -> "FormulaSet":
        self.formulas = [f for f in self.formulas if f.name != str(name)]
        return self

    def clear(self) -> "FormulaSet":
        self.formulas = []
        return self

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(f.name for f in self.formulas)

    @property
    def is_empty(self) -> bool:
        return not self.formulas

    def __len__(self) -> int:
        return len(self.formulas)

    def apply(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[ColumnResult]]:
        """``frame`` plus one column per formula, and a result per column.

        A **copy** — the loaded table is never mutated, so removing a formula
        removes its column rather than leaving it behind, and two screens
        sharing a frame do not grow each other's columns.
        """
        return compute(frame, self.formulas)

    def to_dict(self) -> Dict[str, Any]:
        return {"formulas": [f.to_dict() for f in self.formulas]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormulaSet":
        rows = dict(payload).get("formulas") or []
        return cls([ColumnFormula.from_dict(row) for row in rows])

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "FormulaSet":
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        if not self.formulas:
            return "no computed columns"
        return " · ".join(f.describe() for f in self.formulas)


def compute(frame: pd.DataFrame,
            formulas: Sequence[ColumnFormula]
            ) -> Tuple[pd.DataFrame, List[ColumnResult]]:
    """Add one column per formula to a copy of ``frame``.

    In list order, each formula seeing what the earlier ones added.

    :returns: ``(frame_with_columns, results)``.
    :raises FormulaError: naming the formula that failed. Nothing is added when
        one fails — a half-applied set would leave the user with some of the
        columns they asked for and no way to tell which.
    """
    working = frame.copy()
    results: List[ColumnResult] = []
    for formula in formulas:
        try:
            result = _apply_one(working, formula)
        except FormulaError as exc:
            raise FormulaError(f"{formula.name}: {exc}") from None
        working[formula.name] = result.values
        results.append(result)
    return working, results
