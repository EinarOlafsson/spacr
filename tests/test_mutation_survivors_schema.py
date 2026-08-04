"""Gaps in the schema suite found by AST mutation testing of ``spacr.schema``.

``spacr/schema.py`` came out of the mutation run in very good shape -- of 255
mutants on executed lines, only two survived, and one of those turns out to be
an equivalent mutant (see the module note below). This file closes the one
that is real.

Verified twice: red with the mutant loaded, green without it.

.. note::

   The other survivor is ``schema.py:733`` in :func:`is_row_column_pair`,
   ``if not row_text or not column_text: return False`` mutated to ``and``.
   It is EQUIVALENT, not a test gap: every input the ``or`` short-circuits is
   rejected again further down (an empty row fails ``row_index_from_letters``,
   an empty column fails both the ``c`` prefix test and ``str.isdigit``), so
   the guard is a redundant early exit and no test can tell the two versions
   apart. Recorded here so the next mutation run does not re-litigate it.
"""
from __future__ import annotations

from spacr import schema as S


def test_a_prefixed_pair_is_never_the_positional_passthrough():
    """``return row_text[:1].lower() != 'r' and column_text[:1].lower() != 'c'``
    (schema.py:710, :func:`is_positional_pair`).

    Mutant ``and -> or`` survived the whole schema suite. By the time this
    line runs the two texts are already known to be EQUAL, so ``and`` and
    ``or`` differ on exactly the equal pairs that carry a key prefix:
    ``('r1', 'r1')`` and ``('c1', 'c1')``. With ``or`` both are reported as
    ``parse_well``'s unrecognisable-well passthrough.

    That answer propagates: :func:`is_row_column_pair` returns ``True`` for a
    passthrough without further checks, and it is the guard that stops a
    right-to-left key parse (``parse_prcf``, ``ml._split_prc``) from absorbing
    a deeper key into an underscored plate id. So the mutation turns
    ``..._r1_r1`` into a well, which is the "one level too deep" misparse this
    predicate exists to prevent.

    The existing tests only ever feed it unprefixed equal pairs (``'12'``,
    ``'12'``) and unequal prefixed pairs, so both clauses are always on the
    same side and the operator never shows.
    """
    # Equal AND prefixed -> not a passthrough, whichever slot carries the prefix.
    assert S.is_positional_pair('r1', 'r1') is False
    assert S.is_positional_pair('c1', 'c1') is False
    assert S.is_positional_pair('R1', 'R1') is False
    assert S.is_positional_pair('C1', 'C1') is False

    # ...while the genuine passthrough is still recognised.
    assert S.is_positional_pair('12', '12') is True
    assert S.is_positional_pair('weird', 'weird') is True

    # ...and the consequence the predicate exists for: a prefixed equal pair
    # must not short-circuit is_row_column_pair into "yes, that's a well".
    assert S.is_row_column_pair('r1', 'r1') is False
    assert S.is_row_column_pair('c1', 'c1') is False


def test_row_zero_is_on_no_plate():
    """``... or r_index < 1 or c_index < 1: return None`` (schema.py:864,
    :func:`plate_format_for`).

    Mutant ``r_index < 0`` survived. Well coordinates in this module are
    1-based -- row A is 1, ``letters_from_row_index`` starts there, and
    :func:`is_within_plate_format` states the range as ``1 <= r_index <=
    n_rows``. With ``< 0`` the guard stops rejecting row 0, the loop's
    ``0 <= n_rows`` is trivially true, and ``plate_format_for(0, 1)`` reports
    the smallest standard plate for a position that is on no plate at all.

    The two guards are asymmetric under mutation -- the ``c_index`` half was
    killed, the ``r_index`` half was not -- which is exactly the kind of
    one-sided hole a boundary test closes: assert BOTH halves and both
    spellings (bare index and ``'r0'``/``'c0'`` token).
    """
    assert S.plate_format_for(0, 1) is None
    assert S.plate_format_for(1, 0) is None
    assert S.plate_format_for('r0', 'c1') is None
    assert S.plate_format_for('r1', 'c0') is None
    assert S.plate_format_for(-1, 1) is None

    # The first real well still resolves, so the guard has not been widened.
    assert S.plate_format_for(1, 1) is not None
    # The same boundary in is_within_plate_format. Both halves of ITS guard
    # were mutable too; they were caught only by the run hanging, which is a
    # kill but a slow and uninformative one.
    assert S.is_within_plate_format(0, 1, 96) is False
    assert S.is_within_plate_format(1, 0, 96) is False
    assert S.is_within_plate_format(1, 1, 96) is True
