"""Property-based tests for the plate / row / column / field / object keys.

Why this file exists
--------------------
Every key-parsing bug found in spaCR recently would have been caught by one
of the five properties below, and none of them was caught by an example.

* ``ml._split_prc`` silently accepted a four-token key: a wrong-arity key
  parsed "successfully" into the wrong fields, and every per-well count
  grouped on the result was a plausible wrong number.
* Eleven positional ``prc`` / ``prcf`` splits existed across the codebase,
  each with slightly different behaviour, in place of one canonical parser.
* SQLite compares identifiers case-insensitively, so a real column named
  ``rowID`` shadows the implicit ``rowid`` — which once made a ``DELETE``
  remove an entire table instead of two rows. A key-name case collision is a
  live hazard here, not a theoretical one.

An example test asks "does this key parse?". These ask the questions that
actually go wrong:

**round-trip**
    ``parse(compose(k)) == k`` for every key the writers can build. A parser
    that is not the inverse of the composer means the identity a row was
    written with is not the identity it is read back with, and the join
    between two tables silently returns nothing.

**arity**
    A key with the wrong number of components is *rejected*, never silently
    parsed into the wrong fields. This is the ``_split_prc`` bug, stated as a
    law.

**injectivity**
    Two distinct identities never compose to the same key. This is what
    catches separator-in-component bugs — the class where three fields go in
    and one comes out, which is the failure mode :mod:`spacr.schema` was
    written to end.

**idempotence**
    ``canonicalise(canonicalise(x)) == canonicalise(x)``. A canonicaliser
    that moves on every application means a key changes every time it passes
    through a reader, so no two readers agree on it.

**agreement**
    Where more than one parser still exists — ``schema.parse_prcf`` and
    ``ml._split_prc``, ``schema.canonicalise_columns`` and
    ``utils.canonicalize_measurement_columns`` — they agree on every
    generated input. A differential property test is the cheapest proof that
    a consolidation was safe, and the cheapest alarm for the copies drifting
    apart again.

What the properties found
-------------------------
Three bugs, fixed in :mod:`spacr.schema` alongside this file:

* **round-trip / idempotence.** :func:`spacr.schema.parse_prcfo`
  double-prefixed a non-numeric object label — ``'p_r1_c1_f1_oxy'`` parsed to
  ``'p_r1_c1_f1_ooxy'``, and *that* parsed to ``'ooooxy'``. A key that grows
  every time it is read joins to nothing, and a table read twice keyed its
  rows two ways.
* **arity.** :func:`spacr.schema.parse_prcf` absorbed any surplus component
  into the plate without checking what was left. ``'plate1_r1_c1_f1_f2'``
  parsed to ``plateID='plate1_r1'``, ``rowID='c1'``, ``columnID='f1'`` —
  half the well inside the plate and a field id in the column slot. That is
  exactly the ``_split_prc`` bug, in the module ``_split_prc`` was
  consolidated onto; it now applies the same
  :func:`spacr.schema.is_row_column_pair` guard, which moved here so there is
  one definition of it.
* **arity.** :func:`spacr.schema.parse_prcf` accepted an empty component:
  ``'plate1__c1_f1'`` gave ``rowID=''`` and ``'_r1_c1_f1'`` gave
  ``plateID=''``. An empty component is not a missing token, it is a token
  every field of the plate shares. ``_split_prc`` refused both, with that
  reason in its message; the parser it was consolidated onto did not.

A fourth, found by the injectivity property and fixed in
:mod:`spacr.selection`: ``object_keys`` applied no guard to its components,
so ``("p_x", "r1", "c1", "f1", 1)`` and ``("p", "x_r1", "c1", "f1", 1)`` both
composed to ``"p_x_r1_c1_f1_1"``. It now percent-escapes the separator, which
is reversible where refusing would be unusable — the key is built from data
already on disk and the user cannot go back and rename the plate.

Four more are confirmed, reproduced by an ``xfail(strict=True)`` test each,
and left for their owning module (see each test's docstring). They are
**not** weakened to pass: a property relaxed until it is green is a property
that will not catch the bug it was written for.

The object type
---------------
:data:`spacr.selection.OBJECT_KEY_COLUMNS` was the field plus the object
label, with no table in it — so a nucleus labelled 1 and a pathogen labelled 1
in one field were the same key, and a cell's own children are exactly the
objects most likely to collide. The properties here now pin both halves of the
repair: two types with one label are two keys, and dropping the type gives
back byte for byte the key the previous release wrote, so nothing already
stored has to move.

Determinism
-----------
``derandomize=True`` — hypothesis draws from a fixed seed, so a CI failure
here is reproducible from the printed counterexample and a green run does
not go red on someone else's commit.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# `spacr.ml` and `spacr.utils` hold the parsers this file differentially
# tests against `spacr.schema`. They are imported here rather than inside the
# property bodies because an import that first runs inside a
# hypothesis-controlled frame trips its recursion-limit warning on every
# example.
from spacr import schema as S
from spacr import selection as SEL
from spacr.ml import _is_row_column_pair, _split_prc
from spacr.utils import canonicalize_measurement_columns

# ---------------------------------------------------------------------------
# Deterministic profile
# ---------------------------------------------------------------------------
# Registered and loaded at import. tests/conftest.py is deliberately not
# touched: the profile belongs with the properties it governs, and a project
# that later grows a shared profile can move it without hunting for callers.
settings.register_profile(
    'spacr_keys',
    derandomize=True,
    deadline=None,
    max_examples=300,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile('spacr_keys')

pytestmark = pytest.mark.filterwarnings('ignore::DeprecationWarning')


# ---------------------------------------------------------------------------
# Strategies: realistic keys, and the adversarial ones that break parsers
# ---------------------------------------------------------------------------

SEP = S.KEY_SEPARATOR

#: Plate names spaCR actually sees: a folder name. Unicode is allowed (the
#: plate id is free-form text), the separator is not (``_check_plate``
#: refuses it, and this file proves why).
plate_names = st.one_of(
    st.sampled_from([
        'plate1', 'plate_1'.replace('_', '-'), 'PLATE-4', '20240101',
        'exp1', 'pläte', 'Plate', 'plate', 'p' * 200,
    ]),
    st.text(
        alphabet=st.characters(
            blacklist_characters=SEP,
            blacklist_categories=('Cs', 'Cc', 'Zs', 'Zl', 'Zp'),
        ),
        min_size=1,
        max_size=12,
    ).filter(lambda t: t.strip() and SEP not in t.strip()),
)

#: 1-based indices, covering past row Z (a 1536 plate has 32 rows) and past
#: column 24 (it has 48 of them).
row_indices = st.integers(min_value=1, max_value=64)
column_indices = st.integers(min_value=1, max_value=64)
field_indices = st.integers(min_value=1, max_value=9999)
time_indices = st.integers(min_value=1, max_value=9999)
object_labels = st.integers(min_value=1, max_value=100000)

#: How a number is *spelled* in a file name. Leading zeros, vendor prefixes
#: and surrounding whitespace are spellings of the same number, and a parser
#: that gave them different ids would be as wrong as one that gave three
#: different sites the same id.
_SPELLINGS = ('{n}', '{n:03d}', 's{n}', 'F{n:03d}', 'T{n:04d}', ' {n} ')


@st.composite
def numeric_tokens(draw, indices=field_indices):
    """A parseable integer token, in one of its real-world spellings."""
    value = draw(indices)
    return draw(st.sampled_from(_SPELLINGS)).format(n=value), value


#: Tokens that hold no integer. These take the "preserved token" path — the
#: one that must never invent an ``f0`` and must never merge two of them.
unparseable_tokens = st.one_of(
    st.sampled_from(['xy', 'unknown', 'NA', 'σ', 'a b', 'x' * 100]),
    st.text(
        alphabet=st.characters(
            blacklist_categories=('Cs', 'Cc', 'Nd', 'Zs', 'Zl', 'Zp'),
        ),
        min_size=1,
        max_size=8,
    ).filter(lambda t: t.strip() and S.parse_int_token(t) is None),
)

#: Well names, including the multi-letter rows a 1536 plate really has and
#: the separators plate readers really emit.
well_names = st.one_of(
    st.sampled_from(['A01', 'a1', 'A-01', ' A01 ', 'AA01', 'af48', 'P24',
                     'B 7', 'AF01']),
    st.tuples(row_indices, column_indices).map(
        lambda rc: f'{S.letters_from_row_index(rc[0])}{rc[1]:02d}'),
)

#: Metadata column names, in every case variant spaCR has written. These are
#: a closed vocabulary and their lookup is case-insensitive, because SQLite
#: folds identifier case whether or not pandas does.
metadata_aliases = st.sampled_from([
    'rowID', 'rowid', 'RowId', 'ROWID', 'row', 'Row', 'ROW', 'row_name',
    'row_id', 'columnID', 'column', 'Col', 'col_name', 'plateID', 'plate',
    'Plate_Name', 'fieldID', 'field', 'timeID', 'time_id', 'TimePoint',
])

#: Feature columns. Their rewrites are deliberately case-SENSITIVE: they are
#: generated per object / channel / percentile rather than enumerated, so
#: folding case would let a user column named ``Outside_5_Percentile`` be
#: rewritten out from under them. The properties below must respect that
#: difference rather than assert one rule over both families.
feature_names = st.sampled_from([
    'cell_area', 'cell_periphery_25_percentile', 'Outside_5_Percentile',
    'organelle_summary_organelle_ch0_mean_intensity', 'nucleus_channel_1_std',
])

#: Both families, for the properties that hold over any column name.
case_variants = st.one_of(metadata_aliases, feature_names)


@st.composite
def field_ids(draw, timelapse=None):
    """A :class:`~spacr.schema.FieldID` as the writers build one."""
    if timelapse is None:
        timelapse = draw(st.booleans())
    return S.FieldID(
        plateID=draw(plate_names).strip(),
        rowID=S.row_id(draw(row_indices)),
        columnID=S.column_id(draw(column_indices)),
        fieldID=S.field_id(draw(field_indices)),
        timeID=S.time_id(draw(time_indices)) if timelapse else None,
    )


@st.composite
def object_ids(draw, timelapse=None):
    """An :class:`~spacr.schema.ObjectID` as the writers build one."""
    field = draw(field_ids(timelapse=timelapse))
    return field.with_object(draw(object_labels))


# ===========================================================================
# round-trip
# ===========================================================================

@given(field=field_ids())
def test_prcf_round_trips(field):
    """``parse_prcf(f.prcf) == f`` for every field key a writer can build.

    Not "it parses" — it parses back to *the same identity*. The measured
    failure this stands against: ``png_list`` and ``cell`` describing the
    same field with two different identities, and the join between them
    returning 0 rows out of 2 x 2.
    """
    assert S.parse_prcf(field.prcf) == field


@given(obj=object_ids())
def test_prcfo_round_trips(obj):
    """``parse_prcfo(o.prcfo) == o``, and therefore ``.prcfo`` is stable.

    The counterexample that broke this before the fix::

        compose_prcfo('p', 1, 1, 1, 'xy')  -> 'p_r1_c1_f1_oxy'
        parse_prcfo(...).prcfo             -> 'p_r1_c1_f1_ooxy'

    ``object_id`` reads an already-prefixed *numeric* id back out of its
    prefix, but a preserved non-numeric token has no number to read, so the
    prefix was applied a second time — and a third on the next parse.
    """
    assert S.parse_prcfo(obj.prcfo) == obj


@given(obj=object_ids())
def test_prcfo_parsing_is_idempotent(obj):
    """Reading a key and writing it back is a fixed point, not a ratchet.

    This is the property the ``'ooxy'`` bug violated most destructively: the
    key was not merely wrong, it was *different every time*, so a table read
    twice keyed its rows two ways.
    """
    once = S.parse_prcfo(obj.prcfo).prcfo
    twice = S.parse_prcfo(once).prcfo
    assert once == twice == obj.prcfo


@given(field=field_ids())
def test_compose_agrees_with_the_dataclass(field):
    """``compose_prcf`` and ``FieldID.prcf`` are two spellings of one key."""
    composed = S.compose_prcf(field.plateID, field.rowID, field.columnID,
                              field.fieldID, field.timeID)
    assert composed == field.prcf


@given(obj=object_ids())
def test_prcf_of_a_prcfo_is_the_prcfo_minus_the_object(obj):
    """An object key is its field key plus one component, exactly."""
    assert obj.prcfo == obj.prcf + SEP + obj.objectID
    assert S.parse_prcf(obj.prcf) == obj.field


@given(data=numeric_tokens())
def test_a_number_is_the_same_field_however_it_is_spelled(data):
    """``'3'``, ``'003'``, ``'s3'``, ``'F003'``, ``'T0003'`` are one field.

    The ``f0`` bug in reverse: ``_safe_int_convert`` collapsed every
    unparseable spelling onto one id. Collapsing every *parseable* spelling
    onto one id is the behaviour that is actually wanted, and it has to be
    asserted or the repair could overshoot into three ids for one site.
    """
    token, value = data
    assert S.field_id(token) == f'f{value}'
    assert S.field_index(S.field_id(token)) == value


@given(index=row_indices)
def test_row_letters_round_trip_past_z(index):
    """``letters_from_row_index`` and ``row_index_from_letters`` are inverse.

    Row 27 is ``AA`` on a real 1536 plate. ``utils._map_wells`` raised on it
    and ``_map_wells_png`` returned ``('r1', 'c0')``.
    """
    letters = S.letters_from_row_index(index)
    assert S.row_index_from_letters(letters) == index
    assert S.row_id(letters) == f'r{index}'


@given(row=row_indices, column=column_indices)
def test_well_id_is_the_inverse_of_parse_well(row, column):
    """``parse_well(well_id(r, c)) == (r, c)`` over the whole plate."""
    well = S.well_id(row, column)
    assert S.parse_well(well) == (f'r{row}', f'c{column}')


@given(well=well_names)
def test_parse_well_is_idempotent_through_well_id(well):
    """Renaming a well does not move it."""
    row, column = S.parse_well(well)
    once = S.well_id(row, column)
    assert S.parse_well(once) == (row, column)
    assert S.well_id(*S.parse_well(once)) == once


# ===========================================================================
# arity -- the wrong number of components is REJECTED
# ===========================================================================

@given(field=field_ids())
def test_a_prcf_with_a_surplus_component_is_refused(field):
    """The ``_split_prc`` bug, as a law, for ``parse_prcf``.

    Before the fix::

        parse_prcf('plate1_r1_c1_f1_f2')
        -> plateID='plate1_r1' rowID='c1' columnID='f1' fieldID='f2'

    Half the well inside the plate, a field id in the column slot, no error.
    A surplus component is only ever legitimate when the plate id itself
    contains the separator, and that is recognisable: what is left over must
    still be a row and a column.
    """
    with pytest.raises(S.KeyParseError):
        S.parse_prcf(field.prcf + SEP + 'f2')


@given(obj=object_ids(timelapse=False))
def test_a_prcfo_handed_to_the_prcf_parser_is_refused(obj):
    """One level too deep is a caller bug and must be loud."""
    with pytest.raises(S.KeyParseError):
        S.parse_prcf(obj.prcfo)


@given(field=field_ids())
def test_a_prcf_handed_to_the_prcfo_parser_is_refused(field):
    """One level too shallow: there is no object, so there is no answer."""
    with pytest.raises(S.KeyParseError):
        S.parse_prcfo(field.prcf)


@given(field=field_ids(timelapse=False), n=st.integers(min_value=0,
                                                       max_value=3))
def test_a_truncated_prcf_is_refused(field, n):
    """Dropping components off the right never yields a shallower key.

    ``'plate1_r1_c1'`` is a ``prc``, not a ``prcf`` with a missing field, and
    a parser that returned a ``FieldID`` for it would put ``'c1'`` in the
    field slot for every well of the plate.
    """
    parts = field.prcf.split(SEP)
    truncated = SEP.join(parts[:n])
    with pytest.raises(S.KeyParseError):
        S.parse_prcf(truncated)


@given(field=field_ids(timelapse=False),
       position=st.integers(min_value=0, max_value=3))
def test_an_empty_component_is_refused(field, position):
    """An empty component is not a missing token, it is a shared one.

    Every row keyed on it merges with every other row that also failed to
    parse — the ``f0`` disaster with a different letter.

    This property found a bug. ``parse_prcf('plate1__c1_f1')`` returned
    ``FieldID(plateID='plate1', rowID='', columnID='c1', fieldID='f1')``, and
    ``'_r1_c1_f1'`` returned one with ``plateID=''``. ``ml._split_prc``
    refused both, explicitly and with that reason in its message; the parser
    it was consolidated onto did not. Fixed in :mod:`spacr.schema`.
    """
    parts = field.prcf.split(SEP)
    parts[position] = ''
    with pytest.raises(S.KeyParseError):
        S.parse_prcf(SEP.join(parts))


@given(plate=plate_names, row=row_indices, column=column_indices)
def test_a_plate_containing_the_separator_is_refused(plate, row, column):
    """The key is separator-joined, so the plate may not contain one.

    Refusing at *compose* time is what makes the round-trip and injectivity
    properties above true at all: a plate called ``'a_b'`` and a plate called
    ``'a'`` with row ``'b'`` would otherwise write the same key.
    """
    with pytest.raises(S.KeyParseError):
        S.compose_prc(plate.strip() + SEP + 'x', row, column)


@given(field=field_ids())
def test_an_empty_object_label_is_refused(field):
    """``'o'`` alone names no object.

    Before the ``parse_prcfo`` fix this became ``'oo'`` — an identity built
    out of nothing, and shared by every row whose label was missing.
    """
    with pytest.raises(S.KeyParseError):
        S.parse_prcfo(field.prcf + SEP + S.OBJECT_PREFIX)


# ===========================================================================
# injectivity -- two distinct keys never parse to the same tuple
# ===========================================================================

@given(a=field_ids(), b=field_ids())
def test_distinct_fields_have_distinct_keys(a, b):
    """No two field identities compose to one ``prcf``.

    The separator-in-component bug, stated positively. A collision here is
    two fields becoming one row set with nothing anywhere saying so.
    """
    assert (a == b) == (a.prcf == b.prcf)


@given(a=object_ids(), b=object_ids())
def test_distinct_objects_have_distinct_keys(a, b):
    """No two object identities compose to one ``prcfo``."""
    assert (a == b) == (a.prcfo == b.prcfo)


@given(a=field_ids(), b=field_ids())
def test_the_parser_is_injective_on_the_keys_the_writers_build(a, b):
    """Distinct keys parse to distinct identities, and vice versa.

    Injectivity in the parse direction is the half that catches a parser
    quietly folding two keys together — a ``prcf`` and a ``prcf`` with a
    surplus component both landing on one ``FieldID``.
    """
    parsed_a, parsed_b = S.parse_prcf(a.prcf), S.parse_prcf(b.prcf)
    assert (a.prcf == b.prcf) == (parsed_a == parsed_b)


@given(field=field_ids(timelapse=True))
def test_a_timepoint_is_not_absorbed_into_the_field(field):
    """A timelapse key keeps its five components apart.

    ``ml.py`` split ``prcfo`` left to right into a fixed five columns, so a
    six-part timelapse key put ``'t3'`` in the object slot and misaligned
    every column after it. Right-to-left parsing is what makes the optional
    middle component unambiguous.
    """
    parsed = S.parse_prcf(field.prcf)
    assert parsed.timeID == field.timeID
    assert parsed.fieldID == field.fieldID
    without_time = S.FieldID(field.plateID, field.rowID, field.columnID,
                             field.fieldID, None)
    assert without_time.prcf != field.prcf


@given(a=field_ids(timelapse=False), label_a=object_labels,
       label_b=object_labels)
def test_two_objects_in_one_field_stay_two_objects(a, label_a, label_b):
    """The ``f0`` collision at the object level."""
    key_a, key_b = a.with_object(label_a).prcfo, a.with_object(label_b).prcfo
    assert (label_a == label_b) == (key_a == key_b)


# ===========================================================================
# canonicalisation is idempotent
# ===========================================================================

@given(name=case_variants)
def test_canonical_column_name_is_idempotent(name):
    """``canonical_column_name`` reaches a fixed point in one application.

    A rename map that keeps moving means the column a frame has depends on
    how many readers it has been through.
    """
    once = S.canonical_column_name(name)
    assert S.canonical_column_name(once) == once


@given(name=metadata_aliases)
def test_metadata_column_names_fold_case(name):
    """``'RowID'``, ``'rowid'`` and ``'ROWID'`` are one column.

    SQLite says so whether or not pandas does, and the DELETE that removed a
    whole table instead of two rows is what disagreeing about it costs. A
    database column spelled ``Row`` must therefore be repaired, not left for
    a reader to trip over — pandas reports whatever spelling is stored, which
    is how a frame ends up with no ``rowID`` column on a database that has
    one.
    """
    assert S.canonical_column_name(name) == S.canonical_column_name(name.upper())
    assert S.canonical_column_name(name) == S.canonical_column_name(name.lower())


@given(name=feature_names)
def test_feature_column_names_do_not_fold_case(name):
    """Feature rewrites stay case-sensitive, on purpose.

    They are generated per object type, per channel and per percentile — a
    four-channel run has several hundred — so they are matched by shape
    rather than enumerated. Folding case there would rewrite a user column
    named ``Outside_5_Percentile`` out from under them, which is why this is
    asserted rather than left to be "fixed" for symmetry with the metadata
    vocabulary above.
    """
    assert S.canonical_column_name(name.upper()) == name.upper()


@given(names=st.lists(case_variants, min_size=1, max_size=6, unique=True))
def test_canonicalise_columns_is_idempotent(names):
    """Canonicalising a frame twice is canonicalising it once."""
    frame = pd.DataFrame({name: [1] for name in names})
    once = S.canonicalise_columns(frame)
    twice = S.canonicalise_columns(once)
    assert list(twice.columns) == list(once.columns)


@given(names=st.lists(case_variants, min_size=1, max_size=6, unique=True))
def test_canonicalise_columns_never_loses_a_column(names):
    """Renaming is never allowed to drop data to tidy a name up.

    A frame carrying both spellings keeps both; a human decides which is
    authoritative, and until then both stay reachable.
    """
    frame = pd.DataFrame({name: [i] for i, name in enumerate(names)})
    out = S.canonicalise_columns(frame)
    assert len(out.columns) == len(names)
    assert sorted(out.iloc[0].tolist()) == sorted(range(len(names)))


@given(field=field_ids())
def test_composing_an_already_composed_id_is_a_no_op(field):
    """``row_id('r1') == 'r1'``, not ``'rr1'``, for every id shape."""
    assert S.row_id(field.rowID) == field.rowID
    assert S.column_id(field.columnID) == field.columnID
    assert S.field_id(field.fieldID) == field.fieldID
    if field.timeID is not None:
        assert S.time_id(field.timeID) == field.timeID


@given(label=object_labels)
def test_object_id_does_not_double_prefix(label):
    """``object_id('o7') == 'o7'``, and stays there."""
    once = S.object_id(label)
    assert S.object_id(once) == once
    assert S.object_index(once) == label


# ===========================================================================
# agreement -- every parser that still exists gives the same answer
# ===========================================================================

@given(plate=plate_names, row=row_indices, column=column_indices)
def test_split_prc_agrees_with_the_schema_parser(plate, row, column):
    """``ml._split_prc`` and ``schema`` read one ``prc`` the same way.

    Eleven positional splits were consolidated onto these two; a differential
    test is the cheapest proof the consolidation was safe, and the cheapest
    alarm if the copies drift apart again.
    """
    key = S.compose_prc(plate, row, column)
    assert _split_prc(key) == (plate.strip(), f'r{row}', f'c{column}')


@given(field=field_ids(timelapse=False))
def test_split_prc_and_parse_prcf_reject_the_same_deep_keys(field):
    """Both refuse a key one level too deep, and for the same reason.

    They are the same right-to-left parse at two depths. If one absorbed a
    surplus component that the other refused, a ``prcf`` written by one
    module would be a well key to the other.
    """
    with pytest.raises(S.KeyParseError):
        _split_prc(field.prcf)
    with pytest.raises(S.KeyParseError):
        S.parse_prcf(field.prcf + SEP + field.fieldID)


@given(row=st.one_of(st.sampled_from(['r1', 'r27', 'A', 'AA', 'c1', 'f1',
                                      '12', 'xy', '']),
                     st.text(max_size=5)),
       column=st.one_of(st.sampled_from(['c1', 'c48', '1', '01', 'f1', 'o1',
                                         '12', 'xy', '']),
                        st.text(max_size=5)))
def test_the_row_column_predicate_has_one_definition(row, column):
    """``schema.is_row_column_pair`` and ``ml._is_row_column_pair`` agree.

    ``ml`` keeps a private copy for now. This pins them together so the
    collapse of one onto the other stays a pure move.
    """
    assert S.is_row_column_pair(row, column) == _is_row_column_pair(row, column)


@given(names=st.lists(case_variants, min_size=1, max_size=6, unique=True))
def test_the_two_frame_canonicalisers_agree_on_every_frame(names):
    """``schema.canonicalise_columns`` vs ``utils.canonicalize_measurement_columns``.

    Two functions with the same job, reached from different modules. They
    used to give different answers on the same frame — the schema one knew
    the metadata aliases and not the feature spellings, the utils one the
    reverse — so whichever a caller imported decided what its columns were
    called. ``tests/test_schema.py`` pins one frame; this pins all of them.
    """
    frame = pd.DataFrame({name: [1] for name in names})
    schema_out = S.canonicalise_columns(frame.copy())
    utils_out = canonicalize_measurement_columns(frame.copy())
    assert list(schema_out.columns) == list(utils_out.columns)


@given(field=field_ids(timelapse=False), label=object_labels)
def test_selection_keys_agree_with_the_schema_row_key(field, label):
    """``selection.object_keys`` builds the row key ``schema`` declares.

    Four views over one measurement table have to mean the same thing by "the
    object I pointed at", so the key the UMAP publishes has to be the key the
    plate view resolves.
    """
    assume(all('%' not in value for value in (
        field.plateID, field.rowID, field.columnID, field.fieldID)))
    frame = pd.DataFrame([{
        S.PLATE_KEY: field.plateID,
        S.ROW_KEY: field.rowID,
        S.COLUMN_KEY: field.columnID,
        S.FIELD_KEY: field.fieldID,
        S.OBJECT_LABEL_KEY: label,
    }])
    expected = SEP.join([field.plateID, field.rowID, field.columnID,
                         field.fieldID, str(label)])
    assert list(SEL.object_keys(frame)) == [expected]
    assert SEL.OBJECT_KEY_COLUMNS == \
        S.FIELD_KEY_COLUMNS + (S.OBJECT_LABEL_KEY,)


# ===========================================================================
# CONFIRMED BUGS, reproduced -- not weakened to pass
# ===========================================================================
#
# Each of these is a property that ought to hold, a minimal counterexample
# hypothesis found, and the module that owns the repair. They are
# xfail(strict=True) so that fixing one turns this file red until the xfail
# is removed -- a bug that quietly stops reproducing is a bug nobody notices
# was fixed, and an xfail nobody removes is a property nobody is testing.


@given(token=unparseable_tokens)
def test_a_preserved_timepoint_token_reads_back(token):
    """A preserved time token is still a valid, round-trippable join key."""
    key = S.compose_prcf('p', 1, 1, 1, time=token)
    assert S.parse_prcf(key).prcf == key


@pytest.mark.xfail(strict=True, reason=(
    'BUG (spacr.schema, unfixed): parse_field_stem / parse_object_stem read '
    'a file name LEFT to right at fixed positions, while parse_prcf reads '
    'right to left "which is what makes it correct". Two consequences, both '
    'silent: (1) a plate id containing the separator is mis-slotted -- '
    'parse_field_stem("my_plate_A01_3") -> plateID="my", rowID="plate", '
    'columnID="plate", fieldID="f1", because well=parts[1]="plate" falls '
    'into the positional-well passthrough and field=parts[2]="A01" parses as '
    'the prefixed integer 1; (2) surplus components are dropped, so '
    'parse_field_stem("plate1_A01_3_junk") == parse_field_stem('
    '"plate1_A01_3") -- an injectivity failure, and on a timelapse name read '
    'with timelapse=False it is the "three timepoints go in, one comes out" '
    'bug this module was written to end. The repair is to parse the stem '
    'right to left like every other key parser here, or to refuse a stem '
    'with more components than the requested shape.'))
@given(plate_left=st.sampled_from(['my', 'exp1']),
       plate_right=st.sampled_from(['plate', 'plate1']),
       well=st.sampled_from(['A01', 'AA01']),
       field=field_indices)
def test_a_file_name_is_parsed_right_to_left_like_every_other_key(
        plate_left, plate_right, well, field):
    stem = SEP.join([plate_left, plate_right, well, str(field)])
    parsed = S.parse_field_stem(stem)
    assert parsed.plateID == plate_left + SEP + plate_right
    assert parsed.fieldID == f'f{field}'


@given(names=st.lists(case_variants, min_size=2, max_size=4, unique=True))
def test_canonicalise_columns_never_creates_a_case_collision(names):
    frame = pd.DataFrame({name: [1] for name in names})
    before = [str(c).casefold() for c in frame.columns]
    out = S.canonicalise_columns(frame)
    after = [str(c).casefold() for c in out.columns]
    collisions_before = len(before) - len(set(before))
    collisions_after = len(after) - len(set(after))
    assert collisions_after <= collisions_before, (
        f'{list(frame.columns)} -> {list(out.columns)}')
    # And the frame SQLite gets is one SQLite will take -- PROVIDED the caller
    # did not hand over a frame that was already unwritable. A frame built as
    # ["rowID", "rowid"] collides before canonicalisation touches it, and no
    # non-destructive canonicaliser can repair that: the only ways out are
    # dropping one column or inventing a name for it, and "keep both, let a
    # human decide which is authoritative" is the rule this function is built
    # on. Asserting to_sql unconditionally would be asserting that
    # canonicalise_columns fixes its caller's data, which it does not claim to
    # do -- it claims not to CREATE a collision, which is what is checked above
    # and on every input below.
    if collisions_before == 0:
        conn = sqlite3.connect(':memory:')
        try:
            out.to_sql('cell', conn, index=False)
        finally:
            conn.close()


@given(label=object_labels)
def test_object_keys_are_injective(label):
    """FIXED. ``object_keys`` composed two distinct objects onto one key.

    It joined plateID/rowID/columnID/fieldID/object_label with the schema
    separator and applied no guard to the components, so ``("p_x", "r1",
    "c1", "f1", 1)`` and ``("p", "x_r1", "c1", "f1", 1)`` both produced
    ``"p_x_r1_c1_f1_1"`` — and every view that resolves a key back to a row
    showed or annotated the wrong object. ``schema._check_plate`` refuses a
    separator in a plate id for exactly this reason; ``object_keys`` now
    percent-escapes one instead, because a key is built from data that is
    already on disk and refusing would only turn a wrong answer into a crash
    the user cannot clear.
    """
    frame = pd.DataFrame([
        {S.PLATE_KEY: 'p' + SEP + 'x', S.ROW_KEY: 'r1',
         S.COLUMN_KEY: 'c1', S.FIELD_KEY: 'f1', S.OBJECT_LABEL_KEY: label},
        {S.PLATE_KEY: 'p', S.ROW_KEY: 'x' + SEP + 'r1',
         S.COLUMN_KEY: 'c1', S.FIELD_KEY: 'f1', S.OBJECT_LABEL_KEY: label},
    ])
    keys = list(SEL.object_keys(frame))
    assert len(set(keys)) == 2, keys


@given(a=st.text(alphabet='ab_%', min_size=1, max_size=4),
       b=st.text(alphabet='ab_%', min_size=1, max_size=4),
       label=object_labels)
def test_the_key_escape_is_reversible_so_it_cannot_merge_two_plates(a, b,
                                                                    label):
    """Escaping ``_`` is not enough on its own; ``%`` has to go too.

    An escape that maps ``'_'`` to ``'%5F'`` and leaves ``'%'`` alone merges
    a plate literally named ``'p%5Fx'`` with a plate named ``'p_x'`` — the
    same two-into-one failure one level down, which is the shape of the
    ``_sanitise_token`` bug still pinned below.
    """
    frame = pd.DataFrame([
        {S.PLATE_KEY: a, S.ROW_KEY: 'r1', S.COLUMN_KEY: 'c1',
         S.FIELD_KEY: 'f1', S.OBJECT_LABEL_KEY: label},
        {S.PLATE_KEY: b, S.ROW_KEY: 'r1', S.COLUMN_KEY: 'c1',
         S.FIELD_KEY: 'f1', S.OBJECT_LABEL_KEY: label},
    ])
    keys = list(SEL.object_keys(frame))
    assert (a == b) == (keys[0] == keys[1]), (a, b, keys)


@given(field=field_ids(timelapse=False), label=object_labels,
       left=st.sampled_from(S.OBJECT_TYPES),
       right=st.sampled_from(S.OBJECT_TYPES))
def test_two_object_types_with_one_label_are_two_keys(field, label, left,
                                                      right):
    """The defect the object type was put into the key to end.

    Object tables are one type per table, so a nucleus 1 and a pathogen 1 in
    the same field composed to the identical key. A cell's own children are
    exactly the objects most likely to collide.
    """
    row = {S.PLATE_KEY: field.plateID, S.ROW_KEY: field.rowID,
           S.COLUMN_KEY: field.columnID, S.FIELD_KEY: field.fieldID,
           S.OBJECT_LABEL_KEY: label}
    frame = pd.DataFrame([{**row, S.OBJECT_TYPE_KEY: left},
                          {**row, S.OBJECT_TYPE_KEY: right}])
    keys = list(SEL.object_keys(frame))
    assert (left == right) == (keys[0] == keys[1]), (left, right, keys)


@given(field=field_ids(timelapse=False), label=object_labels,
       object_type=st.sampled_from(S.OBJECT_TYPES))
def test_a_typed_object_key_is_the_prcfo_of_the_same_object(field, label,
                                                            object_type):
    """Two identities converge as soon as the type is known.

    ``selection.object_keys`` joins the label bare and ``compose_prcfo``
    writes it behind ``'o'``, so the two disagreed by one character for every
    object in the database. With a type stated they are the same string, and
    a key copied out of a crop table names the same object as one built from
    a measurement table.
    """
    # selection's documented percent-escape exception has its own injectivity
    # property below; schema's legacy composer deliberately preserves it.
    assume(all('%' not in value for value in (
        field.plateID, field.rowID, field.columnID, field.fieldID)))
    frame = pd.DataFrame([{
        S.PLATE_KEY: field.plateID, S.ROW_KEY: field.rowID,
        S.COLUMN_KEY: field.columnID, S.FIELD_KEY: field.fieldID,
        S.OBJECT_LABEL_KEY: label,
    }])
    key = str(SEL.object_keys(frame, object_type=object_type)[0])
    assert key == S.compose_prcfo(field.plateID, field.rowID, field.columnID,
                                  field.fieldID, label,
                                  object_type=object_type)
    assert SEL.key_object_type(key) == object_type
    assert S.parse_prcfo(key).objectLabel == str(label)


@given(field=field_ids(timelapse=False), label=object_labels,
       object_type=st.sampled_from(S.OBJECT_TYPES))
def test_dropping_the_type_gives_back_exactly_the_key_spacr_used_to_write(
        field, label, object_type):
    """The migration, as a property: no key that exists today moves.

    ``untyped_object_key`` is the inverse of adding a type, and what it
    returns has to be byte for byte the key the previous release composed —
    otherwise every stored selection, exported ``.h5ad`` index and prediction
    bundle would need rewriting, and the ones that were missed would silently
    match nothing.

    ``%`` is excluded because it is the escape character: see
    :func:`test_a_percent_in_an_identifier_is_the_one_key_that_is_respelled`,
    which pins that exception rather than hiding it behind this filter.
    """
    assume('%' not in field.plateID)
    frame = pd.DataFrame([{
        S.PLATE_KEY: field.plateID, S.ROW_KEY: field.rowID,
        S.COLUMN_KEY: field.columnID, S.FIELD_KEY: field.fieldID,
        S.OBJECT_LABEL_KEY: label,
    }])
    legacy = SEP.join([field.plateID, field.rowID, field.columnID,
                       field.fieldID, str(label)])
    assert str(SEL.object_keys(frame)[0]) == legacy
    typed = str(SEL.object_keys(frame, object_type=object_type)[0])
    assert SEL.untyped_object_key(typed) == legacy
    # And so does the prcfo of the same object, which is how a key copied
    # from a crop table meets one built from a measurement table.
    assert SEL.untyped_object_key(
        S.compose_prcfo(field.plateID, field.rowID, field.columnID,
                        field.fieldID, label)) == legacy


def test_a_percent_in_an_identifier_is_the_one_key_that_is_respelled():
    """The cost of a reversible escape, stated rather than hidden.

    ``%`` is the escape character, so an identifier that already contains one
    is written ``%25``. That is the *only* change to any key spaCR has
    written, it is required for the escape to be reversible at all, and the
    alternative — leaving ``%`` alone — merges a plate named ``'p%5Fx'`` with
    a plate named ``'p_x'``, which is the two-into-one failure the escape
    exists to prevent.
    """
    frame = pd.DataFrame([
        {S.PLATE_KEY: 'p%5Fx', S.ROW_KEY: 'r1', S.COLUMN_KEY: 'c1',
         S.FIELD_KEY: 'f1', S.OBJECT_LABEL_KEY: 1},
        {S.PLATE_KEY: 'p_x', S.ROW_KEY: 'r1', S.COLUMN_KEY: 'c1',
         S.FIELD_KEY: 'f1', S.OBJECT_LABEL_KEY: 1},
    ])
    keys = list(SEL.object_keys(frame))
    assert keys == ['p%255Fx_r1_c1_f1_1', 'p%5Fx_r1_c1_f1_1']
    assert len(set(keys)) == 2


@given(left=st.text(alphabet='ab', min_size=1, max_size=3),
       right=st.text(alphabet='ab', min_size=1, max_size=3))
def test_sanitising_a_token_does_not_merge_two_fields(left, right):
    with_separator = S.field_id(left + SEP + right)
    with_hyphen = S.field_id(left + '-' + right)
    assert with_separator != with_hyphen
