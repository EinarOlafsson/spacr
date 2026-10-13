"""Curation edges: a missing track schema, an unreadable user, empty dabs.

Curation is the one place where a crash costs work that cannot be recovered by
re-running anything -- the correction was made by hand. So each of the places
this module protects itself is driven here: the track-column import failing, the
environment refusing to name the user, a dab that changed nothing, and a
listener that raises while the ledger is being told about an edit.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import curation as curation_module
from spacr.curation import (TRACK_COLUMNS, LabelEdit, MaskCuration,
                            _track_columns, _user)
from spacr.layers import LabelsLayer, Spacing


def mask_layer(shape=(9, 9)):
    return LabelsLayer(np.zeros(shape, dtype=np.int64), name='mask',
                       spacing=Spacing.isotropic(len(shape), 1.0, units='px'))


def test_the_track_columns_fall_back_when_the_zstack_schema_is_missing(
        monkeypatch):
    """With :mod:`spacr.zstack` unimportable the module's own list is used.

    The ledger has to be able to describe a track edit even in a cut-down
    install; a curation session that could not start because an optional module
    moved would lose the corrections outright.
    """
    import spacr.zstack as zstack

    monkeypatch.delattr(zstack, 'BASE_TRACK_COLUMNS')
    assert _track_columns() == TRACK_COLUMNS


def test_an_environment_that_refuses_to_be_read_leaves_the_user_blank(
        monkeypatch):
    """``_user`` returns an empty string rather than raising on a broken environ.

    The user name is provenance decoration on the ledger entry. Letting the
    lookup escape would abort the edit it was only annotating.
    """
    def boom(*args, **kwargs):
        raise OSError('environment is unreadable')

    monkeypatch.setattr(curation_module.os.environ, 'get', boom)
    assert _user() == ''


def test_reverting_a_dab_that_changed_nothing_moves_no_elements():
    """An edit holding no previous values restores zero elements.

    Undo walks every dab in a stroke; one that recorded nothing must contribute
    nothing rather than indexing an empty array into the layer.
    """
    layer = mask_layer()
    empty = LabelEdit(index=(np.array([], dtype=np.intp),
                             np.array([], dtype=np.intp)),
                      before=np.array([], dtype=np.int64), after=1)
    assert len(empty) == 0
    assert empty.revert(layer) == 0
    assert int(layer.data.sum()) == 0

    changed = LabelEdit(
        index=(np.array([1, 2], dtype=np.intp),
               np.array([3, 4], dtype=np.intp)),
        before=np.array([7, 8], dtype=np.int64),
        after=1,
    )
    assert len(changed) == 2


def test_a_stroke_whose_dabs_changed_nothing_is_not_written_to_the_ledger():
    """``end_stroke`` returns None when the grouped dabs moved no elements.

    A ledger padded with no-op entries is one nobody reads, and the count of
    curation entries is what distinguishes a corrected dataset from a raw one.
    """
    session = MaskCuration(mask_layer(), artifact='mask.tif')
    session.begin_stroke()
    session._open.append(LabelEdit(
        index=(np.array([], dtype=np.intp), np.array([], dtype=np.intp)),
        before=np.array([], dtype=np.int64), after=1, radius=3.0))

    assert session.end_stroke() is None
    assert len(session.log) == 0
    assert session.can_undo is False


def test_a_listener_that_raises_does_not_take_the_edit_with_it():
    """A subscriber blowing up leaves the ledger entry and the pixels in place.

    The edit has already happened to the data by the time listeners are told;
    letting one view's redraw escape would report a failure for a correction
    that succeeded.
    """
    session = MaskCuration(mask_layer(), artifact='mask.tif')
    seen = []

    class Panel:
        def broken(self, edit):
            raise RuntimeError('redraw failed')

        def working(self, edit):
            seen.append(edit.kind)

    panel = Panel()
    session.subscribe(panel.broken)
    session.subscribe(panel.working)

    changed = session.paint({'y': 4.0, 'x': 4.0}, label=7, radius=1.0)

    assert changed > 0
    assert seen == ['paint']
    assert len(session.log) == 1
    assert int((session.layer.data == 7).sum()) == changed
