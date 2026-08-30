"""``MaskCuration``'s idempotent seams: subscribe, unsubscribe, begin_stroke.

Three guards in ``spacr.curation`` exist so that a *view* can be careless
without corrupting a *session*: a panel that connects twice must not be
notified twice, a panel that disconnects without ever having connected must
not raise from its close handler, and a second ``begin_stroke`` must not
throw away the dabs already in the open stroke. Each is a branch that only
the careless caller takes, so each is pinned here against the careful one.
"""
from __future__ import annotations

import numpy as np

from spacr.curation import MaskCuration
from spacr.layers import LabelsLayer, Spacing


def _session(shape=(1, 11, 11)):
    """A flat, isotropic mask session -- the geometry is not what is tested."""
    layer = LabelsLayer(np.zeros(shape, dtype=np.int64), name="mask",
                        spacing=Spacing.from_map({"z": 1.0, "y": 1.0,
                                                  "x": 1.0}, units="um"))
    return MaskCuration(layer, artifact="mask.tif")


class _Panel:
    """A view that records the ledger entries it is told about."""

    def __init__(self):
        self.seen = []

    def on_edit(self, edit):
        """Receive one :class:`CurationEdit`."""
        self.seen.append(edit)


def test_subscribing_the_same_bound_method_twice_notifies_it_once():
    """``subscribe`` is de-duplicated so a re-opened panel is not doubled up.

    A bound method looked up fresh compares equal to the one already held, so
    the second ``subscribe`` must be a no-op. The contrast is the second
    panel, which really is a new receiver and really is added: without it,
    "one notification" would be indistinguishable from "the subscription was
    dropped".
    """
    session = _session()
    first, second = _Panel(), _Panel()

    session.subscribe(first.on_edit)
    session.subscribe(first.on_edit)          # the de-duplicated re-subscribe
    session.subscribe(second.on_edit)

    session.paint({"z": 0.0, "y": 5.0, "x": 5.0}, label=3, radius=1.0)

    assert len(first.seen) == 1, "the duplicate subscription fired twice"
    assert len(second.seen) == 1, "the distinct listener was not registered"
    assert first.seen[0] is second.seen[0]
    assert first.seen[0].kind == "paint"


def test_unsubscribing_something_that_never_subscribed_is_a_no_op():
    """A panel's close handler need not know whether it ever connected.

    The stranger is removed from nothing and, crucially, takes nobody else
    with it: the listener that *is* registered still hears the next edit.
    """
    session = _session()
    panel, stranger = _Panel(), _Panel()
    session.subscribe(panel.on_edit)

    session.unsubscribe(stranger.on_edit)     # never registered
    session.paint({"z": 0.0, "y": 5.0, "x": 5.0}, label=3, radius=1.0)
    assert len(panel.seen) == 1
    assert stranger.seen == []

    # And the same call on a listener that *is* registered really does remove
    # it, so the branch above is the "absent" half of a working pair.
    session.unsubscribe(panel.on_edit)
    session.paint({"z": 0.0, "y": 5.0, "x": 6.0}, label=3, radius=1.0)
    assert len(panel.seen) == 1


def test_a_second_begin_stroke_keeps_the_dabs_already_in_the_open_one():
    """Re-opening an open stroke must not discard what it already holds.

    A UI that calls ``begin_stroke`` on every mouse-press event -- including
    a press that arrives while a drag is already running -- would otherwise
    silently split one user action into two undo steps, or lose the first
    half. The guard makes the second call a no-op, so the whole drag is one
    ledger entry and one undo.
    """
    session = _session()

    session.begin_stroke()
    session.paint({"z": 0.0, "y": 5.0, "x": 4.0}, label=2, radius=1.0)
    session.begin_stroke()                    # the guarded second open
    session.paint({"z": 0.0, "y": 5.0, "x": 7.0}, label=2, radius=1.0)
    entry = session.end_stroke()

    painted = int(np.count_nonzero(session.layer.data))
    assert entry is not None
    assert len(session.log) == 1, "the drag became more than one ledger entry"
    assert entry.n_changed == painted > 0

    # Both halves came back, which is what proves the first was never dropped.
    session.undo()
    assert int(np.count_nonzero(session.layer.data)) == 0
