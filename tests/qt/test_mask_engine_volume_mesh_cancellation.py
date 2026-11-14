"""Meshing one label of a native volume inside a nested cancellation scope.

:meth:`spacr.qt.mask_engine.VolumeMaskSession.geometry_for_label` checks the
worker's cancellation token at its mesh boundary.  Whatever happens inside an
inner :func:`spacr.cancellation.installed_token` scope -- a mesh that finishes,
one cancelled half way, one refused because the label is not one connected
object -- the outer token must be back on exit and the session (mask, saved
geometry, undo/redo, saved hashes, edit-cancel hook) must be exactly as before.
"""
import numpy as np
import pytest

from spacr.cancellation import (
    CancellationToken,
    PipelineCancelled,
    current_token,
    installed_token,
)
from spacr.qt import mask_engine as engine
from spacr.qt.widgets import gate_spec


def _cube():
    """A 2x2x2 block of label 1: one valid object."""
    return np.ones((2, 2, 2), dtype=np.uint16)


def _two_corners():
    """Label 1 in two opposite corners: not one connected object."""
    labels = np.zeros((3, 3, 3), dtype=np.uint16)
    labels[0, 0, 0] = labels[2, 2, 2] = 1
    return labels


# (case, labels, cancel the inner token once voxel positions are read,
#  expected exception or None)
MESH_CASES = [
    ("an uncancelled mesh is exact", _cube, False, None),
    ("a cancel at the mesh boundary builds no geometry", _cube, True, PipelineCancelled),
    ("a refused mesh raises without changing state", _two_corners, False, ValueError),
]


def test_geometry_for_label_in_a_nested_scope(subtests, monkeypatch):
    """Every mesh outcome restores the outer token and leaves the session untouched."""
    for case_id, make_labels, cancel_mid_mesh, raised in MESH_CASES:
        with subtests.test(msg=case_id):
            labels = make_labels()
            session = engine.VolumeMaskSession(np.zeros(labels.shape), labels)
            hook = lambda: False  # noqa: E731 - the session's edit-cancel hook
            session._cancelled = hook
            expected = (engine.VolumeMaskSession(np.zeros(labels.shape), labels.copy())
                        .geometry_for_label(1).to_dict() if raised is None else None)
            outer, token = CancellationToken(), CancellationToken()
            constructed = []
            real_argwhere, real_gate = engine.np.argwhere, gate_spec.EditableVolumeGate

            def positions(*args, **kwargs):
                result = real_argwhere(*args, **kwargs)
                if cancel_mid_mesh:
                    token.cancel()
                return result

            def construct(*args, **kwargs):
                constructed.append(True)
                return real_gate(*args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(engine.np, "argwhere", positions)
                patch.setattr(gate_spec, "EditableVolumeGate", construct)
                with installed_token(outer):
                    with installed_token(token):
                        if raised is None:
                            actual = session.geometry_for_label(1)
                        else:
                            with pytest.raises(raised):
                                session.geometry_for_label(1)
                    assert current_token() is outer
                assert current_token() is None

            if raised is None:
                assert actual.to_dict() == expected
            if cancel_mid_mesh:
                assert constructed == []
            np.testing.assert_array_equal(session.mask, labels)
            assert session._cancelled is hook
            assert (session.geometries, session._undo, session._redo) == ({}, [], [])
            assert (session._saved_mask_sha256, session._saved_geometry_sha256) == (None, None)
