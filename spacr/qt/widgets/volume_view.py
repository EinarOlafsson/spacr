"""Camera arithmetic for the Gate Editor's 3D volume.

Pure numpy, no Qt and no matplotlib, so it can be tested on its own.

matplotlib describes a 3D camera with three angles -- ``elev``, ``azim`` and
``roll`` -- and builds the screen axes from them in ``Axes3D.get_proj``.
Dragging by adding to ``elev`` and ``azim`` is a turntable: it can never roll
the view, it flips when the elevation passes the pole, and a clamp on the
elevation (what the editor used to do) makes whole orientations unreachable.

This module goes the other way round. The camera is held as the three screen
axes ``u`` (right), ``v`` (up) and ``w`` (out of the screen) in matplotlib's
box coordinates; a drag rotates that frame about the screen axes, which is a
trackball, and :func:`angles_from_axes` turns the frame back into the three
angles ``view_init`` takes. Every orientation is reachable and nothing snaps.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "view_axes", "angles_from_axes", "trackball", "rotate_about_world",
]

Angles = Tuple[float, float, float]


def _rotation(axis, angle: float) -> np.ndarray:
    """Rodrigues' rotation matrix, ``angle`` radians about ``axis``."""
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    s, c = np.sin(angle), np.cos(angle)
    t = 1.0 - c
    return np.array([
        [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
        [t * y * x + z * s, t * y * y + c, t * y * z - x * s],
        [t * z * x - y * s, t * z * y + x * s, t * z * z + c]])


def _norm_angle(degrees: float) -> float:
    """An angle folded into ``[-180, 180)``, as matplotlib folds it."""
    return ((float(degrees) + 180.0) % 360.0) - 180.0


def view_axes(elev: float, azim: float, roll: float = 0.0):
    """The screen axes matplotlib builds from three camera angles.

    Mirrors ``Axes3D.get_proj`` and ``proj3d._view_axes`` for a Z-up axes:
    ``w`` points from the centre of the box to the eye, ``u`` is
    ``V x w`` with ``V`` the vertical axis (negated past the pole), and
    ``roll`` turns ``u`` and ``v`` about ``w``.

    :returns: ``(u, v, w)``, three orthonormal vectors.
    """
    e = np.deg2rad(float(elev))
    a = np.deg2rad(float(azim))
    w = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    vertical = np.array([0.0, 0.0, 1.0])
    if abs(np.deg2rad(_norm_angle(elev))) > np.pi / 2:
        vertical = -vertical
    u = np.cross(vertical, w)
    length = np.linalg.norm(u)
    if length < 1e-12:
        u = np.array([-np.sin(a), np.cos(a), 0.0])
    else:
        u = u / length
    v = np.cross(w, u)
    r = np.deg2rad(_norm_angle(roll))
    if r != 0.0:
        turn = _rotation(w, -r)
        u, v = turn @ u, turn @ v
    return u, v, w


def angles_from_axes(u, w) -> Angles:
    """The ``(elev, azim, roll)`` that make matplotlib draw this frame.

    ``elev`` is kept inside ``[-90, 90]`` so the vertical never flips; the
    roll then carries whatever turn about the line of sight the frame has.
    Looking straight down (or up) the azimuth is free, so it is chosen to
    make the roll zero.

    :param u: the screen's right-pointing axis.
    :param w: the axis pointing out of the screen.
    """
    u = np.asarray(u, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / np.linalg.norm(w)
    u = u - (u @ w) * w
    u = u / np.linalg.norm(u)
    z = float(np.clip(w[2], -1.0, 1.0))
    if abs(z) > 1.0 - 1e-9:
        azim = float(np.degrees(np.arctan2(-u[0], u[1])))
        elev = 89.9999 if z > 0 else -89.9999
        return elev, azim, 0.0
    elev = float(np.degrees(np.arcsin(z)))
    azim = float(np.degrees(np.arctan2(w[1], w[0])))
    base, _v, _w = view_axes(elev, azim, 0.0)
    turn = float(np.arctan2(w @ np.cross(base, u), base @ u))
    return elev, azim, float(-np.degrees(turn))


def trackball(elev: float, azim: float, roll: float,
              right_degrees: float, up_degrees: float) -> Angles:
    """Turn the view as if the volume were grabbed and dragged.

    A drag to the right turns the volume about the screen's vertical so its
    front follows the pointer; a drag up turns it about the screen's
    horizontal. Both are applied to the frame, not to the angles, so the
    two directions compose freely and there is no pole to stop at.

    :param right_degrees: how far the pointer moved right, as an angle.
    :param up_degrees: how far it moved up, as an angle.
    :returns: the new ``(elev, azim, roll)``.
    """
    u, v, w = view_axes(elev, azim, roll)
    if right_degrees:
        turn = _rotation(v, -np.deg2rad(right_degrees))
        u, w = turn @ u, turn @ w
    if up_degrees:
        turn = _rotation(u, np.deg2rad(up_degrees))
        v, w = turn @ v, turn @ w
    return angles_from_axes(u, w)


def rotate_about_world(elev: float, azim: float, roll: float,
                       axis: str, degrees: float) -> Angles:
    """Turn the volume about one of its own data axes.

    :param axis: ``"x"``, ``"y"`` or ``"z"`` -- the measurement the volume
        spins about, which stays put on screen while the other two turn.
    :param degrees: how far; positive turns the volume anticlockwise when
        that axis points at the viewer.
    :returns: the new ``(elev, azim, roll)``.
    """
    index = {"x": 0, "y": 1, "z": 2}[axis]
    direction = np.zeros(3)
    direction[index] = 1.0
    u, v, w = view_axes(elev, azim, roll)
    turn = _rotation(direction, -np.deg2rad(degrees))
    return angles_from_axes(turn @ u, turn @ w)
