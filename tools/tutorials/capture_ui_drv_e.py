"""Helpers of the drv-e lesson drivers (14 Make Masks, 20 Cellpose masks, 43 Illumination).

A host clip must show the pointer doing the actions of its scene. The
drivers of these lessons are long sequences "do something, capture(name)",
so the clip of each scene is opened before its first action and closed where
the still used to be taken. :class:`capture_ui_drv_a.Rec` does that; the calls
here add what the sequences need around it. Outside the GNOME host every call
reproduces the old order exactly (the actions, then ``capture(name)``).
"""
from __future__ import annotations

import functools
import inspect

import capture_ui
from capture_ui_drv_a import Rec


def scene(capture, settle, name, lead=0.8, **capture_kwargs):
    """Open the clip of ``name`` now (host) and return the started :class:`Rec`."""
    return Rec(capture, name, settle, lead, **capture_kwargs).start()


def snap(capture, name):
    """The old still of a scene nobody narrates: Xvfb only, never inside a host clip."""
    if not capture_ui.live(capture):
        capture(name)


def clip_open(capture) -> bool:
    host = getattr(capture, 'host', None)
    return host is not None and getattr(host, 'current', None) is not None


def close_open(capture):
    """Host: end a clip that an error left open, so the real error is reported."""
    if clip_open(capture):
        try:
            capture.end()
        except Exception:  # noqa: BLE001 - the original error matters more
            pass


def guarded(function):
    """Decorator for a scene driver: an exception first closes the open host clip."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def run(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException:
            try:
                close_open(signature.bind(*args, **kwargs).arguments.get('capture'))
            except TypeError:
                pass
            raise
    return run


def prefixed(capture, prefix):
    """``capture`` for scenes named ``prefix + name``; keeps the host attributes.

    A plain wrapper function loses ``capture.host`` and ``capture.begin``, which
    would turn a host recording back into the Xvfb behaviour.
    """
    def wrapped(name, **kwargs):
        return capture(prefix + name, **kwargs)

    for attribute in ('host', 'end', 'supports_live_actions', 'maximum_live_clip_seconds'):
        if hasattr(capture, attribute):
            setattr(wrapped, attribute, getattr(capture, attribute))
    if hasattr(capture, 'begin'):
        wrapped.begin = lambda name: capture.begin(prefix + name)
    if hasattr(capture, 'clip_seconds_of'):
        wrapped.clip_seconds_of = lambda name: capture.clip_seconds_of(prefix + name)
    return wrapped
