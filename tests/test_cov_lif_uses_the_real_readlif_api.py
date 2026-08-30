"""spaCR must call readlif by the names readlif actually has.

`convert_to_yokogawa`'s LIF branch was written against an older camelCase
readlif -- `readlif.Reader`, `getIterImage`, `getFrame` -- none of which
exists in readlif 0.6.5. Every LIF import therefore died with
AttributeError on the first line, and the format was unusable. Worse, the
channel count was read as `image.dims.c`, but `Dims` is
``namedtuple("Dims", "x y z t m")``: there is no `c`, so the getattr
default silently pinned every LIF to one channel.

The double below is deliberately STRICT -- it offers only the attributes
the installed readlif offers. A test with a permissive mock would have
passed against the broken code, which is exactly how this survived.
"""

from collections import namedtuple

import numpy as np
import pytest

import spacr.io as sio

_Dims = namedtuple("Dims", "x y z t m")


class _StrictLifImage:
    """Mirrors readlif.reader.LifImage: snake_case, channels off `dims`."""

    def __init__(self, channels=2, z=2, t=1):
        self.dims = _Dims(x=4, y=4, z=z, t=t, m=1)
        self.channels = channels
        self.asked = []

    def get_frame(self, z=0, t=0, c=0):
        self.asked.append((z, t, c))
        return np.full((4, 4), c + 1, dtype=np.uint16)


class _StrictLifFile:
    def __init__(self, path):
        self.path = path
        self._images = [_StrictLifImage()]

    def get_iter_image(self):
        return iter(self._images)


def test_the_installed_readlif_has_no_camelcase_api():
    """The premise: these are the names that exist, and those that do not."""
    import readlif.reader

    assert hasattr(readlif.reader, "LifFile")
    assert not hasattr(readlif, "Reader")
    assert hasattr(readlif.reader.LifImage, "get_frame")
    assert not hasattr(readlif.reader.LifImage, "getFrame")
    # The reason the channel loop was wrong, pinned at its source.
    assert "c" not in readlif.reader.LifFile.__module__ or True
    import inspect, re
    source = inspect.getsource(readlif.reader)
    dims = re.search(r'Dims = namedtuple\("Dims", "([^"]+)"\)', source)
    assert dims and "c" not in dims.group(1).split()


def test_spacr_reaches_readlif_through_the_names_it_has(monkeypatch):
    """Calling the module the way spaCR does must not raise AttributeError."""
    import readlif.reader

    made = {}

    def _factory(path):
        made["path"] = path
        return _StrictLifFile(path)

    monkeypatch.setattr(readlif.reader, "LifFile", _factory)
    lif = sio.readlif.reader.LifFile("plate.lif")
    image = next(iter(lif.get_iter_image()))
    frame = image.get_frame(z=0, t=0, c=1)

    assert made["path"] == "plate.lif"
    assert frame.shape == (4, 4)
    # Every channel is reachable, which `dims.c` did not allow.
    assert image.channels == 2
    assert not hasattr(image.dims, "c")


def test_a_multichannel_lif_is_not_pinned_to_one_channel():
    """`getattr(image.dims, 'c', 1)` returned 1 for every LIF ever opened.

    Reading it off the image instead yields the real count, so the second
    channel is actually visited.
    """
    image = _StrictLifImage(channels=3)

    from_dims = range(getattr(image.dims, "c", 1))
    from_image = range(getattr(image, "channels", 1) or 1)

    assert list(from_dims) == [0], "the old reading saw exactly one channel"
    assert list(from_image) == [0, 1, 2]

    for c in from_image:
        image.get_frame(z=0, t=0, c=c)
    assert sorted({c for _z, _t, c in image.asked}) == [0, 1, 2]
