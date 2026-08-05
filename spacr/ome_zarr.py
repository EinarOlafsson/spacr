"""OME-Zarr (OME-NGFF) — read and write, with the axes and the units taken seriously.

OME-NGFF is where large bioimaging data is going, and the two properties that
put it there are the two this module is built around.

**Chunked.** The array is stored as a grid of independently compressed blocks,
so a 100 GB plate is readable a tile at a time. Nothing here ever reads an
array to answer a question about it: :func:`read_ome_zarr` opens a handful of
small JSON files and returns the levels, their shapes, their voxel sizes and
their chunk grids without touching a single chunk, and
:meth:`OmeZarrImage.read` with a ``region=`` decodes only the chunks that
region intersects. Every byte of chunk data in the pure-Python path passes
through one function, :func:`_read_chunk_bytes`, precisely so that "it is
lazy" is a countable claim rather than a sentence in a docstring —
``tests/test_ome_zarr.py`` counts its calls.

**Multiscale.** The group holds a resolution pyramid, so drawing a 200 px
plate overview does not decode full resolution. :meth:`OmeZarrImage.level_for_size`
picks the level, and the per-level ``scale`` transformation is what makes the
picked level land in the same world coordinates as level 0.

What spaCR implements itself, and what the extra is for
------------------------------------------------------
The **metadata** — ``multiscales``, ``axes``, ``coordinateTransformations``,
``omero`` — is parsed and written here, in pure Python, because that layout is
the thing worth getting right and a library would only hide it behind a call
that returns "an array" with the voxel size quietly dropped.

The **chunk codec** is where the compiled code lives, and so it is where the
optional extra lives. This module decodes and encodes chunks compressed with
``zlib``, ``gzip``, ``bz2``, ``lzma`` or stored uncompressed using nothing but
the standard library, which is why a spaCR-written OME-Zarr round-trips on a
plain ``pip install spacr``. A file compressed with ``blosc``, ``zstd`` or
``lz4`` — the common defaults of other tools — raises
:class:`ZarrExtraMissing`, whose message names *the codec that was needed* and
the one command that fixes it, instead of a ``ModuleNotFoundError`` five
frames down.

If :mod:`zarr` *is* importable, array access is delegated to it wholesale
(see ``prefer_zarr=`` on the readers). That is the intended path: zarr handles
v3 sharding, filter pipelines and every codec, and it is maintained by people
who do nothing else. **The pure-Python reader is a fallback for the common
case, not a reimplementation of zarr**, and it says so where it gives up.

Axes and units, which is the decision this module exists to get right
--------------------------------------------------------------------
NGFF names its axes and gives each one a UDUNITS-2 unit
(``"micrometer"``, ``"nanometer"``, ``"second"``). :class:`spacr.layers.Spacing`
names its axes and carries one short unit token (``"um"``, ``"px"``) that
spaCR compares *by name* when layers are stacked — that comparison, at
``spacr/layers.py``'s ``LayerStack._check_units``, is the whole reason
``Spacing`` has a ``units`` field at all. Mapping one onto the other has three
decisions in it, and all three are made here rather than left to the caller.

**1. The Spacing is built from the SPACE axes only.** A 5-D NGFF image has
axes ``(t, c, z, y, x)``: the time axis is in seconds, the channel axis has no
unit at all, and z/y/x are in micrometers. A single ``Spacing`` over all five
would have to answer "what unit is this?" with one string, and whatever it
answered would be wrong for three of the axes — a stack whose spacing claims
``"um"`` while one of its axes steps in seconds passes the very check
``layers.py`` exists to make, and then draws a plausible picture of the wrong
thing. So :attr:`OmeZarrImage.spacing` covers z/y/x, and t and c are reported
alongside as :class:`Axis` records through :attr:`OmeZarrImage.other_axes`,
:attr:`OmeZarrImage.time_axis` and :attr:`OmeZarrImage.channel_axis`. Nothing
is lost; it is simply not pretending that seconds and micrometers are the same
kind of number.

**2. Units are translated through an explicit table, and an unknown one is
refused.** :data:`NGFF_UNIT_TO_SPACR` is the mapping, written out. A unit that
is not in it raises :class:`OmeZarrError`. It would be one line to default to
``"px"`` instead, and that one line is how a 0.65 µm pixel becomes a 0.65 px
pixel and every downstream area comes out wrong by a factor of 10^6 — in a
column named ``cell_area`` that still looks entirely reasonable. The error
distinguishes the two cases that need different fixes: a unit NGFF has never
heard of (the file is wrong), and a legal NGFF unit spaCR has no short token
for (convert it, e.g. to ``micrometer``).

**3. No unit is pixels, said out loud.** An NGFF file whose space axes carry
no ``unit`` is legal and common — it is what every tool that never knew the
pixel size writes. That reads back as ``units="px"`` with
:attr:`OmeZarrImage.units_declared` ``False`` and a
:meth:`OmeZarrImage.describe` line that says the file declared none. It is
never silently upgraded to µm, and ``"px"`` is never written out as a unit
string, because NGFF has no pixel unit: writing a px spacing emits axes with
no ``unit``, which is the same convention read back.

Space axes that disagree with *each other* — y in micrometers and x in
nanometers — are refused by name. A single ``Spacing`` has one ``units``, so
there is no honest way to represent that; the numbers would have to be
converted, and converting the caller's data behind their back is not this
module's decision to make.

Versions
--------
**0.4 is implemented properly** — it is what essentially all real data is in,
and it is zarr-format 2. 0.5 (zarr-format 3, metadata under an ``ome`` key in
``zarr.json``) is *tolerated*: its metadata is parsed in full, and its chunks
are decoded for the codec chains the standard library can do
(``bytes``/``transpose``/``gzip``/``crc32c``). Anything else routes through
the same :func:`require_codec` refusal as v2 does. ``.zgroup`` / ``.zarray`` /
``zarr.json`` and the ``zarr_format`` field are read rather than assumed, and
an unsupported ``zarr_format`` is refused by number.

Typical use::

    from spacr.ome_zarr import read_ome_zarr, write_ome_zarr
    from spacr.layers import Spacing

    img = read_ome_zarr("/data/plate.zarr/A/1/0")
    print(img.describe())
    print(img.spacing.describe())            # z 2, y 0.65, x 0.65 um

    level = img.level_for_size(512)          # coarse enough to draw, no coarser
    tile = img.read(level, world_region={"y": (0.0, 100.0),
                                         "x": (0.0, 100.0)})

    write_ome_zarr(
        "/data/out.zarr", stack,
        spacing=Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um"),
        levels=4, channel_names=("DAPI", "GFP"),
    )

Downsampling, and the transformation bug it is easy to write
------------------------------------------------------------
The writer builds a real pyramid by **2x block mean over the two
fastest-varying space axes** (see :func:`write_ome_zarr`). It is a box filter
and **not a Gaussian pyramid**: there is no pre-filter beyond the box, so it
aliases where a properly filtered pyramid would not. It is chosen anyway
because it is separable, exact, dependency-free, and the same ``local_mean``
the rest of the NGFF ecosystem writes — matching what other viewers produce
matters more here than a marginally better kernel. Striding
(``downsample="stride"``) is offered and is **mandatory for label/mask
arrays**: the mean of labels 3 and 5 is 4, an object that does not exist.

Level *k*'s ``scale`` is level 0's ``scale`` multiplied by the cumulative
factor 2^k on each downsampled axis, and leaving it at level 0's value is the
single most common NGFF bug — it renders as a pyramid whose coarse levels are
a quarter the size of the fine ones and drift off the top-left corner as you
zoom. The ``translation`` moves too, and by a different rule per method: a
block mean puts the first output element at the *centre* of the block it
averaged, half a level-0 pixel in from the edge, so level *k* is translated by
``scale_0 * (2^k - 1) / 2``; a strided level samples element 0 exactly and is
not translated at all. Both are written, and asserted in the tests with the
numbers spelled out by hand.
"""
from __future__ import annotations

import base64
import bz2
import gzip
import itertools
import json
import lzma
import math
import os
import shutil
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Union)

import numpy as np

from .layers import LayerError, Spacing

__all__ = [
    "AXIS_CHANNEL",
    "AXIS_SPACE",
    "AXIS_TIME",
    "AXIS_TYPES",
    "Axis",
    "CANONICAL_AXIS_ORDER",
    "CODEC_MISSING_MESSAGE",
    "DEFAULT_CHANNEL_COLORS",
    "DEFAULT_TILE",
    "DEFAULT_Z_CHUNK",
    "Level",
    "NGFF_SPACE_UNITS",
    "NGFF_TIME_UNITS",
    "NGFF_UNIT_TO_SPACR",
    "OmeZarrError",
    "OmeZarrImage",
    "PIXEL_UNITS",
    "SPACR_UNIT_TO_NGFF",
    "STDLIB_CODECS",
    "SUPPORTED_NGFF_VERSIONS",
    "SUPPORTED_ZARR_FORMATS",
    "ZARR_EXTRA",
    "ZARR_MISSING_MESSAGE",
    "ZarrExtraMissing",
    "axes_from_spacing",
    "ngff_unit_to_spacr",
    "read_ome_zarr",
    "read_ome_zarr_array",
    "require_codec",
    "require_zarr",
    "spacing_from_axes",
    "spacr_unit_to_ngff",
    "write_ome_zarr",
]


# ---------------------------------------------------------------------------
# Errors and the optional extra
# ---------------------------------------------------------------------------

class OmeZarrError(ValueError):
    """An OME-Zarr that cannot mean what it says, or a request that cannot be met.

    Raised rather than repaired, for the reason :class:`spacr.layers.LayerError`
    is: every case this covers — an unknown unit, space axes in two different
    units, a ``multiscales`` block that is not there — has a "helpful" fallback
    that produces a plausible-looking image with the wrong scale on it, and a
    wrong scale reaches a figure without ever looking wrong.
    """


class ZarrExtraMissing(OmeZarrError, ImportError):
    """The optional ``zarr`` extra is needed here and is not installed.

    Both parents are deliberate. It is an :class:`ImportError` because a caller
    guarding an optional feature writes ``except ImportError``; it is an
    :class:`OmeZarrError` because from the caller's side this is simply one
    more file that could not be read, and code that wraps a whole read in
    ``except OmeZarrError`` should not miss it.
    """


#: The ``setup.py`` extra that provides :mod:`zarr` and :mod:`numcodecs`.
ZARR_EXTRA = "zarr"

#: Shown when :mod:`zarr` itself is wanted. One sentence of diagnosis and one
#: command, following :data:`spacr.qt._QT_MISSING_MESSAGE`.
ZARR_MISSING_MESSAGE = """\
This needs the optional `zarr` extra, which is not installed in this
environment (missing module: {module}).

Install it with:

    python -m pip install "spacr[zarr]"

spaCR reads and writes OME-Zarr metadata, and stored/zlib/gzip/bz2/lzma
chunks, without it — the extra is only needed for third-party chunk codecs
and for zarr-format 3 features such as sharding.\
"""

#: Shown when a *chunk codec* is what is missing. It names the codec, because
#: "install the extra" without saying what for is an instruction the user
#: cannot check, and because the codec name is the one piece of information
#: that says whether the file is unusual or the install is.
CODEC_MISSING_MESSAGE = """\
This OME-Zarr stores its chunks with the {codec!r} codec, which spaCR cannot
decode with the standard library alone, and the optional `zarr` extra that
provides it is not installed (missing module: {module}).

Install it with:

    python -m pip install "spacr[zarr]"

Codecs that need no extra: {stdlib}, or uncompressed. spaCR-written OME-Zarr
uses one of those by default, so it always round-trips on a plain
`pip install spacr`.\
"""


def _stdlib_zstd():
    """Return :mod:`compression.zstd`, the standard library's zstd on 3.14+.

    Imported **by name through** :func:`importlib.import_module` rather than
    with an ``import`` statement, and the reason is specific:
    ``tests/test_declared_dependencies_match_imports.py`` walks every import
    statement under ``spacr/`` and requires each one to name either a declared
    distribution or a module in ``sys.stdlib_module_names`` — *of the
    interpreter running the test*. ``compression`` entered the standard
    library in 3.14, so on 3.9-3.13 a literal ``import compression.zstd``
    reads to that census as an undeclared third-party package that must be
    added to setup.py, which would be a lie: there is no such distribution to
    install. The string keeps the census honest on every interpreter.

    :returns: the module.
    :raises ImportError: on Python < 3.14. Callers turn that into the
        numcodecs path.
    """
    import importlib
    return importlib.import_module("compression.zstd")


def _stdlib_zstd_decompress(data: bytes) -> bytes:
    """Decompress zstd with the standard library (Python 3.14+ only).

    :param data: the compressed chunk.
    :returns: the decompressed bytes.
    :raises ImportError: on Python < 3.14. See :func:`_stdlib_zstd`.
    """
    return _stdlib_zstd().decompress(data)


def _stdlib_zstd_compress(data: bytes, level: int) -> bytes:
    """Compress zstd with the standard library (Python 3.14+ only).

    :param data: the raw chunk bytes.
    :param level: compression level.
    :returns: the compressed bytes.
    :raises ImportError: on Python < 3.14. See :func:`_stdlib_zstd`.
    """
    return _stdlib_zstd().compress(data, level)


#: Codec ids spaCR can decode with the standard library and no extra. ``zstd``
#: is in the list only from Python 3.14, where it entered the stdlib as
#: :mod:`compression.zstd`; on older interpreters it falls through to
#: numcodecs like any other third-party codec.
STDLIB_CODECS: Dict[str, Callable[[bytes], bytes]] = {
    "zlib": zlib.decompress,
    "gzip": gzip.decompress,
    "bz2": bz2.decompress,
    "lzma": lzma.decompress,
    "zstd": _stdlib_zstd_decompress,
}


def require_zarr():
    """Import and return :mod:`zarr`, or raise a message worth reading.

    :returns: the imported :mod:`zarr` module.
    :raises ZarrExtraMissing: when the extra is not installed, with the
        ``pip install "spacr[zarr]"`` line and a note that most files do not
        need it.
    """
    try:
        import zarr
    except ImportError as exc:
        module = (getattr(exc, "name", None) or "zarr").split(".", 1)[0]
        raise ZarrExtraMissing(
            ZARR_MISSING_MESSAGE.format(module=module)) from exc
    return zarr


def require_codec(codec_id: str,
                  config: Optional[Mapping[str, Any]] = None
                  ) -> Callable[[bytes], bytes]:
    """Return a ``bytes -> bytes`` decoder for a zarr compressor id.

    The standard library is tried first, then :mod:`numcodecs`. Only when both
    fail is anything raised, and what is raised names the codec.

    :param codec_id: the ``id`` from a zarr v2 ``compressor`` block, or the
        ``name`` of a zarr v3 codec — ``"zlib"``, ``"blosc"``, ``"zstd"``.
    :param config: the rest of that block, passed to :mod:`numcodecs` when it
        is needed. Ignored by the standard-library codecs, whose decompressors
        read their parameters out of the stream.
    :returns: a callable taking the stored chunk bytes and returning the raw
        bytes.
    :raises ZarrExtraMissing: when neither the standard library nor an
        installed :mod:`numcodecs` provides the codec.
    :raises OmeZarrError: when ``codec_id`` is empty or not a string.
    """
    if not isinstance(codec_id, str) or not codec_id.strip():
        raise OmeZarrError(
            f"a chunk codec needs a name; got {codec_id!r}. A zarr v2 "
            f"`compressor` is either null (stored) or an object with an "
            f"\"id\" field.")
    name = codec_id.strip().lower()

    stdlib = STDLIB_CODECS.get(name)
    if stdlib is not None:
        try:                       # zstd only exists in the stdlib on 3.14+
            stdlib(b"")
        except ImportError:
            stdlib = None
        except Exception:          # an empty buffer is not valid input; fine
            pass
    if stdlib is not None:
        return stdlib

    try:
        import numcodecs
    except ImportError as exc:
        module = (getattr(exc, "name", None) or "numcodecs").split(".", 1)[0]
        raise ZarrExtraMissing(CODEC_MISSING_MESSAGE.format(
            codec=name, module=module,
            stdlib=", ".join(sorted(STDLIB_CODECS)))) from exc

    spec = dict(config or {})
    spec["id"] = name
    try:
        codec = numcodecs.get_codec(spec)
    except Exception as exc:
        raise OmeZarrError(
            f"numcodecs is installed but could not build the {name!r} codec "
            f"from {spec!r}: {exc}") from exc
    return lambda raw: bytes(memoryview(codec.decode(raw)))


def _encoder(codec_id: Optional[str], level: int
             ) -> Tuple[Optional[Dict[str, Any]], Callable[[bytes], bytes]]:
    """Return ``(compressor block, encoder)`` for the writer.

    :param codec_id: ``None`` for stored (uncompressed), else a codec id.
    :param level: compression level for the codecs that take one.
    :returns: the JSON ``compressor`` value to put in ``.zarray`` and the
        callable that produces the stored bytes.
    :raises ZarrExtraMissing: when the codec needs :mod:`numcodecs`.
    :raises OmeZarrError: on a codec name nothing provides.
    """
    if codec_id is None:
        return None, lambda raw: raw
    name = str(codec_id).strip().lower()
    if name in ("none", "null", "stored", "raw"):
        return None, lambda raw: raw
    if name == "zlib":
        return {"id": "zlib", "level": int(level)}, \
            lambda raw: zlib.compress(raw, int(level))
    if name == "gzip":
        return {"id": "gzip", "level": int(level)}, \
            lambda raw: gzip.compress(raw, int(level))
    if name == "bz2":
        return {"id": "bz2", "level": int(level)}, \
            lambda raw: bz2.compress(raw, int(level))
    if name == "lzma":
        return {"id": "lzma"}, lambda raw: lzma.compress(raw)
    if name == "zstd":
        try:
            _stdlib_zstd_compress(b"", int(level))
        except ImportError:
            pass
        else:
            return {"id": "zstd", "level": int(level)}, \
                lambda raw: _stdlib_zstd_compress(raw, int(level))
    try:
        import numcodecs
    except ImportError as exc:
        module = (getattr(exc, "name", None) or "numcodecs").split(".", 1)[0]
        raise ZarrExtraMissing(CODEC_MISSING_MESSAGE.format(
            codec=name, module=module,
            stdlib=", ".join(sorted(STDLIB_CODECS)))) from exc
    spec: Dict[str, Any] = {"id": name}
    if name in ("blosc", "zstd", "lz4"):
        spec["level"] = int(level)
    try:
        codec = numcodecs.get_codec(spec)
    except Exception as exc:
        raise OmeZarrError(
            f"numcodecs is installed but could not build the {name!r} codec: "
            f"{exc}") from exc
    return codec.get_config(), lambda raw: bytes(memoryview(codec.encode(raw)))


# ---------------------------------------------------------------------------
# Axes and units
# ---------------------------------------------------------------------------

#: NGFF axis type for a spatial axis. Only these axes go into a
#: :class:`spacr.layers.Spacing`.
AXIS_SPACE = "space"

#: NGFF axis type for a time axis.
AXIS_TIME = "time"

#: NGFF axis type for a channel axis. Channel axes carry no unit — a channel
#: index is not a measurement of anything.
AXIS_CHANNEL = "channel"

#: The three axis types NGFF gives meaning to. It is not a closed set: 0.4
#: says ``type`` SHOULD be one of these, so a file using something else is
#: read faithfully and reported through :attr:`OmeZarrImage.other_axes`
#: rather than refused.
AXIS_TYPES: Tuple[str, ...] = (AXIS_TIME, AXIS_CHANNEL, AXIS_SPACE)

#: The axis order NGFF 0.4 requires, outermost first. Time before channel
#: before space, and space ending in ``y, x``. The writer enforces it.
CANONICAL_AXIS_ORDER: Tuple[str, ...] = ("t", "c", "z", "y", "x")

#: NGFF versions this module claims to handle. 0.4 is implemented properly;
#: 0.5 is parsed and read where the standard library can decode its chunks.
SUPPORTED_NGFF_VERSIONS: Tuple[str, ...] = ("0.4", "0.5")

#: Zarr storage formats this module reads. 2 is what OME-NGFF 0.4 is; 3 is
#: what 0.5 is. Read out of ``.zgroup`` / ``.zarray`` / ``zarr.json`` rather
#: than assumed, and refused by number when it is neither.
SUPPORTED_ZARR_FORMATS: Tuple[int, ...] = (2, 3)

#: What a space axis with no declared unit means. NGFF has no pixel unit, so
#: this token is never written to a file: a px spacing emits axes with no
#: ``unit``, which is exactly what is read back as px.
PIXEL_UNITS = "px"

#: Every spatial unit OME-NGFF 0.4 permits (a UDUNITS-2 subset). Used only to
#: tell "your file is wrong" apart from "spaCR has no token for that", which
#: are two different problems with two different fixes.
NGFF_SPACE_UNITS = frozenset({
    "angstrom", "attometer", "centimeter", "decimeter", "exameter",
    "femtometer", "foot", "gigameter", "hectometer", "inch", "kilometer",
    "megameter", "meter", "micrometer", "mile", "millimeter", "nanometer",
    "parsec", "petameter", "picometer", "terameter", "yard", "yoctometer",
    "yottameter", "zeptometer", "zettameter",
})

#: Every temporal unit OME-NGFF 0.4 permits.
NGFF_TIME_UNITS = frozenset({
    "attosecond", "centisecond", "day", "decisecond", "exasecond",
    "femtosecond", "gigasecond", "hectosecond", "hour", "kilosecond",
    "megasecond", "microsecond", "millisecond", "minute", "nanosecond",
    "petasecond", "picosecond", "second", "terasecond", "yoctosecond",
    "yottasecond", "zeptosecond", "zettasecond",
})

#: The canonical half of the unit table: NGFF (UDUNITS-2) name -> the short
#: token :attr:`spacr.layers.Spacing.units` carries. Inverted for writing, so
#: only these entries round-trip; the aliases below are read-only tolerance.
_CANONICAL_UNIT_TO_SPACR: Dict[str, str] = {
    "femtometer": "fm",
    "picometer": "pm",
    "angstrom": "angstrom",
    "nanometer": "nm",
    "micrometer": "um",
    "millimeter": "mm",
    "centimeter": "cm",
    "decimeter": "dm",
    "meter": "m",
    "kilometer": "km",
    "nanosecond": "ns",
    "microsecond": "us",
    "millisecond": "ms",
    "second": "s",
    "minute": "min",
    "hour": "h",
    "day": "d",
}

#: NGFF unit name -> spaCR token, including the off-spec spellings real files
#: contain. NGFF 0.4 requires the UDUNITS-2 name, but tools write ``"micron"``
#: and ``"um"`` anyway, and refusing a file over a spelling nobody disputes
#: the meaning of would be pedantry with a cost. spaCR only ever *writes* the
#: canonical name.
NGFF_UNIT_TO_SPACR: Dict[str, str] = dict(_CANONICAL_UNIT_TO_SPACR)
NGFF_UNIT_TO_SPACR.update({
    "micron": "um", "microns": "um", "um": "um", "µm": "um",
    "μm": "um", "nm": "nm", "mm": "mm", "cm": "cm", "m": "m",
    "pm": "pm", "Å": "angstrom", "sec": "s", "s": "s", "ms": "ms",
    "us": "us", "µs": "us", "min": "min", "h": "h", "hr": "h",
})

#: spaCR token -> NGFF (UDUNITS-2) name, for writing. The strict inverse of
#: the canonical table: a token with no NGFF name is refused by
#: :func:`spacr_unit_to_ngff` rather than guessed at.
SPACR_UNIT_TO_NGFF: Dict[str, str] = {
    token: name for name, token in _CANONICAL_UNIT_TO_SPACR.items()
}

_TYPE_BY_AXIS_NAME: Dict[str, str] = {
    "t": AXIS_TIME, "time": AXIS_TIME,
    "c": AXIS_CHANNEL, "channel": AXIS_CHANNEL, "ch": AXIS_CHANNEL,
    "z": AXIS_SPACE, "y": AXIS_SPACE, "x": AXIS_SPACE,
}


def ngff_unit_to_spacr(unit: Optional[str], *, axis: str = "?") -> str:
    """Translate an NGFF unit name into the token ``Spacing.units`` carries.

    :param unit: the ``unit`` field of an NGFF axis, or ``None``/``""`` when
        the file declares none.
    :param axis: the axis name, used only to make the error message point at
        the offending axis.
    :returns: the spaCR token (``"um"``, ``"nm"``, ``"s"``, ...), or
        :data:`PIXEL_UNITS` when no unit was declared.
    :raises OmeZarrError: on a unit spaCR will not translate. Two distinct
        messages: one for a name NGFF does not define, one for a legal NGFF
        unit spaCR has no short token for — the second tells the user to
        convert rather than implying their file is broken.
    """
    if unit is None:
        return PIXEL_UNITS
    text = str(unit).strip()
    if not text:
        return PIXEL_UNITS
    token = NGFF_UNIT_TO_SPACR.get(text) or NGFF_UNIT_TO_SPACR.get(text.lower())
    if token:
        return token
    known = text.lower() in NGFF_SPACE_UNITS or text.lower() in NGFF_TIME_UNITS
    if known:
        raise OmeZarrError(
            f"axis {axis!r} is measured in {text!r}, which is a legal OME-NGFF "
            f"unit that spaCR has no short token for. spaCR knows: "
            f"{', '.join(sorted(SPACR_UNIT_TO_NGFF))}. Convert the axis (to "
            f"\"micrometer\", say) and rewrite the multiscales metadata; "
            f"spaCR will not guess a conversion factor for you.")
    raise OmeZarrError(
        f"axis {axis!r} declares unit {text!r}, which is not an OME-NGFF unit. "
        f"NGFF units are UDUNITS-2 names such as \"micrometer\", "
        f"\"nanometer\" or \"second\". spaCR refuses rather than falling back "
        f"to pixels: a 0.65 micrometer pixel silently read as 0.65 px makes "
        f"every area downstream wrong by a factor of a million, in a column "
        f"still named cell_area. Fix the unit in the file's .zattrs.")


def spacr_unit_to_ngff(units: str) -> Optional[str]:
    """Translate a :attr:`spacr.layers.Spacing.units` token into an NGFF unit.

    :param units: the spaCR token, e.g. ``"um"`` or ``"px"``.
    :returns: the UDUNITS-2 name to write into ``axes``, or ``None`` for
        pixels — NGFF has no pixel unit, and an axis with no ``unit`` is the
        spec's way of saying the same thing.
    :raises OmeZarrError: on a token with no NGFF name, naming what is
        available.
    """
    text = str(units).strip()
    if not text or text == PIXEL_UNITS:
        return None
    name = SPACR_UNIT_TO_NGFF.get(text)
    if name:
        return name
    # A caller who already holds the NGFF name is not wrong; accept it.
    if text.lower() in NGFF_SPACE_UNITS or text.lower() in NGFF_TIME_UNITS:
        return text.lower()
    raise OmeZarrError(
        f"spacing units {units!r} have no OME-NGFF equivalent. Writable "
        f"units: {', '.join(sorted(SPACR_UNIT_TO_NGFF))}, or {PIXEL_UNITS!r} "
        f"for an unscaled pixel grid (which writes axes with no unit, the "
        f"spec's way of saying the same thing).")


@dataclass(frozen=True)
class Axis:
    """One NGFF axis: what it is called, what kind it is, and how big a step is.

    A faithful record of the file, not an interpretation of it: :attr:`unit`
    holds the NGFF name as written, even in the cases spaCR would not write
    itself (a channel axis with a unit, say). The interpretation happens in
    :meth:`spacr_units` and in :func:`spacing_from_axes`, where it can refuse.

    :param name: the axis name — ``"t"``, ``"c"``, ``"z"``, ``"y"``, ``"x"``.
    :param type: ``"space"``, ``"time"``, ``"channel"`` or, for a file that
        uses something else, whatever it says. Only ``"space"`` axes reach a
        :class:`spacr.layers.Spacing`.
    :param unit: the UDUNITS-2 unit name, or ``None`` when the file declares
        none. ``None`` on a space axis means pixels, explicitly.
    :param scale: world size of one element along this axis at *level 0*. The
        per-level values live on :class:`Level`; this one is the reference the
        :class:`spacr.layers.Spacing` is built from.
    :param translate: world coordinate of element 0 at level 0.
    """

    name: str
    type: str = AXIS_SPACE
    unit: Optional[str] = None
    scale: float = 1.0
    translate: float = 0.0

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise OmeZarrError("an axis needs a name; NGFF requires it")
        object.__setattr__(self, "name", name)
        kind = str(self.type).strip().lower() or AXIS_SPACE
        object.__setattr__(self, "type", kind)
        unit = self.unit
        object.__setattr__(self, "unit",
                           None if unit is None or not str(unit).strip()
                           else str(unit).strip())
        scale = float(self.scale)
        if scale == 0.0 or not math.isfinite(scale):
            raise OmeZarrError(
                f"axis {name!r} has scale {self.scale!r}. A zero or "
                f"non-finite step collapses the axis: every world coordinate "
                f"on it resolves to element 0, and the image is drawn out of "
                f"register with nothing to show for it.")
        object.__setattr__(self, "scale", scale)
        translate = float(self.translate)
        if not math.isfinite(translate):
            raise OmeZarrError(
                f"axis {name!r} has a non-finite translation "
                f"{self.translate!r}")
        object.__setattr__(self, "translate", translate)

    @classmethod
    def space(cls, name: str, scale: float = 1.0, unit: Optional[str] = None,
              translate: float = 0.0) -> "Axis":
        """A spatial axis — the kind that goes into a :class:`spacr.layers.Spacing`."""
        return cls(name=name, type=AXIS_SPACE, unit=unit, scale=scale,
                   translate=translate)

    @classmethod
    def time(cls, name: str = "t", scale: float = 1.0,
             unit: Optional[str] = "second", translate: float = 0.0) -> "Axis":
        """A time axis. Kept out of the spacing; reported beside it."""
        return cls(name=name, type=AXIS_TIME, unit=unit, scale=scale,
                   translate=translate)

    @classmethod
    def channel(cls, name: str = "c") -> "Axis":
        """A channel axis. No unit, ever: a channel index measures nothing."""
        return cls(name=name, type=AXIS_CHANNEL, unit=None, scale=1.0)

    @property
    def is_space(self) -> bool:
        """Whether this axis is spatial, and so part of the spacing."""
        return self.type == AXIS_SPACE

    def spacr_units(self) -> str:
        """This axis's unit as a spaCR token.

        :returns: the token, or :data:`PIXEL_UNITS` when the file declared no
            unit.
        :raises OmeZarrError: on a unit spaCR will not translate.
        """
        return ngff_unit_to_spacr(self.unit, axis=self.name)

    def to_ngff(self) -> Dict[str, Any]:
        """This axis as the ``axes`` entry NGFF wants.

        :returns: ``{"name": ..., "type": ...}`` plus ``"unit"`` when there is
            one. A channel axis never gets a unit, whatever it was read with —
            the spec forbids it and a viewer that trusts it would be misled.
        """
        entry: Dict[str, Any] = {"name": self.name, "type": self.type}
        if self.unit and self.type != AXIS_CHANNEL:
            entry["unit"] = self.unit
        return entry

    def describe(self) -> str:
        """One clause for a status line: ``x: space, 0.65 micrometer/step``.

        A channel axis gets no step, because a channel index is not a
        measurement and ``c: channel, 1/step`` reads like one.
        """
        if self.type == AXIS_CHANNEL:
            return f"{self.name}: {self.type}"
        unit = self.unit or ("pixel (undeclared)" if self.is_space else "")
        step = f"{self.scale:g}" + (f" {unit}/step" if unit else "/step")
        return f"{self.name}: {self.type}, {step}"


def _axis_from_ngff(entry: Any, index: int) -> Axis:
    """Build an :class:`Axis` from one ``axes`` entry, inferring what is absent."""
    if isinstance(entry, str):          # NGFF 0.3 wrote bare names
        name = entry
        kind = _TYPE_BY_AXIS_NAME.get(name.lower(), AXIS_SPACE)
        return Axis(name=name, type=kind)
    if not isinstance(entry, Mapping):
        raise OmeZarrError(
            f"axes[{index}] is {entry!r}; an NGFF axis is an object with a "
            f"\"name\", and should have a \"type\".")
    name = entry.get("name")
    if not name:
        raise OmeZarrError(f"axes[{index}] has no \"name\"; NGFF requires it")
    # `type` is SHOULD, not MUST, in 0.4, and files in the wild omit it. The
    # name carries the answer for every axis NGFF actually defines.
    kind = entry.get("type") or _TYPE_BY_AXIS_NAME.get(str(name).lower(),
                                                       AXIS_SPACE)
    return Axis(name=str(name), type=str(kind), unit=entry.get("unit"))


def spacing_from_axes(axes: Sequence[Axis]) -> Spacing:
    """Build a :class:`spacr.layers.Spacing` from the SPACE axes only.

    The exclusion is the point. A spacing has one ``units`` string and
    :class:`spacr.layers.LayerStack` compares it by name; a spacing that mixed
    a time axis in seconds with a y axis in micrometers would answer that
    comparison with a string that is wrong for one of them, and pass.

    :param axes: every axis of the image, in array order.
    :returns: a spacing over the spatial axes, in the same relative order,
        with :attr:`spacr.layers.Spacing.units` set from their common unit.
    :raises OmeZarrError: when there are no spatial axes; when they carry
        different units from each other (named); or when a unit does not
        translate.
    """
    space = [a for a in axes if a.is_space]
    if not space:
        raise OmeZarrError(
            f"no spatial axes among {[a.name for a in axes]}. An OME-NGFF "
            f"image must have at least two (y and x); without them there is "
            f"no voxel size to speak of.")
    tokens = {a.name: a.spacr_units() for a in space}
    distinct = set(tokens.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{n}={u}" for n, u in tokens.items())
        raise OmeZarrError(
            f"the space axes are not all in the same unit ({detail}). A "
            f"spacr.layers.Spacing carries one unit for every axis, so there "
            f"is no honest way to hold this. Convert the axes to a common "
            f"unit in the file's .zattrs — spaCR will not apply the "
            f"conversion factor for you, because rescaling somebody's "
            f"coordinates behind their back is how a montage ends up half a "
            f"field out of register.")
    try:
        return Spacing(scale=tuple(a.scale for a in space),
                       translate=tuple(a.translate for a in space),
                       axes=tuple(a.name for a in space),
                       units=distinct.pop())
    except LayerError as exc:
        # Spacing's own refusals (a zero voxel size, a duplicated axis name)
        # are the right refusals; they just need to say which file they are
        # about, since the caller asked about a path, not about a Spacing.
        raise OmeZarrError(
            f"the space axes {[a.name for a in space]} do not make a usable "
            f"spacing: {exc}") from exc


def axes_from_spacing(spacing: Spacing, ndim: Optional[int] = None,
                      names: Optional[Sequence[str]] = None) -> Tuple[Axis, ...]:
    """Build the full NGFF axis list for an array from a spacing.

    :param spacing: the spatial spacing. Its axis names become the space axes,
        and its units become theirs.
    :param ndim: the array's dimensionality. When it exceeds the spacing's,
        the leading axes are taken from :data:`CANONICAL_AXIS_ORDER` — the
        only order NGFF 0.4 permits — so a ``(t, c, z, y, x)`` array written
        with a ``(z, y, x)`` spacing needs no extra argument.
    :param names: explicit names for every axis, overriding the derivation.
    :returns: the axes, outermost first.
    :raises OmeZarrError: when the counts cannot be reconciled, or a unit has
        no NGFF name.
    """
    unit = spacr_unit_to_ngff(spacing.units)
    ndim = int(ndim if ndim is not None else spacing.ndim)
    if names is not None:
        names = tuple(str(n) for n in names)
        if len(names) != ndim:
            raise OmeZarrError(
                f"{len(names)} axis names for a {ndim}-dimensional array")
    elif ndim == spacing.ndim:
        names = tuple(spacing.axes)
    elif ndim < spacing.ndim:
        raise OmeZarrError(
            f"the spacing describes {spacing.ndim} axes {spacing.axes} but "
            f"the array has {ndim}. Pass axes= naming the array's own axes.")
    else:
        lead = [n for n in CANONICAL_AXIS_ORDER if n not in spacing.axes]
        extra = ndim - spacing.ndim
        if len(lead) < extra:
            raise OmeZarrError(
                f"cannot name {extra} extra axes for a {ndim}-dimensional "
                f"array from a spacing over {spacing.axes}; pass axes= "
                f"explicitly.")
        names = tuple(lead[len(lead) - extra:]) + tuple(spacing.axes)

    out: List[Axis] = []
    for name in names:
        if spacing.has_axis(name):
            i = spacing.axis_index(name)
            out.append(Axis(name=name, type=AXIS_SPACE, unit=unit,
                            scale=spacing.scale[i],
                            translate=spacing.translate[i]))
        else:
            kind = _TYPE_BY_AXIS_NAME.get(name.lower(), AXIS_SPACE)
            if kind == AXIS_SPACE:
                raise OmeZarrError(
                    f"axis {name!r} is not in the spacing {spacing.axes} and "
                    f"is not a recognised non-spatial axis name (t, c). Add "
                    f"it to the spacing, or rename it.")
            out.append(Axis(name=name, type=kind,
                            unit="second" if kind == AXIS_TIME else None))
    return tuple(out)


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Level:
    """One resolution level of a multiscale image — its metadata, not its data.

    Everything here comes out of one small JSON file. Building the whole
    pyramid's worth of these costs a handful of stats and no chunk reads,
    which is what makes "list the levels and their shapes" free on a 100 GB
    plate.

    :param path: the dataset path inside the group, e.g. ``"0"``.
    :param shape: array shape, in array order.
    :param chunks: chunk shape — the unit of I/O, and therefore the smallest
        region that can be read.
    :param dtype: the numpy dtype string as stored, e.g. ``"<u2"``. Kept as
        written, byte order included.
    :param scale: per-axis world size of one element AT THIS LEVEL, already
        composed with any group-level transformation.
    :param translation: per-axis world coordinate of element 0 at this level,
        likewise composed.
    :param zarr_format: 2 or 3.
    :param compressor: the codec id, or ``None`` when chunks are stored
        uncompressed.
    """

    path: str
    shape: Tuple[int, ...]
    chunks: Tuple[int, ...]
    dtype: str
    scale: Tuple[float, ...]
    translation: Tuple[float, ...]
    zarr_format: int = 2
    compressor: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", str(self.path))
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))
        object.__setattr__(self, "chunks", tuple(int(v) for v in self.chunks))
        object.__setattr__(self, "scale", tuple(float(v) for v in self.scale))
        object.__setattr__(self, "translation",
                           tuple(float(v) for v in self.translation))
        if len(self.chunks) != len(self.shape):
            raise OmeZarrError(
                f"level {self.path!r}: chunks {self.chunks} and shape "
                f"{self.shape} have different ranks")
        if len(self.scale) != len(self.shape):
            raise OmeZarrError(
                f"level {self.path!r}: the scale transformation has "
                f"{len(self.scale)} entries but the array has "
                f"{len(self.shape)} axes. NGFF requires one per axis.")
        if len(self.translation) != len(self.shape):
            raise OmeZarrError(
                f"level {self.path!r}: the translation transformation has "
                f"{len(self.translation)} entries but the array has "
                f"{len(self.shape)} axes. NGFF requires one per axis.")

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        """Uncompressed size of the whole level, in bytes."""
        return int(np.prod(self.shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize

    @property
    def n_chunks(self) -> int:
        """How many chunks the level is stored in."""
        return int(np.prod([-(-s // c) for s, c in zip(self.shape, self.chunks)],
                           dtype=np.int64))

    def describe(self) -> str:
        """One line: ``0: (2, 12, 2048, 2048) uint16, chunks (1, 16, 256, 256), zlib``."""
        codec = self.compressor or "stored"
        return (f"{self.path}: {self.shape} {np.dtype(self.dtype).name}, "
                f"chunks {self.chunks}, {codec}")


# ---------------------------------------------------------------------------
# The zarr v2 chunk layer, in pure Python
# ---------------------------------------------------------------------------

def _read_chunk_bytes(path: Path) -> Optional[bytes]:
    """Read one stored chunk. **The only place chunk data is ever read.**

    Every byte the pure-Python reader decodes comes through here, which is
    what makes laziness testable: count the calls and you have counted the
    chunks touched. ``tests/test_ome_zarr.py`` does exactly that, and asserts
    that a small region does not touch the whole grid.

    :param path: the chunk's file path.
    :returns: the stored bytes, or ``None`` when the chunk does not exist —
        which in zarr means "entirely fill_value", the normal representation
        of empty space, and costs no decode.
    """
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except FileNotFoundError:
        return None
    except IsADirectoryError:
        return None


def _dtype_from_zarr(raw: Any, where: str) -> np.dtype:
    """Turn a ``.zarray`` dtype field into a :class:`numpy.dtype`."""
    if not isinstance(raw, str):
        raise OmeZarrError(
            f"{where}: dtype {raw!r} is a structured or nested dtype. spaCR's "
            f"pure-Python reader handles plain numeric dtypes only; install "
            f"the extra (`python -m pip install \"spacr[zarr]\"`) to read it "
            f"through zarr.")
    try:
        dtype = np.dtype(raw)
    except TypeError as exc:
        raise OmeZarrError(f"{where}: {raw!r} is not a numpy dtype: {exc}") from exc
    if dtype.kind == "O":
        raise OmeZarrError(
            f"{where}: object dtype arrays store pickles, which spaCR will "
            f"not unpickle from a data file.")
    return dtype


def _fill_value(raw: Any, dtype: np.dtype, where: str) -> np.ndarray:
    """Turn a ``.zarray`` fill_value into a 0-d array of ``dtype``."""
    if raw is None:
        return np.zeros((), dtype=dtype)
    if isinstance(raw, str):
        special = {"NaN": np.nan, "Infinity": np.inf, "-Infinity": -np.inf}
        if raw in special:
            if dtype.kind not in "fc":
                raise OmeZarrError(
                    f"{where}: fill_value {raw!r} on a {dtype.str} array")
            return np.array(special[raw], dtype=dtype)
        try:                     # base64, the spec's escape hatch for raw bits
            buf = base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise OmeZarrError(
                f"{where}: fill_value {raw!r} is neither a number, one of "
                f"NaN/Infinity/-Infinity, nor valid base64: {exc}") from exc
        if len(buf) != dtype.itemsize:
            raise OmeZarrError(
                f"{where}: base64 fill_value decodes to {len(buf)} bytes for "
                f"a {dtype.itemsize}-byte dtype")
        return np.frombuffer(buf, dtype=dtype)[0]
    return np.array(raw, dtype=dtype)


@dataclass(frozen=True)
class _ZarrArray:
    """A zarr array's storage metadata plus the region read built on it.

    Covers zarr-format 2 fully for the codecs the standard library provides,
    and zarr-format 3 for the ``bytes``/``transpose``/``gzip``/``crc32c``
    codec chain. It is deliberately small: it exists so a spaCR-written
    OME-Zarr round-trips with no extra installed, not to be another zarr.

    Codecs are resolved **at decode time, not at open time**, which is why
    :attr:`codec_specs` holds specifications rather than callables. Opening a
    blosc-compressed array must still answer "what shape is it, what is the
    voxel size, how many levels are there" without :mod:`numcodecs`
    installed — refusing to read the metadata of a file whose *pixels* need an
    extra would make the extra mandatory for the cheap half of the format.
    """

    root: Path
    shape: Tuple[int, ...]
    chunks: Tuple[int, ...]
    dtype: np.dtype
    fill: np.ndarray
    order: str = "C"
    separator: str = "."
    prefix: str = ""
    zarr_format: int = 2
    #: ``(codec id, configuration)`` in DECODE order — the reverse of the
    #: order they were applied in.
    codec_specs: Tuple[Tuple[str, Mapping[str, Any]], ...] = ()
    transpose: Optional[Tuple[int, ...]] = None
    codec_id: Optional[str] = None

    def decoders(self) -> Tuple[Callable[[bytes], bytes], ...]:
        """Resolve the codec chain, raising here if an extra is needed."""
        return tuple(_resolve_decoder(name, cfg)
                     for name, cfg in self.codec_specs)

    @classmethod
    def open(cls, path: Path) -> "_ZarrArray":
        """Read ``.zarray`` (v2) or ``zarr.json`` (v3) — metadata only."""
        meta_v2 = path / ".zarray"
        meta_v3 = path / "zarr.json"
        if meta_v2.is_file():
            return cls._from_v2(path, _read_json(meta_v2))
        if meta_v3.is_file():
            return cls._from_v3(path, _read_json(meta_v3))
        raise OmeZarrError(
            f"{path} is not a zarr array: neither .zarray (zarr-format 2, "
            f"which OME-NGFF 0.4 uses) nor zarr.json (zarr-format 3) is there.")

    @classmethod
    def _from_v2(cls, path: Path, meta: Mapping[str, Any]) -> "_ZarrArray":
        fmt = meta.get("zarr_format")
        if fmt != 2:
            raise OmeZarrError(
                f"{path}/.zarray declares zarr_format {fmt!r}; a .zarray is "
                f"zarr-format 2 by definition. spaCR reads formats "
                f"{SUPPORTED_ZARR_FORMATS}.")
        dtype = _dtype_from_zarr(meta.get("dtype"), f"{path}/.zarray")
        filters = meta.get("filters")
        if filters:
            raise OmeZarrError(
                f"{path}/.zarray declares a filter pipeline {filters!r}. "
                f"spaCR's pure-Python reader applies no filters; install the "
                f"extra (`python -m pip install \"spacr[zarr]\"`) to read it "
                f"through zarr.")
        compressor = meta.get("compressor")
        codec_id = None
        specs: Tuple[Tuple[str, Mapping[str, Any]], ...] = ()
        if compressor:
            if not isinstance(compressor, Mapping):
                raise OmeZarrError(
                    f"{path}/.zarray: compressor {compressor!r} is neither "
                    f"null (stored) nor an object with an \"id\"")
            codec_id = str(compressor.get("id", "")).lower()
            specs = ((codec_id, dict(compressor)),)
        order = str(meta.get("order", "C")).upper()
        if order not in ("C", "F"):
            raise OmeZarrError(f"{path}/.zarray: order {order!r} is not C or F")
        separator = str(meta.get("dimension_separator", "."))
        if separator not in (".", "/"):
            raise OmeZarrError(
                f"{path}/.zarray: dimension_separator {separator!r}; zarr "
                f"defines '.' and '/' only.")
        if meta.get("shape") is None or meta.get("chunks") is None:
            raise OmeZarrError(
                f"{path}/.zarray has no \"shape\" or no \"chunks\"; both are "
                f"required by zarr-format 2.")
        shape = tuple(int(v) for v in meta["shape"])
        chunks = tuple(int(v) for v in meta["chunks"])
        if len(chunks) != len(shape):
            raise OmeZarrError(
                f"{path}/.zarray: chunks {chunks} and shape {shape} differ in "
                f"rank")
        if any(c < 1 for c in chunks):
            raise OmeZarrError(
                f"{path}/.zarray: chunk shape {chunks} has a zero edge, so "
                f"the chunk grid is undefined")
        return cls(root=path, shape=shape, chunks=chunks, dtype=dtype,
                   fill=_fill_value(meta.get("fill_value"), dtype,
                                    f"{path}/.zarray"),
                   order=order, separator=separator, zarr_format=2,
                   codec_specs=specs, codec_id=codec_id)

    @classmethod
    def _from_v3(cls, path: Path, meta: Mapping[str, Any]) -> "_ZarrArray":
        fmt = meta.get("zarr_format")
        if fmt != 3:
            raise OmeZarrError(
                f"{path}/zarr.json declares zarr_format {fmt!r}; spaCR reads "
                f"{SUPPORTED_ZARR_FORMATS}.")
        if meta.get("node_type") not in (None, "array"):
            raise OmeZarrError(
                f"{path}/zarr.json is a {meta.get('node_type')!r}, not an array")
        dtype = _dtype_from_zarr(_V3_DTYPES.get(str(meta.get("data_type")),
                                                meta.get("data_type")),
                                 f"{path}/zarr.json")
        grid = meta.get("chunk_grid") or {}
        if grid.get("name") != "regular":
            raise OmeZarrError(
                f"{path}/zarr.json uses the {grid.get('name')!r} chunk grid; "
                f"spaCR's pure-Python reader handles \"regular\" only. "
                f"Install the extra to read it through zarr.")
        chunks = tuple(int(v) for v in grid["configuration"]["chunk_shape"])
        enc = meta.get("chunk_key_encoding") or {"name": "default"}
        enc_name = enc.get("name", "default")
        enc_cfg = enc.get("configuration") or {}
        if enc_name == "default":
            separator, prefix = str(enc_cfg.get("separator", "/")), "c"
        elif enc_name == "v2":
            separator, prefix = str(enc_cfg.get("separator", ".")), ""
        else:
            raise OmeZarrError(
                f"{path}/zarr.json uses chunk key encoding {enc_name!r}; "
                f"spaCR handles \"default\" and \"v2\".")
        specs, transpose, dtype = _v3_codec_chain(meta.get("codecs") or [],
                                                  path, dtype)
        return cls(root=path, shape=tuple(int(v) for v in meta["shape"]),
                   chunks=chunks, dtype=dtype,
                   fill=_fill_value(meta.get("fill_value"), dtype,
                                    f"{path}/zarr.json"),
                   order="C", separator=separator, prefix=prefix,
                   zarr_format=3, codec_specs=specs, transpose=transpose,
                   codec_id=specs[0][0] if specs else None)

    # -- reading --------------------------------------------------------
    def chunk_path(self, index: Sequence[int]) -> Path:
        """Where the chunk at grid position ``index`` is stored."""
        key = self.separator.join(str(int(i)) for i in index)
        if self.prefix:
            key = f"{self.prefix}{self.separator}{key}" if key else self.prefix
        return self.root.joinpath(*key.split("/")) if "/" in key \
            else self.root / key

    def _decode(self, raw: bytes, path: Path,
                decoders: Sequence[Callable[[bytes], bytes]]) -> np.ndarray:
        for decoder in decoders:
            raw = decoder(raw)
        stored = self.chunks if self.transpose is None else tuple(
            self.chunks[i] for i in self.transpose)
        expected = int(np.prod(stored, dtype=np.int64)) * self.dtype.itemsize
        if len(raw) != expected:
            raise OmeZarrError(
                f"chunk {path} decodes to {len(raw)} bytes; a {stored} chunk "
                f"of {self.dtype.str} is {expected}. Zarr stores every chunk "
                f"at full size, padded with fill_value, so this file is "
                f"either truncated or uses a filter spaCR did not apply.")
        block = np.frombuffer(raw, dtype=self.dtype).reshape(stored,
                                                             order=self.order)
        if self.transpose is not None:
            block = np.transpose(block, np.argsort(self.transpose))
        return block

    def read_region(self, box: Sequence[Tuple[int, int]]) -> np.ndarray:
        """Read the half-open index box ``[(start, stop), ...]``.

        Only the chunks the box intersects are opened; a chunk that is not
        stored costs one failed ``open`` and is filled with ``fill_value``.

        :param box: one ``(start, stop)`` per axis, already validated.
        :returns: an array of shape ``(stop - start, ...)`` in the array's own
            dtype, byte order included.
        """
        out_shape = tuple(b - a for a, b in box)
        out = np.empty(out_shape, dtype=self.dtype)
        grids = []
        for (start, stop), chunk in zip(box, self.chunks):
            if stop <= start:
                return out                       # empty selection, no I/O
            grids.append(range(start // chunk, (stop - 1) // chunk + 1))
        decoders = self.decoders()
        for index in itertools.product(*grids):
            dest, src = [], []
            for axis, (gi, (start, stop), chunk) in enumerate(
                    zip(index, box, self.chunks)):
                lo, hi = gi * chunk, min((gi + 1) * chunk, self.shape[axis])
                a, b = max(lo, start), min(hi, stop)
                dest.append(slice(a - start, b - start))
                src.append(slice(a - lo, b - lo))
            path = self.chunk_path(index)
            raw = _read_chunk_bytes(path)
            if raw is None:
                out[tuple(dest)] = self.fill
            else:
                out[tuple(dest)] = self._decode(raw, path, decoders)[tuple(src)]
        return out


#: zarr-format 3 ``data_type`` names -> numpy dtype strings. v3 spells the
#: dtype out and puts the byte order in the ``bytes`` codec instead.
_V3_DTYPES = {
    "bool": "|b1", "int8": "|i1", "uint8": "|u1",
    "int16": "<i2", "uint16": "<u2", "int32": "<i4", "uint32": "<u4",
    "int64": "<i8", "uint64": "<u8",
    "float16": "<f2", "float32": "<f4", "float64": "<f8",
    "complex64": "<c8", "complex128": "<c16",
}


def _strip_crc32c(raw: bytes) -> bytes:
    """Drop the four-byte checksum zarr v3's ``crc32c`` codec appends.

    Stripped, **not verified**: there is no crc32c in the standard library,
    and the alternative to stripping is refusing to read a file over a
    checksum spaCR has no way to compute. Said here rather than implied.
    """
    return raw[:-4]


def _v3_codec_chain(codecs: Sequence[Mapping[str, Any]], path: Path,
                    dtype: np.dtype
                    ) -> Tuple[Tuple[Tuple[str, Mapping[str, Any]], ...],
                               Optional[Tuple[int, ...]], np.dtype]:
    """Resolve a zarr v3 codec chain into codec specs, a transpose and a dtype.

    Stored bytes are the output of the last codec, so decoding runs backwards
    along the chain. ``bytes`` fixes the byte order (v3 puts it here rather
    than in the dtype, and reading it wrong yields plausible garbage rather
    than an error, so it is applied to the dtype and returned), ``transpose``
    fixes the element order, and ``crc32c`` appends a checksum that
    :func:`_strip_crc32c` removes.

    :returns: ``(specs, transpose, dtype)`` — specs in decode order, resolved
        to callables later so that opening the metadata never needs an extra.
    """
    specs: List[Tuple[str, Mapping[str, Any]]] = []
    transpose: Optional[Tuple[int, ...]] = None
    for spec in reversed(list(codecs)):
        name = str(spec.get("name", "")).lower()
        cfg = dict(spec.get("configuration") or {})
        if name == "bytes":
            endian = str(cfg.get("endian", "little"))
            if endian not in ("little", "big"):
                raise OmeZarrError(
                    f"{path}/zarr.json: bytes codec endian {endian!r}; zarr "
                    f"v3 defines \"little\" and \"big\".")
            if dtype.itemsize > 1:
                dtype = dtype.newbyteorder("<" if endian == "little" else ">")
            continue
        if name == "transpose":
            transpose = tuple(int(v) for v in cfg.get("order", ()))
            continue
        if name == "crc32c":
            specs.append((name, cfg))
            continue
        if name == "sharding_indexed":
            raise OmeZarrError(
                f"{path}/zarr.json uses sharding, which spaCR's pure-Python "
                f"reader does not implement. Install the extra "
                f"(`python -m pip install \"spacr[zarr]\"`) to read it "
                f"through zarr.")
        specs.append((name, cfg))
    return tuple(specs), transpose, dtype


def _resolve_decoder(name: str, config: Mapping[str, Any]
                     ) -> Callable[[bytes], bytes]:
    """One codec spec to one ``bytes -> bytes`` callable."""
    if name == "crc32c":
        return _strip_crc32c
    return require_codec(name, config)


def _read_json(path: Path) -> Dict[str, Any]:
    """Read one small JSON metadata file, or say which one was unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise OmeZarrError(f"{path} does not exist") from exc
    except json.JSONDecodeError as exc:
        raise OmeZarrError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise OmeZarrError(f"{path} holds {type(data).__name__}, not an object")
    return data


# ---------------------------------------------------------------------------
# The image
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OmeZarrImage:
    """An opened OME-Zarr multiscale image: metadata now, chunks on demand.

    Returned by :func:`read_ome_zarr` after reading only the group's JSON, so
    holding one of these says nothing about how much data has been read — the
    answer is none until :meth:`read` is called, and then only the chunks the
    region touches.

    :param path: the group directory.
    :param axes: every axis, in array order, with level-0 scale and
        translation on each.
    :param levels: the resolution pyramid, level 0 first.
    :param ngff_version: what the file says it is.
    :param name: the ``multiscales`` name, if any.
    :param channel_names: from the ``omero`` block when present — spaCR has
        channel names to fill in, so they are read and written rather than
        dropped.
    :param omero: the raw ``omero`` block, read-only, for the rendering
        settings this module does not interpret (colours, windows, rdefs).
    :param units_declared: ``False`` when the space axes carried no unit, so
        :attr:`spacing` is in pixels *because the file said nothing*, not
        because it said pixels. Never quietly upgraded to micrometers.
    :param multiscale: the raw ``multiscales`` entry, for anything here does
        not model.
    """

    path: str
    axes: Tuple[Axis, ...]
    levels: Tuple[Level, ...]
    ngff_version: str = "0.4"
    name: str = ""
    channel_names: Tuple[str, ...] = ()
    omero: Mapping[str, Any] = field(default_factory=dict)
    units_declared: bool = True
    multiscale: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", str(self.path))
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "channel_names",
                           tuple(str(c) for c in self.channel_names))
        object.__setattr__(self, "omero", MappingProxyType(dict(self.omero)))
        object.__setattr__(self, "multiscale",
                           MappingProxyType(dict(self.multiscale)))
        if not self.levels:
            raise OmeZarrError(
                f"{self.path}: the multiscales block lists no datasets, so "
                f"there is no image here.")
        for level in self.levels:
            if level.ndim != len(self.axes):
                raise OmeZarrError(
                    f"{self.path}: level {level.path!r} has {level.ndim} axes "
                    f"but the multiscales block declares "
                    f"{len(self.axes)} ({[a.name for a in self.axes]}). NGFF "
                    f"requires one axis entry per array dimension.")

    # -- axes -----------------------------------------------------------
    @property
    def axis_names(self) -> Tuple[str, ...]:
        """Axis names in array order, e.g. ``("t", "c", "z", "y", "x")``."""
        return tuple(a.name for a in self.axes)

    @property
    def space_axes(self) -> Tuple[Axis, ...]:
        """The spatial axes — the ones :attr:`spacing` is built from."""
        return tuple(a for a in self.axes if a.is_space)

    @property
    def other_axes(self) -> Tuple[Axis, ...]:
        """The non-spatial axes, reported beside the spacing rather than in it."""
        return tuple(a for a in self.axes if not a.is_space)

    @property
    def time_axis(self) -> Optional[Axis]:
        """The time axis, or ``None``. Its unit is seconds-like, never the
        spacing's."""
        for axis in self.axes:
            if axis.type == AXIS_TIME:
                return axis
        return None

    @property
    def channel_axis(self) -> Optional[Axis]:
        """The channel axis, or ``None``."""
        for axis in self.axes:
            if axis.type == AXIS_CHANNEL:
                return axis
        return None

    @property
    def spacing(self) -> Spacing:
        """Level 0's voxel size, over the SPACE axes only.

        :raises OmeZarrError: when the space axes disagree about units or one
            of them cannot be translated. See :func:`spacing_from_axes`.
        """
        return spacing_from_axes(self.axes)

    def spacing_at(self, level: Union[int, str] = 0) -> Spacing:
        """The voxel size of one level, over the space axes only.

        This is what makes a world coordinate mean the same thing at every
        level: the same world box resolves to the corresponding pixels of the
        overview and of full resolution.

        :param level: level index or dataset path.
        :returns: the spacing at that level.
        """
        lvl = self.level(level)
        space = [(i, a) for i, a in enumerate(self.axes) if a.is_space]
        base = self.spacing                       # validates units once
        return Spacing(scale=tuple(lvl.scale[i] for i, _ in space),
                       translate=tuple(lvl.translation[i] for i, _ in space),
                       axes=tuple(a.name for _, a in space),
                       units=base.units)

    # -- levels ---------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        """Level 0's shape."""
        return self.levels[0].shape

    @property
    def dtype(self) -> np.dtype:
        """Level 0's dtype, byte order included."""
        return np.dtype(self.levels[0].dtype)

    def level(self, level: Union[int, str] = 0) -> Level:
        """Resolve a level index or dataset path to a :class:`Level`.

        :param level: an index into :attr:`levels`, or a dataset ``path``.
        :returns: the level.
        :raises OmeZarrError: when there is no such level.
        """
        if isinstance(level, str):
            for candidate in self.levels:
                if candidate.path == level:
                    return candidate
            raise OmeZarrError(
                f"{self.path}: no level with path {level!r}; this image has "
                f"{[lv.path for lv in self.levels]}")
        try:
            return self.levels[int(level)]
        except (IndexError, ValueError, TypeError):
            raise OmeZarrError(
                f"{self.path}: no level {level!r}; this image has "
                f"{len(self.levels)} ({[lv.path for lv in self.levels]})"
            ) from None

    def level_for_size(self, size: int,
                       axes: Sequence[str] = ("y", "x")) -> int:
        """Index of the coarsest level still at least ``size`` along ``axes``.

        This is the multiscale payoff: drawing a 200 px thumbnail of a 40k x
        40k plate should decode a 300 x 300 level, not decimate the full one.
        The rule is "coarsest that does not need upsampling" — never return a
        level smaller than asked for while a bigger one exists, because
        upsampling to fill the request shows a blur where there is detail. If
        every level is smaller than ``size`` (a small image, a big request),
        the finest is returned.

        :param size: the wanted output size in pixels along ``axes``.
        :param axes: which axes the size refers to. Defaults to y and x.
        :returns: an index into :attr:`levels`.
        :raises OmeZarrError: when ``axes`` names an axis the image lacks.
        """
        size = int(size)
        idx = []
        for name in axes:
            if name not in self.axis_names:
                raise OmeZarrError(
                    f"{self.path}: no axis {name!r} in {self.axis_names}")
            idx.append(self.axis_names.index(name))
        best = 0
        for i, level in enumerate(self.levels):
            if all(level.shape[j] >= size for j in idx):
                best = i
        return best

    # -- reading --------------------------------------------------------
    def resolve_region(self, level: Union[int, str] = 0,
                       region: Any = None,
                       world_region: Optional[Mapping[str, Sequence[float]]] = None
                       ) -> Tuple[Tuple[int, int], ...]:
        """Turn a region request into a half-open index box, one per axis.

        :param level: which level the box is for. World boxes are resolved
            through *that level's* spacing, so the same world region names the
            matching pixels at every resolution.
        :param region: ``None`` for everything, a mapping of axis name to
            ``slice`` / ``(start, stop)`` / ``int``, or a sequence of those
            with one entry per axis. Index regions are refused when they fall
            outside the array — that is a typo, not a view of an edge.
        :param world_region: a mapping of SPACE axis name to ``(low, high)``
            in world units, resolved with :meth:`spacr.layers.Spacing.to_data`
            and clamped to the array. Clamping is right here and refusing is
            right above: a world box legitimately extends past the edge of a
            tile, an index box does not.
        :returns: ``((start, stop), ...)``, one per axis, in array order.
        :raises OmeZarrError: on an unknown axis, a reversed or out-of-range
            index box, or the same axis constrained both ways.
        """
        lvl = self.level(level)
        names = self.axis_names
        box: List[Tuple[int, int]] = [(0, n) for n in lvl.shape]
        touched: Dict[str, str] = {}

        entries: List[Tuple[int, Any]] = []
        if region is not None:
            if isinstance(region, Mapping):
                for key, value in region.items():
                    if key not in names:
                        raise OmeZarrError(
                            f"{self.path}: region names axis {key!r}; this "
                            f"image has {names}")
                    entries.append((names.index(str(key)), value))
                    touched[str(key)] = "region"
            else:
                items = list(region)
                if len(items) != len(names):
                    raise OmeZarrError(
                        f"{self.path}: region has {len(items)} entries for "
                        f"{len(names)} axes {names}. Pass one per axis, or a "
                        f"mapping keyed by axis name.")
                for i, value in enumerate(items):
                    if value is None:
                        continue
                    entries.append((i, value))
                    touched[names[i]] = "region"

        for i, value in entries:
            n = lvl.shape[i]
            if isinstance(value, slice):
                if value.step not in (None, 1):
                    raise OmeZarrError(
                        f"{self.path}: region on axis {names[i]!r} has step "
                        f"{value.step}. A strided read would decode every "
                        f"chunk it steps through anyway; read the box and "
                        f"stride the result, or use a coarser level.")
                start, stop, _ = value.indices(n)
            elif isinstance(value, (int, np.integer)):
                start = int(value)
                start = start + n if start < 0 else start
                stop = start + 1
            else:
                pair = tuple(value)
                if len(pair) != 2:
                    raise OmeZarrError(
                        f"{self.path}: region on axis {names[i]!r} is "
                        f"{value!r}; use a slice, an (start, stop) pair or an "
                        f"integer index.")
                start, stop = int(pair[0]), int(pair[1])
            if not 0 <= start <= stop <= n:
                raise OmeZarrError(
                    f"{self.path}: region {start}:{stop} on axis "
                    f"{names[i]!r} is outside the level's extent 0:{n}. Index "
                    f"regions are refused rather than clipped, because a box "
                    f"that runs off the end is usually a level mix-up: level "
                    f"{lvl.path!r} is {lvl.shape}, level 0 is "
                    f"{self.levels[0].shape}. Use world_region= to ask in "
                    f"world units, which is level-independent.")
            box[i] = (start, stop)

        if world_region:
            spacing = self.spacing_at(level)
            for key, bounds in world_region.items():
                key = str(key)
                if key in touched:
                    raise OmeZarrError(
                        f"{self.path}: axis {key!r} is constrained by both "
                        f"region= and world_region=. Pick one.")
                if not spacing.has_axis(key):
                    raise OmeZarrError(
                        f"{self.path}: world_region names axis {key!r}, which "
                        f"is not a space axis of this image "
                        f"({spacing.axes}). Non-spatial axes (t, c) have no "
                        f"world coordinate here — select them with region=.")
                low, high = (float(bounds[0]), float(bounds[1]))
                if high < low:
                    low, high = high, low
                j = spacing.axis_index(key)
                i = names.index(key)
                n = lvl.shape[i]
                step = spacing.scale[j]
                a = (low - spacing.translate[j]) / step
                b = (high - spacing.translate[j]) / step
                if step < 0:
                    a, b = b, a
                start = max(0, min(n, int(math.floor(a))))
                stop = max(start, min(n, int(math.ceil(b))))
                box[i] = (start, stop)
        return tuple(box)

    def read(self, level: Union[int, str] = 0, region: Any = None, *,
             world_region: Optional[Mapping[str, Sequence[float]]] = None,
             prefer_zarr: bool = True) -> np.ndarray:
        """Read one level, or a region of it, as a numpy array.

        Only the chunks the region intersects are opened. On a 100 GB plate
        that is the difference between a tile and the plate.

        :param level: level index or dataset path. Level 0 is full resolution.
        :param region: see :meth:`resolve_region`.
        :param world_region: a ``{axis: (low, high)}`` box in world units,
            resolved through the level's own spacing.
        :param prefer_zarr: use :mod:`zarr` when it is installed. That is the
            intended path — zarr handles sharding, filters and every codec —
            and the pure-Python reader is the fallback for the common case.
            Pass ``False`` to force the fallback, which is what the tests do.
        :returns: the region, in the file's own dtype and byte order.
        :raises OmeZarrError: on an impossible region, and
            :class:`ZarrExtraMissing` on a codec that needs the extra.
        """
        lvl = self.level(level)
        box = self.resolve_region(level, region, world_region)
        store = Path(self.path) / lvl.path
        if prefer_zarr and _zarr_is_installed():
            return _read_with_zarr(store, box)
        # Re-reading the level's `.zarray` here rather than caching a handle
        # on the image keeps this dataclass frozen and picklable — it can
        # cross into a worker process — at the cost of one small JSON read per
        # call, against however many chunk decodes follow it.
        return _ZarrArray.open(store).read_region(box)

    # -- reporting ------------------------------------------------------
    def describe(self) -> str:
        """A short report: version, axes, spacing and every level.

        The spacing line says ``pixel units (file declares none)`` when that is
        what happened, because "0.65" with no unit beside it has been read as
        micrometers before.
        """
        head = (f"{os.path.basename(self.path) or self.path} — OME-NGFF "
                f"{self.ngff_version}, {len(self.levels)} level"
                f"{'s' if len(self.levels) != 1 else ''}, axes "
                f"{', '.join(self.axis_names)}")
        if self.name:
            head += f" ({self.name})"
        lines = [head]
        try:
            spacing = self.spacing
            note = "" if self.units_declared else "   [file declares no unit]"
            lines.append(f"  spacing: {spacing.describe()}{note}")
        except OmeZarrError as exc:
            lines.append(f"  spacing: unavailable — {exc}")
        for axis in self.other_axes:
            lines.append(f"  {axis.describe()}")
        if self.channel_names:
            lines.append(f"  channels: {', '.join(self.channel_names)}")
        for lvl in self.levels:
            lines.append(f"  level {lvl.describe()}")
        return "\n".join(lines)


def _zarr_is_installed() -> bool:
    """Whether :mod:`zarr` can be imported, without raising if it cannot."""
    try:
        import zarr  # noqa: F401
    except ImportError:
        return False
    return True


def _read_with_zarr(store: Path, box: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Read a box through :mod:`zarr` — the preferred path when it is there.

    :param store: the array directory.
    :param box: ``((start, stop), ...)`` per axis.
    :returns: the region as a numpy array.
    """
    zarr = require_zarr()
    array = zarr.open(str(store), mode="r")
    return np.asarray(array[tuple(slice(a, b) for a, b in box)])


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _compose(group_transforms: Sequence[Mapping[str, Any]],
             dataset_transforms: Sequence[Mapping[str, Any]],
             ndim: int, where: str) -> Tuple[Tuple[float, ...],
                                             Tuple[float, ...]]:
    """Compose a level's coordinate transformations into (scale, translation).

    NGFF applies the dataset's transformations first and the multiscale's
    afterwards, so for the scale/translation pair that means::

        world = ms_scale * (ds_scale * index + ds_translation) + ms_translation

    which is ``scale = ms_scale * ds_scale`` and
    ``translation = ms_scale * ds_translation + ms_translation``. Dropping the
    group-level transformation — the usual shortcut, because most files do not
    have one — silently ignores the plate-wide voxel size in the files that do.
    """
    def _pair(transforms: Sequence[Mapping[str, Any]]
              ) -> Tuple[List[float], List[float]]:
        scale = [1.0] * ndim
        translation = [0.0] * ndim
        for entry in transforms or ():
            if not isinstance(entry, Mapping):
                raise OmeZarrError(
                    f"{where}: coordinateTransformations entry {entry!r} is "
                    f"not an object")
            kind = entry.get("type")
            if kind == "scale":
                values = entry.get("scale")
                if values is None or len(values) != ndim:
                    raise OmeZarrError(
                        f"{where}: a scale transformation needs one value per "
                        f"axis ({ndim}); got {values!r}")
                scale = [s * float(v) for s, v in zip(scale, values)]
            elif kind == "translation":
                values = entry.get("translation")
                if values is None or len(values) != ndim:
                    raise OmeZarrError(
                        f"{where}: a translation transformation needs one "
                        f"value per axis ({ndim}); got {values!r}")
                translation = [t + float(v) for t, v in zip(translation, values)]
            elif kind == "identity":
                continue
            else:
                raise OmeZarrError(
                    f"{where}: coordinateTransformations type {kind!r}. NGFF "
                    f"0.4 permits \"identity\", \"scale\" and \"translation\" "
                    f"in multiscales; anything else needs the transformation "
                    f"applied before spaCR sees it.")
        return scale, translation

    ms_scale, ms_translation = _pair(group_transforms)
    ds_scale, ds_translation = _pair(dataset_transforms)
    scale = tuple(m * d for m, d in zip(ms_scale, ds_scale))
    translation = tuple(m * d + t for m, d, t
                        in zip(ms_scale, ds_translation, ms_translation))
    return scale, translation


def _group_attributes(root: Path) -> Tuple[Dict[str, Any], int]:
    """Read a group's attributes and its zarr format, from whichever file has them."""
    zgroup, zjson = root / ".zgroup", root / "zarr.json"
    if zgroup.is_file():
        meta = _read_json(zgroup)
        fmt = meta.get("zarr_format")
        if fmt != 2:
            raise OmeZarrError(
                f"{zgroup} declares zarr_format {fmt!r}; a .zgroup is "
                f"zarr-format 2 by definition. spaCR reads formats "
                f"{SUPPORTED_ZARR_FORMATS}.")
        attrs = _read_json(root / ".zattrs") if (root / ".zattrs").is_file() else {}
        return attrs, 2
    if zjson.is_file():
        meta = _read_json(zjson)
        fmt = meta.get("zarr_format")
        if fmt not in SUPPORTED_ZARR_FORMATS:
            raise OmeZarrError(
                f"{zjson} declares zarr_format {fmt!r}; spaCR reads "
                f"{SUPPORTED_ZARR_FORMATS}. OME-NGFF 0.4 is zarr-format 2 and "
                f"0.5 is zarr-format 3.")
        if meta.get("node_type") not in (None, "group"):
            raise OmeZarrError(
                f"{zjson} is a {meta.get('node_type')!r}, not a group. Point "
                f"spaCR at the multiscale group, not at one of its arrays.")
        return dict(meta.get("attributes") or {}), int(fmt)
    raise OmeZarrError(
        f"{root} is not a zarr group: it has neither .zgroup (zarr-format 2, "
        f"which OME-NGFF 0.4 uses) nor zarr.json (zarr-format 3). Point spaCR "
        f"at the group holding the `multiscales` metadata — for a plate that "
        f"is <plate>.zarr/<row>/<column>/<field>, not <plate>.zarr itself.")


def read_ome_zarr(path: Union[str, os.PathLike], *,
                  multiscale_index: int = 0) -> OmeZarrImage:
    """Open an OME-Zarr group and read its metadata. **No chunk is touched.**

    Everything this reads is small JSON: the group's ``.zattrs`` (or the
    ``attributes`` of its ``zarr.json``) and one ``.zarray`` per level. So
    asking a 100 GB plate what its levels, shapes, voxel size and channels are
    costs a few kilobytes, which is the property the whole format exists for.

    :param path: the group directory — the one holding ``multiscales``. For a
        plate that is ``<plate>.zarr/<row>/<column>/<field>``.
    :param multiscale_index: which ``multiscales`` entry to read. NGFF permits
        several; spaCR reads the first by default and says so here rather than
        pretending there can only be one.
    :returns: the opened :class:`OmeZarrImage`.
    :raises OmeZarrError: when the directory is not a zarr group, has no
        ``multiscales``, declares a zarr format spaCR does not read, or
        carries axis units spaCR will not translate.
    """
    root = Path(path)
    if not root.exists():
        raise OmeZarrError(f"{root} does not exist")
    if not root.is_dir():
        raise OmeZarrError(
            f"{root} is a file. An OME-Zarr group is a directory (or a "
            f"directory-like store); a single .zarr file is not something "
            f"this format defines.")

    attrs, zarr_format = _group_attributes(root)
    ome = attrs.get("ome") if isinstance(attrs.get("ome"), Mapping) else {}
    multiscales = attrs.get("multiscales") or ome.get("multiscales")
    if not multiscales:
        raise OmeZarrError(
            f"{root} is a zarr group but carries no `multiscales` metadata, "
            f"so it is not an OME-NGFF image. Keys present: "
            f"{sorted(attrs) or 'none'}. A plate or a well group holds its "
            f"images one or two levels down — try "
            f"{root}/<row>/<column>/<field>.")
    if not isinstance(multiscales, list):
        raise OmeZarrError(
            f"{root}: `multiscales` is {type(multiscales).__name__}, not the "
            f"list NGFF requires")
    try:
        entry = multiscales[multiscale_index]
    except IndexError:
        raise OmeZarrError(
            f"{root}: multiscale_index {multiscale_index} but the file has "
            f"{len(multiscales)}") from None
    if not isinstance(entry, Mapping):
        raise OmeZarrError(f"{root}: multiscales[{multiscale_index}] is not an object")

    version = str(entry.get("version") or ome.get("version")
                  or ("0.5" if zarr_format == 3 else "0.4"))
    datasets = entry.get("datasets")
    if not datasets:
        raise OmeZarrError(
            f"{root}: multiscales[{multiscale_index}] lists no `datasets`, so "
            f"nothing says where the arrays are.")

    # Levels first: their rank is what an absent `axes` has to be inferred from.
    arrays = []
    for i, dataset in enumerate(datasets):
        if not isinstance(dataset, Mapping) or not dataset.get("path"):
            raise OmeZarrError(
                f"{root}: datasets[{i}] has no `path`; NGFF requires it")
        sub = root / str(dataset["path"])
        arrays.append((str(dataset["path"]), dataset, _ZarrArray.open(sub)))

    ndim = len(arrays[0][2].shape)
    raw_axes = entry.get("axes")
    if raw_axes:
        axes = [_axis_from_ngff(a, i) for i, a in enumerate(raw_axes)]
    else:
        # 0.1-0.3 had no `axes`. The canonical tczyx tail is what those files
        # meant, and inferring it beats refusing to open old data.
        axes = [Axis(name=n, type=_TYPE_BY_AXIS_NAME.get(n, AXIS_SPACE))
                for n in CANONICAL_AXIS_ORDER[len(CANONICAL_AXIS_ORDER) - ndim:]]
    if len(axes) != ndim:
        raise OmeZarrError(
            f"{root}: `axes` lists {len(axes)} axes "
            f"({[a.name for a in axes]}) but dataset "
            f"{arrays[0][0]!r} is {ndim}-dimensional {arrays[0][2].shape}.")

    group_transforms = entry.get("coordinateTransformations") or []
    levels: List[Level] = []
    for name, dataset, array in arrays:
        if len(array.shape) != ndim:
            raise OmeZarrError(
                f"{root}: dataset {name!r} is {len(array.shape)}-dimensional "
                f"but dataset {arrays[0][0]!r} is {ndim}-dimensional. Every "
                f"level of a multiscale image describes the same data.")
        scale, translation = _compose(
            group_transforms, dataset.get("coordinateTransformations") or [],
            ndim, f"{root}/{name}")
        levels.append(Level(path=name, shape=array.shape, chunks=array.chunks,
                            dtype=array.dtype.str, scale=scale,
                            translation=translation,
                            zarr_format=array.zarr_format,
                            compressor=array.codec_id))

    base = levels[0]
    axes = [Axis(name=a.name, type=a.type, unit=a.unit, scale=base.scale[i],
                 translate=base.translation[i]) for i, a in enumerate(axes)]
    # Validate the units HERE, not on first use of `.spacing`. The code path
    # that reads shapes and never asks for a spacing is exactly the one that
    # would carry an untranslatable unit all the way into a measurement
    # without anything raising, so the refusal has to happen at the door.
    spacing_from_axes(axes)

    omero = attrs.get("omero") or ome.get("omero") or {}
    if not isinstance(omero, Mapping):
        omero = {}
    channel_names: Tuple[str, ...] = ()
    if isinstance(omero.get("channels"), list):
        channel_names = tuple(
            str(c.get("label") or c.get("name") or f"channel_{i}")
            for i, c in enumerate(omero["channels"])
            if isinstance(c, Mapping))

    declared = any(a.unit for a in axes if a.is_space)
    return OmeZarrImage(path=str(root), axes=tuple(axes), levels=tuple(levels),
                        ngff_version=version, name=str(entry.get("name") or ""),
                        channel_names=channel_names, omero=dict(omero),
                        units_declared=bool(declared), multiscale=dict(entry))


def read_ome_zarr_array(path: Union[str, os.PathLike],
                        level: Union[int, str] = 0, region: Any = None, *,
                        world_region: Optional[Mapping[str, Sequence[float]]] = None,
                        prefer_zarr: bool = True) -> np.ndarray:
    """Open an OME-Zarr and read one level (or a region of it) in one call.

    The convenience form of ``read_ome_zarr(path).read(...)``, for when the
    metadata is not wanted afterwards.

    :param path: the group directory.
    :param level: level index or dataset path.
    :param region: an index box — see :meth:`OmeZarrImage.resolve_region`.
    :param world_region: a world-coordinate box, ditto.
    :param prefer_zarr: delegate to :mod:`zarr` when installed.
    :returns: the array, in the file's own dtype.
    """
    return read_ome_zarr(path).read(level, region, world_region=world_region,
                                    prefer_zarr=prefer_zarr)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

#: Default chunk edge for the two fastest-varying spatial axes. 256 x 256 of
#: uint16 is 128 KiB, and with a z chunk of :data:`DEFAULT_Z_CHUNK` a chunk is
#: 2 MiB — inside the few-hundred-KiB-to-few-MiB band that object stores and
#: zarr's own guidance both want, and small enough that a screenful of a plate
#: overview is a handful of reads.
DEFAULT_TILE = 256

#: Default chunk depth along a third spatial axis. Not 1: an orthogonal
#: reslice through a z-stack chunked one plane at a time reads every chunk in
#: the stack. Not the whole axis either, since a single-plane view would then
#: decode the lot.
DEFAULT_Z_CHUNK = 16

#: Colours cycled through when writing ``omero`` channel metadata and the
#: caller named no colours. Hex RGB, the format the ``omero`` block wants.
DEFAULT_CHANNEL_COLORS: Tuple[str, ...] = (
    "00FF00", "FF00FF", "0000FF", "FFFF00", "FF0000", "00FFFF", "FFFFFF",
)


def _default_chunks(shape: Sequence[int], axes: Sequence[Axis]
                    ) -> Tuple[int, ...]:
    """Pick a chunk shape: 1 per t/c, 256 on y/x, 16 on any other space axis."""
    space = [i for i, a in enumerate(axes) if a.is_space]
    fastest = set(space[-2:])
    chunks = []
    for i, (n, axis) in enumerate(zip(shape, axes)):
        if not axis.is_space:
            chunks.append(1)
        elif i in fastest:
            chunks.append(min(int(n), DEFAULT_TILE))
        else:
            chunks.append(min(int(n), DEFAULT_Z_CHUNK))
    return tuple(max(1, int(c)) for c in chunks)


def _downsample_axis(array: np.ndarray, axis: int, method: str) -> np.ndarray:
    """Halve one axis. ``ceil(n / 2)`` elements out, so nothing is cropped."""
    n = array.shape[axis]
    if n < 2:
        return array
    if method == "stride":
        return array[tuple(slice(None) if i != axis else slice(None, None, 2)
                           for i in range(array.ndim))]
    # Captured before anything else touches the array: np.concatenate below
    # returns NATIVE byte order, so `array.dtype` at the end of this function
    # is not the dtype that came in. A silently byte-swapped pyramid level
    # reads back as different numbers, not as an error.
    dtype = array.dtype
    half = (n + 1) // 2
    if n % 2:
        # Duplicate the last element so the tail block is a full pair. Its
        # mean is that element, which is exactly the partial-block mean — the
        # padding is arithmetic bookkeeping, not an invented sample.
        edge = array[tuple(slice(None) if i != axis else slice(n - 1, n)
                           for i in range(array.ndim))]
        array = np.concatenate([array, edge], axis=axis)
    shape = list(array.shape)
    shape[axis:axis + 1] = [half, 2]
    block = array.reshape(shape).mean(axis=axis + 1)
    if np.issubdtype(dtype, np.integer) or dtype == np.bool_:
        return np.rint(block).astype(dtype)
    return block.astype(dtype)


def _pyramid(array: np.ndarray, axes: Sequence[Axis], levels: int,
             method: str, downsample_axes: Sequence[int]) -> List[np.ndarray]:
    """Build the resolution pyramid, level 0 first."""
    out = [array]
    for _ in range(1, levels):
        current = out[-1]
        for axis in downsample_axes:
            current = _downsample_axis(current, axis, method)
        if current.shape == out[-1].shape:
            break        # every downsampled axis is already 1; stop early
        out.append(current)
    return out


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one metadata file, formatted so a human can diff it."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _json_fill(value: Any, dtype: np.dtype) -> Any:
    """JSON form of a fill value: NaN and the infinities become their names."""
    if dtype.kind == "b":
        return bool(value)
    if dtype.kind in "iu":
        return int(value)
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    return number


def _write_zarr_v2_array(root: Path, array: np.ndarray,
                         chunks: Sequence[int], compressor: Optional[str],
                         level: int, order: str, separator: str,
                         fill_value: Any, write_empty_chunks: bool) -> None:
    """Write one zarr-format-2 array: its ``.zarray`` and its chunks."""
    root.mkdir(parents=True, exist_ok=True)
    block, encode = _encoder(compressor, level)
    dtype = array.dtype
    fill = np.array(0 if fill_value is None else fill_value, dtype=dtype)
    _write_json(root / ".zarray", {
        "zarr_format": 2,
        "shape": [int(v) for v in array.shape],
        "chunks": [int(v) for v in chunks],
        "dtype": dtype.str,
        "compressor": block,
        "fill_value": _json_fill(fill, dtype),
        "order": order,
        "filters": None,
        "dimension_separator": separator,
    })
    grids = [range(-(-n // c)) for n, c in zip(array.shape, chunks)]
    for index in itertools.product(*grids):
        slabs = tuple(slice(i * c, min((i + 1) * c, n))
                      for i, c, n in zip(index, chunks, array.shape))
        slab = array[slabs]
        if not write_empty_chunks and slab.size and bool(np.all(slab == fill)):
            continue          # an unwritten chunk reads back as fill_value
        if slab.shape != tuple(chunks):
            padded = np.full(tuple(chunks), fill, dtype=dtype)
            padded[tuple(slice(0, s) for s in slab.shape)] = slab
            slab = padded
        buffer = np.asfortranarray(slab) if order == "F" \
            else np.ascontiguousarray(slab)
        key = separator.join(str(i) for i in index)
        target = root.joinpath(*key.split("/")) if separator == "/" \
            else root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            handle.write(encode(buffer.tobytes(order=order)))


def write_ome_zarr(path: Union[str, os.PathLike], array: Any, *,
                   spacing: Optional[Spacing] = None,
                   axes: Optional[Sequence[Union[str, Axis]]] = None,
                   name: Optional[str] = None,
                   levels: int = 1,
                   downsample: str = "mean",
                   downsample_axes: Optional[Sequence[str]] = None,
                   chunks: Optional[Sequence[int]] = None,
                   compressor: Optional[str] = "zlib",
                   compression_level: int = 5,
                   order: str = "C",
                   dimension_separator: str = "/",
                   fill_value: Any = 0,
                   write_empty_chunks: bool = False,
                   channel_names: Optional[Sequence[str]] = None,
                   channel_colors: Optional[Sequence[str]] = None,
                   overwrite: bool = False,
                   ngff_version: str = "0.4") -> OmeZarrImage:
    """Write an array as an OME-NGFF 0.4 multiscale image.

    The defaults are chosen so the result is readable by anything: zarr-format
    2, ``zlib`` chunks (which need no extra), ``/`` separators, and the
    ``axes``/``coordinateTransformations`` metadata filled in from the
    spacing rather than left at 1.0.

    **The pyramid.** ``levels=1`` writes level 0 alone; more builds each level
    by halving the two fastest-varying space axes of the one above with a 2x
    **block mean** (``downsample="mean"``) or by **striding**
    (``downsample="stride"``). Neither is a Gaussian pyramid — there is no
    pre-filter beyond the box — and the block mean is the default because
    striding aliases: a one-pixel-wide bright structure survives or vanishes
    with the parity of its coordinate, so the same object appears and
    disappears as a user zooms. **Label and mask arrays must use
    ``downsample="stride"``**: the mean of labels 3 and 5 is 4, which is a
    different object, and averaging a boolean mask invents half-membership.
    Levels shrink by ``ceil(n / 2)``, so nothing is cropped off an odd edge.

    **The transformations.** Level *k*'s ``scale`` is level 0's multiplied by
    2^k on each downsampled axis — writing level 0's scale on every level is
    the most common NGFF bug there is, and it renders as coarse levels that
    are a quarter the size of the fine ones and slide off the corner as you
    zoom. The ``translation`` moves too, and differently per method: a block
    mean centres its first output element half a level-0 pixel inside the
    edge, so level *k* gets ``translate_0 + scale_0 * (2^k - 1) / 2``, while a
    strided level samples element 0 exactly and keeps ``translate_0``.

    **The chunks.** Default is 1 along t and c (a viewer draws one channel of
    one timepoint), :data:`DEFAULT_TILE` on y and x, and
    :data:`DEFAULT_Z_CHUNK` on any other space axis — about 2 MiB per chunk
    for uint16. Pass ``chunks=`` to override.

    :param path: the group directory to create.
    :param array: the image data, 2 to 5 dimensional.
    :param spacing: the voxel size, over the SPACE axes. Defaults to an
        isotropic pixel grid, which writes axes with no ``unit`` — legal NGFF
        and honestly what is known.
    :param axes: axis names, or :class:`Axis` objects for full control of the
        non-spatial axes (a time step, say). Derived from ``spacing`` and the
        array's rank when omitted.
    :param name: the ``multiscales`` name. Defaults to the directory name.
    :param levels: how many resolution levels to write.
    :param downsample: ``"mean"`` or ``"stride"``.
    :param downsample_axes: which axes to halve. Defaults to the two
        fastest-varying space axes — halving z as well ruins an anisotropic
        stack, where 12 planes at 2 µm become 2 planes at 16 µm by level 3.
    :param chunks: chunk shape, one per axis.
    :param compressor: ``"zlib"`` (default), ``"gzip"``, ``"bz2"``, ``"lzma"``,
        ``None`` for stored, or any codec :mod:`numcodecs` provides. The
        default needs no extra by design.
    :param compression_level: for the codecs that take one.
    :param order: ``"C"`` or ``"F"`` chunk memory order.
    :param dimension_separator: ``"/"`` (nested directories, the default, and
        much kinder to filesystems at 10^5 chunks) or ``"."`` (flat).
    :param fill_value: what an unwritten chunk reads back as.
    :param write_empty_chunks: write chunks that are entirely ``fill_value``.
        ``False`` (the default) omits them, which is how a sparse mask stops
        costing what a dense one does.
    :param channel_names: channel labels, written into the ``omero`` block
        along with per-channel display windows measured from the data.
    :param channel_colors: hex RGB per channel; defaults to
        :data:`DEFAULT_CHANNEL_COLORS`.
    :param overwrite: replace an existing group. Without it, an existing group
        is refused rather than merged — a pyramid written over another
        pyramid's levels leaves the leftover deeper levels of the old one in
        place, and they read back as part of the new image.
    :param ngff_version: the version to declare. 0.4 is what this writes.
    :returns: the group, reopened with :func:`read_ome_zarr` — so the return
        value is a *read* of what was actually written, not a description of
        what was intended.
    :raises OmeZarrError: on an unwritable axis layout, a unit with no NGFF
        name, or an existing group without ``overwrite=True``.
    """
    data = np.asarray(array)
    if data.ndim < 2 or data.ndim > 5:
        raise OmeZarrError(
            f"OME-NGFF images are 2 to 5 dimensional; this array is "
            f"{data.ndim}-dimensional {data.shape}.")
    if data.dtype.kind not in "biufc":
        raise OmeZarrError(
            f"cannot write dtype {data.dtype!r} as an OME-Zarr; NGFF images "
            f"hold numeric or boolean data.")
    if str(ngff_version) != "0.4":
        raise OmeZarrError(
            f"spaCR writes OME-NGFF 0.4 (zarr-format 2), not "
            f"{ngff_version!r}. 0.4 is what essentially every reader "
            f"supports; spaCR *reads* {SUPPORTED_NGFF_VERSIONS}.")
    method = str(downsample).lower()
    if method not in ("mean", "stride"):
        raise OmeZarrError(
            f"downsample={downsample!r}; use \"mean\" (block mean, for "
            f"intensity images) or \"stride\" (nearest, and the only correct "
            f"one for label/mask arrays).")
    if int(levels) < 1:
        raise OmeZarrError(f"levels={levels}; a pyramid has at least level 0")

    n_space = min(data.ndim, 3)
    if spacing is None:
        spacing = Spacing.isotropic(n_space, 1.0, units=PIXEL_UNITS)
    if not isinstance(spacing, Spacing):
        raise OmeZarrError(
            f"spacing must be a spacr.layers.Spacing, got "
            f"{type(spacing).__name__}. Build one with "
            f"Spacing.from_map({{'z': 2.0, 'y': 0.65, 'x': 0.65}}, "
            f"units='um').")

    if axes is not None and all(isinstance(a, Axis) for a in axes):
        axis_list = tuple(axes)                    # type: ignore[arg-type]
        if len(axis_list) != data.ndim:
            raise OmeZarrError(
                f"{len(axis_list)} axes for a {data.ndim}-dimensional array")
    else:
        names = None if axes is None else [str(a) for a in axes]
        axis_list = axes_from_spacing(spacing, data.ndim, names)

    _validate_ngff_axes(axis_list, data.shape)

    space_positions = [i for i, a in enumerate(axis_list) if a.is_space]
    if downsample_axes is None:
        halved = space_positions[-2:]
    else:
        names = tuple(a.name for a in axis_list)
        halved = []
        for wanted in downsample_axes:
            if str(wanted) not in names:
                raise OmeZarrError(
                    f"downsample_axes names {wanted!r}; the array's axes are "
                    f"{names}")
            halved.append(names.index(str(wanted)))

    target = Path(path)
    _prepare_group_dir(target, overwrite)

    pyramid = _pyramid(data, axis_list, int(levels), method, halved)
    chunk_shape = tuple(int(c) for c in chunks) if chunks is not None \
        else _default_chunks(data.shape, axis_list)
    if len(chunk_shape) != data.ndim:
        raise OmeZarrError(
            f"chunks {chunk_shape} has {len(chunk_shape)} entries for a "
            f"{data.ndim}-dimensional array")

    datasets: List[Dict[str, Any]] = []
    for k, level_array in enumerate(pyramid):
        level_chunks = tuple(min(c, s) if s else 1
                             for c, s in zip(chunk_shape, level_array.shape))
        _write_zarr_v2_array(
            target / str(k), level_array, level_chunks, compressor,
            int(compression_level), str(order).upper(),
            str(dimension_separator), fill_value, bool(write_empty_chunks))
        factor = [2.0 ** k if i in halved else 1.0 for i in range(data.ndim)]
        scale = [a.scale * f for a, f in zip(axis_list, factor)]
        # Block mean: element 0 of level k covers level-0 elements [0, 2^k),
        # whose centre is (2^k - 1)/2. Stride: element 0 IS level-0 element 0.
        shift = 0.0 if method == "stride" else 1.0
        # scale * ((f - 1) / 2) rather than scale * (f - 1) / 2: (f - 1) / 2
        # is exact in binary for a power-of-two f, so this is one rounding
        # instead of two.
        translation = [a.translate + a.scale * ((f - 1.0) / 2.0 * shift)
                       for a, f in zip(axis_list, factor)]
        datasets.append({
            "path": str(k),
            "coordinateTransformations": [
                {"type": "scale", "scale": [float(v) for v in scale]},
                {"type": "translation",
                 "translation": [float(v) for v in translation]},
            ],
        })

    multiscale: Dict[str, Any] = {
        "version": "0.4",
        "name": str(name if name is not None else target.name),
        "axes": [a.to_ngff() for a in axis_list],
        "datasets": datasets,
        # 0.4 defines `type` and `metadata` on a multiscale for exactly this:
        # saying how the pyramid was built, so a reader knows whether the
        # coarse levels can be trusted for a measurement (they cannot).
        "type": "local mean" if method == "mean" else "nearest (stride)",
        "metadata": {
            "method": "spacr.ome_zarr.write_ome_zarr",
            "version": _spacr_version(),
            "description": (
                "2x block mean over the fastest-varying space axes"
                if method == "mean" else
                "2x striding (nearest); label-safe, aliases intensity"),
            "downsample_axes": [axis_list[i].name for i in halved],
        },
    }
    attrs: Dict[str, Any] = {"multiscales": [multiscale]}
    omero = _omero_block(data, axis_list, channel_names, channel_colors,
                         multiscale["name"])
    if omero:
        attrs["omero"] = omero

    _write_json(target / ".zgroup", {"zarr_format": 2})
    _write_json(target / ".zattrs", attrs)
    return read_ome_zarr(target)


def _spacr_version() -> str:
    """spaCR's version string, or ``"unknown"`` if it cannot be determined.

    Recorded in the file's ``multiscales.metadata`` so that a pyramid whose
    downsampling rule changes one day can be told from one written before.
    Never allowed to fail a write: a version string is documentation.
    """
    try:
        from .version import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _validate_ngff_axes(axes: Sequence[Axis], shape: Sequence[int]) -> None:
    """Refuse an axis layout NGFF 0.4 does not permit, saying what to fix."""
    names = [a.name for a in axes]
    if len(set(names)) != len(names):
        raise OmeZarrError(f"axis names must be unique, got {names}")
    if len(names) != len(shape):
        raise OmeZarrError(
            f"{len(names)} axes {names} for a {len(shape)}-dimensional array "
            f"{tuple(shape)}")
    kinds = [a.type for a in axes]
    space = [i for i, k in enumerate(kinds) if k == AXIS_SPACE]
    if len(space) not in (2, 3):
        raise OmeZarrError(
            f"NGFF 0.4 wants 2 or 3 space axes; these axes have "
            f"{len(space)} ({[names[i] for i in space]}).")
    if space != list(range(len(names) - len(space), len(names))):
        raise OmeZarrError(
            f"the space axes must come last: NGFF 0.4 requires the order "
            f"time, channel, space. Got {list(zip(names, kinds))}.")
    order = {AXIS_TIME: 0, AXIS_CHANNEL: 1, AXIS_SPACE: 2}
    ranks = [order.get(k, 1) for k in kinds]
    if ranks != sorted(ranks):
        raise OmeZarrError(
            f"axes must be ordered time, then channel, then space. Got "
            f"{list(zip(names, kinds))}. Transpose the array (np.moveaxis) "
            f"rather than relabelling the axes.")


def _prepare_group_dir(target: Path, overwrite: bool) -> None:
    """Create the group directory, refusing to clobber anything by accident."""
    if target.exists() and not target.is_dir():
        raise OmeZarrError(
            f"{target} exists and is a file; an OME-Zarr group is a directory")
    looks_like_group = (target / ".zgroup").is_file() or \
        (target / "zarr.json").is_file() or (target / ".zattrs").is_file()
    if looks_like_group and not overwrite:
        raise OmeZarrError(
            f"{target} is already a zarr group. Pass overwrite=True to "
            f"replace it. spaCR refuses by default because writing a pyramid "
            f"into an existing one does not replace it: the old group's "
            f"deeper levels stay on disk and are read back as levels of the "
            f"new image, at the old image's scale.")
    if looks_like_group:
        shutil.rmtree(target)
    elif target.exists() and any(target.iterdir()) and not overwrite:
        raise OmeZarrError(
            f"{target} exists and is not empty, and is not a zarr group "
            f"either. Pass overwrite=True only if you meant this path — "
            f"spaCR will not delete a directory it did not recognise.")
    target.mkdir(parents=True, exist_ok=True)


def _omero_block(data: np.ndarray, axes: Sequence[Axis],
                 channel_names: Optional[Sequence[str]],
                 channel_colors: Optional[Sequence[str]],
                 name: str) -> Dict[str, Any]:
    """Build the ``omero`` rendering block, or ``{}`` when there is nothing to say.

    Written only when the caller supplied channel names — spaCR has them, and
    dropping them turns a two-channel image into "channel 0" and "channel 1"
    in every viewer that opens it. The display window is measured from the
    data rather than assumed from the dtype, because a uint16 image with a
    0-4000 range renders as black under a 0-65535 window.
    """
    if not channel_names:
        return {}
    index = next((i for i, a in enumerate(axes) if a.type == AXIS_CHANNEL),
                 None)
    if index is None:
        raise OmeZarrError(
            f"channel_names={list(channel_names)} but the axes "
            f"{[a.name for a in axes]} have no channel axis to name.")
    n = data.shape[index]
    labels = [str(c) for c in channel_names]
    if len(labels) != n:
        raise OmeZarrError(
            f"{len(labels)} channel names for {n} channels along axis "
            f"{axes[index].name!r}")
    colors = list(channel_colors) if channel_colors else [
        DEFAULT_CHANNEL_COLORS[i % len(DEFAULT_CHANNEL_COLORS)]
        for i in range(n)]
    if len(colors) != n:
        raise OmeZarrError(f"{len(colors)} channel colours for {n} channels")
    channels = []
    for i, label in enumerate(labels):
        plane = data[tuple(slice(None) if a != index else i
                           for a in range(data.ndim))]
        low = float(np.min(plane)) if plane.size else 0.0
        high = float(np.max(plane)) if plane.size else 1.0
        channels.append({
            "label": label,
            "color": str(colors[i]).lstrip("#").upper(),
            "active": True,
            "window": {"start": low, "end": high, "min": low, "max": high},
        })
    return {"version": "0.4", "name": name, "channels": channels,
            "rdefs": {"model": "color" if n > 1 else "greyscale"}}
