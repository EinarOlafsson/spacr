"""Tiny Gatan DigitalMicrograph files written from scratch, for the TEM tests.

The layout is the one :func:`spacr.io.read_dm` parses: a big-endian header,
then nested tag groups. Files written here were checked against ncempy, an
independent reader (GPL, so not a spaCR dependency): it returns the same
pixels, scale and units for DM3 and DM4 of every type below.
"""
from __future__ import annotations

import struct

import numpy as np

_CODES = {np.dtype("<i2"): 2, np.dtype("<i4"): 3, np.dtype("<u2"): 4,
          np.dtype("<u4"): 5, np.dtype("<f4"): 6, np.dtype("<f8"): 7,
          np.dtype("u1"): 10, np.dtype("i1"): 9}
_DATATYPE = {np.dtype("<i2"): 1, np.dtype("<f4"): 2, np.dtype("u1"): 6,
             np.dtype("<i4"): 7, np.dtype("i1"): 9, np.dtype("<u2"): 10,
             np.dtype("<u4"): 11, np.dtype("<f8"): 12}


def write_dm(path, image, *, version=4, scale=1.0, unit="nm",
             thumbnail=True, datatype=None, dims=None, metadata=False,
             little=True):
    """Write ``image`` as a DM3 or DM4 file.

    :param path: where to write.
    :param image: the 2-D array; its dtype must be one of :data:`_CODES`.
    :param version: 3 or 4.
    :param scale: the calibration of both axes, in ``unit`` per pixel.
    :param unit: the calibration unit, stored as UTF-16 code units.
    :param thumbnail: write a packed-RGBA thumbnail first, as DM does.
    :param datatype: the DM ``DataType`` to record; derived from the dtype
        when ``None``.
    :param dims: the ``Dimensions`` to record; the image's when ``None``.
    :param metadata: add root tags of the kinds an acquisition carries
        beside the image: a struct, a string and an array of structs.
    :param little: declare little-endian tag data (as every PC writes);
        False writes the pixels and values big-endian.
    """
    size = ">I" if version == 3 else ">Q"
    order = "<" if little else ">"
    image = np.ascontiguousarray(image)

    def num(value):
        return struct.pack(size, value)

    def data(info, payload):
        return (b"%%%%" + num(len(info)) + b"".join(num(i) for i in info)
                + payload)

    def code_of(dtype):
        return _CODES[np.dtype(dtype).newbyteorder("<")]

    def scalar(value, dtype):
        dtype = np.dtype(dtype).newbyteorder(order)
        return data([code_of(dtype)], np.asarray(value, dtype).tobytes())

    def array(values):
        values = np.ascontiguousarray(values)
        values = values.astype(values.dtype.newbyteorder(order))
        return data([20, code_of(values.dtype), values.size], values.tobytes())

    def entry(name, body, is_group):
        raw = name.encode("latin-1")
        head = (bytes([20 if is_group else 21]) + struct.pack(">H", len(raw))
                + raw)
        if version == 4:
            head += struct.pack(">Q", len(body))
        return head + body

    def group(items):
        body = b"\x00\x01" + num(len(items))
        for name, (kind, value) in items:
            body += entry(name, value, kind == "g")
        return body

    def g(items):
        return ("g", group(items))

    def d(body):
        return ("d", body)

    units = np.array([ord(c) for c in unit], "<u2")
    axis = g([("Origin", d(scalar(0.0, "f4"))),
              ("Scale", d(scalar(scale, "f4"))),
              ("Units", d(array(units)))])
    images = []
    if thumbnail:
        images.append(("", g([("ImageData", g([
            ("Data", d(array(np.zeros(16, "<u4")))),
            ("DataType", d(scalar(23, "i4"))),
            ("Dimensions", g([("", d(scalar(4, "u4"))),
                              ("", d(scalar(4, "u4")))]))]))])))
    shape = dims if dims is not None else (image.shape[1], image.shape[0])
    code = datatype if datatype is not None else _DATATYPE[image.dtype]
    images.append(("", g([("ImageData", g([
        ("Calibrations", g([("Dimension", g([("", axis), ("", axis)]))])),
        ("Data", d(array(image.ravel()))),
        ("DataType", d(scalar(code, "i4"))),
        ("Dimensions", g([("", d(scalar(v, "u4"))) for v in shape]))]))])))
    extra = []
    if metadata:
        pair = np.array([1.5, 2.5], np.dtype("f4").newbyteorder(order))
        points = np.array([(1, 2), (3, 4)],
                          np.dtype([("a", order + "i4"), ("b", order + "i4")]))
        extra = [
            ("Pair", d(data([15, 0, 2, 0, 6, 0, 6], pair.tobytes()))),
            ("Label", d(data([18, 5], b"hello"))),
            ("Points", d(data([20, 15, 0, 2, 0, 3, 0, 3, 2], points.tobytes()))),
        ]
    root = group([("ImageList", g(images))] + extra)
    header = (struct.pack(">i", version) + num(len(root) + 16)
              + struct.pack(">i", 1 if little else 0))
    with open(path, "wb") as handle:
        handle.write(header + root + b"\x00" * 8)
