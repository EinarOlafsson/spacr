"""Table-driven tests of the Yokogawa (CellVoyager) renaming converters in ``spacr.io``.

One general test per converter, each walking a case table with pytest's
``subtests`` fixture.  ``convert_to_yokogawa`` turns whole images (TIFF,
ND2, CZI, LIF) into ``plate<n>_<well>_T..F..L01..C...tif`` planes with a
well per source file; ``convert_separate_files_to_yokogawa`` renames
one-plane-per-file acquisitions by a regex.  The vendor readers are small
doubles mirroring their real surfaces.
"""
from __future__ import annotations

import os
import types

import numpy as np
import pandas as pd
import pytest
import tifffile

import spacr.io as IO


def _tif(path, value=1, shape=(2, 2)):
    tifffile.imwrite(str(path), np.full(shape, value, np.uint16))


class _StubND2:
    """Minimal stand-in for nd2reader.ND2Reader (2 z-levels, 1 channel)."""

    def __init__(self, path, fail=False):
        self.fail = fail
        self.metadata = {"frames": [0], "fields_of_view": [0],
                         "z_levels": [0, 1], "channels": ["DAPI"]}

    def get_frame_2D(self, t=0, v=0, z=0, c=0):
        if self.fail:
            raise IndexError("truncated ND2")
        return np.full((4, 4), z + 1, np.uint16)


class _StubCziDoc:
    """pylibCZIrw document double: 2 channels, plane value = channel + 1."""

    def __init__(self, scenes):
        self._scenes = scenes
        self.total_bounding_box = {"T": (0, 1), "C": (0, 2), "Z": (0, 1)}

    @property
    def scenes_bounding_rectangle(self):
        return self._scenes

    def read(self, plane=None, scene=None):
        return np.full((1, 4, 4, 1), plane["C"] + 1, np.uint16)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _LifImage:
    """readlif 0.6.5 LifImage: dims x y z t m, channels on the image."""

    def __init__(self, channels=2, drop_z=(), value=None):
        self.dims = types.SimpleNamespace(x=4, y=4, z=2, t=1, m=1)
        self.channels = channels
        self.drop_z = drop_z
        self.value = value

    def get_frame(self, z=0, t=0, c=0):
        if z in self.drop_z:
            raise IndexError("missing plane")
        return np.full((4, 4), self.value or z + 1, np.uint16)


def _readlif(images_for):
    """``readlif.reader.LifFile(path).get_iter_image()`` -> images_for(path)."""
    class _LifFile:
        def __init__(self, path):
            self.path = path

        def get_iter_image(self):
            return list(images_for(self.path))

    return types.SimpleNamespace(reader=types.SimpleNamespace(LifFile=_LifFile))


def _raise(exc):
    def boom(*_a, **_k):
        raise exc
    return boom


def _log(folder):
    path = folder / "rename_log.csv"
    if not path.read_text(encoding="utf-8").strip():
        return []
    df = pd.read_csv(path)
    original = "Original File" if "Original File" in df else "Original File(s)"
    return list(zip(df[original], df["Renamed TIFF"]))


def _maxima(folder, names):
    return [int(tifffile.imread(str(folder / n)).max()) for n in names]


def test_convert_to_yokogawa_gives_each_source_file_its_own_well(
        subtests, tmp_path, monkeypatch, capsys):
    """Every plane of a source lands in that source's well; failures are named."""
    planes = np.arange(3 * 7 * 8, dtype=np.uint16).reshape(3, 7, 8)
    zstack = np.zeros((6, 4, 4), np.uint16)
    zstack[3] = 77

    def two_channel(value):
        return np.stack([np.full((8, 9), value, np.uint16),
                         np.full((8, 9), value + 1, np.uint16)])

    def write(name, arr):
        return lambda d: tifffile.imwrite(str(d / name), arr,
                                          photometric="minisblack")

    def plain(name, arr):
        # No photometric hint: tifffile guesses a short last axis as colour
        # samples, which the converter must recognise as a guess.
        return lambda d: tifffile.imwrite(str(d / name), arr)

    def stub(name):
        return lambda d: (d / name).write_bytes(b"stub")

    lif_dark_full = _readlif(lambda p: [_LifImage(
        channels=1, drop_z=(0, 1) if "dark" in os.path.basename(p) else (),
        value=6)])
    cases = [
        # id, builders, doubles {attr: value}, log [(original, renamed)],
        # maxima of the renamed files in log order, printed fragments
        ("three planes of one file share one well",
         [write("multichannel.tif", planes)], {},
         [("multichannel.tif", f"plate1_A01_T0001F001L01C0{c}.tif")
          for c in (1, 2, 3)], [int(p.max()) for p in planes], []),
        ("each original gets one stable well",
         [write("b.tif", two_channel(20)), write("a.tif", two_channel(10))], {},
         [("a.tif", "plate1_A01_T0001F001L01C01.tif"),
          ("a.tif", "plate1_A01_T0001F001L01C02.tif"),
          ("b.tif", "plate1_A02_T0001F001L01C01.tif"),
          ("b.tif", "plate1_A02_T0001F001L01C02.tif")], [10, 11, 20, 21], []),
        ("three 2-D sources, three wells",
         [lambda d: [_tif(d / f"{n}.tif", i + 1, (8, 8))
                     for i, n in enumerate(("alpha", "beta", "gamma"))]], {},
         [(f"{n}.tif", f"plate1_A0{i + 1}_T0001F001L01C01.tif")
          for i, n in enumerate(("alpha", "beta", "gamma"))], [1, 2, 3], []),
        ("z-stack MIP, 4-D split per time, others reported",
         [plain("zzz.tif", zstack), plain("ttt.tif", np.ones((2, 3, 4, 4),
                                                             np.uint16)),
          plain("five.tif", np.ones((2, 2, 2, 4, 4), np.uint16)),
          lambda d: (d / "broken.tif").write_bytes(b"not a tiff at all")], {},
         [("ttt.tif", "plate1_A03_T0001F001L01C01.tif"),
          ("ttt.tif", "plate1_A03_T0002F001L01C01.tif"),
          ("zzz.tif", "plate1_A04_T0001F001L01C01.tif")], [1, 1, 77],
         ["Unsupported TIFF dimensions",
          "Error processing standard image file broken.tif"]),
        ("ND2 is max-projected over Z", [stub("movie.nd2")],
         {"ND2Reader": _StubND2},
         [("movie.nd2", "plate1_A01_T0001F001L01C01.tif")], [2], []),
        ("ND2 with missing frames writes nothing", [stub("movie.nd2")],
         {"ND2Reader": lambda p: _StubND2(p, fail=True)}, [], [],
         ["incomplete data structure"]),
        ("ND2 reader failure", [stub("movie.nd2")],
         {"ND2Reader": _raise(OSError("cannot open ND2"))}, [], [],
         ["Error processing ND2 file movie.nd2"]),
        ("CZI scenes get their own wells and fields", [stub("img.czi")],
         {"pyczi": types.SimpleNamespace(open_czi=lambda p: _StubCziDoc(
             {0: (0, 0, 4, 4), 1: (4, 0, 4, 4)}))},
         [("img.czi", "plate1_A02_T0001F001L01A01Z01C01.tif"),
          ("img.czi", "plate1_A02_T0001F001L01A01Z01C02.tif"),
          ("img.czi", "plate1_A03_T0001F002L01A02Z01C01.tif"),
          ("img.czi", "plate1_A03_T0001F002L01A02Z01C02.tif")],
         [1, 2, 1, 2], []),
        ("CZI without scenes is one field", [stub("img.czi")],
         {"pyczi": types.SimpleNamespace(open_czi=lambda p: _StubCziDoc({}))},
         [("img.czi", "plate1_A02_T0001F001L01A01Z01C01.tif"),
          ("img.czi", "plate1_A02_T0001F001L01A01Z01C02.tif")], [1, 2], []),
        ("CZI reader failure", [stub("img.czi")],
         {"pyczi": types.SimpleNamespace(
             open_czi=_raise(OSError("cannot open CZI")))}, [], [],
         ["Error processing CZI file img.czi"]),
        ("LIF channels are max-projected over Z", [stub("stack.lif")],
         {"readlif": _readlif(lambda p: [_LifImage()])},
         [("stack.lif", "plate1_A01_T0001F001L01C01.tif"),
          ("stack.lif", "plate1_A01_T0001F001L01C02.tif")], [2, 2], []),
        ("a missing LIF plane is skipped", [stub("stack.lif")],
         {"readlif": _readlif(lambda p: [_LifImage(drop_z=(1,))])},
         [("stack.lif", "plate1_A01_T0001F001L01C01.tif"),
          ("stack.lif", "plate1_A01_T0001F001L01C02.tif")], [1, 1],
         ["Missing frame"]),
        ("a LIF channel with no planes writes nothing",
         [stub("a_dark.lif"), stub("b_full.lif")], {"readlif": lif_dark_full},
         [("b_full.lif", "plate1_A02_T0001F001L01C01.tif")], [6],
         ["Missing frame"]),
        ("LIF reader failure", [stub("stack.lif")],
         {"readlif": _readlif(_raise(OSError("cannot open LIF")))}, [], [],
         ["Error processing LIF file stack.lif"]),
    ]
    for i, (case_id, builders, doubles, log, maxima, said) in enumerate(cases):
        with subtests.test(msg=case_id):
            folder = tmp_path / f"c{i}"
            folder.mkdir()
            for build in builders:
                build(folder)
            with monkeypatch.context() as m:
                for attr, value in doubles.items():
                    m.setattr(IO, attr, value)
                capsys.readouterr()
                IO.convert_to_yokogawa(str(folder))
            out = capsys.readouterr().out
            got = _log(folder)
            assert got == log
            assert sorted(p.name for p in folder.glob("plate*.tif")) == \
                sorted(n for _o, n in log)
            assert _maxima(folder, [n for _o, n in log]) == maxima
            for fragment in said:
                assert fragment in out
            if case_id.startswith("a LIF channel"):
                assert out.count("Missing frame") == 2

    with subtests.test(msg="385 files overflow onto plate2"):
        folder = tmp_path / "overflow"
        folder.mkdir()
        for k in range(385):
            _tif(folder / f"img_{k:04d}.tif")
        ledger = IO.convert_to_yokogawa(str(folder))
        assert ledger.failures == []
        assert (folder / "plate2_A01_T0001F001L01C01.tif").is_file()
        assert len(list(folder.glob("plate1_*.tif"))) == 384
        assert len(_log(folder)) == 385

    with subtests.test(msg="an already converted folder is refused untouched"):
        folder = tmp_path / "converted"
        folder.mkdir()
        _tif(folder / "a_raw.tif", 3)
        _tif(folder / "plate9_B02_T0001F001L01C01.tif", 7)
        (folder / "notes.txt").write_text("not an image", encoding="utf-8")
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        with pytest.raises(ValueError, match="already contains converted images"):
            IO.convert_to_yokogawa(str(folder))
        assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def test_convert_separate_files_to_yokogawa_maps_wells_by_regex(
        subtests, tmp_path, capsys):
    """Each source well gets one Yokogawa well; unmatched files are skipped."""
    numbered = (r"W(?P<wellID>\d+)_F(?P<fieldID>\d+)_T(?P<timeID>\d+)"
                r"_C(?P<chanID>\d+)_Z(?P<sliceID>\d+)")
    named = (r"(?P<wellID>[A-Za-z0-9\-]+)_F(?P<fieldID>\d+)_T(?P<timeID>\d+)"
             r"_C(?P<chanID>\d+)_Z(?P<sliceID>\d+)")
    cases = [
        # id, {file: value}, regex, {renamed: value}, printed
        ("two single-slice regions",
         {"W1_F1_T1_C1_Z1.tif": 5, "W2_F1_T1_C1_Z1.tif": 9}, numbered,
         {"plate1_A01_T0001F001L01C01.tif": 5,
          "plate1_A02_T0001F001L01C01.tif": 9}, None),
        ("two spellings of one address do not collide",
         {"A01_F1_T1_C1_Z1.tif": 5, "a-1_F1_T1_C1_Z1.tif": 9}, named,
         {"plate1_A01_T0001F001L01C01.tif": 5, None: 9},
         "is not a plate address"),
        ("unmatched and well-less files are skipped",
         {"F1.tif": 1, "notes.txt": None},
         r"(W(?P<wellID>\d+))?F(?P<fieldID>\d+)\.tif", {}, None),
    ]
    for i, (case_id, files, regex, renamed, said) in enumerate(cases):
        with subtests.test(msg=case_id):
            folder = tmp_path / f"c{i}"
            folder.mkdir()
            for name, value in files.items():
                if value is None:
                    (folder / name).write_text("hello", encoding="utf-8")
                else:
                    _tif(folder / name, value)
            capsys.readouterr()
            IO.convert_separate_files_to_yokogawa(str(folder), regex)
            out = capsys.readouterr().out
            produced = sorted(p.name for p in folder.glob("plate*.tif"))
            log = _log(folder)
            assert len(produced) == len(log) == len(renamed)
            assert sorted(n for _o, n in log) == produced
            values = dict(zip(produced, _maxima(folder, produced)))
            for name, value in renamed.items():
                if name is not None:
                    assert values.pop(name) == value
            assert sorted(values.values()) == sorted(
                v for k, v in renamed.items() if k is None)
            if said:
                assert said in out

    with subtests.test(msg="the 385th region rolls over onto plate2"):
        folder = tmp_path / "overflow"
        folder.mkdir()
        for k in range(1, 386):
            _tif(folder / f"W{k}_F1_T1_C1_Z1.tif")
        IO.convert_separate_files_to_yokogawa(str(folder), numbered)
        assert (folder / "plate2_A01_T0001F001L01C01.tif").is_file()
        assert len(list(folder.glob("plate1_*.tif"))) == 384
        assert len(_log(folder)) == 385
