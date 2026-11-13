"""Table-driven tests of the preprocessing entry points in ``spacr.io``.

One general test per unit (``preprocess_img_data`` gets three: channel
mapping, run scenarios, and existing-output / refusal handling), each
walking a case table with pytest's ``subtests`` fixture.  The heavy
collaborators (normalisation, plotting, movies, the file organiser) are
replaced by recording doubles unless a case needs the real one; the plates
are tiny synthetic TIFF folders under ``tmp_path``.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest
import tifffile

import spacr.io as IO
from tests.test_raw_volumetric_tiffs_f551 import (_settings as _vol_settings,
                                                   _write as _vol_write)


@pytest.fixture(autouse=True)
def _no_figure_leak():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _tiny_img(seed=0, shape=(16, 16)):
    """Small deterministic uint16 image with a bright square (never flat)."""
    img = np.random.default_rng(seed).integers(50, 200, size=shape,
                                               dtype=np.uint16)
    img[4:10, 4:10] = 40000
    return img


class _Rec:
    """Callable that records every call and returns a fixed value."""

    def __init__(self, ret=None):
        self.calls = []
        self.ret = ret

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.ret

    @property
    def n(self):
        return len(self.calls)


def _fake_rename(n_files=4, n_channels=2, shape=(8, 8)):
    """Stub organiser: writes ``n_files`` stacks and reports ``n_channels``."""
    rec = _Rec(ret=n_channels)

    def _f(src, regex, batch_size, metadata_type, img_format, timelapse=False,
           save_original_images=True, metadata_well_map=None, source_plate=None):
        rec.calls.append(((src, regex, batch_size, metadata_type, img_format),
                          {"timelapse": timelapse,
                           "metadata_well_map": metadata_well_map,
                           "source_plate": source_plate}))
        stack = os.path.join(src, "stack")
        os.makedirs(stack, exist_ok=True)
        for i in range(n_files):
            arr = np.arange(shape[0] * shape[1] * n_channels, dtype=np.uint16)
            np.save(os.path.join(stack, f"plate1_A01_00{i}_T0001.npy"),
                    arr.reshape(shape[0], shape[1], n_channels) + i)
        return n_channels

    return _f, rec


def _patch(monkeypatch, rename=None, keep=()):
    """Replace the heavy collaborators with recorders (except ``keep``)."""
    import spacr.plot as PLOT

    recs = {"concat": _Rec(ret="masks"), "plot": _Rec(), "movies": _Rec(),
            "merge": _Rec(ret=2)}
    targets = {"concat": (IO, "concatenate_and_normalize"),
               "plot": (PLOT, "plot_arrays"),
               "movies": (IO, "_create_movies_from_npy_per_channel"),
               "merge": (IO, "_merge_channels")}
    for key, (mod, attr) in targets.items():
        if key not in keep:
            monkeypatch.setattr(mod, attr, recs[key])
    if rename is not None:
        monkeypatch.setattr(IO, "_rename_and_organize_image_files", rename)
    return recs


def _settings(src, **over):
    s = {"src": str(src), "metadata_type": "cellvoyager", "custom_regex": None,
         "channels": [0, 1], "nucleus_channel": 0, "cell_channel": 1,
         "pathogen_channel": None, "organelle_channel": None, "plot": False,
         "batch_size": 1, "test_mode": False, "timelapse": False,
         "normalize": True}
    s.update(over)
    return s


def _plate_with_stack(root):
    """A plate whose ``stack/`` already holds one (4, 4, 2) field."""
    stack = root / "stack"
    stack.mkdir(parents=True)
    np.save(stack / "fov.npy", np.zeros((4, 4, 2), dtype=np.uint16))
    return root


# ---------------------------------------------------------------------------
# preprocess_img_data
# ---------------------------------------------------------------------------

def test_preprocess_img_data_maps_object_channels_onto_stack_positions(
        subtests, tmp_path, monkeypatch):
    """Object channels dedupe in order; cellpose_* name their stack position."""
    keys = ("nucleus", "cell", "pathogen", "organelle")
    cases = [
        # id, (nucleus, cell, pathogen, organelle), concat channels,
        # {role: cellpose position} (absent roles get no key)
        ("ints", (0, 1, None, None), [0, 1], {"nucleus": 0, "cell": 1}),
        ("duplicate and None collapse", (1, 0, 1, None), [1, 0],
         {"nucleus": 0, "cell": 1, "pathogen": 0}),
        ("digit strings", ("1", "0", None, None), [1, 0],
         {"nucleus": 0, "cell": 1}),
        ("uncoercible values are dropped", ((0, 1), "abc", 2, None), [2],
         {"pathogen": 0}),
    ]
    for i, (case_id, chans, concat, mapped) in enumerate(cases):
        with subtests.test(msg=case_id):
            src = _plate_with_stack(tmp_path / f"p{i}")
            with monkeypatch.context() as m:
                recs = _patch(m)
                out, out_src = IO.preprocess_img_data(_settings(
                    src, **{f"{k}_channel": v for k, v in zip(keys, chans)}))
            assert out_src == str(src)
            assert recs["merge"].n == 0
            kwargs = recs["concat"].calls[0][1]
            assert (kwargs["src"], kwargs["channels"]) == (str(src / "stack"),
                                                           concat)
            assert kwargs["save_dtype"] is np.float32
            assert {k: out[f"cellpose_{k}_channel"] for k in keys
                    if f"cellpose_{k}_channel" in out} == mapped
            assert out["lower_percentile"] == 2


def test_preprocess_img_data_reports_batches_channels_and_movies(
        subtests, tmp_path, monkeypatch, capsys):
    """The organised stack drives batch warnings, channel fix-up and movies."""
    cases = [
        # id, (n stacks, n channel folders), settings overrides,
        # printed, not printed, out channels, plot calls, movie calls
        ("trailing batch of one", (11, 2), {"batch_size": 5, "plot": True},
         ["all images: 11,  full batch: 2, last batch: 1",
          "Last batch of size 1 detected"], [], [0, 1], 1, 0),
        ("single image batch", (1, 2), {"batch_size": 5, "plot": True},
         ["Only one batch of size 1 detected"], [], [0, 1], 1, 0),
        ("image count, not path length",
         (4, 2), {"batch_size": "path-1", "plot": True}, [],
         ["Last batch of size 1 detected"], [0, 1], 1, 0),
        ("channel folders rewrite channels", (4, 3), {"channels": [0, 1]},
         ["Number of channels does not match number of channel folders",
          "Changing channels from [0, 1] to [0, 1, 2]"], [], [0, 1, 2], 0, 0),
        ("timelapse makes movies", (2, 2), {"timelapse": True, "fps": 7}, [],
         [], [0, 1], 0, 1),
    ]
    for i, (case_id, (n_files, n_chan), over, said, unsaid, channels, n_plot,
            n_movies) in enumerate(cases):
        with subtests.test(msg=case_id):
            src = tmp_path / f"p{i}" / "plate1"
            src.mkdir(parents=True)
            tifffile.imwrite(str(src / "plate1_A01_T0001F001L01A01Z01C01.tif"),
                             _tiny_img())
            if over.get("batch_size") == "path-1":
                over = dict(over, batch_size=len(str(src / "stack")) - 1)
            rename, rename_rec = _fake_rename(n_files=n_files,
                                              n_channels=n_chan)
            with monkeypatch.context() as m:
                recs = _patch(m, rename=rename)
                capsys.readouterr()
                out, out_src = IO.preprocess_img_data(_settings(src, **over))
            text = capsys.readouterr().out
            assert out_src == str(src)
            assert rename_rec.calls[0][1] == {
                "timelapse": bool(over.get("timelapse")),
                "metadata_well_map": "", "source_plate": "plate1"}
            assert out["channels"] == channels
            assert (recs["concat"].n, recs["plot"].n, recs["movies"].n) == \
                (1, n_plot, n_movies)
            if n_movies:
                assert recs["movies"].calls[0] == ((str(src / "stack"),),
                                                   {"fps": 7})
            for fragment in said:
                assert fragment in text
            for fragment in unsaid:
                assert fragment not in text


def test_preprocess_img_data_reuses_existing_output_and_names_failures(
        subtests, tmp_path, monkeypatch, capsys, yokogawa_cellvoyager_dir):
    """Existing stacks/masks short-circuit; test mode subsets; empty runs refuse."""
    import spacr.illumination as illumination

    with subtests.test(msg="masks folder short-circuits, empty dirs pruned"):
        src = tmp_path / "masks_done"
        src.mkdir()
        (src / "notes.txt").write_text("not an image", encoding="utf-8")
        for name in ("stack", "channel_stack", "masks"):
            (src / name).mkdir()
            (src / name / "keep.bin").write_bytes(b"x")
        (src / "empty_dir").mkdir()
        with monkeypatch.context() as m:
            recs = _patch(m)
            settings = _settings(src)
            capsys.readouterr()
            out, out_src = IO.preprocess_img_data(settings)
        text = capsys.readouterr().out
        assert out is settings and out_src == str(src)
        assert "lower_percentile" not in out and "save_dtype" not in out
        assert (recs["concat"].n, recs["merge"].n) == (0, 0)
        assert not (src / "empty_dir").exists()
        for fragment in ("Could not find any", "Found existing stack folder.",
                         "Found existing channel_stack folder.",
                         "Found existing masks folder. Skipping preprocessing"):
            assert fragment in text

    with subtests.test(msg="existing masks with illumination check provenance"):
        src = tmp_path / "illum_masks"
        (src / "masks").mkdir(parents=True)
        np.savez_compressed(src / "masks" / "stack_0_norm.npz",
                            data=np.zeros((2, 4, 4, 2), np.float32),
                            filenames=np.asarray(["f0.npy", "f1.npy"]),
                            source_channels=np.asarray([0, 1], np.int64))
        seen = []
        with monkeypatch.context() as m:
            m.setattr(illumination, "load_segmentation_illumination_resume",
                      lambda settings, **kw: seen.append(kw) or object())
            settings = _settings(src, illumination_correction=True,
                                 illumination_model="")
            out, out_src = IO.preprocess_img_data(settings)
        assert out is settings and out_src == str(src)
        assert seen == [{
            "provenance_path": str(src / "illumination"
                                   / "segmentation_application.json"),
            "pipeline_style": "v1", "expected_fields": ("f0", "f1"),
            "verbose": True}]

    with subtests.test(msg="existing stack prepares illumination first"):
        src = _plate_with_stack(tmp_path / "illum_stack")
        session, seen = object(), []
        with monkeypatch.context() as m:
            recs = _patch(m)
            m.setattr(illumination, "prepare_segmentation_illumination",
                      lambda settings, **kw: seen.append(kw) or session)
            IO.preprocess_img_data(_settings(src, illumination_correction=True))
        assert seen == [{"src": str(src / "stack"), "channels": [0, 1],
                         "pipeline_style": "v1"}]
        assert recs["concat"].calls[0][1]["illumination_session"] is session

    with subtests.test(msg="preprocessing only never touches old masks"):
        src = _plate_with_stack(tmp_path / "pre_only")
        old = src / "masks" / "cell_mask_stack" / "fov.npy"
        old.parent.mkdir(parents=True)
        np.save(old, np.full((4, 4), 7, np.uint16))

        def forbidden(*a, **k):
            raise AssertionError("preprocessing-only must not prepare")

        with monkeypatch.context() as m:
            _patch(m)
            m.setattr(illumination, "prepare_segmentation_illumination",
                      forbidden)
            out, out_src = IO.preprocess_img_data(_settings(
                src, masks=False, illumination_correction=True,
                illumination_model=""))
        assert out_src == str(src)
        assert out.get("_illumination_v1_rebuild_required") is None
        np.testing.assert_array_equal(np.load(old), np.full((4, 4), 7))

    with subtests.test(msg="channel subfolders are merged into stack/"):
        src = tmp_path / "chan_folders"
        src.mkdir()
        (src / "notes.txt").write_text("not an image", encoding="utf-8")
        imgs = {}
        for chan in ("0", "1"):
            (src / chan).mkdir()
            for k, name in enumerate(("fovA.tif", "fovB.tif")):
                imgs[chan, name] = _tiny_img(seed=int(chan) * 10 + k)
                tifffile.imwrite(str(src / chan / name), imgs[chan, name])
        with monkeypatch.context() as m:
            recs = _patch(m, keep=("merge",))
            _, out_src = IO.preprocess_img_data(_settings(src))
        assert out_src == str(src)
        assert sorted(p.name for p in (src / "stack").glob("*.npy")) == [
            "fovA.npy", "fovB.npy"]
        merged = np.load(src / "stack" / "fovA.npy")
        np.testing.assert_array_equal(
            merged, np.dstack([imgs["0", "fovA.tif"], imgs["1", "fovA.tif"]]))
        assert recs["concat"].calls[0][1]["src"] == str(src / "stack")

    with subtests.test(msg="test mode with >=100 entries deletes empty test/"):
        src = tmp_path / "many"
        src.mkdir()
        for k, (field, chan) in enumerate([("001", "01"), ("001", "02"),
                                           ("002", "01"), ("002", "02")]):
            tifffile.imwrite(
                str(src / f"plate1_A01_T0001F{field}L01A01Z01C{chan}.tif"),
                _tiny_img(seed=k))
        for k in range(100):
            (src / f"pad_{k:03d}.txt").write_text("x", encoding="utf-8")
        (src / "test").mkdir()
        (src / "junk").mkdir()
        with monkeypatch.context() as m:
            recs = _patch(m)
            capsys.readouterr()
            out, out_src = IO.preprocess_img_data(_settings(
                src, test_mode=True, test_images=10, random_test=True))
        text = capsys.readouterr().out
        assert "Deleted test directory" in text and "Found 4 tif files" in text
        assert (src / "junk").is_dir()
        assert (out_src, out["plot"]) == (str(src / "test"), True)
        assert len(list((src / "test" / "stack").glob("*.npy"))) == 2
        assert recs["concat"].n == 1

    cv = yokogawa_cellvoyager_dir["src"]
    with subtests.test(msg="test mode on a clean plate"):
        with monkeypatch.context() as m:
            recs = _patch(m)
            capsys.readouterr()
            out, out_src = IO.preprocess_img_data(_settings(
                cv, test_mode=True, test_images=1, random_test=True))
        assert "Running spacr in test mode" in capsys.readouterr().out
        assert (out_src, out["src"], out["plot"]) == (str(cv / "test"),
                                                      str(cv / "test"), True)
        stacks = sorted((cv / "test" / "stack").glob("*.npy"))
        assert [np.load(str(p)).shape for p in stacks] == [(128, 128, 2)]
        assert len(list((cv / "test" / "orig").glob("*.tif"))) == 2
        assert len(list(cv.glob("*.tif"))) == 8
        assert recs["plot"].calls[0][0][0] == str(cv / "test" / "stack")
        assert recs["concat"].calls[0][1]["src"] == str(cv / "test" / "stack")

    with subtests.test(msg="an undeletable test folder is reported"):
        (cv / "test" / "stale.txt").write_text("blocks rmdir", encoding="utf-8")
        with monkeypatch.context() as m:
            _patch(m)
            capsys.readouterr()
            _, out_src = IO.preprocess_img_data(_settings(
                cv, test_mode=True, test_images=1, random_test=True))
        text = capsys.readouterr().out
        assert "Error deleting test directory" in text
        assert "Delete manually before running test mode" in text
        assert out_src == str(cv / "test")

    def boom(*_a, **_k):
        raise RuntimeError("organiser found nothing to do")

    screen = tmp_path / "screen"
    for plate in ("plate1", "plate2"):
        (screen / plate).mkdir(parents=True)
        (screen / plate / "keep.txt").write_text("x", encoding="utf-8")
    unmatched = tmp_path / "unmatched"
    unmatched.mkdir()
    tifffile.imwrite(str(unmatched / "scan_0001.tif"), np.ones((8, 8), np.uint16))
    swept = tmp_path / "swept"
    swept.mkdir()
    (swept / "img_A01_f1.tif").write_bytes(b"")

    def sweep(*_a, **_k):
        shutil.rmtree(str(swept))
        raise RuntimeError("source folder was swept")

    refusals = [
        # id, src, organiser double, fragments in message, absent fragments,
        # printed fragment
        ("a folder of plates", screen, boom,
         ["No image stacks were produced", "0 image file(s)", "plate1",
          "plate2", "point src at one of them"], [],
         "organiser found nothing to do"),
        ("an unmatched filename keeps its image", unmatched, None,
         ["No image stacks were produced", "1 image file(s)"],
         ["point src at one of them"], None),
        ("a source swept mid-run", swept, sweep,
         ["No image stacks were produced", str(swept), "0 image file(s)"],
         [], None),
    ]
    for case_id, src, organiser, said, unsaid, printed in refusals:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as m:
                _patch(m)
                if organiser is not None:
                    m.setattr(IO, "_rename_and_organize_image_files", organiser)
                capsys.readouterr()
                with pytest.raises(FileNotFoundError) as caught:
                    IO.preprocess_img_data(_settings(src))
            message = str(caught.value)
            for fragment in said:
                assert fragment in message
            for fragment in unsaid:
                assert fragment not in message
            if printed:
                assert printed in capsys.readouterr().out


def test_preprocess_img_data_end_to_end_on_cellvoyager_tiffs(
        yokogawa_cellvoyager_dir):
    """Raw CellVoyager TIFFs -> stack/*.npy -> masks/*_norm.npz, all real."""
    src = yokogawa_cellvoyager_dir["src"]
    out, out_src = IO.preprocess_img_data(_settings(
        src, randomize=False, lower_percentile=2,
        nucleus_background=100, nucleus_signal_to_noise=5,
        remove_background_nucleus=False, cell_background=100,
        cell_signal_to_noise=5, remove_background_cell=False))
    assert out_src == str(src)
    stacks = sorted((src / "stack").glob("*.npy"))
    assert [np.load(p).shape for p in stacks] == [(128, 128, 2)] * 4
    assert len(list((src / "orig").glob("*.tif"))) == 8
    npzs = sorted((src / "masks").glob("*_norm.npz"))
    assert len(npzs) == 4
    with np.load(npzs[0]) as z:
        data, names = z["data"], z["filenames"]
    assert (data.shape, data.dtype) == ((1, 128, 128, 2), np.float32)
    assert 0.0 == float(data.min()) < float(data.max()) <= 1.0
    assert len(names) == 1 and str(names[0]).endswith(".npy")
    assert (out["cellpose_nucleus_channel"], out["cellpose_cell_channel"],
            out["save_dtype"]) == (0, 1, "uint16")


# ---------------------------------------------------------------------------
# _preprocess_volume_tiffs
# ---------------------------------------------------------------------------

def test_preprocess_volume_tiffs_refuses_every_request_it_cannot_honour(
        subtests, tmp_path, monkeypatch):
    """Volumetric ingest refuses bad requests, inputs and receipts by name."""
    receipt_name = ".spacr_volume_ingest.json"

    def fresh(case):
        root = tmp_path / case
        root.mkdir()
        return root

    def ingested(root):
        _vol_write(root)
        IO._preprocess_volume_tiffs(_vol_settings(root))
        return root / "stack" / receipt_name

    requests = [
        ({"timelapse": True}, "does not combine time and Z"),
        ({"z_axis": 2}, "z_axis must be 0"),
        ({"test_mode": True}, "turn test_mode off"),
        ({"illumination_correction": True}, "does not yet support"),
        ({"nucleus_channel": 5}, "index the sorted source-channel"),
    ]
    for k, (changes, match) in enumerate(requests):
        with subtests.test(msg=f"request {sorted(changes)}"):
            root = fresh(f"r{k}")
            _vol_write(root)
            with pytest.raises(ValueError, match=match):
                IO._preprocess_volume_tiffs(_vol_settings(root, **changes))

    def no_volumes(root):
        (root / "notes.png").write_bytes(b"x")

    def unmatched(root):
        _vol_write(root)
        tifffile.imwrite(root / "random.tif", np.zeros((4, 8, 9), np.uint16),
                         metadata={"axes": "ZYX"})

    def nonfinite(root):
        path, _ = _vol_write(root)
        volume = np.ones((4, 8, 9), np.float32)
        volume[0, 0, 0] = np.nan
        tifffile.imwrite(path, volume, metadata={"axes": "ZYX"})

    def complex_volume(root):
        path, _ = _vol_write(root)
        tifffile.imwrite(path, np.zeros((4, 8, 9), np.complex64),
                         metadata={"axes": "ZYX"})

    def multi_series(root):
        path, _ = _vol_write(root)
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.zeros((4, 8, 9), np.uint16), metadata={"axes": "ZYX"})
            writer.write(np.zeros((2, 3), np.uint8))

    def extra_stack(root):
        ingested(root)
        np.save(root / "stack" / "plate1_A01_9_1.npy", np.zeros(1))

    def receipt_edit(**fields):
        def build(root):
            receipt = ingested(root)
            data = json.loads(receipt.read_text(encoding="utf-8"))
            receipt.write_text(json.dumps(dict(data, **fields)),
                               encoding="utf-8")
        return build

    def oversized(root):
        ingested(root).write_text(" " * (16 * 1024 * 1024 + 2),
                                  encoding="utf-8")

    inputs = [
        ("no TIFF volumes", no_volumes, "metadata-labelled ZYX TIFF"),
        ("unmatched filename", unmatched, "filename metadata pattern"),
        ("nonfinite intensities", nonfinite, "nonfinite intensities"),
        ("complex intensities", complex_volume, "real numeric intensities"),
        ("multiple series", multi_series, "multiple TIFF series"),
        ("extra stack in the output", extra_stack, "inventory differs"),
        ("receipt of another version", receipt_edit(version=2),
         "unsupported format"),
        ("receipt without stacks", receipt_edit(stacks={}),
         "receipt is incomplete"),
        ("oversized receipt", oversized, "exceeds 16 MiB"),
    ]
    for case_id, build, match in inputs:
        with subtests.test(msg=case_id):
            root = fresh(case_id.replace(" ", "_"))
            build(root)
            with pytest.raises(ValueError, match=match):
                IO._preprocess_volume_tiffs(_vol_settings(root))

    real_link, real_rmdir, real_hash = os.link, os.rmdir, IO._volume_file_hash

    def failing_link(source, destination):
        shutil.rmtree(os.path.dirname(source))
        raise OSError("link refused")

    def stuck_rmdir(path, *a, **k):
        if os.path.basename(str(path)) == "stack":
            raise OSError("busy")
        return real_rmdir(path, *a, **k)

    def receipt_refused(source, destination):
        if destination.endswith(receipt_name):
            raise OSError("receipt refused")
        return real_link(source, destination)

    def link_then_tidy(source, destination):
        real_link(source, destination)
        if destination.endswith(receipt_name):
            shutil.rmtree(os.path.dirname(source))

    def drifting(written):
        calls = []

        def hash_(target):
            digest = real_hash(target)
            if str(target) == str(written[0]):
                calls.append(1)
                if len(calls) == 3:
                    return "changed"
            return digest
        return hash_

    publication = [
        # id, {attr: double}, exception, match, stack/ left behind,
        # receipt published
        ("publication fails and cleanup stumbles",
         {"link": failing_link, "rmdir": stuck_rmdir}, OSError,
         "link refused", None, False),
        ("late receipt failure unpublishes", {"link": receipt_refused},
         OSError, "receipt refused", False, False),
        ("input changed before publication", {"hash": drifting}, ValueError,
         "changed before stack publication", False, False),
        ("staging gone after publication is fine", {"link": link_then_tidy},
         None, None, True, True),
    ]
    for case_id, doubles, exc, match, stack_left, published in publication:
        with subtests.test(msg=case_id):
            root = fresh(case_id.replace(" ", "_"))
            path, _ = _vol_write(root)
            with monkeypatch.context() as m:
                for attr, double in doubles.items():
                    if attr == "hash":
                        m.setattr(IO, "_volume_file_hash", double([path]))
                    else:
                        m.setattr(IO.os, attr, double)
                if exc is None:
                    IO._preprocess_volume_tiffs(_vol_settings(root))
                else:
                    with pytest.raises(exc, match=match):
                        IO._preprocess_volume_tiffs(_vol_settings(root))
            if stack_left is not None:
                assert (root / "stack").exists() is stack_left
            assert (root / "stack" / receipt_name).exists() is published


# ---------------------------------------------------------------------------
# _get_avg_object_size
# ---------------------------------------------------------------------------

def test_get_avg_object_size_averages_over_every_mask_and_names_skips(
        subtests, capsys):
    """Objects per mask and mean area; empty or wrong-rank masks are named."""
    one = np.zeros((10, 10), np.int32)
    one[2:5, 2:5] = 1
    two = np.zeros((10, 10), np.int32)
    two[1:4, 1:4] = 1
    two[6:8, 6:8] = 2
    empty = np.zeros((10, 10), np.int32)
    rank4 = np.ones((2, 2, 2, 2), np.int32)
    rank1 = np.array([1, 2, 3], np.int32)
    cases = [
        # id, masks, (objects per mask, mean area), warning lines
        ("no masks", [], (0, 0), []),
        ("two objects", [two], (2, 6.5), []),
        ("same mask twice", [two, two], (2, 6.5), []),
        ("empty mask", [empty], (0, 0), ["Mask 0 is empty."]),
        ("rank-1 mask", [rank1], (0, 0), ["Mask 0 has invalid dimension: 1"]),
        ("rank-4 mask", [rank4], (0, 0), ["Mask 0 has invalid dimension: 4"]),
        ("one good, one empty, one rank-4", [one, empty, rank4], (1 / 3, 9.0),
         ["Mask 1 is empty.", "Mask 2 has invalid dimension: 4"]),
        ("two objects among skips", [two, empty, rank4], (2 / 3, 6.5),
         ["Mask 1 is empty.", "Mask 2 has invalid dimension: 4"]),
    ]
    for case_id, masks, expected, warned in cases:
        with subtests.test(msg=case_id):
            capsys.readouterr()
            got = IO._get_avg_object_size(masks)
            assert got == pytest.approx(expected)
            out = capsys.readouterr().out
            assert out.count("Warning: Mask") == len(warned)
            for line in warned:
                assert line in out
