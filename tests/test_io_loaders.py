"""Table-driven tests of the image loaders and PyTorch data plumbing in ``spacr.io``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: the Cellpose image/label loaders, the percentile
normaliser, ``spacrDataset`` / ``spacrDataLoader`` (preloading, pinning,
stalls and cleanup), ``generate_loaders`` and the fixed-scale preprocessing
contract.  Images are tiny synthetic TIFF/PNG files; everything is CPU-only.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time as _time
import types

import numpy as np
import pytest
import tifffile
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

torch = pytest.importorskip("torch")

import spacr.io as IO  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _ramp(shape, dtype=np.uint16, mod=500, start=0):
    """Deterministic non-constant array of the requested shape."""
    n = int(np.prod(shape))
    return ((np.arange(start, start + n) % mod) + 1).reshape(shape).astype(dtype)


def _tif(path, arr):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    tifffile.imwrite(str(path), arr)
    return str(path)


def _labels(shape=(16, 16), n=3):
    """A label mask with ``n`` small rectangles labelled 1..n."""
    m = np.zeros(shape, np.uint16)
    for i in range(1, n + 1):
        m[i * 3:i * 3 + 2, 2:5] = i
    return m


def _none_for(monkeypatch, marker):
    """Make cellpose.io.imread return None for paths containing ``marker``."""
    import cellpose.io

    real = cellpose.io.imread
    monkeypatch.setattr(cellpose.io, "imread",
                        lambda p, *a, **k: None if marker in str(p)
                        else real(p, *a, **k))


def _png(path, rng, size=8):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (size, size, 3)).astype(np.uint8)
                    ).save(path)
    return str(path)


def _class_dirs(root, rng, classes=("nc", "pc"), n=2, size=8, stem="plate1_"):
    for cls in classes:
        for i in range(n):
            _png(root / cls / f"{stem}{cls}_{i}.png", rng, size=size)
    return root


def _stub_pinning(monkeypatch):
    """Record tensor pinning instead of doing it (CPU-only)."""
    from torch.utils.data._utils import pin_memory as torch_pin

    calls = []

    def fake_pin(self, *args, **kwargs):
        calls.append(tuple(self.shape))
        return self

    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin, raising=False)
    monkeypatch.setattr(torch_pin, "pin_memory",
                        lambda data, device=None: data, raising=False)
    return calls


# ---------------------------------------------------------------------------
# _load_images_and_labels
# ---------------------------------------------------------------------------

def test_load_images_and_labels_keeps_names_in_step_with_pixels(
        subtests, tmp_path, monkeypatch, capsys):
    """Unloadable files drop with their names; images scale to [0, 1] if >1."""
    raw = _ramp((16, 16))
    inverted = np.iinfo(np.uint16).max - raw
    img = lambda name, arr=raw: _tif(tmp_path / "i" / name, arr)  # noqa: E731
    lbl = lambda name, n=3, sub="l": _tif(tmp_path / sub / name,  # noqa: E731
                                          _labels(n=n))
    flt = lambda name, v: _tif(tmp_path / "f" / name,  # noqa: E731
                               np.full((4, 4), v, np.float32))
    cases = [
        # id, image files, label files, invert, broken marker,
        # expected image names, label names, image maxima, label maxima,
        # printed fragment
        ("pairs, broken image drops the pair",
         [img("ok.tif"), img("bad_img.tif")],
         [lbl("ok.tif"), lbl("bad_img.tif")], False, "bad_img",
         ["ok.tif"], ["ok.tif"], [1.0], [3], "Could not load image"),
        ("pairs, broken label drops the pair",
         [img("a.tif"), img("b.tif")], [lbl("a.tif"), lbl("bad_lbl.tif")],
         False, "bad_lbl", ["a.tif"], ["a.tif"], [1.0], [3],
         "Could not load label"),
        ("labels only keep the given order",
         [], [lbl("z.tif", sub="l2"), lbl("bad_lbl.tif", sub="l2"),
             lbl("a.tif", n=2, sub="l2")], False,
         "bad_lbl", [], ["z.tif", "a.tif"], [], [3, 2], "Could not load label"),
        ("images only, broken one skipped",
         [img("a.tif"), img("bad_img.tif")], [], True, "bad_img",
         ["a.tif"], [], [1.0], [], "Could not load image"),
        ("0-1 floats are not rescaled, >1 are",
         [flt("low.tif", 0.25), flt("high.tif", 8.0)],
         [flt("low_mask.tif", 1.0), flt("high_mask.tif", 1.0)], False, None,
         ["low.tif", "high.tif"], ["low_mask.tif", "high_mask.tif"],
         [0.25, 1.0], [1, 1], None),
        ("0-1 floats, images only",
         [flt("low.tif", 0.25), flt("high.tif", 8.0)], [], False, None,
         ["low.tif", "high.tif"], [], [0.25, 1.0], [], None),
    ]
    for case_id, ifiles, lfiles, invert, broken, inames, lnames, imax, lmax, \
            said in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as m:
                if broken:
                    _none_for(m, broken)
                capsys.readouterr()
                images, labels, image_names, label_names = \
                    IO._load_images_and_labels(ifiles, lfiles, invert=invert)
            assert image_names == inames
            assert label_names == lnames
            assert [float(np.max(x)) for x in images] == pytest.approx(imax)
            assert [int(np.max(x)) for x in labels] == lmax
            if said:
                assert said in capsys.readouterr().out

    with subtests.test(msg="invert flips, then divides by the new max"):
        for lfiles in ([], [_tif(tmp_path / "inv" / "a.tif", _labels())]):
            images, labels = IO._load_images_and_labels(
                [_tif(tmp_path / "inv_i" / "a.tif", raw)], lfiles,
                invert=True)[:2]
            np.testing.assert_allclose(images[0], inverted / inverted.max())
        np.testing.assert_array_equal(labels[0], _labels())
        assert labels[0].dtype == np.uint16


# ---------------------------------------------------------------------------
# _load_normalized_images_and_labels
# ---------------------------------------------------------------------------

def test_load_normalized_images_and_labels_rescales_by_percentiles(
        subtests, tmp_path, monkeypatch, capsys):
    """Percentiles are used when given as a numeric list, else estimated."""
    raws = [_ramp((16, 16), start=7 * k) for k in range(2)]
    ifiles = [_tif(tmp_path / "i" / f"f{k}.tif", r) for k, r in enumerate(raws)]
    lfiles = [_tif(tmp_path / "l" / f"f{k}.tif", _labels()) for k in range(2)]
    cases = [
        # id, kwargs, auto percentiles printed
        ("non-numeric list", {"percentiles": ["a", "b"]}, True),
        ("tuple is not accepted", {"percentiles": (1, 99)}, True),
        ("None", {"percentiles": None}, True),
        ("numeric list", {"percentiles": [2, 98]}, False),
    ]
    for case_id, kwargs, auto in cases:
        with subtests.test(msg=case_id):
            capsys.readouterr()
            norm, labels, inames, lnames, dims = \
                IO._load_normalized_images_and_labels(
                    ifiles, lfiles, background=0, Signal_to_noise=10, **kwargs)
            out = capsys.readouterr().out
            assert ("Average 1st percentiles" in out) is auto
            assert dims == [(16, 16), (16, 16)]
            assert inames == lnames == ["f0.tif", "f1.tif"]
            assert [n.shape for n in norm] == [(16, 16, 1)] * 2
            assert norm[0].dtype == np.float64
            assert float(norm[0].min()) == 0.0 and float(norm[0].max()) == 1.0
            assert [(lab.shape, lab.dtype, int(lab.max())) for lab in labels] \
                == [((16, 16), np.uint8, 3)] * 2

    with subtests.test(msg="invert reverses the ordering, no labels"):
        plain = IO._load_normalized_images_and_labels(
            ifiles[:1], None, percentiles=None, background=0,
            Signal_to_noise=10)[0][0][..., 0]
        inv, labels, _n, lnames, _d = IO._load_normalized_images_and_labels(
            ifiles[:1], None, percentiles=None, invert=True, background=0,
            Signal_to_noise=10)
        assert labels == [] and lnames == []
        assert np.argmax(plain) == np.argmin(inv[0][..., 0])
        assert np.argmin(plain) == np.argmax(inv[0][..., 0])

    with subtests.test(msg="unreachable signal threshold collapses to p1"):
        capsys.readouterr()
        norm = IO._load_normalized_images_and_labels(
            ifiles[:1], None, percentiles=None, background=10000,
            Signal_to_noise=10)[0]
        out = capsys.readouterr().out
        p1 = out.split("Average 1st percentiles: ")[1].split("], Average")[0]
        p99 = out.split("Average 99th percentiles: ")[1].split("]")[0]
        assert p1 == p99
        assert str(float(np.percentile(raws[0], 2))) in p1
        assert np.allclose(norm[0], 1.0)

    with subtests.test(msg="channel subset and background removal"):
        rgb = _ramp((16, 16, 3), mod=400)
        norm, _l, _n, _ln, dims = IO._load_normalized_images_and_labels(
            [_tif(tmp_path / "rgb" / "rgb.tif", rgb)], None, channels=[0, 2],
            percentiles=[2, 98], remove_background=True, background=100,
            Signal_to_noise=10)
        assert dims == [(16, 16)]
        assert norm[0].shape == (16, 16, 2)
        below = rgb[..., 0] < 100
        assert int(below.sum()) == 66
        assert np.all(norm[0][..., 0][below] == 0.0)

    with subtests.test(msg="visualize draws the 2x2 resize comparison"):
        shown = {}

        def fake_show(*a, **kw):
            fig = plt.gcf()
            shown["titles"] = [ax.get_title() for ax in fig.axes]

        monkeypatch.setattr(plt, "show", fake_show)
        norm, labels = IO._load_normalized_images_and_labels(
            ifiles[:1], lfiles[:1], percentiles=[2, 98], visualize=True,
            background=0, Signal_to_noise=10, target_height=8,
            target_width=8)[:2]
        assert len(shown["titles"]) == 4
        assert {"Original Image", "Resized Label"} <= set(shown["titles"])
        assert norm[0].shape == (8, 8, 1)
        assert labels[0].shape == (8, 8)


# ---------------------------------------------------------------------------
# spacrDataset
# ---------------------------------------------------------------------------

def test_spacr_dataset_serves_images_from_disk_or_ram(
        subtests, tmp_path, rng, monkeypatch):
    """pin_memory=True preloads every image; __getitem__ then never reads disk."""
    monkeypatch.setattr(IO, "cpu_count", lambda: 1)
    with subtests.test(msg="class folders preload in order"):
        root = _class_dirs(tmp_path / "train", rng, n=3)
        ds = IO.spacrDataset(str(root), loader_classes=["nc", "pc"],
                             shuffle=False, pin_memory=True)
        assert len(ds) == len(ds.images) == 6
        assert [(im.mode, im.size) for im in ds.images] == [("RGB", (8, 8))] * 6
        img, label, filename = ds[2]
        assert img is ds.images[2]
        assert filename == ds.filenames[2]
        assert [os.path.basename(os.path.dirname(f)) for f in ds.filenames] \
            == ["nc"] * 3 + ["pc"] * 3
        assert ds.labels == [0, 0, 0, 1, 1, 1]
        assert label == 0
    with subtests.test(msg="specific files survive deletion"):
        files = [_png(tmp_path / "flat" / f"plateX_{i}.png", rng)
                 for i in range(3)]
        ds = IO.spacrDataset(str(tmp_path / "flat"), loader_classes=["a"],
                             shuffle=False, pin_memory=True,
                             specific_files=files, specific_labels=[0, 1, 0])
        for f in files:
            os.remove(f)
        img, label, filename = ds[1]
        assert (img.size, label, filename) == ((8, 8), 1, files[1])


# ---------------------------------------------------------------------------
# spacrDataLoader
# ---------------------------------------------------------------------------

class _EmptyQueue:
    """A queue whose producer never delivers."""

    def __init__(self):
        self.timeouts = []

    def get(self, timeout=None):
        self.timeouts.append(timeout)
        raise queue.Empty

    def get_nowait(self):
        raise queue.Empty


def test_spacr_dataloader_delivers_complete_passes_or_says_why_not(
        subtests, tmp_path, rng, monkeypatch, caplog):
    """Every pass yields every batch; stalls warn, lost producers raise."""
    from torchvision import transforms

    with subtests.test(msg="a new pass after a break is complete"):
        dl = IO.spacrDataLoader([1, 2, 3], batch_size=1, num_workers=0,
                                preload_batches=1)
        try:
            for index, _batch in enumerate(dl):
                if index == 2:
                    break
            assert [int(b) for b in dl] == [1, 2, 3]
            assert [int(b) for b in dl] == [1, 2, 3]
        finally:
            dl.cleanup()

    with subtests.test(msg="a slow producer is waited for, with a warning"):
        class _Slow(torch.utils.data.Dataset):
            def __len__(self):
                return 3

            def __getitem__(self, index):
                if index == 2:
                    _time.sleep(0.3)
                return torch.tensor(index)

        dl = IO.spacrDataLoader(_Slow(), batch_size=1, num_workers=0,
                                preload_batches=1)
        dl._poll_seconds, dl._stall_warning_seconds = 0.02, 0.05
        try:
            with caplog.at_level(logging.WARNING, logger="spacr.io"):
                assert [int(b) for b in dl] == [0, 1, 2]
            assert "the producer is still running" in caplog.text
        finally:
            dl.cleanup()

    with subtests.test(msg="a gone producer is an incomplete pass"):
        dl = IO.spacrDataLoader([1, 2], batch_size=1, num_workers=0,
                                preload_batches=1)
        try:
            dl.batch_queue = _EmptyQueue()
            dl.current_batch_index = 0
            with pytest.raises(RuntimeError, match="the pass is incomplete"):
                next(dl)
            assert dl.batch_queue.timeouts == [dl._poll_seconds]
            assert dl.current_batch_index == 0
        finally:
            dl.cleanup()

    with subtests.test(msg="a pass stopped by cleanup ends quietly"):
        dl = IO.spacrDataLoader([1, 2, 3], batch_size=1, num_workers=0,
                                preload_batches=1)
        iter(dl)
        dl.cleanup()
        assert dl._stop_signal.is_set()
        assert not dl.thread.is_alive()
        dl.batch_queue = _EmptyQueue()
        with pytest.raises(StopIteration):
            next(dl)
        assert dl._iteration_active is False

    root = _class_dirs(tmp_path / "train", rng, n=2)
    with subtests.test(msg="pin_memory pins images and labels of every batch"):
        with monkeypatch.context() as m:
            calls = _stub_pinning(m)
            ds = IO.spacrDataset(str(root), loader_classes=["nc", "pc"],
                                 transform=transforms.ToTensor(), shuffle=False)
            dl = IO.spacrDataLoader(ds, batch_size=2, preload_batches=1,
                                    pin_memory=True)
            try:
                batches = list(iter(dl))
            finally:
                dl.cleanup()
        assert [type(b) for b in batches] == [list, list]
        images, labels, names = batches[0]
        assert (tuple(images.shape), labels.tolist()) == ((2, 3, 8, 8), [0, 0])
        assert [os.path.basename(n) for n in names] == ["plate1_nc_0.png",
                                                        "plate1_nc_1.png"]
        assert calls == [(2, 3, 8, 8), (2,), (2, 3, 8, 8), (2,)]

    with subtests.test(msg="_pin_memory_batch pins tensors, passes the rest"):
        with monkeypatch.context() as m:
            calls = _stub_pinning(m)
            dl = IO.spacrDataLoader([1, 2], batch_size=1, preload_batches=1)
            try:
                t, bare, payload = torch.zeros(2, 2), torch.ones(3), {"x": 1}
                out = dl._pin_memory_batch([t, "name"])
                assert out[0] is t and out[1] == "name"
                assert dl._pin_memory_batch(bare) is bare
                assert dl._pin_memory_batch(payload) is payload
                assert dl._pin_memory_batch((torch.zeros(1), 7))[1] == 7
                assert calls == [(2, 2), (3,), (1,)]
            finally:
                dl.cleanup()

    with subtests.test(msg="__del__ stops the producer, idempotently"):
        ds = IO.spacrDataset(str(root), loader_classes=["nc", "pc"],
                             transform=transforms.ToTensor(), shuffle=False)
        dl = IO.spacrDataLoader(ds, batch_size=2, preload_batches=1)
        assert next(iter(dl))[0].shape == (2, 3, 8, 8)
        assert dl._stop_event is False and dl.thread is not None
        dl.__del__()
        assert dl._stop_event is True and not dl.thread.is_alive()
        dl.__del__()
        assert dl._stop_event is True

    with subtests.test(msg="cleanup reports a preloader that will not stop"):
        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.start()
        ticks = iter([0.0, 1.0, 1000.0, 1000.0, 1000.0])
        q = queue.Queue()
        q.put(object())
        fake = types.SimpleNamespace(
            _iteration_active=True, _stop_event=False,
            _stop_signal=threading.Event(), thread=worker, batch_queue=q)
        with monkeypatch.context() as m:
            m.setattr(IO.time, "monotonic", lambda: next(ticks, 1000.0))
            caplog.clear()
            with caplog.at_level(logging.ERROR, logger="spacr.io"):
                IO.spacrDataLoader.cleanup(fake)
        release.set()
        worker.join(timeout=5)
        assert "did not stop within five seconds" in caplog.text
        assert fake._stop_signal.is_set()
        assert fake._iteration_active is False
        assert q.empty()


# ---------------------------------------------------------------------------
# generate_loaders
# ---------------------------------------------------------------------------

def _raw_dataset(root, classes=("nc", "pc"),
                 wells=("A01", "A02", "B01", "B02"), plates=("plate1", "plate2")):
    """Declared-raw uint16 crops, one per (plate, well), in train/<class>."""
    import cv2
    from spacr.crops import _CROP_FORMAT_DECLARED_RAW, write_crop_folder_marker

    k = 0
    for cls_i, cls in enumerate(classes):
        folder = root / "train" / cls
        folder.mkdir(parents=True)
        write_crop_folder_marker(str(folder), _CROP_FORMAT_DECLARED_RAW)
        for p_i, plate in enumerate(plates):
            for well in wells:
                level = 1000 * (p_i + 1) * (cls_i + 1)
                arr = np.zeros((16, 16, 3), np.uint16)
                arr[4:12, 4:12] = np.random.default_rng(k).integers(
                    level // 2, level, size=(8, 8, 3))
                cv2.imwrite(str(folder / f"{plate}_{well}_1_{10 * cls_i}.png"),
                            np.ascontiguousarray(arr[:, :, ::-1]))
                k += 1
    return root


def test_generate_loaders_builds_splits_or_refuses_by_name(
        subtests, tmp_path, rng, monkeypatch, capsys):
    """Train/val/test loaders hold the expected crops; bad layouts are named."""
    import spacr.resource_log as resource_log
    from spacr import classification_pixels as cp

    guarded = []
    monkeypatch.setattr(resource_log, "_guard_workers",
                        lambda task, workers, unit, **k:
                        guarded.append(workers) or 0)
    cases = [
        # id, layout (split, n per class, classes), kwargs, train len, val len
        ("augment expands the train split", ("train", 4, ("nc", "pc")),
         {"mode": "train", "classes": ["nc", "pc"], "validation_split": 0.25,
          "augment": True}, 6 * 8, 2),
        ("test mode with default classes", ("test", 4, ("nc", "pc")),
         {"mode": "test", "classes": None, "channels": None,
          "verbose": True}, 8, None),
    ]
    for i, (case_id, (split, n, classes), kwargs, n_train, n_val) in \
            enumerate(cases):
        with subtests.test(msg=case_id):
            root = tmp_path / f"c{i}"
            _class_dirs(root / split, rng, classes=classes, n=n, size=32,
                        stem="")
            loader, val, fig = IO.generate_loaders(
                str(root), image_size=16, batch_size=2, n_jobs=0, **kwargs)
            assert len(loader.dataset) == n_train
            assert fig is None
            if n_val is None:
                assert val == []
                images, labels, _paths = next(iter(loader))
                assert images.shape[1] == 3
                assert set(labels.tolist()) <= {0, 1}
            else:
                assert len(val.dataset) == n_val

    with subtests.test(msg="worker counts go through the guard"):
        root = tmp_path / "guard"
        for name in ("nc", "pc"):
            for well in ("A01", "A02", "A03", "A04"):
                path = root / "train" / name / f"plate1_{well}_1_o1.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), (30, 80, 200)).save(path)
        IO.generate_loaders(str(root), classes=["nc", "pc"], n_jobs=2,
                            image_size=8, validation_split=0.25)
        IO.generate_cv_loaders(str(root), 2, batch_size=2, image_size=8,
                               n_jobs=3)
        assert {2, 3} <= set(guarded)

    with subtests.test(msg="fixed scaling without a split carries the contract"):
        root = _raw_dataset(tmp_path / "raw")
        train, val, fig = IO.generate_loaders(
            str(root), mode="train", image_size=32, batch_size=4,
            classes=["nc", "pc"], validation_split=0.0, crop_scaling="fixed",
            input_statistics="symmetric")
        contract = train.preprocessing_contract
        assert contract["crop_loading_policy"] == cp.RAW_FIXED_SCALE
        assert os.path.isfile(contract["scales_file"])
        assert fig is None

    with subtests.test(msg="an invalid mode returns None, saying so"):
        capsys.readouterr()
        assert IO.generate_loaders(str(tmp_path), mode="validate") is None
        assert "is not valid" in capsys.readouterr().out

    with subtests.test(msg="no split folder"):
        with pytest.raises(FileNotFoundError, match="Generate Training Data"):
            IO.generate_loaders(str(tmp_path / "nothing"), mode="train",
                                n_jobs=0)

    with subtests.test(msg="a missing class folder lists what is there"):
        root = _class_dirs(tmp_path / "half" / "train", rng, classes=("nc",),
                           n=2, size=32, stem="")
        with pytest.raises(FileNotFoundError) as caught:
            IO.generate_loaders(str(root.parent), mode="train",
                                classes=["nc", "pc"], n_jobs=0)
        assert "Missing:   ['pc']" in str(caught.value)
        assert "Available: ['nc']" in str(caught.value)


# ---------------------------------------------------------------------------
# _fixed_scale_contract
# ---------------------------------------------------------------------------

def test_fixed_scale_contract_records_scales_and_input_statistics(
        subtests, tmp_path):
    """The contract carries the scale file, its md5 and the chosen mean/std."""
    from spacr import classification_pixels as cp
    from spacr.normalization import IMAGENET_MEAN

    root = _raw_dataset(tmp_path / "ds")
    files = sorted(str(p) for p in (root / "train").rglob("*.png"))
    common = {"src": str(root), "image_size": 32, "channels": [1, 2, 3],
              "augment": False}
    first = IO._fixed_scale_contract(files, **common)

    with subtests.test(msg="an existing scale file is read, not rewritten"):
        again = IO._fixed_scale_contract(files, scale_file=first["scales_file"],
                                         **common)
        assert (again["scales_file"], again["scales_md5"], again["scales"]) == \
            (first["scales_file"], first["scales_md5"], first["scales"])
        assert cp._read_crop_scales(first["scales_file"])[1] == \
            first["scales_md5"]

    cases = [
        ("none", files, {"input_statistics": "none"}, None),
        ("imagenet without weights", files,
         {"input_statistics": "imagenet", "model_type": "custom"},
         list(IMAGENET_MEAN)),
        ("dataset over more than 512 crops",
         [files[i % len(files)] for i in range(513)],
         {"input_statistics": "dataset"}, "dataset"),
    ]
    for case_id, names, kwargs, mean in cases:
        with subtests.test(msg=case_id):
            contract = IO._fixed_scale_contract(names, **common, **kwargs)
            if mean is None:
                assert contract["mean"] is None and contract["std"] is None
            elif mean == "dataset":
                assert len(contract["mean"]) == len(contract["std"]) == 3
                assert all(0 < m < 1.5 for m in contract["mean"])
                assert all(s > 0 for s in contract["std"])
            else:
                assert contract["mean"] == pytest.approx(mean)
