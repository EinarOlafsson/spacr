"""Table-driven tests of ``spacr.object.generate_organelle_masks_sam``.

Two general tests walk case tables with pytest's ``subtests`` fixture: the
classical morphology routes (refusals, channel choice, resume, per-cell
masking, preprocessing, filters, saving, plotting) and the deep-learning
routes (Cellpose and U-Net).  The ``.npz`` stacks are deterministic blob
fields, the Cellpose model is a double declaring the installed 4.0.7
``eval`` signature (``channel_axis=MISSING_CHANNEL_AXIS``, checked by
:func:`tests.conftest.check_cellpose_eval_call`), and the U-Net is a saved
1x1 convolution.  CPU-only and offline.
"""
from __future__ import annotations

import os
import re
import sqlite3

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import spacr.object as O  # noqa: E402
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """Keep every device decision on the CPU, whatever the host has."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SPACR_DEVICE", "cpu")


CENTERS = ((16, 16), (16, 48), (48, 16), (48, 48))


def _blob_field(centers=CENTERS, radius=3):
    """64x64 uint16 field: background 100 with 29-px disks of 3000."""
    yy, xx = np.mgrid[:64, :64]
    img = np.full((64, 64), 100, dtype=np.uint16)
    for cy, cx in centers:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 3000
    return img


def _make_npz(src, filenames, channels=("blob",), name="stack1", centers=CENTERS):
    """One merged stack; ``channels`` None gives a channel-less (N, H, W) stack."""
    os.makedirs(src, exist_ok=True)
    flat = np.full((64, 64), 100, dtype=np.uint16)
    imgs = []
    for _ in filenames:
        if channels is None:
            imgs.append(_blob_field(centers))
        else:
            imgs.append(np.stack([_blob_field(centers) if c == "blob" else flat
                                  for c in channels], axis=-1))
    np.savez(os.path.join(src, f"{name}.npz"), data=np.stack(imgs),
             filenames=np.array(list(filenames)))


def _settings(**over):
    s = {"verbose": False, "save": True, "plot": False, "batch_size": 4, "n_jobs": 1,
         "nucleus_channel": None, "cell_channel": None, "pathogen_channel": None,
         "organelle_channel": 0, "organelle_morphology": "spots",
         "organelle_method": "otsu", "organelle_model_name": "cpsam"}
    s.update(over)
    return s


def _n_objects(mask):
    return len(set(np.unique(mask).tolist()) - {0})


def _counts(db_path):
    if not os.path.exists(db_path):
        return None
    con = sqlite3.connect(str(db_path))
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "object_counts" not in tables:
            return None
        return sorted(con.execute("SELECT file_name, count_type, object_count FROM object_counts"))
    finally:
        con.close()


def _run_case(tmp_path, case_id, monkeypatch, capsys, case):
    """Build the case's stacks, run the generator and collect what it left."""
    root = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", case_id)
    src = root / "masks"
    for stack in case.get("stacks", [dict(files=["f1.npy"])]):
        stack = dict(stack)
        _make_npz(str(src), stack.pop("files"), **stack)
    if case.get("no_stack"):
        src.mkdir(parents=True, exist_ok=True)
        (src / "not_a_stack.txt").write_text("ignored", encoding="utf-8")
    out_dir = src / "organelle_mask_stack"
    for fn, value in case.get("existing", {}).items():
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / fn, np.full((64, 64), value, dtype=np.uint16))
    if "cell_mask" in case:
        (root / "cell_mask_stack").mkdir(parents=True)
        np.save(root / "cell_mask_stack" / "f1.npy", case["cell_mask"])
    log = []
    capsys.readouterr()
    with monkeypatch.context() as mp:
        if "patch" in case:
            case["patch"](mp, log, root)
        settings = case["settings"](root) if callable(case["settings"]) else case["settings"]
        if "raises" in case:
            with pytest.raises(ValueError, match=case["raises"]):
                O.generate_organelle_masks_sam(str(src), settings, "organelle")
            returned = "raised"
        else:
            returned = O.generate_organelle_masks_sam(str(src), settings, "organelle")
    files = sorted(os.listdir(out_dir)) if out_dir.is_dir() else None
    masks = {f: np.load(out_dir / f) for f in files or []}
    return {"returned": returned, "files": files, "masks": masks,
            "objects": {f: _n_objects(m) for f, m in masks.items()},
            "rows": _counts(str(root / "measurements" / "measurements.db")),
            "measurements": (root / "measurements").exists(),
            "out": capsys.readouterr().out, "log": log, "figures": len(plt.get_fignums())}


def _check(obs, expect):
    for key, want in expect.items():
        if key == "out_has":
            for fragment in want:
                assert fragment in obs["out"], fragment
        elif key == "check":
            want(obs)
        else:
            assert obs[key] == want, key


def _rows(*pairs):
    return [(f, "organelle", n) for f, n in pairs]


def _first_mask_is_four_distinct_blobs(obs):
    mask = obs["masks"]["f1.npy"]
    assert (mask.shape, mask.dtype, int(mask[0, 0])) == ((64, 64), np.dtype(np.uint16), 0)
    labels = {int(mask[cy, cx]) for cy, cx in CENTERS}
    assert 0 not in labels and len(labels) == 4


def _only_top_blobs(obs):
    mask = obs["masks"]["f1.npy"]
    assert int(mask[32:, :].max()) == 0
    assert all(mask[cy, cx] != 0 for cy, cx in CENTERS[:2])


def _seen_parallel(mp, log, root):
    def fake(img_batch, classical_settings, n_jobs=1):
        log.append({"n_jobs": n_jobs, "shape": img_batch.shape, "dtype": img_batch.dtype,
                    "morphology": classical_settings["organelle_morphology"],
                    "method": classical_settings["organelle_method"],
                    "dl_keys": sorted({"organelle_model_name", "organelle_unet_model_path",
                                       "organelle_channel"} & set(classical_settings))})
        return [np.zeros(img_batch.shape[1:], dtype=np.int32) for _ in range(len(img_batch))]
    mp.setattr(O, "_segment_classical_parallel", fake)


def _segmenter_returns(value):
    return lambda mp, log, root: mp.setattr(O, "_segment_classical_parallel",
                                            lambda *a, **k: value)


def _plot_spy(mp, log, root):
    import spacr.plot as PL
    mp.setattr(PL, "plot_organelle_output", lambda img_batch, masks, settings, **kw:
               log.append({"shape": img_batch.shape, "n_masks": len(masks),
                           "morphology": settings["organelle_morphology"], **kw}))


def _no_show(mp, log, root):
    mp.setattr(plt, "show", lambda *a, **k: None)


CLASSICAL_CASES = [
    ("no .npz in src returns before the database exists", {
        "stacks": [], "no_stack": True, "settings": _settings(),
        "expect": {"returned": None, "files": None, "measurements": False,
                   "out_has": ["No .npz files found in"]}}),
    ("an unknown morphology is refused before any IO", {
        "settings": _settings(organelle_morphology="mesh"),
        "raises": "organelle_morphology must be one of",
        "expect": {"measurements": False, "files": None}}),
    ("a method the morphology does not offer is refused", {
        "settings": _settings(organelle_morphology="irregular", organelle_method="log"),
        "raises": re.escape("For morphology='irregular', method must be one of"),
        "expect": {"measurements": False}}),
    ("spots/otsu writes uint16 masks and counts", {
        "stacks": [dict(files=["f1.npy", "f2.npy"])], "settings": _settings(),
        "expect": {"files": ["f1.npy", "f2.npy"], "objects": {"f1.npy": 4, "f2.npy": 4},
                   "rows": _rows(("f1.npy", 4), ("f2.npy", 4)),
                   "check": _first_mask_is_four_distinct_blobs,
                   "out_has": ["Found 4.0 organelle/FOV, average size: 25.0 px2 "
                               "[batch 1/1,", "n_jobs=1]"]}}),
    ("batch_size 1 processes every field", {
        "stacks": [dict(files=["a.npy", "b.npy", "c.npy"])], "settings": _settings(batch_size=1),
        "expect": {"files": ["a.npy", "b.npy", "c.npy"],
                   "rows": _rows(("a.npy", 4), ("b.npy", 4), ("c.npy", 4))}}),
    ("two stacks share one output folder", {
        "stacks": [dict(files=["a.npy"], name="stack1"), dict(files=["b.npy"], name="stack2")],
        "settings": _settings(),
        "expect": {"files": ["a.npy", "b.npy"], "out_has": ["[batch 2/2,"]}}),
    ("verbose shows the organelle settings; n_jobs is clamped to 1", {
        "settings": _settings(verbose=True, n_jobs=0),
        "expect": {"files": ["f1.npy"],
                   "out_has": ["setting_key", "setting_value", "organelle_morphology",
                               "n_jobs=1]"]}}),
    ("the raw organelle channel is remapped to its dense position", {
        "stacks": [dict(files=["f1.npy"], channels=("flat", "blob"))],
        "settings": _settings(nucleus_channel=1, organelle_channel=3),
        "expect": {"objects": {"f1.npy": 4}}}),
    ("a recorded organelle position wins over the raw channel", {
        "stacks": [dict(files=["plate1_A01_1.npy"], channels=("blob", "flat"))],
        "settings": _settings(organelle_channel=1, cellpose_organelle_channel=0),
        "expect": {"objects": {"plate1_A01_1.npy": 4}}}),
    ("no organelle channel uses channel 0 of a 4-D stack", {
        "stacks": [dict(files=["f1.npy"], channels=("blob", "flat"))],
        "settings": _settings(organelle_channel=None),
        "expect": {"objects": {"f1.npy": 4}}}),
    ("a 3-D stack with a channel set is segmented whole", {
        "stacks": [dict(files=["f1.npy"], channels=None)],
        "settings": _settings(organelle_channel=0),
        "expect": {"objects": {"f1.npy": 4}, "check": _first_mask_is_four_distinct_blobs}}),
    ("a 3-D stack without a channel is segmented whole", {
        "stacks": [dict(files=["f1.npy"], channels=None)],
        "settings": _settings(organelle_channel=None),
        "expect": {"objects": {"f1.npy": 4}, "check": _first_mask_is_four_distinct_blobs}}),
    ("a fully processed stack is skipped untouched", {
        "stacks": [dict(files=["f1.npy", "f2.npy"])], "existing": {"f1.npy": 7, "f2.npy": 7},
        "settings": _settings(),
        "expect": {"rows": None,
                   "check": lambda obs: [np.testing.assert_array_equal(m, 7)
                                         for m in obs["masks"].values()],
                   "out_has": ["All files in stack1.npz already processed. Skipping."]}}),
    ("a partly processed stack runs the missing field only", {
        "stacks": [dict(files=["f1.npy", "f2.npy"])], "existing": {"f1.npy": 0},
        "settings": _settings(),
        "expect": {"objects": {"f1.npy": 0, "f2.npy": 4}, "rows": _rows(("f2.npy", 4)),
                   "out_has": ["1 of 2 files in stack1.npz already processed. "
                               "Skipping those."]}}),
    ("per-cell masking zeroes organelles outside cells", {
        "cell_mask": np.pad(np.ones((32, 64), np.uint16), ((0, 32), (0, 0))),
        "settings": _settings(organelle_mask_within_cells=True),
        "expect": {"objects": {"f1.npy": 2}, "check": _only_top_blobs,
                   "out_has": ["Per-cell masking enabled, using cell masks from"]}}),
    ("per-cell masking without cell masks warns and segments everything", {
        "settings": _settings(organelle_mask_within_cells=True),
        "expect": {"objects": {"f1.npy": 4},
                   "out_has": ["Warning: organelle_mask_within_cells=True but no "
                               "cell_mask_stack found at"]}}),
    ("rolling ball + CLAHE still find every blob", {
        "settings": _settings(organelle_rolling_ball=True, organelle_rolling_ball_radius=10,
                              organelle_clahe=True),
        "expect": {"check": _first_mask_is_four_distinct_blobs}}),
    ("max_area below every object empties the masks", {
        "settings": _settings(organelle_max_area=5),
        "expect": {"objects": {"f1.npy": 0}, "rows": _rows(("f1.npy", 0)),
                   "out_has": ["Found 0.0 organelle/FOV, average size: 0.0 px2"]}}),
    ("remove_border=False keeps an edge-clipped blob", {
        "stacks": [dict(files=["f1.npy"], centers=CENTERS + ((2, 32),))],
        "settings": _settings(organelle_remove_border=False),
        "expect": {"objects": {"f1.npy": 5},
                   "check": lambda obs: obs["masks"]["f1.npy"][2, 32] != 0 or pytest.fail()}}),
    ("remove_border=True drops the edge-clipped blob", {
        "stacks": [dict(files=["f1.npy"], centers=CENTERS + ((2, 32),))],
        "settings": _settings(organelle_remove_border=True),
        "expect": {"objects": {"f1.npy": 4},
                   "check": lambda obs: int(obs["masks"]["f1.npy"][0:6, :].max()) == 0
                   or pytest.fail()}}),
    ("save=False keeps counts and writes nothing", {
        "settings": _settings(save=False),
        "expect": {"files": [], "rows": _rows(("f1.npy", 4))}}),
    ("a segmenter returning None skips the batch", {
        "settings": _settings(), "patch": _segmenter_returns(None),
        "expect": {"files": [], "rows": None}}),
    ("a segmenter returning [] skips the batch", {
        "settings": _settings(), "patch": _segmenter_returns([]),
        "expect": {"files": [], "rows": None}}),
    ("workers get the float32 plane and only the classical settings", {
        "stacks": [dict(files=["f1.npy", "f2.npy"])], "settings": _settings(n_jobs=-4),
        "patch": _seen_parallel,
        "expect": {"log": [{"n_jobs": 1, "shape": (2, 64, 64), "dtype": np.dtype(np.float32),
                            "morphology": "spots", "method": "otsu", "dl_keys": []}],
                   "objects": {"f1.npy": 0, "f2.npy": 0}}}),
    ("network/hysteresis segments every blob", {
        "settings": _settings(organelle_morphology="network", organelle_method="hysteresis",
                              organelle_min_area=5),
        "expect": {"objects": {"f1.npy": 4}}}),
    ("irregular/otsu segments every blob", {
        "settings": _settings(organelle_morphology="irregular", organelle_method="otsu",
                              organelle_min_area=5),
        "expect": {"objects": {"f1.npy": 4}}}),
    ("ring/otsu segments every blob", {
        "settings": _settings(organelle_morphology="ring", organelle_method="otsu",
                              organelle_min_area=5),
        "expect": {"objects": {"f1.npy": 4}}}),
    ("plot=True hands the batch and masks to plot_organelle_output", {
        "stacks": [dict(files=["f1.npy", "f2.npy"])],
        "settings": _settings(plot=True, examples_to_plot=5), "patch": _plot_spy,
        "expect": {"log": [{"shape": (2, 64, 64), "n_masks": 2, "morphology": "spots",
                            "nr": 2, "cmap": "inferno", "figuresize": 10,
                            "print_object_number": True}]}}),
    ("plot=True renders one real figure", {
        "settings": _settings(plot=True), "patch": _no_show,
        "expect": {"figures": 1, "objects": {"f1.npy": 4}}}),
]


def test_generate_organelle_masks_sam_classical_routes(subtests, tmp_path, monkeypatch, capsys):
    """Classical morphology runs refuse bad settings, pick the right plane, resume,
    mask within cells, filter, save, count and plot exactly as configured."""
    for case_id, case in CLASSICAL_CASES:
        with subtests.test(msg=case_id):
            obs = _run_case(tmp_path, case_id, monkeypatch, capsys, case)
            _check(obs, case["expect"])
            plt.close("all")


# ---------------------------------------------------------------------------
# deep-learning routes
# ---------------------------------------------------------------------------

class _FakeCellposeModel:
    """``CellposeModel`` double, installed 4.0.7 ``eval`` signature, three return values.

    Every image gets two squares: label 1 (100 px) and label 2 (64 px).
    """

    def __init__(self):
        self.calls = []

    def eval(self, x, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
             invert=False, rescale=None, diameter=None, flow_threshold=0.4,
             cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
             max_size_fraction=0.4, niter=None, augment=False,
             tile_overlap=0.1, bsize=256, compute_masks=True, progress=None):
        check_cellpose_eval_call(x, channel_axis)
        self.calls.append({"n": len(x), "batch_size": batch_size, "normalize": normalize,
                           "channel_axis": channel_axis, "diameter": diameter,
                           "flow_threshold": flow_threshold,
                           "cellprob_threshold": cellprob_threshold, "resample": resample})
        masks = []
        for _ in x:
            m = np.zeros((64, 64), dtype=np.int32)
            m[5:15, 5:15] = 1
            m[40:48, 40:48] = 2
            masks.append(m)
        n = len(x)
        flows = [np.zeros((n, 64, 64, 3), np.float32), np.zeros((3, n, 64, 64), np.float32),
                 np.zeros((n, 64, 64), np.float32), np.zeros((n, 64, 64), np.float32)]
        return masks, flows, None


def _save_tiny_unet(path, out_channels=1, weight=5.0):
    """A 1x1 conv: sigmoid(w * z-scored x) > 0.5 exactly where x is above the mean."""
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Conv2d(1, out_channels, kernel_size=1, bias=True))
    with torch.no_grad():
        model[0].weight.fill_(0.0)
        model[0].weight[0, 0, 0, 0] = weight
        if out_channels > 1:
            model[0].weight[1, 0, 0, 0] = -weight
        model[0].bias.fill_(0.0)
    torch.save(model, str(path))
    return str(path)


def test_generate_organelle_masks_sam_deep_learning_routes(subtests, tmp_path, monkeypatch, capsys):
    """Cellpose is chosen once with the organelle object settings and evaluated per
    batch; the U-Net route segments with the saved model and refuses a missing path."""
    import spacr.utils as U

    model = {}

    def choose(mp, log, root):
        model["m"] = _FakeCellposeModel()

        def fake_choose_model(model_name, device, object_type=None, restore_type=None,
                              object_settings=None):
            log.append({"model_name": model_name, "object_type": object_type,
                        "device": str(device), "restore_type": restore_type,
                        "object_settings": object_settings})
            return model["m"]
        mp.setattr(U, "_choose_model", fake_choose_model)

    cellpose_settings = {"model_name": "cpsam", "diameter": 30, "minimum_size": 10,
                         "maximum_size": None, "resample": True, "filter_size": False,
                         "filter_intensity": False, "remove_border_objects": False,
                         "merge": False}
    cellpose_eval = {"n": 2, "batch_size": 2, "normalize": False, "channel_axis": -1,
                     "diameter": None, "flow_threshold": 0.4, "cellprob_threshold": 0.0,
                     "resample": True}
    cases = [
        ("cellpose: one model, organelle settings, its masks saved", {
            "stacks": [dict(files=["f1.npy", "f2.npy"])],
            "settings": _settings(organelle_method="cellpose"), "patch": choose,
            "evals": [cellpose_eval],
            "expect": {"log": [{"model_name": "cpsam", "object_type": "organelle",
                                "device": "cpu", "restore_type": None,
                                "object_settings": cellpose_settings}],
                       "objects": {"f1.npy": 2, "f2.npy": 2},
                       "rows": _rows(("f1.npy", 2), ("f2.npy", 2)),
                       "check": lambda obs: [int(obs["masks"]["f1.npy"][r, c])
                                             for r, c in ((10, 10), (44, 44), (30, 30))]
                       == [1, 2, 0] or pytest.fail(),
                       "out_has": ["n_jobs=GPU]"]}}),
        ("cellpose: loaded once for two stacks, evaluated per stack", {
            "stacks": [dict(files=["a.npy"], name="stack1"), dict(files=["b.npy"], name="stack2")],
            "settings": _settings(organelle_method="cellpose"), "patch": choose,
            "evals": [dict(cellpose_eval, n=1, batch_size=1)] * 2,
            "expect": {"files": ["a.npy", "b.npy"],
                       "check": lambda obs: len(obs["log"]) == 1 or pytest.fail()}}),
        ("unet: the saved model finds every blob", {
            "settings": lambda root: _settings(
                organelle_morphology="network", organelle_method="unet",
                organelle_unet_model_path=_save_tiny_unet(root / "unet.pt")),
            "expect": {"objects": {"f1.npy": 4},
                       "check": lambda obs: [bool(obs["masks"]["f1.npy"][cy, cx])
                                             for cy, cx in ((0, 0),) + CENTERS]
                       == [False, True, True, True, True] or pytest.fail()}}),
        ("unet: two-channel logits use channel 0, skeleton thinned", {
            "settings": lambda root: _settings(
                organelle_morphology="network", organelle_method="unet",
                organelle_skeletonize=True, organelle_min_area=1,
                organelle_unet_model_path=_save_tiny_unet(root / "unet2.pt", out_channels=2)),
            "expect": {"objects": {"f1.npy": 4},
                       # skeleton + 1-px dilation: 14 px per blob instead of the 29-px disk
                       "check": lambda obs: int((obs["masks"]["f1.npy"] > 0).sum()) == 4 * 14
                       or pytest.fail("skeleton area")}}),
        ("unet: no model path is refused", {
            "settings": _settings(organelle_morphology="network", organelle_method="unet"),
            "raises": "organelle_unet_model_path",
            "expect": {"files": None}}),
    ]
    for case_id, case in cases:
        with subtests.test(msg=case_id):
            model.clear()
            obs = _run_case(tmp_path, case_id, monkeypatch, capsys, case)
            _check(obs, case["expect"])
            if "evals" in case:
                assert model["m"].calls == case["evals"]
