"""Cellpose-4 leftovers in spacrops.py and submodules.py.

Two things Cellpose 4 (the SAM release) changed that these modules had not
caught up with:

* ``cellpose.models.Cellpose`` is gone. The pre-SAM wrapper class went with
  the pre-SAM model zoo, leaving only ``CellposeModel``, so
  ``spacrStitcher(outline_source='cellpose')`` raised ``AttributeError:
  module 'cellpose.models' has no attribute 'Cellpose'`` before it segmented
  anything. ``CellposeModel.eval`` also returns three values now, not four,
  so the ``masks, _, _, _ = ...`` unpack on the next line was broken too.
* ``train_cellpose`` fine-tunes ``cpsam``, but stamped every checkpoint it
  wrote with ``_cyto_`` -- the name of the Cellpose-3 model it used to
  fine-tune.

No weights are loaded here: ``cellpose.models.CellposeModel`` is replaced
with a recorder so the call shape is what is under test. CPU only, offline.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call


# ---------------------------------------------------------------------------
# a CellposeModel stand-in that records how it was built and called
# ---------------------------------------------------------------------------

class _FakeCellposeModel:
    """Records construction kwargs and eval calls; returns a fixed labelling."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.eval_calls = []
        self.n_returns = 3
        type(self).instances.append(self)

    def eval(self, x, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
        # channel_axis is named rather than swallowed by **kwargs. These two
        # call sites (spacrops._cellpose_labels, submodules' test loops) are
        # the ones that deliberately DO NOT pass an axis -- they hand Cellpose
        # a plain 2-D image and let it auto-detect -- so the contract here is
        # "whatever axis arrives must be one convert_image accepts", not
        # "an axis must arrive". If either site starts naming an axis, the
        # value is checked from that moment on rather than silently absorbed.
        check_cellpose_eval_call(x, channel_axis,
                                 require_channel_axis=False)
        self.eval_calls.append((x, {"channel_axis": channel_axis, **kwargs}
                                if channel_axis is not MISSING_CHANNEL_AXIS
                                else kwargs))
        labels = np.zeros(np.asarray(x).shape[:2], dtype=np.int32)
        labels[4:12, 4:12] = 1
        labels[20:28, 20:28] = 2
        flows, styles, diams = [None], np.zeros(256), 30.0
        if self.n_returns == 4:
            return labels, flows, styles, diams
        return labels, flows, styles


@pytest.fixture
def fake_cp(monkeypatch):
    """Swap cellpose.models.CellposeModel for the recorder."""
    from cellpose import models as cp_models
    from spacr.utils import reset_cellpose_model_reports

    _FakeCellposeModel.instances = []
    monkeypatch.setattr(cp_models, "CellposeModel", _FakeCellposeModel)
    # The substitution notices are latched per run; start clean so a test
    # that asserts on them is not silenced by an earlier test.
    reset_cellpose_model_reports()
    yield _FakeCellposeModel
    reset_cellpose_model_reports()


def _img(h=32, w=32):
    img = np.full((h, w), 20, np.uint8)
    img[4:12, 4:12] = 220
    img[20:28, 20:28] = 200
    return img


def _stitcher(tmp_path, **kw):
    from spacr.spacrops import spacrStitcher
    kw.setdefault("outdir", str(tmp_path / "sbs_out"))
    return spacrStitcher(**kw)


# ---------------------------------------------------------------------------
# the class that no longer exists
# ---------------------------------------------------------------------------

def test_installed_cellpose_has_no_Cellpose_class():
    """The premise: what spacrops.py called is not there any more."""
    from cellpose import models as cp_models
    assert not hasattr(cp_models, "Cellpose")
    assert hasattr(cp_models, "CellposeModel")


def test_spacrops_no_longer_calls_the_removed_class():
    """AST, not grep: a docstring may quote the old call, code may not."""
    import ast
    import spacr.spacrops as spacrops

    tree = ast.parse(open(spacrops.__file__, encoding="utf-8").read())
    called = set()
    model_types = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                called.add(fn.attr)
            for kw in node.keywords:
                if kw.arg == "model_type" and isinstance(kw.value, ast.Constant):
                    model_types.add(kw.value.value)
    assert "Cellpose" not in called, "models.Cellpose was removed in Cellpose 4"
    assert not model_types, (
        f"model_type= is accepted-and-ignored by Cellpose 4; still passed "
        f"{model_types}")


# ---------------------------------------------------------------------------
# the ported path
# ---------------------------------------------------------------------------

def test_foreground_mask_segments_with_CellposeModel(tmp_path, fake_cp):
    st = _stitcher(tmp_path, outline_source="cellpose")
    mask = st._foreground_mask(_img())

    assert mask.dtype == np.bool_
    assert mask.sum() == 8 * 8 * 2
    assert len(fake_cp.instances) == 1
    built = fake_cp.instances[0].kwargs
    assert built["pretrained_model"] == "cpsam"
    # model_type / diam_mean are accepted-and-ignored by Cellpose 4; not passed.
    assert "model_type" not in built
    assert "diam_mean" not in built


def test_eval_is_called_without_the_removed_channels_kwarg(tmp_path, fake_cp):
    st = _stitcher(tmp_path, outline_source="cellpose")
    st._foreground_mask(_img())
    _x, kwargs = fake_cp.instances[0].eval_calls[0]
    assert "channels" not in kwargs
    assert kwargs["diameter"] is None
    assert kwargs["flow_threshold"] == 0.4
    assert kwargs["cellprob_threshold"] == 0.0


def test_cellpose_diameter_reaches_eval(tmp_path, fake_cp):
    """diameter is the one pre-SAM knob Cellpose 4 still honours."""
    st = _stitcher(tmp_path, outline_source="cellpose", cellpose_diameter=45)
    st._foreground_mask(_img())
    _x, kwargs = fake_cp.instances[0].eval_calls[0]
    assert kwargs["diameter"] == 45.0


def test_three_and_four_value_eval_returns_both_work(tmp_path, fake_cp):
    """Cellpose 4 returns (masks, flows, styles); 3 returned a 4th, diams.

    The old `masks, _, _, _ = model.eval(...)` handled only the second.
    """
    st = _stitcher(tmp_path, outline_source="cellpose")
    three = st._cellpose_labels(_img())
    fake_cp.instances[0].n_returns = 4
    four = st._cellpose_labels(_img())
    assert np.array_equal(three, four)
    assert set(np.unique(three)) == {0, 1, 2}


def test_the_model_is_built_once_not_once_per_tile(tmp_path, fake_cp):
    """It used to be constructed inside the per-tile mask helpers.

    cpsam is a 1.2 GB checkpoint; reloading it per image in a well is the
    difference between a stitch that finishes and one that does not.
    """
    st = _stitcher(tmp_path, outline_source="cellpose")
    for _ in range(4):
        st._foreground_mask(_img())
    st._outline_mask(_img())
    assert len(fake_cp.instances) == 1
    assert len(fake_cp.instances[0].eval_calls) == 5


def test_outline_mask_uses_the_same_ported_path(tmp_path, fake_cp):
    st = _stitcher(tmp_path, outline_source="cellpose", line_thickness=1)
    edges = st._outline_mask(_img())
    assert edges.dtype == np.bool_
    assert edges.any(), "two 8x8 squares must produce outline pixels"
    assert len(fake_cp.instances) == 1


def test_otsu_and_none_never_touch_cellpose(tmp_path, fake_cp):
    for source in ("otsu", "none"):
        st = _stitcher(tmp_path / source, outline_source=source)
        st._foreground_mask(_img())
        st._outline_mask(_img())
    assert fake_cp.instances == []


# ---------------------------------------------------------------------------
# user checkpoints
# ---------------------------------------------------------------------------

def test_a_user_checkpoint_is_loaded_as_given(tmp_path, fake_cp, capsys):
    ckpt = tmp_path / "mine_cpsam_e500_X512_Y512.CP_model"
    ckpt.write_bytes(b"not really weights")
    st = _stitcher(tmp_path, outline_source="cellpose", cellpose_model=str(ckpt))
    st._foreground_mask(_img())
    assert fake_cp.instances[0].kwargs["pretrained_model"] == str(ckpt)
    assert "fine-tuned" in capsys.readouterr().out


def test_a_missing_checkpoint_path_stops_the_run(tmp_path, fake_cp):
    """Cellpose would quietly fall back to stock cpsam; spaCR refuses to."""
    st = _stitcher(tmp_path, outline_source="cellpose",
                   cellpose_model=str(tmp_path / "gone.CP_model"))
    with pytest.raises(FileNotFoundError, match="checkpoint path"):
        st._foreground_mask(_img())
    assert fake_cp.instances == []


def test_a_legacy_model_name_is_mapped_forward_not_passed_on(tmp_path, fake_cp, capsys):
    """'nuclei' is what the old code hard-coded; it names no weights now."""
    st = _stitcher(tmp_path, outline_source="cellpose", cellpose_model="nuclei")
    st._foreground_mask(_img())
    assert fake_cp.instances[0].kwargs["pretrained_model"] == "cpsam"
    out = capsys.readouterr().out
    assert "nuclei" in out and "cpsam" in out


def test_defaults_declare_the_new_keys():
    from spacr.spacrops import get_preprocess_ops_settings
    s = get_preprocess_ops_settings({})
    assert s["cellpose_model"] == "cpsam"
    assert s["cellpose_diameter"] is None


def test_stitcher_defaults_match_the_settings_defaults(tmp_path):
    st = _stitcher(tmp_path)
    assert st.cellpose_model == "cpsam"
    assert st.cellpose_diameter is None
    assert st._cp_model is None, "the model must be built lazily"


# ---------------------------------------------------------------------------
# submodules.train_cellpose -- the checkpoint name
# ---------------------------------------------------------------------------

def test_trained_checkpoint_is_named_cpsam_not_cyto(tmp_path, monkeypatch):
    """train_cellpose fine-tunes cpsam, so the filename must say cpsam."""
    import spacr.submodules as submodules

    seen = {}

    class _Model:
        def __init__(self, **kw):
            seen["model_kwargs"] = kw
            self.net = object()

    def _train_seg(net, **kw):
        seen["train_kwargs"] = kw

    monkeypatch.setattr(submodules.cp_models, "CellposeModel", _Model)
    monkeypatch.setattr(submodules.train_cp, "train_seg", _train_seg)
    monkeypatch.setattr(submodules, "plot_cellpose_batch", lambda *a, **k: None)

    import tifffile
    for sub in ("images", "masks"):
        (tmp_path / "train" / sub).mkdir(parents=True)
    for i in range(2):
        name = f"img{i}.tif"
        tifffile.imwrite(tmp_path / "train" / "images" / name,
                         np.full((32, 32), 100, np.uint16))
        lbl = np.zeros((32, 32), np.uint16)
        lbl[4:12, 4:12] = 1
        tifffile.imwrite(tmp_path / "train" / "masks" / name, lbl)

    submodules.train_cellpose({
        "src": str(tmp_path), "model_name": "mymodel", "n_epochs": 20,
        "target_size": 16, "augment": False, "batch_size": 2,
        "learning_rate": 0.05, "weight_decay": 1e-4,
    })

    name = seen["train_kwargs"]["model_name"]
    assert name == "mymodel_cpsam_e20_X16_Y16.CP_model"
    assert "_cyto_" not in name
    # It really is the SAM checkpoint being fine-tuned.
    assert seen["model_kwargs"]["pretrained_model"] == "cpsam"
    # The settings snapshot is written under the same name.
    assert (tmp_path / "settings" /
            "mymodel_cpsam_e20_X16_Y16.CP_model.csv").exists()


def test_checkpoints_written_under_the_old_name_still_resolve(tmp_path):
    """Nothing parses the infix -- old names keep working.

    model_zoo recognises a Cellpose checkpoint by its ``.CP_model`` suffix,
    so a checkpoint trained before the rename is still discovered, still
    classified as a Cellpose model, and still loadable by path.
    """
    from spacr import model_zoo
    from spacr.utils import _resolve_cellpose_pretrained

    old = tmp_path / "models" / "cellpose_model" / "toxo_cyto_e25_X512_Y512.CP_model"
    new = tmp_path / "models" / "cellpose_model" / "toxo_cpsam_e25_X512_Y512.CP_model"
    old.parent.mkdir(parents=True)
    for p in (old, new):
        p.write_bytes(b"weights")
        assert model_zoo.classify_kind(str(p)) == "cellpose"
        assert _resolve_cellpose_pretrained(str(p)) == str(p)
