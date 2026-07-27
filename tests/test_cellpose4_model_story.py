"""What spaCR says, and does, about Cellpose 4 having exactly one model.

Cellpose 4 ships ``cpsam`` and nothing else: ``models.MODEL_NAMES ==
['cpsam']``, ``CellposeModel(model_type=...)`` and ``CellposeModel(diam_mean=
...)`` are logged as "not used in v4.0.1+" and dropped, and ``eval(channels=
...)`` is deprecated. ``diameter`` is the one parameter of that group that
still does something — ``eval`` rescales the image by ``30. / diameter``.

Three things follow, and each is pinned here:

1. The substitution notice must name the object type that was actually
   requested. It used to default to 'cell', so asking for the nucleus model
   printed "using 'cpsam' for cell" — two facts in one line, one of them made
   up.

2. It must be said once per object type per run, not once per field.

3. A path to a checkpoint the user trained in Train Cellpose must survive all
   the way to ``pretrained_model``. ``_choose_model`` once hard-coded 'cpsam'
   and discarded every such model silently, and
   ``generate_cellpose_masks_sam`` — the pipeline's DEFAULT path — still did
   until this file was written.

Everything runs on CPU with a recording double in place of Cellpose. No
weights are downloaded and no GPU is touched.
"""
from __future__ import annotations

import sqlite3
import types
from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

import spacr.utils as U
import spacr.settings as S


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_reports():
    """Each test starts a fresh 'run' so the once-per-run cache never leaks."""
    U.reset_cellpose_model_reports()
    yield
    U.reset_cellpose_model_reports()


class _RecordingCellposeModel:
    """Records the kwargs Cellpose would have been constructed with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.pretrained_model = kwargs.get("pretrained_model")


@pytest.fixture
def fake_cellpose(monkeypatch):
    """Swap the Cellpose constructor for a recorder (no weights, no GPU)."""
    monkeypatch.setattr(U.cp_models, "CellposeModel", _RecordingCellposeModel)
    return _RecordingCellposeModel


# ---------------------------------------------------------------------------
# 1. The message names the object type that was asked for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("object_type", ["cell", "nucleus", "pathogen",
                                         "organelle"])
def test_the_notice_names_the_object_type_that_was_requested(
        fake_cellpose, capsys, object_type):
    """Asking for the nucleus model must not report 'for cell'."""
    U._choose_model("nuclei", device="cpu", object_type=object_type)
    out = capsys.readouterr().out
    assert f"for {object_type}" in out
    for other in ("cell", "nucleus", "pathogen", "organelle"):
        if other != object_type:
            assert f"for {other}" not in out


def test_an_unstated_object_type_is_left_unstated(fake_cellpose, capsys):
    """The default used to be 'cell', so a call that named no object type
    still announced one. Saying nothing is the only honest option."""
    U._choose_model("nucleus", device="cpu")
    out = capsys.readouterr().out
    assert "nucleus" in out and "cpsam" in out
    assert " for " not in out


def test_the_notice_says_what_was_asked_for_and_what_will_run(
        fake_cellpose, capsys):
    U._choose_model("cyto2", device="cpu", object_type="cell")
    out = capsys.readouterr().out
    assert "'cyto2'" in out          # what the settings asked for
    assert "predates Cellpose-SAM" in out
    assert "'cpsam'" in out          # what will actually run
    assert "for cell" in out


def test_a_missing_checkpoint_error_names_the_object_type(fake_cellpose,
                                                          tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        U._choose_model(str(tmp_path / "gone" / "m.pth"), device="cpu",
                        object_type="pathogen")
    assert "for pathogen" in str(exc.value)


def test_a_missing_checkpoint_error_invents_no_object_type(fake_cellpose,
                                                           tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        U._choose_model(str(tmp_path / "gone" / "m.pth"), device="cpu")
    msg = str(exc.value)
    for object_type in ("cell", "nucleus", "pathogen", "organelle"):
        assert f"for {object_type}" not in msg


# ---------------------------------------------------------------------------
# 2. Once per object type per run, not once per field
# ---------------------------------------------------------------------------

def test_the_notice_is_printed_once_however_many_fields(fake_cellpose, capsys):
    """A 1000-field plate printed the same warning a thousand times."""
    for _ in range(1000):
        U._choose_model("cyto3", device="cpu", object_type="cell")
    out = capsys.readouterr().out
    assert out.count("predates Cellpose-SAM") == 1


def test_each_object_type_still_gets_its_own_line(fake_cellpose, capsys):
    """Deduplication must not hide that two different objects were mapped."""
    for object_type in ("cell", "nucleus", "pathogen"):
        for _ in range(5):
            U._choose_model("cyto", device="cpu", object_type=object_type)
    out = capsys.readouterr().out
    assert out.count("predates Cellpose-SAM") == 3
    for object_type in ("cell", "nucleus", "pathogen"):
        assert f"for {object_type}" in out


def test_a_different_model_name_is_reported_again(fake_cellpose, capsys):
    U._choose_model("cyto", device="cpu", object_type="cell")
    U._choose_model("cyto2", device="cpu", object_type="cell")
    out = capsys.readouterr().out
    assert out.count("predates Cellpose-SAM") == 2


def test_reset_re_arms_the_notice_for_the_next_run(fake_cellpose, capsys):
    """A GUI session segmenting a second plate must not inherit the first
    run's silence — preprocess_generate_masks resets at the top."""
    U._choose_model("nuclei", device="cpu", object_type="nucleus")
    U._choose_model("nuclei", device="cpu", object_type="nucleus")
    assert capsys.readouterr().out.count("predates Cellpose-SAM") == 1

    U.reset_cellpose_model_reports()
    U._choose_model("nuclei", device="cpu", object_type="nucleus")
    assert capsys.readouterr().out.count("predates Cellpose-SAM") == 1


def test_the_unknown_model_notice_is_also_once_per_run(fake_cellpose, capsys):
    for _ in range(50):
        U._choose_model("no_such_model", device="cpu", object_type="cell")
    assert capsys.readouterr().out.count("Unknown Cellpose model") == 1


def test_the_checkpoint_notice_is_also_once_per_run(fake_cellpose, tmp_path,
                                                    capsys):
    ckpt = tmp_path / "my_cells"
    ckpt.write_bytes(b"weights")
    for _ in range(50):
        U._choose_model(str(ckpt), device="cpu", object_type="cell")
    assert capsys.readouterr().out.count("Loading fine-tuned") == 1


def test_report_once_returns_whether_it_printed(capsys):
    assert U._report_cellpose_once(("k",), "said once") is True
    assert U._report_cellpose_once(("k",), "said once") is False
    assert capsys.readouterr().out.count("said once") == 1


# ---------------------------------------------------------------------------
# 3. A user-trained checkpoint survives — the regression that must not return
# ---------------------------------------------------------------------------

def test_a_checkpoint_path_survives_choose_model(fake_cellpose, tmp_path):
    """pretrained_model was once hard-coded to 'cpsam', so every model from
    Train Cellpose was discarded and the stock weights ran instead."""
    ckpt = tmp_path / "trained_on_my_cells.pth"
    ckpt.write_bytes(b"not really weights, but it exists")

    model = U._choose_model(str(ckpt), device="cpu", object_type="cell")

    assert model.kwargs["pretrained_model"] == str(ckpt)
    # ...and none of the arguments Cellpose 4 drops on the floor.
    assert "model_type" not in model.kwargs
    assert "diam_mean" not in model.kwargs


def test_a_checkpoint_path_survives_resolution(tmp_path):
    ckpt = tmp_path / "trained"
    ckpt.write_bytes(b"x")
    assert U._resolve_cellpose_pretrained(str(ckpt)) == str(ckpt)


def test_a_checkpoint_path_survives_the_settings_layer(tmp_path):
    """normalize_cellpose_model_name maps legacy names forward but must never
    touch a path — that is the one model choice Cellpose 4 still honours."""
    ckpt = tmp_path / "trained.pth"
    ckpt.write_bytes(b"x")
    assert S.normalize_cellpose_model_name(str(ckpt)) == str(ckpt)
    # even a path that happens to be named after a removed model
    legacy_named = tmp_path / "cyto2"
    legacy_named.write_bytes(b"x")
    assert S.normalize_cellpose_model_name(str(legacy_named)) == str(legacy_named)


@pytest.mark.parametrize("object_type", ["cell", "nucleus", "pathogen"])
def test_a_checkpoint_named_in_settings_reaches_object_settings(
        tmp_path, object_type):
    ckpt = tmp_path / f"my_{object_type}s.pth"
    ckpt.write_bytes(b"x")
    settings = S.set_default_settings_preprocess_generate_masks(
        {"src": str(tmp_path), f"{object_type}_model_name": str(ckpt),
         "verbose": False})
    out = S._get_object_settings(object_type, settings)
    assert out["model_name"] == str(ckpt)


# ---------------------------------------------------------------------------
# ...all the way through the pipeline's DEFAULT segmentation path
# ---------------------------------------------------------------------------

def _write_npz(src: Path, n=2, h=32, w=32, c=2, seed=0):
    """Write one pre-batched npz exactly like spaCR's preprocessing does."""
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 4000, size=(n, h, w, c)).astype(np.uint16)
    filenames = np.array([f"plate1_A01_{i + 1}.npy" for i in range(n)])
    np.savez(src / "batch1.npz", data=data, filenames=filenames)


@pytest.fixture
def sam_pipeline(monkeypatch):
    """CPU-only stand-in for the model generate_cellpose_masks_sam builds."""
    import torch
    import spacr.object as O
    import spacr.plot as PL

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(PL, "plot_cellpose4_output", lambda *a, **k: None)

    holder = {"model": None}

    class _M:
        def __init__(self, gpu=None, pretrained_model=None, device=None, **kw):
            self.pretrained_model = pretrained_model
            holder["model"] = self

        def eval(self, x=None, **kwargs):
            holder.setdefault("eval_kwargs", []).append(kwargs)
            imgs = [np.asarray(im) for im in x]
            masks, flows = [], []
            for im in imgs:
                m = np.zeros(im.shape[:2], dtype=np.uint16)
                m[2:8, 2:8] = 1
                masks.append(m)
                flows.append(np.zeros(im.shape[:2], dtype=np.float32))
            return masks, flows, None, None

    monkeypatch.setattr(O, "cp_models",
                        types.SimpleNamespace(CellposeModel=_M))
    return holder


def _mask_settings(src, **over):
    s = {"src": str(src), "cell_channel": 0, "nucleus_channel": 1,
         "pathogen_channel": None, "magnification": 20, "batch_size": 50,
         "verbose": False, "plot": False, "save": True, "timelapse": False,
         "n_jobs": 1, "cell_min_object_area": 0,
         "nucleus_min_object_area": 0, "pathogen_min_object_area": 0}
    s.update(over)
    return s


def test_the_sam_generator_loads_the_users_checkpoint(tmp_path, sam_pipeline):
    """generate_cellpose_masks_sam is the pipeline's DEFAULT path and it
    hard-coded pretrained_model='cpsam', so a model trained in spaCR's own
    Train Cellpose module was discarded on every run without a word."""
    import spacr.object as O

    ckpt = tmp_path / "my_cells.pth"
    ckpt.write_bytes(b"x")
    src = tmp_path / "stack"
    _write_npz(src)

    O.generate_cellpose_masks_sam(
        str(src), _mask_settings(src, cell_model_name=str(ckpt)), "cell")

    assert sam_pipeline["model"].pretrained_model == str(ckpt)


def test_the_sam_generator_still_defaults_to_cpsam(tmp_path, sam_pipeline):
    import spacr.object as O

    src = tmp_path / "stack"
    _write_npz(src)
    O.generate_cellpose_masks_sam(str(src), _mask_settings(src), "cell")
    assert sam_pipeline["model"].pretrained_model == "cpsam"


def test_the_sam_generator_honours_the_older_pathogen_model_key(tmp_path,
                                                                sam_pipeline):
    """pathogen_model predates the per-object *_model_name keys; the v1
    generator honoured it and the SAM one must not disagree."""
    import spacr.object as O

    ckpt = tmp_path / "my_pathogens.pth"
    ckpt.write_bytes(b"x")
    src = tmp_path / "stack"
    _write_npz(src)

    O.generate_cellpose_masks_sam(
        str(src), _mask_settings(src, pathogen_channel=1,
                                 pathogen_model=str(ckpt)), "pathogen")

    assert sam_pipeline["model"].pretrained_model == str(ckpt)


def test_the_sam_generator_stops_on_a_checkpoint_that_is_not_there(
        tmp_path, sam_pipeline):
    """Falling back to cpsam would segment a whole plate with the wrong
    weights and say nothing."""
    import spacr.object as O

    src = tmp_path / "stack"
    _write_npz(src)
    missing = str(tmp_path / "gone" / "m.pth")

    with pytest.raises(FileNotFoundError) as exc:
        O.generate_cellpose_masks_sam(
            str(src), _mask_settings(src, cell_model_name=missing), "cell")
    assert "cpsam" in str(exc.value)
    assert not (src / "cell_mask_stack").exists() or \
        list((src / "cell_mask_stack").iterdir()) == []


# ---------------------------------------------------------------------------
# 4. Old settings files keep loading: legacy names are aliases, not choices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("legacy", list(U.LEGACY_CELLPOSE_MODELS))
def test_every_legacy_name_still_loads_and_maps_to_cpsam(legacy):
    assert S.normalize_cellpose_model_name(legacy) == "cpsam"


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_blank_model_setting_means_the_stock_model(blank):
    assert S.normalize_cellpose_model_name(blank) == "cpsam"


def test_the_settings_layer_reports_the_substitution_once(capsys):
    for _ in range(20):
        S.normalize_cellpose_model_name("cyto3", object_type="cell",
                                        key="cell_model_name")
    out = capsys.readouterr().out
    assert out.count("predates Cellpose-SAM") == 1
    assert "cell_model_name" in out     # names the setting to edit
    assert "for cell" in out


def test_a_legacy_settings_file_segments_with_cpsam(tmp_path, sam_pipeline,
                                                     capsys):
    """The whole point of keeping the aliases: a settings CSV written against
    Cellpose 3 still runs, it just runs the model that exists."""
    import spacr.object as O

    src = tmp_path / "stack"
    _write_npz(src)
    O.generate_cellpose_masks_sam(
        str(src), _mask_settings(src, nucleus_model_name="nuclei"), "nucleus")

    assert sam_pipeline["model"].pretrained_model == "cpsam"
    out = capsys.readouterr().out
    assert "nuclei" in out and "for nucleus" in out
    assert "for cell" not in out


def test_object_settings_default_to_cpsam_when_the_key_is_absent():
    """Hand-built settings dicts predating these keys must still work."""
    base = {"magnification": 20, "verbose": False, "merge_pathogens": False,
            "nucleus_channel": None}
    assert U._get_object_settings("cell", base)["model_name"] == "cpsam"


# ---------------------------------------------------------------------------
# 5. What Cellpose 4 still honours, and what it drops
# ---------------------------------------------------------------------------

def test_choose_model_never_passes_the_arguments_cellpose4_drops(
        fake_cellpose, tmp_path):
    """model_type and diam_mean are logged 'not used in v4.0.1+' and dropped.
    Passing them makes a run look configured when it is not."""
    ckpt = tmp_path / "m"
    ckpt.write_bytes(b"x")
    for name in ("cpsam", "cyto2", str(ckpt)):
        model = U._choose_model(name, device="cpu", object_type="cell")
        assert "model_type" not in model.kwargs
        assert "diam_mean" not in model.kwargs
        assert "nchan" not in model.kwargs


def test_diameter_is_still_passed_to_eval_because_cellpose4_honours_it(
        tmp_path, sam_pipeline):
    """Unlike model_type and diam_mean, diameter still does something:
    CellposeModel.eval rescales the image by 30./diameter."""
    import spacr.object as O

    src = tmp_path / "stack"
    _write_npz(src)

    O.generate_cellpose_masks_sam(
        str(src), _mask_settings(src, cell_diameter=60), "cell")

    assert sam_pipeline["eval_kwargs"][0]["diameter"] == 60, (
        "cell_diameter must reach eval — it is the one parameter of this "
        "group Cellpose 4 does not ignore")

    settings = S.set_default_settings_preprocess_generate_masks(
        {"src": str(src), "cell_diameter": 60, "verbose": False})
    assert S._get_object_settings("cell", settings)["diameter"] == 60.0


def test_the_docs_say_which_parameters_cellpose4_still_reads():
    """The tooltips are the only place most users learn this."""
    for key in ("cell_model_name", "nucleus_model_name",
                "pathogen_model_name"):
        assert key in S.tooltips, key
    cell = S.tooltips["cell_model_name"]
    assert "diameter" in cell and "30/diameter" in cell
    assert "model_type" in cell and "diam_mean" in cell
    assert "not used in v4.0.1+" in cell
    assert "cpsam" in S.tooltips["model_name"]
    assert "30/diameter" in S.tooltips["model_name"]


def test_the_model_choices_constant_offers_only_what_exists():
    assert S.CELLPOSE_MODEL_CHOICES == ("cpsam",)
    for legacy in ("cyto", "cyto2", "cyto3", "nuclei"):
        assert legacy not in S.CELLPOSE_MODEL_CHOICES


# ---------------------------------------------------------------------------
# 6. spacr_cellpose: the finetune tool
# ---------------------------------------------------------------------------

def test_identify_masks_finetune_loads_the_custom_model_without_diam_mean(
        tmp_path, monkeypatch):
    """It used to pass model_type=None and diam_mean=diameter; Cellpose 4
    logs 'not used in v4.0.1+' for both and drops them."""
    import tifffile
    import spacr.spacr_cellpose as SC

    built = {}

    class _M:
        def __init__(self, **kw):
            built.update(kw)
            self.pretrained_model = kw.get("pretrained_model")

        def eval(self, x=None, **kwargs):
            built.setdefault("eval_kwargs", []).append(kwargs)
            arr = np.asarray(x)
            h, w = arr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint16)
            mask[2:5, 2:5] = 1
            flows = [np.zeros((h, w, 3), np.float32),
                     np.zeros((3, h, w), np.float32),
                     np.zeros((h, w), np.float32),
                     np.zeros((h, w), np.float32)]
            return mask, flows, None, None

    monkeypatch.setattr(SC, "cp_models",
                        types.SimpleNamespace(CellposeModel=_M))
    monkeypatch.setattr(SC, "display", lambda *a, **k: None, raising=False)

    rng = np.random.default_rng(0)
    tifffile.imwrite(str(tmp_path / "img_0.tif"),
                     rng.integers(0, 2000, (16, 16, 3)).astype(np.uint16))
    ckpt = tmp_path / "custom.pth"
    ckpt.write_bytes(b"x")

    SC.identify_masks_finetune({
        "src": str(tmp_path), "model_name": "cyto", "custom_model": str(ckpt),
        "diameter": 30, "flow_threshold": 0.4, "CP_prob": 0.0,
        "grayscale": False, "save": False, "normalize": True,
        "channels": [0, 1, 2], "percentiles": [2, 98], "invert": False,
        "verbose": False, "resize": False, "target_height": 16,
        "target_width": 16, "remove_background": False, "background": 100,
        "Signal_to_noise": 5, "rescale": False, "resample": False,
        "fill_in": False, "batch_size": 2, "plot": False,
    })

    assert built["pretrained_model"] == str(ckpt)
    assert "model_type" not in built
    assert "diam_mean" not in built
    # diameter still reaches eval, where Cellpose 4 acts on it
    assert built["eval_kwargs"][0]["diameter"] == 30
    # ...and the deprecated channel pair does not
    assert "channels" not in built["eval_kwargs"][0]


def test_identify_masks_finetune_maps_a_legacy_stock_name(tmp_path,
                                                           monkeypatch, capsys):
    import tifffile
    import spacr.spacr_cellpose as SC

    built = {}

    class _M:
        def __init__(self, **kw):
            built.update(kw)
            self.pretrained_model = kw.get("pretrained_model")

        def eval(self, x=None, **kwargs):
            arr = np.asarray(x)
            h, w = arr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint16)
            flows = [np.zeros((h, w, 3), np.float32),
                     np.zeros((3, h, w), np.float32),
                     np.zeros((h, w), np.float32),
                     np.zeros((h, w), np.float32)]
            return mask, flows, None, None

    monkeypatch.setattr(SC, "cp_models",
                        types.SimpleNamespace(CellposeModel=_M))
    monkeypatch.setattr(SC, "display", lambda *a, **k: None, raising=False)

    rng = np.random.default_rng(0)
    tifffile.imwrite(str(tmp_path / "img_0.tif"),
                     rng.integers(0, 2000, (16, 16, 3)).astype(np.uint16))

    SC.identify_masks_finetune({
        "src": str(tmp_path), "model_name": "cyto2", "custom_model": None,
        "diameter": 30, "flow_threshold": 0.4, "CP_prob": 0.0,
        "grayscale": False, "save": False, "normalize": True,
        "channels": [0, 1, 2], "percentiles": [2, 98], "invert": False,
        "verbose": False, "resize": False, "target_height": 16,
        "target_width": 16, "remove_background": False, "background": 100,
        "Signal_to_noise": 5, "rescale": False, "resample": False,
        "fill_in": False, "batch_size": 2, "plot": False,
    })

    assert built["pretrained_model"] == "cpsam"
    assert "predates Cellpose-SAM" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 7. The masks still get written — none of the above broke segmentation
# ---------------------------------------------------------------------------

def test_the_run_still_writes_masks_and_counts(tmp_path, sam_pipeline):
    import spacr.object as O

    src = tmp_path / "stack"
    _write_npz(src, n=2)
    O.generate_cellpose_masks_sam(str(src), _mask_settings(src), "cell")

    masks = sorted(p.name for p in (src / "cell_mask_stack").iterdir())
    assert len(masks) == 2

    db = Path(src).parent / "measurements" / "measurements.db"
    con = sqlite3.connect(str(db))
    try:
        rows = con.execute(
            "SELECT count_type, object_count FROM object_counts").fetchall()
    finally:
        con.close()
    assert rows and all(count == 1 for _, count in rows)
