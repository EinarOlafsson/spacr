"""CPU coverage for spacr.deep_spacr's SmoothGrad attribution, the
``visualize_smooth_grad`` driver, ``save_top_class_examples`` and
``merge_predictions_into_db``.

Everything runs on a tiny hand-built ``nn.Sequential`` so the gradients are
analytically known (for a linear head the input gradient is exactly the
weight row), which lets the tests assert real numbers instead of shapes.
Matplotlib is forced to Agg and every figure is closed by an autouse
fixture, so ``plt.show()`` inside the product code never blocks.
"""
from __future__ import annotations

import os
import sqlite3
import tarfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend switch)

import torch.nn as nn  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


def _linear_model(n_in, n_out, weight=None, bias=0.0):
    """Flatten -> Linear model with fully deterministic parameters."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(n_in, n_out))
    with torch.no_grad():
        if weight is None:
            weight = torch.arange(n_out * n_in, dtype=torch.float32).reshape(n_out, n_in)
            weight = weight / float(n_out * n_in)
        model[1].weight.copy_(weight)
        model[1].bias.fill_(bias)
    return model


def _write_png(path, rng, size=8):
    arr = rng.integers(0, 255, (size, size, 3)).astype(np.uint8)
    Image.fromarray(arr).save(str(path))
    return str(path)


def _save_model(model, path):
    torch.save(model, str(path))
    return str(path)


def _expected_smooth_grad_map(weight_row, size):
    """The map visualize_smooth_grad must produce for a linear head.

    d(logit_c)/d(pixel) == weight[c] for a Flatten->Linear model, so the
    SmoothGrad average over noisy copies is exactly ``|weight[c]|``; the
    driver then averages over the 3 colour channels.
    """
    w = weight_row.detach().numpy().reshape(3, size, size)
    return np.abs(w).mean(axis=0)


def _make_db(path, rows, table="png_list", extra_cols=""):
    con = sqlite3.connect(str(path))
    con.execute(f"CREATE TABLE {table} (png_path TEXT, prcfo TEXT{extra_cols})")
    con.executemany(f"INSERT INTO {table} (png_path, prcfo) VALUES (?,?)", rows)
    con.commit()
    con.close()
    return str(path)


def _table_info(db_path, table="png_list"):
    con = sqlite3.connect(str(db_path))
    info = list(con.execute(f"PRAGMA table_info({table})"))
    con.close()
    return {r[1]: r[2] for r in info}


def _fetch(db_path, cols, table="png_list"):
    con = sqlite3.connect(str(db_path))
    rows = list(con.execute(f"SELECT {cols} FROM {table} ORDER BY rowid"))
    con.close()
    return rows


# ---------------------------------------------------------------------------
# SmoothGrad.__init__
# ---------------------------------------------------------------------------

def test_smoothgrad_init_defaults_and_overrides():
    """__init__ stores the model plus both noise knobs verbatim."""
    from spacr.deep_spacr import SmoothGrad
    model = _linear_model(12, 2)

    default = SmoothGrad(model)
    assert default.model is model
    assert default.n_samples == 50
    assert default.stdev_spread == pytest.approx(0.15)

    custom = SmoothGrad(model, n_samples=7, stdev_spread=0.4)
    assert custom.n_samples == 7
    assert custom.stdev_spread == pytest.approx(0.4)
    # the two instances must not share state
    assert default.n_samples == 50


# ---------------------------------------------------------------------------
# SmoothGrad.compute_smooth_grad
# ---------------------------------------------------------------------------

def test_compute_smooth_grad_equals_abs_weights_for_linear_head():
    """For a linear head the averaged gradient is exactly |W[target]|.

    Signed weights prove the trailing ``.abs()`` is applied, and the
    model is left in eval mode.
    """
    from spacr.deep_spacr import SmoothGrad
    torch.manual_seed(0)
    n_in = 3 * 4 * 4
    weight = torch.linspace(-1.0, 1.0, 3 * n_in).reshape(3, n_in)
    model = _linear_model(n_in, 3, weight=weight)
    model.train()  # compute_smooth_grad must flip this to eval

    x = torch.rand(1, 3, 4, 4)
    sg = SmoothGrad(model, n_samples=5)
    out = sg.compute_smooth_grad(x, 2)

    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert bool((out >= 0).all())
    expected = weight[2].abs().reshape(1, 3, 4, 4)
    assert torch.allclose(out, expected, atol=1e-5)
    assert model.training is False
    # the caller's tensor is neither mutated nor given a grad
    assert x.grad is None
    assert x.requires_grad is False


def test_compute_smooth_grad_targets_the_requested_class():
    """Changing target_class changes the map (the logit index is honoured)."""
    from spacr.deep_spacr import SmoothGrad
    torch.manual_seed(0)
    n_in = 3 * 4 * 4
    weight = torch.linspace(-2.0, 2.0, 2 * n_in).reshape(2, n_in)
    model = _linear_model(n_in, 2, weight=weight)
    sg = SmoothGrad(model, n_samples=3)
    x = torch.rand(1, 3, 4, 4)

    g0 = sg.compute_smooth_grad(x, 0)
    g1 = sg.compute_smooth_grad(x, 1)
    assert torch.allclose(g0, weight[0].abs().reshape(1, 3, 4, 4), atol=1e-5)
    assert torch.allclose(g1, weight[1].abs().reshape(1, 3, 4, 4), atol=1e-5)
    assert not torch.allclose(g0, g1)


def test_compute_smooth_grad_zero_noise_reduces_to_plain_gradient():
    """A flat input gives stdev == 0, so the average is the raw |gradient|.

    Uses a ReLU net so the answer is genuinely input-dependent — this
    pins the ``/ n_samples`` normalisation, not just the accumulation.
    """
    from spacr.deep_spacr import SmoothGrad
    torch.manual_seed(0)
    n_in = 3 * 4 * 4
    model = nn.Sequential(nn.Flatten(), nn.Linear(n_in, 5), nn.ReLU(), nn.Linear(5, 2))
    model.eval()

    x = torch.full((1, 3, 4, 4), 0.7)
    assert float(x.max() - x.min()) == 0.0  # -> stdev == 0 -> noise == 0

    ref = x.clone().requires_grad_()
    model(ref)[0, 1].backward()
    plain = ref.grad.abs()

    out = SmoothGrad(model, n_samples=6).compute_smooth_grad(x, 1)
    assert torch.allclose(out, plain, atol=1e-6)
    assert float(out.abs().sum()) > 0.0


def test_compute_smooth_grad_noise_actually_smooths():
    """With a ReLU net, non-zero stdev_spread yields a different map."""
    from spacr.deep_spacr import SmoothGrad
    torch.manual_seed(0)
    n_in = 3 * 4 * 4
    lin1 = nn.Linear(n_in, 6)
    with torch.no_grad():
        lin1.weight.copy_(torch.linspace(-1, 1, 6 * n_in).reshape(6, n_in))
        lin1.bias.zero_()
    lin2 = nn.Linear(6, 2)
    with torch.no_grad():
        lin2.weight.copy_(torch.linspace(-1, 1, 12).reshape(2, 6))
        lin2.bias.zero_()
    model = nn.Sequential(nn.Flatten(), lin1, nn.ReLU(), lin2)

    # inputs straddling 0 so noise flips ReLU units
    x = torch.linspace(-0.05, 0.05, n_in).reshape(1, 3, 4, 4)

    torch.manual_seed(1)
    quiet = SmoothGrad(model, n_samples=8, stdev_spread=0.0).compute_smooth_grad(x, 1)
    torch.manual_seed(1)
    noisy = SmoothGrad(model, n_samples=8, stdev_spread=2.0).compute_smooth_grad(x, 1)

    assert quiet.shape == noisy.shape == x.shape
    assert not torch.allclose(quiet, noisy, atol=1e-4)
    assert bool((noisy >= 0).all())


@pytest.mark.xfail(strict=True,
                   reason="BUG: compute_smooth_grad hardcodes output[0, target_class], "
                          "so for a batched input only sample 0 gets gradients")
def test_compute_smooth_grad_supports_batched_input():
    """A batch must attribute every sample, not only the first one.

    The docstring advertises "single sample or batch"; today rows 1..N-1
    come back as all-zero because only ``output[0, target]`` is
    back-propagated.
    """
    from spacr.deep_spacr import SmoothGrad
    torch.manual_seed(0)
    n_in = 3 * 4 * 4
    weight = torch.linspace(-1.0, 1.0, 2 * n_in).reshape(2, n_in)
    model = _linear_model(n_in, 2, weight=weight)

    x = torch.rand(3, 3, 4, 4)
    out = SmoothGrad(model, n_samples=2).compute_smooth_grad(x, 1)

    assert out.shape == x.shape
    per_sample = out.flatten(1).sum(dim=1)
    assert bool((per_sample > 0).all()), f"zero attribution rows: {per_sample.tolist()}"


# ---------------------------------------------------------------------------
# visualize_smooth_grad
# ---------------------------------------------------------------------------

def test_visualize_smooth_grad_saves_expected_map(tmp_path, rng):
    """save_smooth_grad=True writes smooth_grad_<file>.png with the exact map."""
    from spacr.deep_spacr import visualize_smooth_grad
    size = 8
    n_in = 3 * size * size
    weight = torch.arange(2 * n_in, dtype=torch.float32).reshape(2, n_in) / float(2 * n_in)
    model = _linear_model(n_in, 2, weight=weight)
    model_path = _save_model(model, tmp_path / "m.pth")

    src = tmp_path / "src"
    src.mkdir()
    _write_png(src / "a.png", rng, size=size)
    (src / "notes.txt").write_text("not an image")  # exercises the .png filter

    save_dir = tmp_path / "sg_out"  # does not exist -> makedirs branch
    out = visualize_smooth_grad(str(src), model_path, 1, image_size=size,
                                save_smooth_grad=True, save_dir=str(save_dir))

    assert out is None
    written = sorted(p.name for p in save_dir.iterdir())
    assert written == ["smooth_grad_a.png"], "the .txt file must be skipped"

    saved = Image.open(save_dir / "smooth_grad_a.png")
    assert saved.size == (size, size)
    assert saved.mode == "L"
    expected = (_expected_smooth_grad_map(weight[1], size) * 255).astype(np.uint8)
    got = np.array(saved)
    assert got.shape == (size, size)
    assert np.abs(got.astype(int) - expected.astype(int)).max() <= 1
    # class 1 weights live in the upper half of [0,1) -> a bright map
    assert got.min() > 100


def test_visualize_smooth_grad_plots_without_saving(tmp_path, rng, monkeypatch):
    """save_smooth_grad=False plots 3 panels per image and writes nothing."""
    import spacr.deep_spacr as ds
    size = 8
    n_in = 3 * size * size
    weight = torch.arange(2 * n_in, dtype=torch.float32).reshape(2, n_in) / float(2 * n_in)
    model = _linear_model(n_in, 2, weight=weight)
    model_path = _save_model(model, tmp_path / "m.pth")

    src = tmp_path / "src"
    src.mkdir()
    _write_png(src / "a.png", rng, size=size)

    captured = []

    def _capture_show(*args, **kwargs):
        fig = plt.gcf()
        panels = []
        for ax in fig.axes:
            panels.append({
                "title": ax.get_title(),
                "data": None if not ax.images else np.asarray(ax.images[0].get_array()),
                "cmap": None if not ax.images else ax.images[0].get_cmap().name,
            })
        captured.append(panels)

    monkeypatch.setattr(ds.plt, "show", _capture_show)

    save_dir = tmp_path / "never_created"
    ds.visualize_smooth_grad(str(src), model_path, 0, image_size=size,
                             normalize=False, save_smooth_grad=False,
                             save_dir=str(save_dir))

    assert not save_dir.exists()
    assert len(captured) == 1
    panels = captured[0]
    assert [p["title"] for p in panels] == ["Original Image", "SmoothGrad", "Overlay"]

    original, smooth, overlay = (p["data"] for p in panels)
    assert original.shape == (size, size, 3)
    assert smooth.shape == (size, size)
    assert panels[1]["cmap"] == "hot"
    expected = _expected_smooth_grad_map(weight[0], size)
    assert np.allclose(smooth, expected, atol=1e-5)
    assert overlay.shape == (size, size, 3)
    assert float(overlay.min()) >= 0.0 and float(overlay.max()) <= 1.0


def test_visualize_smooth_grad_no_pngs_is_a_noop(tmp_path, monkeypatch):
    """A source folder with no PNGs produces no figures and no files."""
    import spacr.deep_spacr as ds
    size = 8
    model = _linear_model(3 * size * size, 2)
    model_path = _save_model(model, tmp_path / "m.pth")

    src = tmp_path / "empty"
    src.mkdir()
    (src / "readme.md").write_text("nothing to see")
    (src / "table.csv").write_text("a,b\n1,2\n")

    shown = []
    monkeypatch.setattr(ds.plt, "show", lambda *a, **k: shown.append(1))

    save_dir = tmp_path / "sg_none"
    ds.visualize_smooth_grad(str(src), model_path, 0, image_size=size,
                             save_smooth_grad=False, save_dir=str(save_dir))

    assert shown == []
    assert not save_dir.exists()
    assert sorted(p.name for p in src.iterdir()) == ["readme.md", "table.csv"]


def test_visualize_smooth_grad_reuses_existing_save_dir(tmp_path, rng):
    """An already-existing save_dir is reused rather than re-created."""
    from spacr.deep_spacr import visualize_smooth_grad
    size = 8
    model = _linear_model(3 * size * size, 2)
    model_path = _save_model(model, tmp_path / "m.pth")

    src = tmp_path / "src"
    src.mkdir()
    _write_png(src / "b.png", rng, size=size)

    save_dir = tmp_path / "already_here"
    save_dir.mkdir()
    (save_dir / "keep_me.txt").write_text("previous run")

    visualize_smooth_grad(str(src), model_path, 0, image_size=size,
                          save_smooth_grad=True, save_dir=str(save_dir))

    names = sorted(p.name for p in save_dir.iterdir())
    assert names == ["keep_me.txt", "smooth_grad_b.png"]
    assert (save_dir / "keep_me.txt").read_text() == "previous run"


@pytest.mark.xfail(strict=True,
                   reason="BUG: the overlay blends the ORIGINAL-resolution image with the "
                          "image_size-resolution map, so any src image whose size differs "
                          "from image_size raises ValueError (broadcast)")
def test_visualize_smooth_grad_handles_image_size_mismatch(tmp_path, rng):
    """Source images larger than image_size must still be visualised.

    preprocess_image returns the UNRESIZED PIL image alongside the
    resized tensor, so ``overlay * 0.5 + smooth_grad_map_rgb * 0.5``
    blends a (H,W,3) array with an (image_size,image_size,3) one.
    """
    from spacr.deep_spacr import visualize_smooth_grad
    model = _linear_model(3 * 8 * 8, 2)
    model_path = _save_model(model, tmp_path / "m.pth")

    src = tmp_path / "src_big"
    src.mkdir()
    _write_png(src / "big.png", rng, size=16)  # 16 != image_size=8

    visualize_smooth_grad(str(src), model_path, 1, image_size=8,
                          save_smooth_grad=False)


# ---------------------------------------------------------------------------
# save_top_class_examples
# ---------------------------------------------------------------------------

def _tar_of_pngs(tmp_path, rng, names, size=8):
    d = tmp_path / "imgs"
    d.mkdir(exist_ok=True)
    payload = {}
    for nm in names:
        p = d / os.path.basename(nm)
        _write_png(p, rng, size=size)
        payload[nm] = p.read_bytes()
    tar_path = tmp_path / "ds.tar"
    with tarfile.open(tar_path, "w") as t:
        for nm in names:
            t.add(d / os.path.basename(nm), arcname=nm)
    return str(tar_path), payload


def test_save_top_class_examples_picks_the_extremes(tmp_path, rng):
    """class_0 gets the lowest preds, class_1 the highest; bytes match the tar."""
    from spacr.deep_spacr import save_top_class_examples
    names = [f"crops/o{i}.png" for i in range(6)]
    tar_path, payload = _tar_of_pngs(tmp_path, rng, names)
    df = pd.DataFrame({"path": names, "pred": [0.01, 0.02, 0.5, 0.6, 0.98, 0.99]})

    dst = tmp_path / "top"
    out = save_top_class_examples(df, tar_path, str(dst), n=2)

    assert out == str(dst)
    assert sorted(p.name for p in (dst / "class_0").iterdir()) == ["o0.png", "o1.png"]
    assert sorted(p.name for p in (dst / "class_1").iterdir()) == ["o4.png", "o5.png"]
    # the extracted bytes are the real tar members, not placeholders
    assert (dst / "class_0" / "o0.png").read_bytes() == payload["crops/o0.png"]
    assert (dst / "class_1" / "o5.png").read_bytes() == payload["crops/o5.png"]


def test_save_top_class_examples_duplicate_extremes(tmp_path, rng, capsys):
    """One image can be both the lowest and the highest; it is copied twice."""
    from spacr.deep_spacr import save_top_class_examples
    names = ["only.png"]
    tar_path, payload = _tar_of_pngs(tmp_path, rng, names)
    df = pd.DataFrame({"path": names, "pred": [0.42]})

    dst = tmp_path / "dup"
    save_top_class_examples(df, tar_path, str(dst), n=5)

    assert (dst / "class_0" / "only.png").read_bytes() == payload["only.png"]
    assert (dst / "class_1" / "only.png").read_bytes() == payload["only.png"]
    assert "Saved 2 top-confidence example images" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# merge_predictions_into_db
# ---------------------------------------------------------------------------

def test_merge_predictions_missing_db_returns_none(tmp_path, capsys):
    """A non-existent database is reported and skipped, not raised on."""
    from spacr.deep_spacr import merge_predictions_into_db
    missing = tmp_path / "nope" / "measurements.db"
    df = pd.DataFrame({"path": ["a.png"], "pred": [0.9], "cv_predictions": [1]})

    assert merge_predictions_into_db(df, str(missing)) is None
    assert "Database not found" in capsys.readouterr().out
    assert not missing.exists()


def test_merge_predictions_updates_matching_rows(tmp_path, capsys):
    """Rows are matched on basename and get both pred + class written."""
    from spacr.deep_spacr import merge_predictions_into_db
    db = _make_db(tmp_path / "measurements.db", [
        ("/data/plate1/merged/a.png", "p1_A01_1_o1"),
        ("/data/plate1/merged/b.png", "p1_A01_1_o2"),
        ("/data/plate1/merged/unseen.png", "p1_A01_1_o3"),
    ])
    df = pd.DataFrame({
        # tar member names: relative, with a folder prefix the DB does not have
        "path": ["crops/a.png", "crops/b.png", "crops/not_in_db.png"],
        "pred": [0.1, 0.85, 0.5],
        "cv_predictions": [0, 1, 1],
    })

    matched = merge_predictions_into_db(df, db)

    assert matched == 2
    cols = _table_info(db)
    assert cols["pred"] == "REAL"
    assert cols["cv_predictions"] == "INTEGER"
    rows = _fetch(db, "png_path, pred, cv_predictions")
    assert rows[0][1] == pytest.approx(0.1) and rows[0][2] == 0
    assert rows[1][1] == pytest.approx(0.85) and rows[1][2] == 1
    assert rows[2][1] is None and rows[2][2] is None
    assert "2/3 rows matched" in capsys.readouterr().out
    # the caller's DataFrame is untouched (the function copies)
    assert "_join_key" not in df.columns


def test_merge_predictions_reuses_existing_columns(tmp_path):
    """A pre-existing prediction column is overwritten, not duplicated."""
    from spacr.deep_spacr import merge_predictions_into_db
    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE png_list (png_path TEXT, prcfo TEXT, pred REAL)")
    con.executemany("INSERT INTO png_list VALUES (?,?,?)",
                    [("/x/a.png", "p1", 0.999), ("/x/b.png", "p2", 0.111)])
    con.commit()
    con.close()

    df = pd.DataFrame({"path": ["a.png", "b.png"],
                       "pred": [0.2, 0.7],
                       "cv_predictions": [0, 1]})
    matched = merge_predictions_into_db(df, str(db))

    assert matched == 2
    names = [r[1] for r in sqlite3.connect(str(db)).execute("PRAGMA table_info(png_list)")]
    assert names.count("pred") == 1, "ALTER TABLE must not duplicate the column"
    assert names.count("cv_predictions") == 1
    rows = _fetch(db, "pred, cv_predictions")
    assert [r[0] for r in rows] == [pytest.approx(0.2), pytest.approx(0.7)]
    assert [r[1] for r in rows] == [0, 1]


def test_merge_predictions_skips_null_png_path(tmp_path):
    """A NULL png_path row is skipped instead of crashing on basename()."""
    from spacr.deep_spacr import merge_predictions_into_db
    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE png_list (png_path TEXT, prcfo TEXT)")
    con.executemany("INSERT INTO png_list VALUES (?,?)",
                    [(None, "p1"), ("/x/a.png", "p2"), ("", "p3")])
    con.commit()
    con.close()

    df = pd.DataFrame({"path": ["a.png"], "pred": [0.33], "cv_predictions": [0]})
    matched = merge_predictions_into_db(df, str(db))

    assert matched == 1
    rows = _fetch(db, "png_path, pred")
    assert rows[0][1] is None      # NULL path untouched
    assert rows[1][1] == pytest.approx(0.33)
    assert rows[2][1] is None      # '' path untouched


def test_merge_predictions_no_matches_leaves_table_clean(tmp_path, capsys):
    """Zero overlap -> returns 0 and every prediction cell stays NULL."""
    from spacr.deep_spacr import merge_predictions_into_db
    db = _make_db(tmp_path / "measurements.db",
                  [("/x/a.png", "p1"), ("/x/b.png", "p2")])
    df = pd.DataFrame({"path": ["zzz.png"], "pred": [0.9], "cv_predictions": [1]})

    matched = merge_predictions_into_db(df, db)

    assert matched == 0
    assert set(_table_info(db)) >= {"pred", "cv_predictions"}
    assert _fetch(db, "pred, cv_predictions") == [(None, None), (None, None)]
    assert "0/2 rows matched" in capsys.readouterr().out


def test_merge_predictions_custom_table_and_columns(tmp_path):
    """table / pred_col / class_col are all honoured."""
    from spacr.deep_spacr import merge_predictions_into_db
    db = _make_db(tmp_path / "measurements.db",
                  [("/x/a.png", "p1")], table="object_png")
    df = pd.DataFrame({"path": ["a.png"], "score": [0.6], "label": [1]})

    matched = merge_predictions_into_db(df, db, table="object_png",
                                        pred_col="score", class_col="label")

    assert matched == 1
    cols = _table_info(db, table="object_png")
    assert cols["score"] == "REAL" and cols["label"] == "INTEGER"
    assert "pred" not in cols and "cv_predictions" not in cols
    assert _fetch(db, "score, label", table="object_png") == [(pytest.approx(0.6), 1)]


def test_merge_predictions_missing_table_raises(tmp_path):
    """An existing DB without the target table fails loudly on the SELECT."""
    from spacr.deep_spacr import merge_predictions_into_db
    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE something_else (x INTEGER)")
    con.commit()
    con.close()

    df = pd.DataFrame({"path": ["a.png"], "pred": [0.1], "cv_predictions": [0]})
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        merge_predictions_into_db(df, str(db))
