"""Real-data tests for the analysis submodules + toxo plotting.

The user's directive: "test the other modules especially those in
submodules and toxo."

Two families here:

  * **Image-based cellpose submodules** (GPU) — train_cellpose on a
    small synthetic image/mask set for 2 epochs, then apply_cellpose_model
    over a folder of images. These mirror what a user does when they
    fine-tune Cellpose in spaCR.

  * **Toxo dataframe/plotting functions** (CPU) — feed realistic
    dataframes matching the real column schemas to plot_gene_phenotypes,
    plot_gene_heatmaps and custom_volcano_plot, and assert a figure PDF
    lands on disk. Plus count_phenotypes on a real measurements.db.

The cellpose tests are @slow + @gpu; the toxo/plotting tests are CPU
so they run in the default suite (matplotlib Agg backend, no display).
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pytest

# Force a non-interactive backend before pyplot is imported anywhere so
# these tests never try to open a window.
import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# GPU guard
# ---------------------------------------------------------------------------

def _require_gpu_cellpose():
    # importorskip, not `except Exception`: "the package is not installed"
    # raises ImportError and is an environment fact, while "the package is
    # installed and detonates on import" is a bug and must fail here.
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA — cellpose submodule tests are GPU-only")
    pytest.importorskip("cellpose")


# ---------------------------------------------------------------------------
# Synthetic image + mask set for train_cellpose
# ---------------------------------------------------------------------------

def _make_cellpose_train_set(root: Path, n: int = 6, size: int = 96):
    """Write ``root/train/images/*.tif`` + ``root/train/masks/*.tif``.

    Each image has a handful of Gaussian blobs; the paired mask labels
    each blob with a distinct integer (a trivially-learnable target)."""
    import tifffile
    img_dir = root / "train" / "images"
    mask_dir = root / "train" / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    y, x = np.ogrid[:size, :size]
    for k in range(n):
        img = rng.integers(80, 160, size=(size, size)).astype(np.uint16)
        mask = np.zeros((size, size), dtype=np.uint16)
        centres = rng.integers(15, size - 15, size=(6, 2))
        for lbl, (cy, cx) in enumerate(centres, start=1):
            g = np.exp(-((x - int(cx)) ** 2 + (y - int(cy)) ** 2)
                           / (2 * 5 ** 2))
            img = np.clip(img.astype(np.float32) + g * 2500, 0, 65535
                             ).astype(np.uint16)
            mask[g > 0.5] = lbl
        name = f"img_{k:02d}.tif"
        tifffile.imwrite(str(img_dir / name), img)
        tifffile.imwrite(str(mask_dir / name), mask)
    return root


# ---------------------------------------------------------------------------
# train_cellpose
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
def test_train_cellpose_writes_model(tmp_path):
    _require_gpu_cellpose()
    from spacr.submodules import train_cellpose
    root = _make_cellpose_train_set(tmp_path / "cp_train")
    settings = {
        "src": str(root),
        "model_name": "e2e_test",
        "target_size": 96,
        "n_epochs": 2,
        "batch_size": 2,
        "learning_rate": 0.1,
        "weight_decay": 1e-5,
        "augment": False,
        "verbose": False,
    }
    # Unguarded: _make_cellpose_train_set builds the image/mask pairs these
    # settings point at, and the GPU precondition is already handled by
    # _require_gpu_cellpose above. "bailed on synthetic set" was train_cellpose
    # failing, reported as an excuse.
    train_cellpose(settings)
    # A trained model directory should exist with content.
    model_dir = root / "models" / "cellpose_model"
    assert model_dir.is_dir()
    produced = list(model_dir.rglob("*"))
    assert any(p.is_file() for p in produced), (
        "train_cellpose wrote no model files")


# ---------------------------------------------------------------------------
# apply_cellpose_model
# ---------------------------------------------------------------------------

#: What this test found the moment its ``except Exception: pytest.skip(
#: "apply_cellpose_model bailed on synthetic imgs")`` came off.
#:
#: ``spacr/submodules.py:632`` calls ``model.eval(..., bsize=224)``. 224 was
#: the Cellpose-3 default tile size; Cellpose-SAM's ViT carries a positional
#: embedding of shape ``(1, 32, 32, 1024)`` -- a 256 px window at 8 px patches
#: -- so a 224 px tile arrives as 28x28 patches and
#: ``cellpose/vit_sam.py:61``'s ``x = x + self.encoder.pos_embed`` raises
#: ``RuntimeError: The size of tensor a (28) must match the size of tensor b
#: (32) at non-singleton dimension 2``.
#:
#: Confirmed directly against the installed Cellpose, independent of spaCR:
#: the same eval call raises at ``bsize=224`` and succeeds at ``bsize=256``.
#: It is not input-size dependent -- bsize is the tile fed to the net, so
#: apply_cellpose_model cannot segment anything at all on Cellpose 4.x/SAM.
#:
#: ``spacr/submodules.py:432`` (``test_cellpose_model``) has the identical
#: hard-coded 224 and is the same bug.
#:
#: Product fix, not made here: drop the hard-coded bsize (let Cellpose choose)
#: or set it to the backbone's window. strict=True fails this test the moment
#: that lands, so the pin cannot outlive the bug.
CELLPOSE_SAM_BSIZE_BUG = (
    "spacr/submodules.py:632 (and :432) hard-code bsize=224 in "
    "CellposeModel.eval; Cellpose-SAM's pos_embed is 32x32 patches (256 px), "
    "so 224 gives 28x28 and vit_sam.py:61 raises RuntimeError: The size of "
    "tensor a (28) must match the size of tensor b (32). Verified against the "
    "installed cellpose: bsize=224 raises, bsize=256 works."
)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.xfail(strict=True, reason=CELLPOSE_SAM_BSIZE_BUG)
def test_apply_cellpose_model_writes_results(tmp_path):
    _require_gpu_cellpose()
    from spacr.submodules import apply_cellpose_model
    import tifffile
    # Flat folder of images for inference.
    img_dir = tmp_path / "apply_imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    # 256 px so Cellpose-SAM's internal tiling (bsize) divides evenly —
    # tiny tiles (96 px) trip a tensor-shape mismatch in the SAM
    # backbone.
    S = 256
    y, x = np.ogrid[:S, :S]
    for k in range(4):
        img = rng.integers(80, 160, size=(S, S)).astype(np.uint16)
        for cy, cx in rng.integers(20, S - 20, size=(12, 2)):
            g = np.exp(-((x - int(cx)) ** 2 + (y - int(cy)) ** 2)
                           / (2 * 6 ** 2)) * 2500
            img = np.clip(img.astype(np.float32) + g, 0, 65535
                             ).astype(np.uint16)
        tifffile.imwrite(str(img_dir / f"apply_{k:02d}.tif"), img)

    settings = {
        "src": str(img_dir),
        "model_path": "cyto",   # stock model
        "batch_size": 2,
        "FT": 0.4,
        "CP_probability": 0.0,
        "circularize": False,
        "save": True,
        "diameter": 30,
        "normalize": True,
        "percentiles": [2, 98],
        "target_height": 96,
        "target_width": 96,
        "verbose": False,
    }
    # Unguarded: the images above are sized for Cellpose-SAM's tiling on
    # purpose, so a failure here is apply_cellpose_model's -- and it is. See
    # CELLPOSE_SAM_BSIZE_BUG above.
    apply_cellpose_model(settings)
    csvs = list((img_dir).rglob("*.csv"))
    assert csvs, "apply_cellpose_model wrote no result CSVs"


# ---------------------------------------------------------------------------
# count_phenotypes on a real measurements.db
# ---------------------------------------------------------------------------

def _make_png_list_db(db_path: Path):
    """Create a minimal measurements.db with a png_list table carrying
    plate metadata + an annotation column, matching what count_phenotypes
    reads."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE png_list (
                png_path TEXT, plateID TEXT, rowID TEXT,
                columnID TEXT, value INTEGER
            )""")
        rows = []
        for r in range(1, 4):
            for c in range(1, 4):
                for i in range(5):
                    rows.append((f"/x/p1_r{r}_c{c}_{i}.png", "plate1",
                                    f"r{r}", f"c{c}", (i % 2) + 1))
        conn.executemany(
            "INSERT INTO png_list VALUES (?,?,?,?,?)", rows)
        conn.commit()


def test_count_phenotypes_real_db(tmp_path, monkeypatch):
    """count_phenotypes on a real png_list DB should write
    phenotype_counts.csv."""
    from spacr import submodules as SUB
    # count_phenotypes calls display() — stub it so a headless run
    # doesn't choke.
    monkeypatch.setattr(SUB, "display", lambda *a, **k: None,
                          raising=False)
    db = tmp_path / "measurements" / "measurements.db"
    _make_png_list_db(db)
    settings = {"src": str(db), "annotation_column": "value"}
    # Unguarded: _make_png_list_db writes the real png_list schema and
    # display() is stubbed above, so "needs more than the stub db" was
    # count_phenotypes failing rather than the fixture being thin.
    SUB.count_phenotypes(settings)
    out = list(tmp_path.rglob("phenotype_counts.csv"))
    assert out, "count_phenotypes wrote no phenotype_counts.csv"


# ---------------------------------------------------------------------------
# toxo plotting functions
# ---------------------------------------------------------------------------

def _toxo_phenotype_df(n: int = 40):
    import pandas as pd
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "Gene ID": [f"TGGT1_{200000 + i}" for i in range(n)],
        "T.gondii GT1 CRISPR Phenotype - Mean Phenotype":
            rng.normal(0, 1, n),
        "T.gondii GT1 CRISPR Phenotype - Standard Error":
            np.abs(rng.normal(0.1, 0.05, n)),
    })


def test_toxo_plot_gene_phenotypes_saves_pdf(tmp_path):
    from spacr import toxo
    df = _toxo_phenotype_df()
    genes = df["Gene ID"].iloc[:3].tolist()
    save = tmp_path / "gene_phenotypes.pdf"
    toxo.plot_gene_phenotypes(
        data=df, gene_list=genes, save_path=str(save))
    assert save.exists() and save.stat().st_size > 0


def test_toxo_plot_gene_heatmaps_saves_pdf(tmp_path):
    from spacr import toxo
    import pandas as pd
    rng = np.random.default_rng(6)
    n = 30
    cols = ["metric_a", "metric_b", "metric_c"]
    df = pd.DataFrame({"Gene ID": [f"TGGT1_{200000 + i}"
                                       for i in range(n)]})
    for c in cols:
        df[c] = rng.normal(0, 1, n)
    # plot_gene_heatmaps extracts the numeric portion of TGGT1_<id>
    # from the data's Gene ID and matches gene_list against THAT, so
    # gene_list must be the numeric IDs.
    genes = [g.split("_")[1] for g in df["Gene ID"].iloc[:5].tolist()]
    save = tmp_path / "gene_heatmaps.pdf"
    toxo.plot_gene_heatmaps(
        data=df, gene_list=genes, columns=cols,
        save_path=str(save))
    assert save.exists() and save.stat().st_size > 0


def test_toxo_custom_volcano_plot_saves_pdf(tmp_path):
    from spacr import toxo
    import pandas as pd
    rng = np.random.default_rng(7)
    n = 50
    # custom_volcano_plot derives `variable` from `feature`, then
    # gene_nr = variable.split('_')[0]. Make feature "<gene_nr>_g1" so
    # gene_nr matches the metadata's numeric gene_nr.
    data = pd.DataFrame({
        "feature": [f"{200000 + i}_g1" for i in range(n)],
        "coefficient": rng.normal(0, 0.2, n),
        "p_value": np.clip(np.abs(rng.normal(0.05, 0.05, n)), 1e-6, 1),
    })
    metadata = pd.DataFrame({
        "gene_nr": [str(200000 + i) for i in range(n)],
        "tagm_location": rng.choice(
            ["nucleus", "cytoplasm", "apicoplast", "rhoptry"], n),
    })
    save = tmp_path / "volcano.pdf"
    toxo.custom_volcano_plot(
        data_path=data, metadata_path=metadata,
        metadata_column="tagm_location", save_path=str(save))
    assert save.exists() and save.stat().st_size > 0
