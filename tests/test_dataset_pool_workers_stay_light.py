"""The tar and Cellpose dataset pools start workers that never import torch.

On Windows every pool worker is spawned: a new interpreter that imports the
module of the function it runs. ``generate_dataset`` ran
``spacr.utils.add_images_to_tar`` and ``prepare_cellpose_dataset`` ran
``spacr.io.process_instruction``, so each worker imported ``spacr.utils`` or
``spacr.io`` and with them torch, about 0.7 GiB committed per worker before
it read one image. Under an 8 GiB job the workers failed with MemoryError or
"WinError 1455 paging file too small" in torch's shm.dll. Both pools now run
private workers in ``spacr._mask_workers``, and the RAM guard counts what a
spawned worker commits importing them.

The import check runs in a fresh interpreter, which is exactly what a
spawned worker is: it unpickles the worker by reference, runs it on real
files and reports which modules it had to load.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

from spacr import _mask_workers as W
from spacr import resource_log as R

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 ** 2

_CHILD = r"""
import json, multiprocessing, pickle, sys
init, tar_worker, pair_worker = (pickle.loads(bytes.fromhex(h)) for h in sys.argv[1:4])
paths, tar_path, entry = json.loads(sys.argv[4])
init(multiprocessing.Value('i', 0), multiprocessing.Lock())
tar_worker(paths, tar_path, len(paths))
done = pair_worker(entry)
print(json.dumps({'done': done,
                  'heavy': sorted(m for m in ('torch', 'torchvision', 'spacr.io', 'spacr.utils',
                                              'matplotlib', 'pandas') if m in sys.modules)}))
"""


def test_a_spawned_dataset_worker_runs_both_jobs_without_importing_torch(tmp_path):
    paths = []
    for index in range(12):
        path = tmp_path / f"crop_{index}.png"
        Image.fromarray(np.full((4, 4), index, np.uint8)).save(path)
        paths.append(str(path))
    image = np.arange(30, dtype=np.uint16).reshape(5, 6)
    mask = (image > 10).astype(np.uint16)
    tifffile.imwrite(tmp_path / "img.tif", image)
    tifffile.imwrite(tmp_path / "msk.tif", mask)
    entry = {"src_img": str(tmp_path / "img.tif"), "src_msk": str(tmp_path / "msk.tif"),
             "dst_img": str(tmp_path / "out_img.tif"), "dst_msk": str(tmp_path / "out_msk.tif"),
             "augment": "rotate90"}
    workers = [pickle.dumps(f).hex() for f in (W._dataset_tar_init, W._dataset_tar_worker,
                                               W._cellpose_pair_worker)]
    env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, *workers,
         json.dumps([paths, str(tmp_path / "chunk.tar"), entry])],
        capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"done": 1, "heavy": []}
    assert "Progress: 10/12, operation_type: generating .tar dataset, None" in result.stdout
    with tarfile.open(tmp_path / "chunk.tar") as tar:
        assert sorted(tar.getnames()) == sorted(os.path.basename(p) for p in paths)
    import cv2
    assert np.array_equal(tifffile.imread(tmp_path / "out_img.tif"),
                          cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    assert np.array_equal(tifffile.imread(tmp_path / "out_msk.tif"),
                          cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE))


class _InlinePool:
    """Runs a pool's work in this process and records what it was given."""

    seen = []

    def __init__(self, processes=None, initializer=None, initargs=()):
        type(self).seen.append(("init", initializer))
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def starmap(self, function, arguments):
        type(self).seen.append(("task", function))
        return [function(*item) for item in arguments]

    def imap_unordered(self, function, items):
        type(self).seen.append(("task", function))
        return (function(item) for item in items)

    def close(self):
        pass

    def join(self):
        pass


@pytest.fixture
def inline_pool(monkeypatch):
    _InlinePool.seen = []
    monkeypatch.setattr(R, "_parallel_pool", _InlinePool)
    import spacr.io as IO
    monkeypatch.setattr(IO, "Pool", _InlinePool)
    return _InlinePool.seen


def test_generate_dataset_hands_its_pool_only_the_light_worker(tmp_path, inline_pool):
    import spacr.io as IO

    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    png_dir = src / "data" / "cell_png"
    png_dir.mkdir(parents=True)
    paths = []
    for index in range(6):
        path = png_dir / f"plate1_r{index % 5 + 1}_c1_f1_o{index + 1}.png"
        Image.fromarray(np.full((4, 4), index, np.uint8)).save(path)
        paths.append(str(path))
    with sqlite3.connect(src / "measurements" / "measurements.db") as connection:
        pd.DataFrame({"png_path": paths, "cell_id": [f"o{i + 1}" for i in range(6)]}
                     ).to_sql("png_list", connection, index=False)
    tar = IO.generate_dataset({"src": str(src), "file_metadata": None,
                               "experiment": "exp", "sample": None})
    with tarfile.open(tar) as archive:
        assert {os.path.basename(p) for p in paths} <= set(archive.getnames())
    used = [function for _kind, function in inline_pool if function is not None]
    assert used and all(function.__module__ == "spacr._mask_workers" for function in used)


def test_prepare_cellpose_dataset_hands_its_pool_only_the_light_worker(tmp_path, inline_pool):
    from spacr.io import prepare_cellpose_dataset

    rng = np.random.default_rng(725)
    root = tmp_path / "sets"
    for name in ("dsA", "dsB"):
        (root / name / "masks").mkdir(parents=True)
        for index in range(4):
            tifffile.imwrite(root / name / f"{name}_{index}.tif",
                             rng.integers(0, 500, (8, 8)).astype(np.uint16))
            tifffile.imwrite(root / name / "masks" / f"{name}_{index}.tif",
                             np.ones((8, 8), np.uint16))
    prepare_cellpose_dataset(str(root), augment_data=False, train_fraction=0.75, n_jobs=2)
    assert any((root / "cellpose_dataset" / "train").rglob("*.tif"))
    used = [function for kind, function in inline_pool if kind == "task"]
    assert used == [W._cellpose_pair_worker]


def test_a_spawned_worker_is_charged_its_import_and_a_forked_one_is_not():
    assert R._worker_import_bytes("dataset", "fork", "linux") == 0
    assert R._worker_import_bytes("dataset", "spawn", "linux") == 24 * MIB
    assert R._worker_import_bytes("cellpose_dataset", "forkserver", "linux") == 66 * MIB
    assert R._worker_import_bytes("cellpose_dataset", "spawn", "win32") == 132 * MIB
    assert R._worker_import_bytes("dataset", "spawn", "darwin") == 36 * MIB
    assert R._worker_import_bytes("measure", "spawn", "win32") == 0
    default = (mp.get_start_method(allow_none=True) or mp.get_all_start_methods()[0])
    assert R._worker_import_bytes("dataset") == R._worker_import_bytes("dataset", default)


def test_the_guard_sizes_a_spawned_pool_by_input_plus_import(monkeypatch):
    total = 16 * 1024 ** 3
    reserve = R._ram_reserve_bytes(total)
    spare = 1024 * MIB
    memory = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=reserve + spare,
                                                                     total=total))
    unit = 2 * MIB
    per_input = int(unit * R._RAM_WORKER_MULTIPLIERS["cellpose_dataset"])
    monkeypatch.setattr(R, "_worker_import_bytes", lambda module: 0)
    assert R._guard_workers("cellpose_dataset", 64, unit, settings={"ram_guard": True},
                            psutil_module=memory) == spare // per_input
    monkeypatch.setattr(R, "_worker_import_bytes",
                        lambda module: 132 * MIB if module == "cellpose_dataset" else 0)
    guarded = R._guard_workers("cellpose_dataset", 64, unit, settings={"ram_guard": True}, psutil_module=memory)
    assert guarded == spare // (per_input + 132 * MIB) < spare // per_input
    plan = R._ram_plan(unit, 64, module="cellpose_dataset", psutil_module=memory)
    assert plan["per_worker"] == per_input + 132 * MIB
