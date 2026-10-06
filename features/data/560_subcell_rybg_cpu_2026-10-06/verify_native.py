"""Compare native-size SubCell R/Y/B/G inference with its official loader."""

import ast
import gc
import hashlib
import importlib.util
import json
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from spacr.embeddings import CHANNEL_PROJECT, EmbeddingSpec, _foundation_encoder


def digest(path):
    """Hash a pinned source or weight file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_normalizer(path):
    """Load the authors' exact min-max function from pinned dataset source."""
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "min_max_norm_fn")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 str(path), "exec"), namespace)
    return namespace["min_max_norm_fn"]


def input_crop(size):
    """Return one native crop with isolated extrema and a broad baseline."""
    crop = np.full((1, size, size, 4), 100.0, dtype=np.float32)
    crop[0, 2, 3, 0] = 0.0
    crop[0, size - 12, size - 11, 3] = 1000.0
    return crop


def main():
    """Run two original-pixel sizes through both strict-loaded public models."""
    base = Path(__file__).resolve().parent
    checkpoint = base / "torch/hub/checkpoints/all_channels_ViT-ProtS-Pool.pth"
    source = base / "vit_model.py"
    config_path = base / "model_config.yaml"
    dataset = base / "upstream_dataset.py"
    normalize = official_normalizer(dataset)
    spec = EmbeddingSpec(backbone="subcell_rybg", channel_policy=CHANNEL_PROJECT,
                         channels=(0, 1, 2, 3), normalize=False,
                         batch_size=1, device="cpu")
    encode = _foundation_encoder(spec)
    crops = {size: input_crop(size) for size in (64, 512)}
    spacr_features = {size: encode(crop) for size, crop in crops.items()}
    del encode
    gc.collect()

    module_spec = importlib.util.spec_from_file_location("subcell_native_author", source)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    config = yaml.safe_load(config_path.read_text())["model_config"]
    model = module.ViTPoolClassifier(config)
    model.load_model_dict(str(checkpoint), [])
    model.eval()
    comparisons = {}
    with torch.no_grad():
        for size, crop in crops.items():
            original = crop[0].transpose(2, 0, 1)
            prepared = normalize(original)
            assert prepared.shape == (4, size, size)
            tensor = torch.from_numpy(prepared[None].copy())
            reference = model(tensor).pool_op.detach().numpy()
            delta = np.abs(spacr_features[size] - reference)
            assert reference.shape == spacr_features[size].shape == (1, 1536)
            assert np.isfinite(reference).all()
            assert float(delta.max()) < 1e-5
            comparisons[str(size)] = {
                "original_shape": list(crop.shape),
                "official_model_input_shape": list(tensor.shape),
                "feature_shape": list(reference.shape),
                "max_abs_difference": float(delta.max()),
                "mean_abs_difference": float(delta.mean()),
            }
    receipt = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           text=True).strip(),
        "spacr_source_sha256": digest(Path.cwd() / "spacr/embeddings.py"),
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "official_model_source_sha256": digest(source),
        "official_model_config_sha256": digest(config_path),
        "official_dataset_sha256": digest(dataset),
        "comparisons": comparisons,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_visible_devices": "",
    }
    (base / "native_parity.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
