from pathlib import Path
import hashlib
import json
import datetime
import yaml
base = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/560-subcell-rybg-preparation-r1')
expected = {
    'vit_model.py': ('2b18c329d0f1b6d959b900b01c354ebb5dd4b2b2f6f9968ce77875fba3199267', 'https://raw.githubusercontent.com/CellProfiling/SubCellPortable/c4e0a9106ecffdd2ad91fbbd60bda6dc9fc74009/vit_model.py'),
    'upstream_dataset.py': ('c3eaaa5df1b44b38c8cfeafe1cb0e9c1c83a9dceb5b4d255518094550b4a1c3c', 'https://raw.githubusercontent.com/CellProfiling/SubCellPortable/c4e0a9106ecffdd2ad91fbbd60bda6dc9fc74009/dataset.py'),
    'model_config.yaml': ('4af4ce5fd3cfee73c70435e9c62d73e19fd55b64ecb72f95bd9a9f8c148bdf5d', 'https://raw.githubusercontent.com/CellProfiling/SubCellPortable/c4e0a9106ecffdd2ad91fbbd60bda6dc9fc74009/models/rybg/vit_supcon_model/model_config.yaml'),
    'torch/hub/checkpoints/all_channels_ViT-ProtS-Pool.pth': ('6a2d117bcacaa0697034d06d6922580ca7cd996564f270529ec97e6502559845', 'https://czi-subcell-public.s3.amazonaws.com/models/all_channels_ViT-ProtS-Pool.pth'),
}
rows = {}
for name, (expected_sha, url) in expected.items():
    path = base / name
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == expected_sha, name
    rows[name] = {'sha256': sha, 'bytes': path.stat().st_size, 'original_url': url}
    print('PASS original bytes', name, sha)
assert rows['torch/hub/checkpoints/all_channels_ViT-ProtS-Pool.pth']['bytes'] == 349009018
config = yaml.safe_load((base / 'model_config.yaml').read_text())['model_config']
print('Original model configuration:', config)
report = {'item': 560, 'acquired_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': rows,
          'all_four_original_hashes_exact_to_published_Home_CPU_receipt': True,
          'upstream_commit': 'c4e0a9106ecffdd2ad91fbbd60bda6dc9fc74009',
          'Home_source_checkpoint_pending_integration': '3ce39ac7bde621dffc34c82f18779ce1a21a61e0',
          'no_GPU_run_or_biological_benchmark_claim': True}
(base / 'acquisition.json').write_text(json.dumps(report, indent=2) + '\n')
