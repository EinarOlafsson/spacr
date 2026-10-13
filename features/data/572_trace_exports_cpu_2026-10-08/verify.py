from pathlib import Path
import hashlib, json, sys
root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
for name, expected in manifest["sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
if "--source" in sys.argv:
    repo = root.parents[2]
    receipt = json.loads((root / "receipt.json").read_text())
    for name, expected in receipt["source_sha256"].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected, name
print("verified", len(manifest["sha256"]), "payloads")
