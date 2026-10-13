"""Archive existing observations without rerunning Qt or the tile experiments."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

repo = Path(sys.argv[1]).resolve()
target = repo / "features/data/663_rejected_fungal_tile_gil_cpu_2026-10-07"
target.mkdir(parents=True, exist_ok=True)
fresh = Path(__file__).resolve().parent
old = Path("/mnt/wd4tb/scratch/theme-refinement-20261006/mycelium-path-cache-20261006")
for name in ("gil_probe.py", "gil_receipt.json", "gil_probe.log", "decision.json"):
    shutil.copyfile(fresh / name, target / name)
for name in ("tile_full_timing.json", "tile_full_expanded.json", "tile_two_gate.json"):
    shutil.copyfile(old / name, target / name)
shutil.copyfile(Path(__file__), target / "archive.py")
(target / "README.md").write_text("""# Rejected native tile/GIL feasibility

No production, quality, cache-budget or workflow changes. Hard native 24 FPS
remains OPEN. This archive adds one installed-binding observation; earlier tile
receipts are historical evidence and were not rerun for this checkpoint.

The CUDA-hidden, offscreen, capped-4-GiB probe uses PySide6 6.11.2 and Python
3.12.13. One synthetic 3840×2160 `drawPath` call lasts 4523.50 ms. A Python
heartbeat makes zero ticks strictly inside the call with 10-ms endpoint margins;
the subsequent 100-ms sleep control permits 76 interior ticks. The thread joins.
The one-second Python switch interval limits incidental boundary scheduling.
This is evidence that this installed binding holds the GIL during rasterization,
not a universal statement about other PySide versions or production FPS. Peak
probe RSS is 77,972 KiB. Script, binary/typesystem identities, environment and
argv are bound in the receipts. The probe does not import the application.

Earlier source ac146d3c passes 48 expanded sequential comparisons when each
tile retains full native dimensions, original coordinates and integer clips.
Translated two-tile drawing changes six seam pixels at clock 3600.33. Earlier
balanced full-coordinate two-thread medians are 59.49→98.08 ms at detail 2 and
86.54→142.90 ms at detail 1; peak RSS is 939,528 KiB. Those JSON receipts are
included as historical evidence, not current-source performance acceptance;
the original renderer and full tile scripts remain in the referenced scratch
directory and are not duplicated here. Current f36b33c9 cache guards reject
clipped/non-owned painters, so independent tiles also bypass mature reuse.

The investigation is complete/rejected. There is no justified new tile or live
benchmark. Root requested renderer experiments stop pending an actual CI target.

`verify_proof.py` verifies filesystem payloads, or `--git REV` verifies committed
Git blobs without requiring this private archive commit to survive cherry-pick.
Only hash verification was run during archival; Qt experiments were not rerun.
""")
(target / "verify_proof.py").write_text('''"""Verify portable archive payload hashes in a filesystem or Git revision."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--git")
args = parser.parse_args()
directory = Path(__file__).resolve().parent
repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=directory, text=True).strip())
relative = directory.relative_to(repo).as_posix()

def read(name):
    if args.git:
        return subprocess.check_output(["git", "show", f"{args.git}:{relative}/{name}"], cwd=repo)
    return (directory / name).read_bytes()

manifest = json.loads(read("manifest.json"))
for name, record in manifest["payloads"].items():
    payload = read(name)
    assert len(payload) == record["bytes"], name
    assert hashlib.sha256(payload).hexdigest() == record["sha256"], name
print(json.dumps({"verified_payloads": len(manifest["payloads"]), "bytes": sum(row["bytes"] for row in manifest["payloads"].values()), "mode": args.git or "filesystem"}))
''')
payloads = {path.name: {"bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(target.iterdir()) if path.name != "manifest.json"}
(target / "manifest.json").write_text(json.dumps({"status": "REJECTED", "production_changes": False,
    "hard_24fps_acceptance": False, "payloads": payloads}, indent=2) + "\n")
print(json.dumps({"payloads": len(payloads), "bytes": sum(row["bytes"] for row in payloads.values())}))
