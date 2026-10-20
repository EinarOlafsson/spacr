"""Read immutable tracked byte budgets without a checkout, clone or build."""
import json
import subprocess
import sys

for ref in sys.argv[1:]:
    revision = subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    raw = subprocess.check_output(["git", "ls-tree", "-r", "-l", "-z", revision])
    sizes = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        header, _path = item.split(b"\t", 1)
        mode, kind, _oid, size = header.split()
        assert kind == b"blob" and mode in (b"100644", b"100755")
        sizes.append(int(size))
    total = sum(sizes)
    print(json.dumps({"source": revision, "tracked_files": len(sizes),
                      "bytes": total, "MiB": total / 1048576,
                      "claim_3416_drift": abs(3416 * 1048576 - total) / total}))
