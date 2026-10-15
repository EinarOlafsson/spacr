"""Actual unmodified comparison with exact-source binding and readonly original DB."""
from pathlib import Path
import hashlib
import json
import os
import runpy
import socket
import sqlite3
import sys

root = Path(sys.argv[1]).resolve()
packet = json.loads(Path(sys.argv[2]).read_text())
for name, expected in packet["source_sha256"].items():
    if name.startswith(("tools/", "spacr/")):
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert os.environ.get("SPACR_DEVICE") == "cpu"
for family in (socket.AF_INET, socket.AF_INET6):
    try:
        probe = socket.socket(family, socket.SOCK_STREAM)
    except OSError:
        pass
    else:
        probe.close()
        raise RuntimeError("Required process-family network socket restriction is absent")
left, right = socket.socketpair()
left.close()
right.close()
database = Path(packet["database"]["path"]).resolve()
original = sqlite3.connect
connections = []
def readonly_connect(path, *args, **kwargs):
    actual = Path(os.fsdecode(path)).resolve()
    if actual == database:
        path = database.as_uri() + "?mode=ro"
        kwargs["uri"] = True
        connection = original(path, *args, **kwargs)
        connection.execute("PRAGMA query_only=ON")
        connections.append(str(database))
        return connection
    return original(path, *args, **kwargs)
sqlite3.connect = readonly_connect
sys.argv = packet["proposal"]["command"][1:]
sys.argv[0] = str(root / sys.argv[0])
sys.path.insert(0, str(root))
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    print(json.dumps({"operation": "readonly_database_connections", "count": len(connections), "database": str(database)}), flush=True)
