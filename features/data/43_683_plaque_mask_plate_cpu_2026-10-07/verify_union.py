"""Check the source-bound hosted/focused union without rerunning tests."""
import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--git", action="store_true")
args = parser.parse_args()
folder = Path(__file__).resolve().parent
root = folder.parents[2]
prefix = folder.relative_to(root).as_posix()

def read(name):
    if args.git:
        return subprocess.check_output(["git", "-C", str(root), "show", "HEAD:" + prefix + "/" + name])
    return (folder / name).read_bytes()

receipt = json.loads(read("receipt.json"))
host = json.loads(gzip.decompress(read("hosted-four.json.gz")))["files"]
focused = json.loads(gzip.decompress(read("focused.json.gz")))["files"]
for name, expected in receipt["four_module_union"].items():
    source = gzip.decompress(read(name.replace("/", "__") + ".gz"))
    assert hashlib.sha256(source).hexdigest() == receipt["source_hashes"][name]
    current = subprocess.check_output(["git", "-C", str(root), "show", "HEAD:" + name])
    assert current == source, "production changed: " + name
    old = host[name]
    new = focused[name]
    lines = sorted(set(old["missing_lines"]) - set(new["executed_lines"]))
    arcs = sorted(set(map(tuple, old["missing_branches"])) - set(map(tuple, new["executed_branches"])))
    assert lines == expected["remaining_lines"]
    assert [list(arc) for arc in arcs] == expected["remaining_branches"]
    baseline = expected["unchanged_baseline"]
    assert len(lines) <= baseline["uncovered_statements"]
    assert len(arcs) <= baseline["uncovered_branches"]
    assert old["summary"]["excluded_lines"] <= baseline["excluded_lines"]
    assert source.count(b"pragma: no cover") <= baseline["pragma_no_cover"]
    print(name, len(lines), len(arcs), "within original allowance")
print("Verified four unchanged-source numerical unions")
