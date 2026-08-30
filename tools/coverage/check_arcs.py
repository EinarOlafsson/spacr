"""Did the tests just run actually reach the arcs they claim to close?

Usage: check_arcs.py <coverage.json> <module.py>:<a>,<b> [more...]
An arc is given as "line-line"; a bare line is a statement.
Prints one row per target and exits 1 if any is still missing.
"""
import json, sys
cov = json.load(open(sys.argv[1]))
bad = 0
for spec in sys.argv[2:]:
    path, targets = spec.split(":", 1)
    key = [p for p in cov["files"] if p.endswith(path)]
    if not key:
        print(f"  {'NO DATA':>8}  {path}"); bad += 1; continue
    f = cov["files"][key[0]]
    ex_lines = set(f["executed_lines"])
    ex_arcs = {tuple(a) for a in (f.get("executed_branches") or [])}
    miss_arcs = {tuple(a) for a in (f.get("missing_branches") or [])}
    for t in targets.split(","):
        # An arc target may be NEGATIVE: coverage.py writes an exit from a
        # function as `line -> -<first line of the function>`, so a naive
        # split on "-" produces an empty field and dies. Split on the first
        # separator only, and keep whatever sign the rest carries.
        if "-" in t[1:]:
            head, _, tail = t[1:].partition("-")
            a, b = int(t[0] + head), int(tail)
            ok = (a, b) in ex_arcs and (a, b) not in miss_arcs
            label = f"arc {a}->{b}"
        else:
            ok = int(t) in ex_lines
            label = f"line {t}"
        print(f"  {'REACHED' if ok else 'MISSING':>8}  {path} {label}")
        bad += 0 if ok else 1
sys.exit(1 if bad else 0)
