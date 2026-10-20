"""Refresh the shared progress table without model calls or chat messages."""
import argparse
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True,
                                   stderr=subprocess.PIPE, timeout=90)


def write(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def refresh(root, output, state):
    ref = "origin/nightly"
    sha = git(root, "rev-parse", ref).strip()
    renderer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if state.get("source_sha") == sha and state.get("renderer_sha256") == renderer:
        return state
    names = git(root, "ls-tree", "-r", "--name-only", ref, "features/data", "features/new")
    ledgers = sorted(name for name in names.splitlines()
                     if re.fullmatch(r"features/data/progress_\d{4}-\d{2}-\d{2}_\d{4}\.md", name))
    ledger = git(root, "show", f"{ref}:{ledgers[-1]}")
    rows = state.get("rows", {})
    for line in ledger.splitlines():
        if not line.startswith("| ["):
            continue
        cells = [value.strip() for value in line.strip("|").split("|")]
        match = re.fullmatch(r"\[(\d+)\]\(.*?/((?:features)/.*?)\)", cells[0])
        if not match:
            continue
        number, path = match.groups()
        if "100%" in cells[2] and number not in rows:
            continue
        previous = rows.get(number, {})
        rows[number] = dict(path=path, owner=re.sub(r"\bme\b", "WS", cells[1]),
                            percent=cells[2], remaining=cells[3],
                            description=cells[4].replace("**", ""),
                            done=previous.get("done", False))
        if "100%" in cells[2]:
            rows[number]["done"] = True
    old_done = set(state.get("completed", []))
    highest = max(map(int, rows), default=0)
    for path in names.splitlines():
        match = re.fullmatch(r"features/new/(\d+)_.*\.txt", path)
        if not match or int(match[1]) <= highest:
            continue
        source = git(root, "show", f"{ref}:{path}")
        owner = re.search(r"^Owner:\s*(.+)$", source, re.M)
        description = re.search(r"^Description:\s*(.+)$", source, re.M)
        rows[match[1]] = dict(path=path, owner=owner[1] if owner else "Unassigned",
                             percent="—", remaining="Unknown", done=False,
                             description=description[1] if description else source.splitlines()[0])
    for number, row in rows.items():
        source = git(root, "show", f"{ref}:{row['path']}")
        if re.search(r"^Status:\s*(?:COMPLETE|DONE)\s+100%(?:\s|$)", source, re.I | re.M):
            row["done"] = True
        if row["done"]:
            row["percent"], row["remaining"] = "✅ 100%", "0"
    completed = {number for number, row in rows.items() if row["done"]}
    newly = sorted(completed - old_done, key=int)
    stamp = dt.datetime.now(ZoneInfo("America/Detroit")).strftime("%Y-%m-%d %H:%M:%S %Z")
    note = (f"Updated {stamp}; committed nightly source {sha[:12]}. "
            f"Descriptions come from {Path(ledgers[-1]).stem}; completion reads current item status. "
            "— means unmeasured. Completion requires an explicit recorded 100% status; "
            "passing a subset of tests does not finish its parent item.")
    header = "| item | owner | percent done | time left | discription |\n|---|---|---:|---|---|"
    table = [header]
    html_rows = []
    for number in sorted(rows, key=int):
        row = rows[number]
        item = ("F" if "/future/" in row["path"] else "N") + number
        values = [item, row["owner"], row["percent"], row["remaining"], row["description"]]
        table.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
        html_rows.append("<tr>" + "".join("<td>" + html.escape(value) + "</td>" for value in values) + "</tr>")
    document = note + "\n\n" + "\n".join(table) + "\n"
    write(output / "progress.md", document)
    page = ('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20">'
            '<title>spaCR progress</title><style>body{font:15px system-ui;margin:24px;'
            'background:#171b22;color:#e3e9ef}table{border-collapse:collapse;width:100%}'
            'th,td{padding:10px;text-align:left;border-bottom:1px solid #3a4350}'
            'th{position:sticky;top:0;background:#171b22}</style><h1>spaCR progress</h1><p>'
            + html.escape(note) + '</p><table><thead><tr>'
            + ''.join('<th>' + value + '</th>' for value in
                      ['item', 'owner', 'percent done', 'time left', 'discription'])
            + '</tr></thead><tbody>' + ''.join(html_rows) + '</tbody></table>')
    write(output / "progress.html", page)
    if newly:
        snapshot = output / ("completed-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".md")
        write(snapshot, "Newly completed: " + ", ".join(newly) + "\n\n" + document)
        with (output / "events.log").open("a", encoding="utf-8") as log:
            log.write("Newly completed: " + ", ".join(newly) + "\n\n" + document + "\n")
        print("Completed:", ", ".join(newly), flush=True)
    state = dict(source_sha=sha, renderer_sha256=renderer, rows=rows, completed=sorted(completed, key=int),
                 updated=stamp, table_sha256=hashlib.sha256(document.encode()).hexdigest())
    write(output / "state.json", json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--hours", type=float, default=12, help="0 watches until stopped")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    saved = args.output / "state.json"
    state = json.loads(saved.read_text()) if saved.exists() else {}
    deadline = time.monotonic() + args.hours * 3600 if args.hours > 0 else float("inf")
    while True:
        try:
            if args.watch:
                subprocess.run(["git", "fetch", "origin", "nightly"], cwd=root,
                               capture_output=True, check=True, timeout=90)
            state = refresh(root, args.output, state)
        except Exception as exc:
            print(type(exc).__name__, str(exc), flush=True)
            if not args.watch:
                raise
        if not args.watch or time.monotonic() >= deadline:
            break
        time.sleep(min(300, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
