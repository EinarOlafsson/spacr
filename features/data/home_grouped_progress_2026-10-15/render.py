"""Render the maintainer's grouped progress table from its canonical status."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
state = json.loads((root / "status.json").read_text(encoding="utf-8"))
rows = state["rows"]
assert len({row["item"] for row in rows}) == len(rows), "Duplicate numbered item"
lines = [
    "CURRENT COMBINED HOME PROGRESS — " + state["updated_date"],
    "",
    "Remaining items and recorded completions; one row per numbered item.",
    "Hours are estimates; external unknowns remain explicit.",
    "",
    "| Item | Percent done | Time left (hours) | Owner | Detailed description |",
    "|---|---:|---:|---|---|",
]
for row in rows:
    values = [row[key] for key in (
        "item", "percent_done", "time_left_hours", "owner", "description")]
    lines.append("| " + " | ".join(
        str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
(root / "PROGRESS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
