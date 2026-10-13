"""Current publication drift must be visible outside historical review lists."""
import json
import subprocess
import sys
from pathlib import Path


def test_current_alpha_and_unlisted_lessons_are_audited_without_network(tmp_path):
    root = Path(__file__).resolve().parents[1]
    lessons = tmp_path / "lessons"
    lessons.mkdir()
    (lessons / "reviews").mkdir()
    identities = ("86_alpha_organism_modules", "87_alpha_features", "88_new_screen")
    published = [
        {"id": identity, "scenes": [{"narration": "Open the current screen."}]}
        for identity in identities
    ]
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"lessons": published}))
    for lesson in published:
        authored = json.loads(json.dumps(lesson))
        if authored["id"] != identities[0]:
            authored["scenes"][0]["narration"] = "Inspect the new live plate card."
        (lessons / f'{authored["id"]}.json').write_text(json.dumps(authored))
    report_path = tmp_path / "report.json"
    subprocess.run(
        [sys.executable, str(root / "tools/tutorials/audit_user_walkthroughs.py"),
         "--live-catalog", catalog.as_uri(), "--main-catalog", catalog.as_uri(),
         "--prepared-catalog", str(catalog), "--lesson-root", str(lessons),
         "--output", str(report_path)],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(report_path.read_text())
    assert report["narration_and_caption_refresh_required"] == list(identities[1:])
    assert report["manual_review_pending"] == [identities[2]]
    rows = {row["lesson"]: row for row in report["rows"]}
    assert rows[identities[0]]["newly_authored"]
    assert rows[identities[0]]["published_rewrite"]
    assert rows[identities[1]]["newly_authored"]
    assert not rows[identities[1]]["prepared_rewrite"]
