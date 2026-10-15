import hashlib
import json
import sys
from pathlib import Path

from spacr.qt.help_api_index import API_ENTRIES
from spacr.qt.help_index import HelpEntry, search
from spacr.qt.help_search import api_url, local_docstring

root = Path.cwd()
keys = {
    "spacr.qt.widgets.measurement_compare_dialog.MeasurementComparePanel.workspace_state": "spacr/qt/widgets/measurement_compare_dialog",
    "spacr.qt.widgets.measurement_compare_dialog.MeasurementComparePanel.apply_workspace_state": "spacr/qt/widgets/measurement_compare_dialog",
    "spacr.qt.screens.embeddings.EmbeddingsScreen._pretrain_dino._progress": "spacr/qt/screens/embeddings",
    "spacr.qt.screens.embeddings.EmbeddingsScreen._pretrain_dino._done": "spacr/qt/screens/embeddings",
}
manifest = json.loads((root / "docs/source/_static/i18n/api/en.json").read_text())["symbols"]
rows = []
for key, module in keys.items():
    matches = [summary for symbol, summary in API_ENTRIES if symbol == key]
    assert len(matches) == 1, (key, matches)
    summary = matches[0]
    local = local_docstring(key)
    assert local and local.startswith(summary), (key, summary, local)
    assert manifest[key]["text"].startswith(summary), key
    expected = "/" + module + "/index.html"
    assert api_url(key, "en").endswith(expected + "#" + key), key
    assert api_url(key, "ko").endswith(expected + "?lang=ko#" + key), key
    rows.append(HelpEntry(kind="api", title=key, subtitle="API reference", description=summary, payload={"symbol":key}))
    assert module.replace("/", ".") not in sys.modules, module
for row in rows:
    found = search(rows, row.title, limit=8)
    assert found and found[0].title == row.title, row.title
print(json.dumps({"checked_symbols":list(keys), "result":"PASS", "scope":"Four exact generated API entries, published English, offline source prose, en/ko URL anchors and exact-name ranking. No target module imports, global helper scan, generator, network or rendered-site claim.", "help_index_sha256":hashlib.sha256((root / "spacr/qt/help_api_index.py").read_bytes()).hexdigest()}, indent=2))
