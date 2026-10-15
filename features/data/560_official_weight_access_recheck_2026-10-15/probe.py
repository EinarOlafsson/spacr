"""Read official access metadata without requesting credentials or model files."""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

urls = {
    "official_readme": "https://raw.githubusercontent.com/facebookresearch/dinov2/main/README.md",
    "zenodo_record": "https://zenodo.org/api/records/14767885",
}
result = {"checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "sources": {}}
for name, url in urls.items():
    request = urllib.request.Request(url, headers={"User-Agent": "spacr-readonly-access-recheck"})
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read()
        record = {"url": url, "status": response.status, "sha256": hashlib.sha256(body).hexdigest()}
    if name == "official_readme":
        text = body.decode()
        record["official_request_link_present"] = "https://ai.meta.com/resources/models-and-libraries/cell-dino-downloads/" in text
        record["accepted_email_route_present"] = "once accepted, an e-mail will be sent" in text
    else:
        document = json.loads(body)
        record["access_right"] = document.get("metadata", {}).get("access_right")
        record["listed_files"] = len(document.get("files", []))
        record["record_updated"] = document.get("updated")
    result["sources"][name] = record
result["actions"] = "Read-only official documentation/record metadata; no access request, login, license submission or model download."
Path(__file__).with_name("receipt.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
