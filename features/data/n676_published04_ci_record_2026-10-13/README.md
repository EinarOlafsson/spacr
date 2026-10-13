# Verified published lesson 04 record for current CI

The complete checkpoint under `checkpoint/` is byte-identical to the existing
generated checkpoint in
held commit ea61dbab2 and the original candidate:
`/media/carruthers/mnt3/claude/n676_stages/host/stage-04-r23/release-candidate-append-ieltxi1i`.
The smallest local CI commit is 0e4c60968c46cd1a8d494dbf7cf3de3949e22983, based
on df1e07d70, changing only the two target JSON files. It is not pushed.

The immutable dataset revision is
6b04d1f04ebf22c4a621624f67442f0b24230c68. Its verified tag resolves to that
revision in a fresh HfApi read. The preserved original readback matched all
5,133 file metadata records and downloaded SHA-256 values, reading
7,393,785,118 bytes with zero failures. The receipt binds the complete manifest
SHA-256 8dd2e98facc69ea62721498db0b3b30f9b38c760b9d7bd65fa36795c850e8660.
The existing published playback log covers all 84 routes; this repair does
not repeat the full download or make a new quality claim for the remake goal.

Replacing only these generated files in an isolated copy of current nightly
passes all 32 tests/test_docs_media_budget.py cases, including the current
HTML/HF-root assertion. The raw capped CPU test output is included.

Home can import the generated artifacts with the existing checked-copy tool:

```python
from pathlib import Path
import json
import sys
root = Path.cwd()
sys.path[:0] = [str(root), str(root / "tools/tutorials")]
from build_release_candidate import copy_checked
source = root / "features/data/n676_published04_ci_record_2026-10-13"
records = json.loads((source / "manifest.json").read_text())["files"]
for record in records:
    destination = root / record["target"]
    copy_checked(source / record["artifact"], destination, [],
                 destination.parent, record["sha256"])
```

The source metadata remains exactly as generated and recorded, including the
original readback and all complete manifest entries. Home selects and integrates the CI-required checkpoint files; all
capture/narration tools remain held.

The full export has **148 files / 112,781,553 bytes**: all **145 recorded
checkpoint files / 112,468,718 bytes**, plus the unchanged checkpoint index,
publication receipt and published browser evidence. `export-manifest.json`
records every relative path, length and SHA-256. The largest file is
33,292,958 bytes; the directory preserves the original tree without a tarball.
All source and copied hashes were checked. Candidate and published browser
evidence binds the same release manifest; the published check records 84
passed cases. Its scope is the recorded published snapshot before Pages
deployment, with the existing verify-published log retained separately.

For full import, validate every export-manifest entry and checked-copy its
`checkpoint/<path>` into `tools/tutorials/release_candidate/<path>`. Home
can select the CI-required subset while retaining all original evidence.
The two portable JSON files above also match their checkpoint counterparts.
