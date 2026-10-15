"""Check integrity and coverage of the recorded independent review."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    receipt = json.loads((root / "receipt.json").read_text())
    for name, expected in receipt["evidence"].items():
        assert digest((root / name).read_bytes()) == expected, name
    manifest = json.loads((root / "assignments.json").read_text())
    pairs = json.loads((root / "reviewed-pairs.json").read_text())
    assigned = {
        block["label"]: block
        for block in manifest["blocks"]
        if block["owner"] in receipt["drafts"]
    }
    assert len(assigned) == 30
    assert len(pairs) == 270
    assert len({(pair["language"], pair["label"]) for pair in pairs}) == 270
    assert collections.Counter(pair["language"] for pair in pairs) == {
        language: 30 for language in manifest["languages"]
    }
    assert collections.Counter(pair["author"] for pair in pairs) == {
        author: 135 for author in receipt["drafts"]
    }
    for pair in pairs:
        block = assigned[pair["label"]]
        assert pair["author"] == block["owner"]
        for name in ("source", "context_source"):
            assert pair[name] == block[name]
        for name, hash_name in (
            ("source", "source_sha256"),
            ("context_source", "context_sha256"),
            ("target", "target_sha256"),
        ):
            assert digest(pair[name].encode()) == pair[hash_name]
    excluded = {block["label"] for block in receipt["excluded_blocks"]}
    assert len(excluded) == 3
    assert not excluded.intersection(pair["label"] for pair in pairs)
    followup = json.loads((root / "guard-precision-followup.json").read_text())
    for change in followup:
        assert digest(change["old"].encode()) == change["old_sha256"]
        assert digest(change["new"].encode()) == change["new_sha256"]
        matching = [
            pair for pair in pairs
            if pair["language"] == change["language"]
            and pair["label"] == change["label"]
        ]
        assert len(matching) == 1
        assert matching[0]["target"] == change["new"]
    if args.drafts_dir:
        for author, draft in receipt["drafts"].items():
            name = Path(draft["file"]).name
            assert digest((args.drafts_dir / name).read_bytes()) == draft["sha256"], author
    print(json.dumps({
        "status": "PASS",
        "reviewed_pairs": len(pairs),
        "source_blocks": len(assigned),
        "excluded_blocks": len(excluded),
        "draft_hashes_checked": bool(args.drafts_dir),
        "boundary": "Integrity verification of an existing manual review; no semantic review or normal writer rerun.",
    }))


if __name__ == "__main__":
    main()
