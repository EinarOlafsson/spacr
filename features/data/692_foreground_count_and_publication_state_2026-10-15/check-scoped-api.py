import hashlib
import json
from pathlib import Path
import sys

root = Path.cwd()
sys.path.insert(0, str(root / "tools"))
import build_i18n_catalogs as runtime

def forbidden(*args, **kwargs):
    raise AssertionError("N692 scoped API check must not collect global runtime sources or decode a model")

runtime.canonical_sources = forbidden
import build_documentation_i18n as api
contract = json.loads((root / "tests/data/release_contracts/692_three_d_mask_drawing_api_arrivals_2026-10-11.json").read_text())
en = json.loads((root / "docs/source/_static/i18n/api/en.json").read_text())["symbols"]
result = {}
for language in ("sv", "de", "es", "zh_CN", "fr", "pt", "hi", "ko", "is"):
    reviewed = api.reviewed_api_block_translations(contract, language)
    compiled = json.loads((root / f"docs/source/_static/i18n/api/{language}.json").read_text())["symbols"]
    count = 0
    for symbol, source in contract.items():
        entry = compiled[symbol]
        assert entry["source_sha256"] == en[symbol]["source_sha256"]
        assert entry["source_blocks_sha256"] == en[symbol]["source_blocks_sha256"]
        source_blocks, _ = api.translatable_blocks(source)
        target_blocks, _ = api.translatable_blocks(entry["text"])
        assert len(source_blocks) == len(target_blocks)
        for original, translated in zip(source_blocks, target_blocks):
            if api._api_block_requires_translation(original):
                assert original in reviewed, (language, symbol, original)
            if original in reviewed:
                assert reviewed[original] == translated, (language, symbol, original)
                count += 1
    result[language] = {"symbols": len(contract), "exact_normal_reviewed_compiled_blocks": count}
    print(language, result[language], flush=True)
Path("/mnt/wd4tb/scratch/n692-current-state-reconciliation-20261015/scoped-api-proof.json").write_text(json.dumps(result, indent=2) + "\n")
