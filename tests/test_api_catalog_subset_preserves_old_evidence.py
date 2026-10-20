"""A narrow API refresh cannot relabel unrelated model-input provenance."""
import hashlib
import json
import sys
from pathlib import Path


def test_scoped_api_writer_preserves_missing_baseline_and_old_model_inputs(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    import build_documentation_i18n as builder
    monkeypatch.setattr(builder, 'API_DIR', tmp_path)
    unrelated = {'source_sha256': 'historical-source',
                 'source_blocks_sha256': ['historical-block'],
                 'translation_source_blocks_sha256': ['actual-older-model-input'],
                 'text': 'Tidigare översättning'}
    before = {'schema': 2, 'language': 'sv', 'generator': 'historical-generator',
              'license': 'historical-license',
              'symbols': {'spacr.previous_symbol': unrelated}}
    path = tmp_path / 'sv.json'
    path.write_text(json.dumps(before))
    source = 'The updater verifies its actual application before restarting.'
    builder.write_language({'spacr.installer_update': source}, 'sv',
                           {'spacr.installer_update': 'Uppdateraren verifierar programmet före omstart.'},
                           preserve_other_symbols=True)
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after['symbols']['spacr.previous_symbol'] == unrelated
    assert {k: v for k, v in after.items() if k != 'symbols'} == {
        k: v for k, v in before.items() if k != 'symbols'}
    assert set(after['symbols']) == {'spacr.previous_symbol', 'spacr.installer_update'}
    assert after['symbols']['spacr.installer_update']['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
