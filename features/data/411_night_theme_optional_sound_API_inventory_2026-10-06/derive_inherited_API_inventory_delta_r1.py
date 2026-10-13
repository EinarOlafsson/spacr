from pathlib import Path
from unittest.mock import patch
import hashlib
import importlib.util
import json
import subprocess
import sys

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
anchor = '963fc99633417d52ba11d56008fc251509d2a687'
path = Path('tests/test_docstring_correctness.py').resolve()
spec = importlib.util.spec_from_file_location('_source_bound_API_drift', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
changed = subprocess.check_output(['git', 'diff', '--name-only', anchor, 'HEAD', '--', 'spacr'], text=True).splitlines()
texts = {Path(name).resolve(): subprocess.check_output(['git', 'show', anchor + ':' + name], text=True)
         for name in changed if name.endswith('.py') and '/i18n_catalogs/' not in name}
read_text = Path.read_text

def anchored_text(self, *args, **kwargs):
    return texts.get(self.resolve(), read_text(self, *args, **kwargs))

def rows():
    module._public_callable_inventory.cache_clear()
    return {row.symbol: row for row in module._public_callables()}

def record(row):
    return {'symbol': row.symbol, 'category': row.category, 'exposure': row.exposure,
            'parameters': sorted(row.parameters), 'required_parameters': sorted(row.required_parameters),
            'accepted_documented_parameters': sorted(row.accepted_documented_parameters),
            'accepts_arbitrary_keywords': row.accepts_arbitrary_keywords,
            'variant_count': row.variant_count, 'docless_variant_count': row.docless_variant_count,
            'constructor_prose_variant_count': row.constructor_prose_variant_count}

def digest(values):
    lines = [f"{row.symbol}\0{row.category}\0{row.exposure}\0"
             f"{','.join(sorted(row.parameters))}\0"
             f"{','.join(sorted(row.required_parameters))}\0"
             f"{','.join(sorted(row.accepted_documented_parameters))}\0"
             f"{int(row.accepts_arbitrary_keywords)}\0"
             f"{row.variant_count}\0{row.docless_variant_count}\0"
             f"{row.constructor_prose_variant_count}" for row in values]
    return hashlib.sha256('\n'.join(sorted(lines)).encode()).hexdigest()

current = rows()
with patch.object(Path, 'read_text', anchored_text):
    original = rows()
yolo = {'spacr.qt.mask_engine.' + name for name in (
    'yolo_box_lines', 'save_yolo_boxes', 'load_yolo_boxes', 'export_yolo_boxes')}
assert set(current) - set(original) == yolo and not set(original) - set(current)
prior_current = {key: row for key, row in current.items() if key not in yolo}
differences = {key: {'before': record(original[key]), 'after': record(row)}
               for key, row in prior_current.items() if record(row) != record(original[key])}
expected = 'bfe15fe1e3b29a17b80ebd01d0a016c909264a4eb519a3f379433774adf64abb'
assert digest(original.values()) == expected
restored = dict(prior_current)
for key in differences:
    restored[key] = original[key]
assert digest(restored.values()) == expected
result = {'anchor_commit': anchor, 'prior_count': len(original), 'new_count_excluding_YOLO': len(prior_current),
          'prior_parameter_count': sum(len(row.parameters) for row in original.values()),
          'current_parameter_count_excluding_YOLO': sum(len(row.parameters) for row in prior_current.values()),
          'prior_required_parameter_count': sum(len(row.required_parameters) for row in original.values()),
          'current_required_parameter_count_excluding_YOLO': sum(len(row.required_parameters) for row in prior_current.values()),
          'prior_digest': expected, 'current_digest_excluding_YOLO': digest(prior_current.values()),
          'full_inverse_digest_exact': True, 'changed_contracts': differences,
          'added_YOLO_contracts': {key: record(current[key]) for key in sorted(yolo)}}
(scratch / 'foundation-inherited-API-inventory-delta-r1.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
