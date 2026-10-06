from pathlib import Path
from unittest.mock import patch
import hashlib
import importlib.util
import json
import sys
import traceback

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
path = Path('tests/test_docstring_correctness.py').resolve()
spec = importlib.util.spec_from_file_location('_foundation_callable_inventory_guard', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
source = Path('spacr/embeddings.py').resolve()
original = (scratch / 'foundation-api-refresh-baseline-r1/embeddings.py').read_text()
read_text = Path.read_text

def previous_text(self, *args, **kwargs):
    return original if self.resolve() == source else read_text(self, *args, **kwargs)

def snapshot():
    module._public_callable_inventory.cache_clear()
    rows = list(module._public_callables())
    layout = {item.symbol: {
        'category': item.category, 'exposure': item.exposure,
        'parameters': sorted(item.parameters),
        'required_parameters': sorted(item.required_parameters),
        'variant_count': item.variant_count,
        'docless_variant_count': item.docless_variant_count,
    } for item in rows}
    yolo = {'spacr.qt.mask_engine.' + name for name in (
        'yolo_box_lines', 'save_yolo_boxes', 'load_yolo_boxes', 'export_yolo_boxes')}
    admitted = [item for item in rows if item.symbol not in yolo]
    assert sum(len(item.parameters) for item in admitted) == 19860
    try:
        module.test_public_callable_inventory_is_source_derived_not_docstring_derived()
    except AssertionError as failure:
        last = traceback.extract_tb(failure.__traceback__)[-1]
        assert last.filename == str(path) and last.lineno == 2705, last
    else:
        raise AssertionError('the inherited exact pin mismatch was not reproduced')
    return layout

current = snapshot()
with patch.object(Path, 'read_text', previous_text):
    prior = snapshot()
assert current == prior
result = {
    'unchanged_pre_repair_and_current_source_fail_the_same_parameter_pin': True,
    'actual_parameters_excluding_four_named_YOLO_helpers': 19860,
    'existing_pin': 19859,
    'exact_failing_test': 'tests/test_docstring_correctness.py::test_public_callable_inventory_is_source_derived_not_docstring_derived',
    'exact_failing_line': 2705,
    'all_complete_callable_signature_and_boundary_rows_unchanged': len(current),
    'complete_boundary_sha256': hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest(),
    'no_pin_or_guard_changed': True,
}
(scratch / 'foundation-inherited-callable-inventory-failure-r1.json').write_text(json.dumps(result, indent=2) + '\n')
print('PASS independent baseline reproduction: same inherited 19860 versus 19859 pin; all', len(current), 'complete boundary rows unchanged', flush=True)
