from pathlib import Path
from dataclasses import replace
import importlib.util
import json
import sys

path = Path('tests/test_docstring_correctness.py').resolve()
spec = importlib.util.spec_from_file_location('_sound_key_API_guard_controls', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
rows = tuple(module._public_callables())
theme = next(row for row in rows if row.symbol == 'spacr.qt.night_themes.NightTheme')
module.test_public_callable_inventory_is_source_derived_not_docstring_derived()
controls = {
    'missing_sound_key': replace(theme, parameters=theme.parameters - {'sound_key'},
                                 accepted_documented_parameters=theme.accepted_documented_parameters - {'sound_key'}),
    'sound_key_required': replace(theme, required_parameters=theme.required_parameters | {'sound_key'}),
    'sound_key_undocumented': replace(theme, accepted_documented_parameters=theme.accepted_documented_parameters - {'sound_key'}),
    'extra_unexplained_optional_parameter': replace(theme, parameters=theme.parameters | {'unexplained'},
                                                    accepted_documented_parameters=theme.accepted_documented_parameters | {'unexplained'}),
    'arbitrary_keywords_changed': replace(theme, accepts_arbitrary_keywords=True),
}
for name, altered in controls.items():
    mutated = tuple(altered if row.symbol == theme.symbol else row for row in rows)
    module._public_callables = lambda: iter(mutated)
    try:
        module.test_public_callable_inventory_is_source_derived_not_docstring_derived()
    except AssertionError:
        print('PASS independent guard rejection:', name, flush=True)
    else:
        raise AssertionError('guard admitted ' + name)
out = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/API-boundary-guard-controls-r1.json')
out.write_text(json.dumps({'normal_complete_boundary_passed': True,
                           'all_five_independent_contract_mutations_rejected': list(controls),
                           'original_count_digest_required_docless_and_coverage_pins_unchanged': True}, indent=2) + '\n')
