"""Exercise only the diagnostic branch and its real source-preservation finally."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1]


def record_ast():
    return next(n for n in ast.parse((TOOLS / 'capture_trellis.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'record_trellis')


def run_scope(stop, *, source_changed=False, private_changed=False):
    original = record_ast()
    outer = next(n for n in original.body if isinstance(n, ast.Try))
    position = next(i for i, n in enumerate(outer.body)
                    if isinstance(n, ast.For) and "('free', '06')" in ast.unparse(n.iter))
    # Execute the actual modes loop and next full-route check, with its unchanged finally.
    branch = copy.deepcopy(outer)
    branch.body = branch.body[position:position + 2]
    fn = ast.FunctionDef(name='actual_scope', args=ast.arguments(posonlyargs=[], args=[],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=[branch], decorator_list=[])
    events = []; saved = []; proof = dict(accepted=False, published=False)
    def check(name, **kwargs):
        events.append(name)
        if name == '09_shared_restored':
            raise RuntimeError('continued full route')
    def preserve(*args):
        events.append('source_preservation')
        if source_changed:
            raise ValueError('source changed')
    values = dict(stop_after_free_axes=stop, proof=proof, desktop=None,
        work=Path('/unchanged-private'), source=Path('/original'), captures=Path('/capture'),
        original={'source_bundle': {}, 'database_sha256': 'original-db'},
        _digest=lambda path: 'changed-db' if private_changed else 'original-db',
        require_unchanged_source=preserve, events=[], Path=Path,
        screen=SimpleNamespace(active_jobs=lambda: 0), idle=lambda: events.append('idle'),
        open_popup=lambda *args: None, panel=SimpleNamespace(_scale_x=None, _scale_y=None),
        combo=lambda *args, **kwargs: None, check=check,
        write_json=lambda path, value: saved.append(copy.deepcopy(value)))
    code = compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                   str(TOOLS / 'capture_trellis.py'), 'exec')
    exec(code, values)
    return values['actual_scope'], proof, events, saved


def test_probe_returns_only_after_free_check_and_preserves_false_full_acceptance():
    run, proof, events, saved = run_scope(True)
    run()
    assert events == ['06_both_axes_free', 'source_preservation', 'idle']
    assert proof['accepted'] is False and proof['published'] is False
    assert proof['diagnostic_only'] is True and proof['publication_eligible'] is False
    assert proof['stopped_after'] == '06_both_axes_free'
    assert proof['original_unchanged'] is True and proof['private_database_unchanged'] is True
    assert proof['remaining_workers'] == 0 and saved == [proof]


def test_default_continues_all_modes_and_full_route_without_diagnostic_fields():
    run, proof, events, saved = run_scope(False)
    with pytest.raises(RuntimeError, match='continued full route'):
        run()
    assert events[:4] == ['06_both_axes_free', '07_both_axes_row', '08_both_axes_col', '09_shared_restored']
    assert 'diagnostic_only' not in proof and 'stopped_after' not in proof
    assert proof['original_unchanged'] is True and saved == [proof]
    defaults = record_ast().args.kw_defaults
    assert len(defaults) == 1 and ast.literal_eval(defaults[0]) is False


@pytest.mark.parametrize('source_changed,private_changed,error', [
    (True, False, 'source changed'), (False, True, 'Private measurement database changed')])
def test_probe_cannot_escape_source_or_private_database_preservation(source_changed, private_changed, error):
    run, proof, events, saved = run_scope(True, source_changed=source_changed, private_changed=private_changed)
    with pytest.raises(ValueError, match=error):
        run()
    assert proof['accepted'] is False and proof['published'] is False
    if private_changed:
        assert proof['private_database_unchanged'] is False and saved == [proof]


@pytest.mark.parametrize('module,run,download,preview,valid', [
    ('trellis', False, False, False, True), ('graph_builder', False, False, False, False),
    ('trellis', True, False, False, False), ('trellis', False, True, False, False),
    ('trellis', False, False, True, False)])
def test_explicit_flag_is_wired_and_restricted_to_diagnostic_trellis(module, run, download, preview, valid):
    tree = ast.parse((TOOLS / 'capture_refresh.py').read_text())
    declarations = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and any(isinstance(a, ast.Constant) and a.value == '--trellis-stop-after-free-axes' for a in n.args)]
    assert len(declarations) == 1
    assert any(k.arg == 'action' and ast.literal_eval(k.value) == 'store_true' for k in declarations[0].keywords)
    guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and ast.unparse(n.test).startswith('args.trellis_stop_after_free_axes and'))
    def reject(message): raise ValueError(message)
    env = dict(args=SimpleNamespace(trellis_stop_after_free_axes=True, module=module,
        run=run, download=download, preview=preview), parser=SimpleNamespace(error=reject))
    code = compile(ast.Module(body=[guard], type_ignores=[]), 'actual-parser-guard', 'exec')
    if valid: exec(code, env)
    else:
        with pytest.raises(ValueError, match='requires trellis'): exec(code, env)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == 'record_trellis')
    assert [(k.arg, ast.unparse(k.value)) for k in call.keywords] == [
        ('stop_after_free_axes', 'args.trellis_stop_after_free_axes')]
