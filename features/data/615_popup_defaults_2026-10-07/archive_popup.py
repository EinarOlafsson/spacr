from pathlib import Path
import configparser
import ast
import gzip
import hashlib
import json
import shutil

repo = Path('/media/carruthers/mnt3/codex/spacr-worktrees/popup-defaults-20261007')
scratch = Path('/media/carruthers/mnt3/codex/scratch/popup-defaults-20261007')
output = repo / 'features/data/615_popup_defaults_2026-10-07'
output.mkdir(parents=True, exist_ok=True)
appearance_keys = {
    'prefs': ('theme', 'ambient_enabled', 'ambient_theme', 'ambient_palette',
              'ambient_primary', 'ambient_accent', 'ambient_resolution',
              'ambient_speed', 'ambient_size', 'ambient_density',
              'ambient_gravity_radius', 'ambient_blink_percent',
              'field_popup_wave_frequency', 'pane_opacity', 'field_fade'),
    'rim': ('enabled', 'length_fraction', 'length_px', 'lag', 'alignment',
            'mode', 'period_s', 'popup_backdrop', 'popup_backdrop_darkness'),
}
profile = configparser.ConfigParser(interpolation=None)
profile.read('/home/carruthers/.config/spacr/qt.conf')
appearance = {
    f'{section}/{key}': profile.get(section, key)
    for section, keys in appearance_keys.items()
    for key in keys if profile.has_option(section, key)
}
(output / 'saved_appearance.json').write_text(
    json.dumps(appearance, indent=2, sort_keys=True) + '\n')
logs = list(scratch.glob('*.log'))
for path in logs:
    if not path.name.endswith('-tool.log'):
        assert 'Terminal exit status:' in path.read_text(), path
    shutil.copy2(path, output / path.name)
capture = scratch / 'native-preferences-r1/captures/popup_defaults_20261007_r1'
for path in capture.iterdir():
    if path.is_file() and path.suffix in ('.png', '.json'):
        destination = output / 'native' / path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
for relative in (
    'spacr/qt/preferences.py', 'spacr/qt/widgets/ambient.py',
    'spacr/qt/widgets/setup_card.py', 'tests/qt/test_field_popup_waves.py',
    'tests/qt/test_popup_backdrop_apply.py',
    'tests/qt/test_requested_appearance_defaults.py',
):
    destination = output / 'source' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / relative, destination)
shutil.copy2(Path(__file__), output / 'archive_popup.py')
def executable_tree(path):
    if not path.exists() and path.with_suffix(path.suffix + '.gz').exists():
        text = gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes()).decode()
    else:
        text = path.read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    node.body.pop(0)
    return ast.dump(tree, include_attributes=False)
for relative in ('spacr/qt/preferences.py', 'spacr/qt/widgets/ambient.py',
                 'spacr/qt/widgets/setup_card.py'):
    assert executable_tree(output / 'source-native' / relative) == executable_tree(
        output / 'source' / relative), relative
for path in list(output.rglob('*')):
    if path.is_file() and path.suffix in ('.log', '.py') and path.name != 'archive_popup.py':
        path.with_suffix(path.suffix + '.gz').write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        path.unlink()
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {
    'date': '2026-10-07',
    'base_commit': 'bb7f7585cd9fa2a886e4509010b35f258c529f77',
    'changes': ['Immediate first enabled popup wave',
                'Page opacity on settings cards',
                'Saved appearance defaults: gravity 15%, waves 5/min, rim 17%'],
    'tests': {'popup_r3': {'passed': 52, 'exit_status': 0},
              'neighbors_r2': {'passed': 72, 'exit_status': 0},
              'source_guards_r2': {'passed': 18, 'exit_status': 0},
              'overlapping_cohorts': True},
    'native_capture': {'exit_status': 0, 'resolution': [3840, 2160],
                       'private_fresh_profile': True,
                       'visually_reviewed': '13d_appearance_animation.png'},
    'generation': {'english_api_symbols': 13214,
                   'english_api_audit_exit_status': 0,
                   'help_generation_exit_status': 0,
                   'runtime': {'generation_exit_status': 0,
                               'audit_exit_status': 0,
                               'languages': 9, 'settings': 1243,
                               'categories': 237, 'ui': 7239, 'modules': 77}},
    'retained_failures': ['popup-r1.log: incorrect straight-RGB assertion',
                          'popup-neighbors-r1.log: nonexistent test filename',
                          'source-guards-r1.log: two missing API parameter descriptions'],
    'native_executable_source_matches_final': True,
    'excluded_claims': ['Native compositor stacking', 'Native 24 FPS',
                        'Original installed Save/puncta crash causation',
                        'Full translated API/guides/tutorial publication'],
    'source_sha256': {str(path.relative_to(repo)): digest(path)
                      for path in (repo / 'spacr/qt/preferences.py',
                                   repo / 'spacr/qt/widgets/ambient.py',
                                   repo / 'spacr/qt/widgets/setup_card.py')},
    'artifacts': {str(path.relative_to(output)): digest(path)
                  for path in sorted(output.rglob('*'))
                  if path.is_file() and path.name != 'receipt.json'},
}
(output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('Archived', len(receipt['artifacts']), 'owned artifacts')
