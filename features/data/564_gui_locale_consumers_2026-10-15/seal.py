from pathlib import Path
import ast
import gzip
import hashlib
import json
import shutil
import subprocess

ROOT = Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015')
SCRATCH = Path('/mnt/wd4tb/scratch/f564-gui-consumer-20261015')
DEST = ROOT / 'features/data/564_gui_locale_consumers_2026-10-15'
DEST.mkdir(parents=True, exist_ok=False)
PIN = '7e0dbb3c4e32750100b6dfa9dc1a9021553d9434'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == PIN
assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).strip()
LANGUAGES = ['sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr']
log = (SCRATCH / 'consumer.log').read_text()
for language in LANGUAGES:
    assert f'{language}: 2 actual labels, typed bodies, flow links and defaults; 3 diagnostic lookups PASS' in log
assert (SCRATCH / 'consumer.exit').read_text().strip() == '1'
assert 'assert not network_attempts' in log
trace = json.loads((SCRATCH / 'startup-trace.json').read_text())
assert len(trace) == 1
assert any('console_panel.py' in line and '_fetch' in line for line in trace[0]['stack'])
files = [
    'spacr/settings.py', 'spacr/attribution.py', 'spacr/deep_spacr.py',
    'spacr/qt/screens/settings_model.py', 'spacr/qt/screens/app_screen.py',
    'spacr/qt/screens/settings_flow_index.py', 'spacr/qt/i18n.py',
    'spacr/qt/i18n_catalogs/__init__.py', 'spacr/qt/widgets/console_panel.py',
    'spacr/crashreport.py', 'docs/source/_generated/settings_flow.rst',
] + [f'spacr/qt/i18n_catalogs/{language}.py' for language in ['en', *LANGUAGES]]
files += [f'docs/i18n/reviewed/runtime/{language}/2026-10-15-f564-diffusion-controls-home-b.json' for language in LANGUAGES]
sha = lambda value: hashlib.sha256(value).hexdigest()
hashes = {}
for path in files:
    current = (ROOT / path).read_bytes()
    assert current == subprocess.check_output(['git', 'show', f'{PIN}:{path}'], cwd=ROOT)
    hashes[path] = sha(current)
for name in ('check_consumers.py', 'trace_startup.py', 'run.sh', 'consumer.exit', 'startup-trace.json'):
    shutil.copy2(SCRATCH / name, DEST / name)
for name in ('consumer.log', 'consumer.time', 'trace.log', 'trace.time'):
    (DEST / f'{name}.gz').write_bytes(gzip.compress((SCRATCH / name).read_bytes(), mtime=0))
receipt = {
    'requested_consumer_subchecks': 'PASS: all nine explicit language completion lines; no requested caption/link/default assertion failed',
    'whole_harness_terminal': {'exit': 1, 'session': 57223, 'wall_seconds': 4.82, 'peak_rss_kib': 1136164,
        'reason': 'Post-check self-imposed zero-attempt assertion detected one blocked background hosted-AI allowance connection.'},
    'trace_only_terminal': {'exit': 0, 'session': 88727, 'meaning': 'Identifies startup caller only; does not replay or accept the GUI locale cohort.'},
    'tested_commit': PIN,
    'coordinator_commit_merged_normally': '6fb0afd27',
    'runtime_artifacts': '95778240b', 'derived_artifacts': 'a4e29dd15',
    'languages': LANGUAGES, 'actual_control_language_cases': 18,
    'reviewed_setting_labels_consumed': 18, 'reviewed_tooltip_bodies_consumed': 18,
    'diagnostic_tr_lookups': 27, 'reviewed_targets_consumed': 63,
    'checks': [
        'Actual AppScreen activation lazy Counterfactuals heading and production _setting_label_for_field labels.',
        'Normal retranslate_widget_tree after persisted private language choice, with exact reviewed label/body equality.',
        'Rendered tooltip contains localized float type, body and API caption, generated settings-flow anchor and language query.',
        'Canonical English metadata remains unchanged; real spin values remain 0.95 and 0.0.',
        'Three normal tr diagnostic lookups per language, without calling backend diagnostics or model code.'
    ],
    'isolation': {'memory_cap': '4G', 'threads': 1, 'qt': 'offscreen', 'gpu': 'hidden', 'lock': 'home-b heavy.lock',
        'preferences': 'private INI via preferences._settings; XDG config in scratch',
        'network': 'Python socket.connect audit refused HTTPS before connection. No completed HTTPS request; DNS lookup absence is not established.',
        'blocked_caller': 'ConsolePanel.refresh_ai_usage → _fetch → crashreport.fetch_allowance → urllib.request.urlopen',
        'no_production_changes': True, 'no_locale_replay': True},
    'source_and_artifact_sha256': hashes,
    'source_stability': 'All listed files compared byte-for-byte to the tested commit after both terminal runs. The original in-process post-check hash comparison was not reached after its final assertion.',
    'limits': ['This is actual offscreen widget localization acceptance, not native UI, training, model, biological, hosted service, whole F564, full CI or release acceptance.',
               'Do not relabel original consumer run as exit 0; the later trace is diagnostic only.'],
}
receipt['evidence_sha256'] = {p.name: sha(p.read_bytes()) for p in sorted(DEST.iterdir())}
(DEST / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
shutil.copy2(__file__, DEST / 'seal.py')
print(json.dumps({'requested_cases': 18, 'targets': 63, 'consumer_exit_retained': 1, 'trace_exit': 0, 'source_files_exact': len(hashes)}))
