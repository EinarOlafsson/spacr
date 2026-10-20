"""Prepare all nine ordinary preflights, then normally write six suffix repairs."""
import ast
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())
assert config['ready'] is True, 'Frozen plan only: coordinator authorization required'
root = Path(config['root']).resolve()
assert Path.cwd().resolve() == root
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == config['input_head']
assert not subprocess.check_output(['git','status','--porcelain']), 'Input checkout must be clean'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def check(pins):
    for name, digest in pins.items():
        assert sha(root / name) == digest, ('Input drift', name)
check(config['input_pins'])
check(config['catalog_baseline_pins'])
check(config['protected_pins'])
def refused(*args, **kwargs):
    raise RuntimeError('Model decoding forbidden in scoped punctuation repair')
original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.')[0] in {'transformers','safetensors','onnxruntime','huggingface_hub'}:
        raise RuntimeError('Model/provider import forbidden: ' + name)
    result = original_import(name, globals, locals, fromlist, level)
    for module_name, attribute in [('torch','load'),('torch.serialization','load'),('torch.jit','load')]:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, attribute):
            setattr(module, attribute, refused)
    return result
builtins.__import__ = guarded_import
def audit_event(event, args):
    if event in {'socket.connect','socket.getaddrinfo'}:
        raise RuntimeError('Network forbidden in punctuation repair')
    if (event == 'subprocess.Popen' and Path(str(args[0])).name != 'git'
            and not (str(args[0]) == '/sbin/ldconfig'
                     and args[1] == ['/sbin/ldconfig', '-p'])):
        raise RuntimeError('External process forbidden in punctuation repair')
sys.addaudithook(audit_event)
sys.path[:0] = [str(root / 'tools'), str(root)]
import build_i18n_catalogs as b
assert Path(b.__file__).resolve() == root / 'tools/build_i18n_catalogs.py'
b._translate_batches = refused
original_canonical = b.canonical_sources
sources = original_canonical()
expected_hashes = b._source_hashes(sources)
source = config['source']
assert source in sources['ui']
assert hashlib.sha256(source.encode()).hexdigest() == config['source_sha256']
# Genuine actual context is shared; ordinary validators/loaders are unchanged.
b.canonical_sources = lambda: sources
tables = {'SETTING_LABELS':'setting_labels','SETTING_TOOLTIPS':'setting_tooltips',
          'MODULE_SUMMARIES':'module_summaries','CATEGORY_HELP':'categories','UI':'ui'}
def read_catalog(path):
    module = types.ModuleType('snapshot_' + path.stem)
    # Generated catalogs contain literal assignments; execute no imported code.
    parsed = ast.parse(path.read_bytes())
    assert all(isinstance(n,(ast.Assign,ast.AnnAssign,ast.Expr)) for n in parsed.body)
    exec(compile(parsed,str(path),'exec'),module.__dict__)
    return module
prepared = {}
receipt = {'input_head':config['input_head'],'driver_sha256':sha(Path(__file__)),
           'config_sha256':sha(config_path),'locales':{}}
for language in b.MODEL_SPECS:
    reviewed = b.reviewed_runtime_translations(language)
    current = read_catalog(root / f'spacr/qt/i18n_catalogs/{language}.py')
    target = config['new_targets'].get(language,current.UI[source])
    assert reviewed[source] == target
    assert b._reviewed_translation(source,language) == target
    assert b._contextualize(target,language,source) == target
    assert b._translation_candidate_valid(source,target,language,force=False)
    if language in config['new_targets']:
        assert current.UI[source] == config['old_targets'][language]
    translations, foreign, hashes = {}, {}, {}
    for table, section in tables.items():
        values = sources[section]
        pairs = values.items() if isinstance(values,dict) else ((v,v) for v in values)
        present = set()
        for key, text in pairs:
            identity = (table,str(key)); present.add(key)
            assert key in getattr(current,table), (language,'missing row',identity)
            assert current.SOURCE_HASHES.get(identity) == expected_hashes[identity], (language,'stale row',identity)
            value = target if text == source and language in config['new_targets'] else getattr(current,table)[key]
            assert text not in translations or translations[text] == value, (language,'conflicting source',text)
            translations[text] = value
            if text != source or language not in config['new_targets']:
                foreign[identity] = getattr(current,table)[key]
            hashes[identity] = current.SOURCE_HASHES[identity]
        assert set(getattr(current,table)) == present, (language,'unrelated row deletion',table)
    assert hashes == current.SOURCE_HASHES
    prepared[language] = (translations,foreign,hashes,target)
    receipt['locales'][language] = {'preflight':'ordinary loader/context/candidate PASS',
                                  'owned_changes':int(language in config['new_targets']),
                                  'foreign_values_preserved':len(foreign)}
    print(language + ': normal preflight passed',flush=True)
# No writes occur until every locale passes.
check(config['input_pins']); check(config['catalog_baseline_pins']); check(config['protected_pins'])
for language in config['new_targets']:
    translations,foreign,hashes,target = prepared[language]
    path = b.write_language(language,sources,translations)
    new = read_catalog(path)
    assert new.UI[source] == target
    assert all(getattr(new,table)[key] == value for (table,key),value in foreign.items())
    assert new.SOURCE_HASHES == hashes
# All three unaffected locales, EN/API/Help/arrival files retain exact bytes.
check(config['protected_pins'])
assert b.audit(sources,tuple(b.MODEL_SPECS)) == 0
fresh = original_canonical()
assert fresh == sources
assert b._source_hashes(fresh) == expected_hashes
check(config['input_pins']); check(config['protected_pins'])
allowed = {f'spacr/qt/i18n_catalogs/{language}.py' for language in config['new_targets']}
changed = set(subprocess.check_output(['git','diff','--name-only'],text=True).splitlines())
assert changed <= allowed, ('Unexpected repository changes', changed - allowed)
receipt.update({'full_nine_runtime_audit_exit':0,'fresh_actual_canonical_equal':True,
                'all_source_hashes_unchanged':True,'protected_bytes_unchanged':config['protected_pins'],
                'actual_changed_paths':sorted(changed),'no_EN_API_Help_arrival_rewrite':True})
Path(config['receipt']).write_text(json.dumps(receipt,indent=2)+'\n')
print('Six punctuation-only normal runtime repair PASS',flush=True)
