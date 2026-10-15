import ast
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
root = Path.cwd()
scratch = Path(__file__).resolve().parent
out = root / 'features/data/564_diffusion_ui_reviews_2026-10-15'
out.mkdir(parents=True,exist_ok=True)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
tool_path = 'tools/build_i18n_catalogs.py'
old = subprocess.check_output(['git','show',f'HEAD:{tool_path}'])
new = (root / tool_path).read_bytes()
a,b = ast.parse(old),ast.parse(new)
added = {'_cf_structure_weight','_cf_diffusion_strength','_cf_structure_loss'}
matched=[]
for n in ast.walk(b):
    if isinstance(n,ast.Set) and added <= {v.value for v in n.elts if isinstance(v,ast.Constant)}:
        matched.append([v.value for v in n.elts])
        n.elts=[v for v in n.elts if v.value not in added]
assert len(matched)==1
assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
test_path='tests/test_runtime_backend_caption_owners.py'
old_tests=ast.parse(subprocess.check_output(['git','show',f'HEAD:{test_path}']))
new_tests=ast.parse((root/test_path).read_bytes())
old_funcs={n.name:ast.dump(n,include_attributes=False) for n in old_tests.body if isinstance(n,ast.FunctionDef)}
new_funcs={n.name:ast.dump(n,include_attributes=False) for n in new_tests.body if isinstance(n,ast.FunctionDef)}
assert all(new_funcs[k]==v for k,v in old_funcs.items())
input_hashes={}
record_bindings=[]
for path in sorted((scratch/'inputs').glob('*/*.json')):
    dest=root/'docs/i18n/reviewed/runtime'/path.parent.name/path.name
    assert not dest.exists(),dest
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,dest)
    input_hashes[str(dest.relative_to(root))]=sha(dest.read_bytes())
    for row in json.loads(dest.read_text())['records']:
        record_bindings.append({'language':path.parent.name,'table':row['table'],'key':row['key'],'source_sha256':row['source_sha256'],'target_sha256':sha(row['translation'].encode())})
assert len(input_hashes)==9 and len(record_bindings)==63
artifacts={}
for stem in ['baseline','final','inputs']:
    for suffix in ['sh','log','xml','time','exit']:
        p=scratch/(stem+'.'+suffix)
        if not p.exists():
            assert stem=='inputs' and suffix=='xml'
            continue
        raw=p.read_bytes();compressed=suffix in ['log','xml']
        stored=gzip.compress(raw,mtime=0) if compressed else raw
        name=p.name+('.gz' if compressed else '')
        (out/name).write_bytes(stored)
        artifacts[name]={'sha256':sha(stored),'raw_sha256':sha(raw)}
for name in ['targets.py','prepare_inputs.py','sources.json','api-visibility.json','input-check.json','icelandic-peer-correction.json']:
    shutil.copyfile(scratch/name,out/name)
visibility=json.loads((scratch/'api-visibility.json').read_text())
source_manifest=json.loads((scratch/'sources.json').read_text())
receipt={
 'status':'READY: private normal review inputs and exact extractor dependency; no catalogs/publication',
 'gui_source_commit':'6d1842c2d',
 'frozen_backend':source_manifest['backend'],
 'source_sha256':{**source_manifest['source_files_sha256'],tool_path:sha(new),'spacr/qt/screens/settings_model.py':sha((root/'spacr/qt/screens/settings_model.py').read_bytes()),test_path:sha((root/test_path).read_bytes())},
 'source_count':7,'review_record_count':63,'languages':9,
 'tables':{'setting_labels':2,'setting_tooltips':2,'ui':3},
 'normal_review_gates':{'result':'All nine normal reviewed_runtime_translations loaders PASS seven exact records each','scope':'Isolated proposed inputs with exact source dictionaries obtained from frozen AST literal values and normal _humanize/_strip_type_prefix helpers. Source hashes/schema/context/candidate/syntax/semantic/script gates unchanged. No full canonical inventory or writer.','wall_seconds':2.32,'peak_rss_kib':226120,'cap':'2G','threads':1,'exit':0,'session':72601,'terminal_polled':True},
 'normal_api_visibility':{'changed_private_bodies':6,'eligible_changed_blocks':visibility['eligible_changed_blocks'],'unchanged_enabled_nested_entries':33,'proof':'Actual normal public-symbol/name policy and nested inventory/entries over the three frozen backend files; exact nested docstrings/signatures unchanged. No API targets invented.'},
 'extractor_change':{'new_top_level_owners':sorted(added),'all_attribution_owners':matched[0],'proof':'Removing only three new set constants recovers complete prior builder AST; all prior owning test function ASTs remain exact.','baseline':'3 failed, 3 passed, 39 deselected in1.13s; exact new literals absent, exclusions pass','final':'45 passed in1.20s','wall_seconds':2.80,'peak_rss_kib':462312,'cap':'2G','threads':1,'exit':0,'session':95383,'terminal_polled':True},
 'independent_semantic_review':{'reviewer':'Root coordinator','result':'All63 accepted after one explicit Icelandic tooltip sentence correction','correction_receipt':'icelandic-peer-correction.json','boundary':'Technical bilingual semantics only; no native-user or biological acceptance claim'},
 'review_files':input_hashes,
 'record_bindings':record_bindings,
 'artifacts':artifacts,
 'remaining':['Coordinator normal full source collection/catalog generation/audit and foreign-record preservation','Normal settings-flow output closure','Backend owner coherent source integration/publication','Scientific/biological/native and whole-feature limits remain open'],
 'not_run':['Unchanged backend21/legacy93 or GUI7 acceptance','Full canonical source collector or catalog writer','Model decoder/download/training/inference']
}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'seal.py').write_text(Path(__file__).read_text())
print(json.dumps({'inputs':len(input_hashes),'records':len(record_bindings),'new_owners':sorted(added),'eligible_api_blocks':0}))
