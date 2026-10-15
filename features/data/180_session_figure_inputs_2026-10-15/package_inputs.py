import ast,hashlib,json,runpy,shutil,subprocess,sys
from pathlib import Path
root=Path.cwd();s=Path('/mnt/wd4tb/scratch/home-b-n180-figure-carry-20261015')
sys.path.insert(0,str(root/'tools'))
import build_documentation_i18n as b
import build_i18n_catalogs as runtime
from build_guide_i18n import inline_markup_problems
m=json.loads((s/'sources.json').read_text());targets=runpy.run_path(str(s/'targets.py'))['TARGETS'];problems=[]
sha=lambda x:hashlib.sha256(x).hexdigest()
for name,want in m['source_files'].items():assert sha((root/name).read_bytes())==want,name
for lang,values in targets.items():
 assert len(values)==8
 for row,value in zip(m['changed_blocks'],values[:7]):
  for field in ('source','context'):
   if not b._reviewed_api_block_valid(row[field],value,lang):problems.append([lang,row['label'],field,'normal candidate refusal'])
  for problem in inline_markup_problems(value):problems.append([lang,row['label'],'inline',problem])
(s/'candidate-check.json').write_text(json.dumps({'checked':63,'source_context_checks':126,'inline_checks':63,'problems':problems},ensure_ascii=False,indent=2)+'\n')
assert not problems,problems
caption=m['runtime_arrivals'][0]['source']
node=ast.parse((root/'spacr/qt/widgets/figure_queue.py').read_text())
matching=[]
for call in ast.walk(node):
 if isinstance(call,ast.Call) and runtime._call_name(call)=='tr':
  for arg in runtime._candidate_arguments(call,'tr'):
   for value in runtime._literal_strings(arg,{}):
    if value==caption and runtime._looks_translatable(value):matching.append(value)
assert matching==[caption]
input_root=s/'isolated-reviewed-inputs'
source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
new_paths={};pending={}
for lang,values in targets.items():
 records=[dict(row,translation=value,review_basis='Exact-source technical translation of session figure inventory, queue-owned asset retention, reference mode, immutable originals/bundles, missing preview reporting and staged restoration; no native-speaker claim.') for row,value in zip(m['changed_blocks'],values[:7])]
 api={'schema':1,'language':lang,'review_kind':'Technical translation; coordinator independent semantic review; no native-speaker signoff','source_commit':source_commit,'source_file_sha256':m['source_files'],'records':records}
 r=m['runtime_arrivals'][0]
 run={'schema':1,'language':lang,'review_kind':'Technical translation; coordinator independent semantic review; no native-speaker signoff','source_commit':source_commit,'source_file_sha256':m['source_files'],'records':[dict(r,translation=values[7])]}
 for kind,payload in [('api',api),('runtime',run)]:
  relative=Path('docs/i18n/reviewed')/kind/lang/'2026-10-15-n180-session-figures.json'
  p=input_root/kind/lang/relative.name;p.parent.mkdir(parents=True,exist_ok=True);raw=(json.dumps(payload,ensure_ascii=False,indent=2)+'\n').encode();p.write_bytes(raw);pending[relative]=raw
b.REVIEWED_API_DIR=input_root/'api'
runtime.REVIEWED_RUNTIME_DIR=input_root/'runtime'
runtime.canonical_sources=lambda:{'ui':(caption,)}
runtime.reviewed_runtime_translations.cache_clear()
for lang,values in targets.items():
 got=b.reviewed_api_block_translations(m['source_docs'],lang)
 for row,value in zip(m['changed_blocks'],values[:7]):assert got[row['source']]==value
 assert runtime.reviewed_runtime_translations(lang)[caption]==values[7]
for path,raw in pending.items():
 assert not (root/path).exists(),path
 (root/path).write_bytes(raw);new_paths[str(path)]=sha(raw)
o=root/'features/data/180_session_figure_inputs_2026-10-15';o.mkdir(exist_ok=True)
for n in ['package_inputs.py','targets.py','sources.json','candidate-check.json']:(o/n).write_bytes((s/n).read_bytes())
receipt={'schema':1,'source_commit':source_commit,'source_files':m['source_files'],'new_api_symbols':m['new_symbols'],'api_blocks':7,'api_records':63,'runtime_sources':1,'runtime_records':9,'normal_gates':{'source_context_candidates':126,'inline_markup':63,'normal_scoped_api_review_loader_locales':9,'normal_runtime_review_loader_locales':9},'runtime_gate_scope':'Normal loader over isolated new review files and one explicit AST-extracted current tr literal, using normal _candidate_arguments/_literal_strings/_looks_translatable. No full canonical capture or global source-currency claim. Coordinator normal runtime producer still required.','prior_reviews':'No retirement; two newly eligible methods and one new caption. All pre-existing review files untouched.','input_files':new_paths,'catalog_changes':'None. No models, collector, writer, Help or release operation.','peer_status':'Coordinator accepted first48 exact draft pairs; final24 independent read pending at packaging time.'}
(o/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print('PASS63 API+9runtime records;126source/context+63inline gates;9normal API+9 isolated runtime loaders; no generated outputs')
