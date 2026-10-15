from pathlib import Path
import ast,hashlib,json,subprocess,types
root=Path('/mnt/wd4tb/spacr-worktrees/codex-f567-parent-identity-20261015');base='1055f93c9a2f69a51c6d00f7825cc3b2564fa144';tested='6b8415a8f2498d178e63b38833e505f42949aa89';incoming='71fb782dc7f09cd088cf5fb10bef75445a16ce41';out=root/'features/data/536_calibration_boundary_2026-10-15'
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0',*args],cwd=root)
def tree(pin):return {e.split(b'\t',1)[1].decode():e.split(b'\t',1)[0].split()[-1].decode() for e in git('ls-tree','-rz',pin).split(b'\0') if e}
own=set(git('diff','--name-only',base,tested).decode().splitlines());public,actual=tree(incoming),tree('HEAD');preserved=[p for p in public if p not in own];assert all(public[p]==actual[p] for p in preserved),'Unrelated incoming blob changed'
normal=json.loads((out/'merged-runtime-receipt.json').read_text());assert normal['normal_full_nine_runtime_audit_exit']==0
hashes={p:h for p,h in normal['source_file_sha256_unchanged'].items() if p not in {'tools/build_documentation_i18n.py','docs/source/_static/i18n/api/en.json'}}
for p,h in hashes.items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
old_tool=ast.parse(git('show','7ca1994091554a6d9f16a2c628be868640307b86:tools/build_documentation_i18n.py'));new_tool=ast.parse((root/'tools/build_documentation_i18n.py').read_text())
def functions(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
old_functions,new_functions=functions(old_tool),functions(new_tool)
assert set(new_functions)-set(old_functions)=={'_readme_picker_links_to_index'}
assert set(old_functions)<=set(new_functions)
assert all(new_functions[n]==body for n,body in old_functions.items() if n!='audit')
for p in ['tools/build_documentation_i18n.py','docs/source/_static/i18n/api/en.json']:hashes[p]=hashlib.sha256((root/p).read_bytes()).hexdigest()
for p in ['docs/source/_static/i18n/api','spacr/qt/i18n.py','tools/build_documentation_i18n.py']:assert git('rev-parse','HEAD:'+p)==git('rev-parse',incoming+':'+p),p
api=(root/'docs/source/_static/i18n/api/en.json').read_text();assert all(n not in api for n in ['measure._wound_calibration','measure._wound_settings_check'])
source='Wound calibration {name} must be a finite number greater than zero, or blank.';table_names=['SETTING_LABELS','SETTING_TOOLTIPS','MODULE_SUMMARIES','CATEGORY_HELP','UI'];locale_preservation={}
for language in ['en','sv','de','es','zh_CN','pt','hi','fr','is','ko']:
 p='spacr/qt/i18n_catalogs/'+language+'.py';old=types.ModuleType('incoming');new=types.ModuleType('actual');exec(compile(git('show',incoming+':'+p),p,'exec'),old.__dict__);exec(compile((root/p).read_bytes(),p,'exec'),new.__dict__);count=0
 for name in (['SETTING_LABELS','SETTING_TOOLTIPS','MODULE_SUMMARIES'] if language=='en' else table_names):
  before=getattr(old,name);after=getattr(new,name)
  assert all(key in after and after[key]==value for key,value in before.items()),(language,name)
  assert set(after)-set(before)==({source} if name=='UI' else set()),(language,name,'unexpected keydelta')
  count+=len(before)
 assert all(new.SOURCE_HASHES[k]==v for k,v in old.SOURCE_HASHES.items()),language
 if language!='en':
  review=root/'docs/i18n/reviewed/runtime'/language/'2026-10-15-wound-calibration.json';record=json.loads(review.read_text())['records'][0];assert new.UI[source]==record['translation'];assert record['source']==source and record['source_sha256']==hashlib.sha256(source.encode()).hexdigest()
 else:
  assert new.CATEGORY_SOURCES==old.CATEGORY_SOURCES
  assert set(new.UI_SOURCES)-set(old.UI_SOURCES)=={source} and set(old.UI_SOURCES)<=set(new.UI_SOURCES)
  count+=len(old.CATEGORY_SOURCES)+len(old.UI_SOURCES)
 locale_preservation[language]=count
proof={'immutable_incoming_public':incoming,'tested_coherent_premerge_closure':tested,'final_normal_merged_runtime_audit_exit':0,'unrelated_incoming_public_blobs_exact':len(preserved),'all_incoming_foreign_runtime_targets_and_source_hashes_exact':locale_preservation,'owned_final_delta':'Exactly one wound calibration UI row per catalog, exact approved reviews; all incoming HintBar additions/retirements preserved.','final_source_file_sha256':hashes,'foreign_API_tool_change':'Current tool adds README HTML picker checker and changes audit only; every other existing function/class AST including normal API collector/visibility is exact old7ca. Historical full API census remains source-bound; incoming generated API preserved, both private wound targets absent.','incoming_API_preserved_owned_private_symbols_absent':True,'checks':'30owning and18actual locale callbacks retained source-bound and not rerun. Genuine incoming N655 HintBar runtime delta required one normal merged scoped9writer/full9audit/arrival with fresh original collector equality and frozen7source/tool/test/API hashes. Prior failed old-catalog-byte proof stopped before staging/push; no gate bypass.','limits':'Metadata input boundary/localization only; no biological calibration, whole F536, globalCI or release acceptance.','validated_tree_ids':{p:git('rev-parse','HEAD:'+p).decode().strip() for p in ['spacr','tools','docs','tests']},'artifact_sha256':hashes}
(out/'final-publication-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'incoming':incoming,'unrelated_preserved':len(preserved),'runtime_preserved':locale_preservation}))
