import ast,hashlib,importlib.util,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]).resolve() if len(sys.argv)>1 else pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as builder
receipt_path=pathlib.Path(sys.argv[2]) if len(sys.argv)>2 else None
def data(text):
 tree=ast.parse(text)
 return {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'UI','SOURCE_HASHES'}}
before=data(subprocess.check_output(['git','show','868adff0f^:spacr/qt/i18n_catalogs/en.py'],text=True,cwd=root))
after=data(subprocess.check_output(['git','show','868adff0f:spacr/qt/i18n_catalogs/en.py'],text=True,cwd=root))
current=data((root/'spacr/qt/i18n_catalogs/en.py').read_text())
added=set(after['SOURCE_HASHES'])-set(before['SOURCE_HASHES'])
retired=sorted(added-set(current['SOURCE_HASHES']))
assert retired==[('UI','Add {side} store folder'),('UI','Add {side} tables')]
replacements={('UI',s) for s in ['Add score tables','Add count tables','Add score store folder','Add count store folder']}
selected=sorted((added&set(current['SOURCE_HASHES']))|replacements)
assert len(selected)==42
source_files=['spacr/qt/screens/settings_model.py','spacr/qt/screens/app_screen.py','spacr/qt/widgets/file_list.py','spacr/qt/widgets/measurement_scan_panel.py']
strings=set()
for f in source_files:
 strings.update(n.value for n in ast.walk(ast.parse((root/f).read_text())) if isinstance(n,ast.Constant) and isinstance(n.value,str))
assert all(table=='UI' and s in strings for table,s in selected)
rows=[]
for language in builder.MODEL_SPECS:
 catalog=data((root/f'spacr/qt/i18n_catalogs/{language}.py').read_text())
 path=root/f'docs/i18n/reviewed/runtime/{language}/2026-10-10-n615-captions-store-figure-curation.json'
 payload=json.loads(path.read_text())
 records={r['source']:r for r in payload['records']}
 builder._REVIEWED_RUNTIME_LOADING.add(language)
 try:
  for table,source in selected:
   record=records[source]
   target=catalog[table][source]
   digest=hashlib.sha256(source.encode()).hexdigest()
   assert record['table']=='ui' and record['key']==source
   assert record['source_sha256']==current['SOURCE_HASHES'][table,source]==catalog['SOURCE_HASHES'][table,source]==digest
   assert target==record['translation'] and target!=source
   assert builder._contextualize(target,language,source)==target,(language,source,'contextualize')
   rejected=builder._translation_rejection_reasons(source,target,language,force=builder._looks_translatable(source))
   assert not rejected,(language,source,rejected)
   rows.append({'language':language,'source':source,'source_sha256':digest,'translation_sha256':hashlib.sha256(target.encode()).hexdigest(),'review_file':str(path.relative_to(root))})
 finally:
  builder._REVIEWED_RUNTIME_LOADING.discard(language)
 print(language,42,'current reviewed captions pass normal candidate gates',flush=True)
receipt={'item':'F576','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=root).strip(),'original_publication':'868adff0fda82f6358f2f0e3e55120e2fcda6f5a','translation_completion_commit':'6fc7353a16a','caption_count':42,'locale_count':9,'entries_passed':len(rows),'retired_generic_templates':[s for _,s in retired],'current_explicit_replacements':[s for _,s in sorted(replacements)],'verification_scope':'Current source AST literals; exact English, review and catalog SHA bindings; installed translations equal original reviewed records; normal contextualization and structural, script, protected-syntax and semantic candidate gates, with the review loader recursion guard exactly as used by tools/check_reviewed_runtime_evidence.py. No candidate gate changes or regeneration.','source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in source_files},'attribution':'Existing review attribution retained. Mechanical technical readback by Codex; no new translations or human/native-speaker approval claimed.','entries':rows}
if receipt_path is not None:
 receipt_path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
print('PASS:',len(rows),'current reviewed locale entries; no missing captions')
