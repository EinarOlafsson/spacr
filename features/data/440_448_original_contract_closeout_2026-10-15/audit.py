from pathlib import Path
import ast,hashlib,json,subprocess,xml.etree.ElementTree as ET
root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/n440-n448-original-closeout-20261015');sha=lambda b:hashlib.sha256(b).hexdigest();evidence=root/'features/data/448_440_421_runtime_closure_2026-10-15';r=json.loads((evidence/'receipt.json').read_text());old='8fc2aa5601847ccd8f7cff287c36c6a9ee00ba53'
paths=['spacr/model_zoo.py','spacr/qt/widgets/model_zoo_picker.py','spacr/qt/screens/make_masks.py'];sourcepins={}
for p in paths:
 actual=sha((root/p).read_bytes());assert actual==r['source_sha256'][p];sourcepins[p]=actual
for lang,v in r['locales'].items():assert sha((root/f'spacr/qt/i18n_catalogs/{lang}.py').read_bytes())==v['catalog_sha256']
assert r['normal_nine_runtime_audit_exit']==0 and r['owned_sources']==23 and r['retirement_count']==9 and r['normal_arrivals']['baseline']==9833
approved=json.loads((evidence/'approved-inputs.json').read_text())
for p,v in approved['owned_files'].items():assert sha((root/p).read_bytes())==v['sha256'],p
picker=json.loads((evidence/'n440-compiled-consumer.json').read_text());assert picker['result']=='PASS' and picker['source_file_sha256']==sourcepins[paths[1]] and len(picker['languages'])==9
for row in picker['languages']:assert row['six_compiled_outputs_exact'] and row['english_roundtrip_preserved'] and row['default_two']==['cellposeSAM','spaCR']
make=ast.parse((root/paths[2]).read_text());method=next(n for n in ast.walk(make) if isinstance(n,ast.FunctionDef) and n.name=='_advance_after_verdict');status={n.args[0].value for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tr' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)};assert len(status)==8
trtexts={n.args[0].value for n in ast.walk(make) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tr' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)}
buttons={'Keep','Discard','Mark this field as one to discard and move to the next. Nothing is deleted: the verdict goes to csv/keep_discard.csv beside the images, and the field, its mask and its objects stay as they are.','Mark this field as one to keep and move to the next. The verdict is written to csv/keep_discard.csv beside the images, with the image, its mask and the number of objects the mask holds right now.'};assert buttons<=trtexts
pickertree=ast.parse((root/paths[1]).read_text());warning=next(n for n in pickertree.body if isinstance(n,ast.FunctionDef) and n.name=='confirm_community_uploads');call=next(n for n in ast.walk(warning) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='warning');warnings={ast.literal_eval(call.args[i]) for i in [1,2]}
warningprior=json.loads((root/'features/data/440_current_locale_gap_2026-10-15/warning-preservation-proof.json').read_text());proof={}
for lang in r['locales']:
 asttree=ast.parse((root/f'spacr/qt/i18n_catalogs/{lang}.py').read_text());tables={n.targets[0].id:ast.literal_eval(n.value) for n in asttree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['UI','SOURCE_HASHES']}
 for source in status|buttons|warnings:assert source in tables['UI'] and tables['SOURCE_HASHES'][('UI',source)]==sha(source.encode()),(lang,source)
 for record in warningprior['final_exact_scope'][lang]['review_files']:assert sha((root/record['path']).read_bytes())==record['sha256']
 proof[lang]={'N448status8_current_compiled_hashes':True,'N448buttons_tooltips4_current_compiled_hashes':True,'N440warnings2_current_compiled_hashes_and_prior_normalreview_inputs':True,'wholecatalog_exact_normal_audit_receipt':r['locales'][lang]['catalog_sha256']}
functions={'spacr/model_zoo.py':['source_of','group_by_source','entries_from_sources'],'spacr/qt/mask_engine.py':['curation_folder','curation_csv_path','read_curation','curation_verdict','record_curation'],'spacr/qt/screens/make_masks.py':['_on_curate','_curation_paths','_show_curation_verdict','_refresh_curation_buttons','_objects_now']};hosted={}
for path,names in functions.items():
 a=ast.parse((root/path).read_bytes());b=ast.parse(subprocess.check_output(['git','show',old+':'+path]));checks={}
 for name in names:
  aa=next(n for n in ast.walk(a) if isinstance(n,ast.FunctionDef) and n.name==name);bb=next(n for n in ast.walk(b) if isinstance(n,ast.FunctionDef) and n.name==name);checks[name]={'current_ast_sha256':sha(ast.dump(aa).encode()),'hosted_ast_sha256':sha(ast.dump(bb).encode()),'identical':ast.dump(aa)==ast.dump(bb)}
 hosted[path]=checks
files=['tests/qt/test_the_model_zoo_picker_groups_by_source.py','tests/test_a_field_is_kept_or_discarded_and_the_csv_says_so.py','tests/qt/test_keep_and_discard_mark_the_field.py'];testpins={}
for p in files:
 before=subprocess.check_output(['git','show',old+':'+p]);current=(root/p).read_bytes();assert before==current;pins={'sha256':sha(current),'exact_hosted_source':True};testpins[p]=pins
log=Path('/mnt/wd4tb/scratch/ci-38699600639-current-triage-20261015/failed.log');passlines=[line for line in log.read_text().splitlines() if 'PASSED tests/' in line and any(p+'::' in line for p in files)];unique={p:sorted({line.split(p+'::')[1].split()[0] for line in passlines if p+'::' in line}) for p in files}
xmldata={}
for name in ['consumers.xml','n448-final.xml']:
 p=evidence/name;t=ET.parse(p);cases=[c for c in t.iter('testcase') if 'test_n448_' in c.attrib.get('classname','')];passed=[c.attrib['name'] for c in cases if c.find('failure') is None and c.find('error') is None and c.find('skipped') is None];fail=[c.attrib['name'] for c in cases if c.find('failure') is not None];xmldata[name]={'sha256':sha(p.read_bytes()),'N448_passed':passed,'N448_failed':fail}
assert len(xmldata['consumers.xml']['N448_passed'])==16 and len(xmldata['n448-final.xml']['N448_passed'])==9 and not xmldata['n448-final.xml']['N448_failed'];assert (evidence/'n448-final.exit').read_text().strip()=='0' and (evidence/'n440-compiled-consumer.exit').read_text().strip()=='0'
receipt={'public_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'normal_source_and_ninecatalog_pins_current':sourcepins,'normal_runtime_receipt_sha256':sha((evidence/'receipt.json').read_bytes()),'normal_nine_audit':0,'normal_arrival_baseline':9833,'owned_translation_files_pins_verified':len(approved['owned_files']),'N440_actual_compiled_consumer54':{'receipt_sha256':sha((evidence/'n440-compiled-consumer.json').read_bytes()),'all9_exact':True,'current_source_hash':picker['source_file_sha256']},'existing_source_status_warning_button_scope':proof,'hosted_AST_comparison':hosted,'hosted_owning_testfiles':testpins,'hosted_raw_PASS_lines':passlines,'hosted_unique_owning_nodes':unique,'existing_N448_actual_consumer_XML':xmldata,'scope':'Stdlib-only read-only source/normalcatalog/review pin and existing exact producer/consumer evidence reconciliation; no tests/generator/collector/model/network/source/status/index mutations. Allsourcebranches/semantics remain unchanged.'}
(scratch/'source-evidence-proof.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'N440compiledconsumer54':'PASS currentexactsource','N448actualconsumers':'16+9PASS retained first9testpathfailures','hosted_unique':{p:len(v) for p,v in unique.items()},'changed_owning_AST':{p:[n for n,v in d.items() if not v['identical']] for p,d in hosted.items()}}),flush=True)
