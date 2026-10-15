from pathlib import Path
import ast,hashlib,json,re,subprocess,sys,tempfile
r=Path.cwd();s=Path('/mnt/wd4tb/scratch/n690-current-contract-audit-20261015');old='8fc2aa5601847ccd8f7cff287c36c6a9ee00ba53';sha=lambda b:hashlib.sha256(b).hexdigest()
def nodes(raw):
 out={}
 def walk(body,p=''):
  for n in body:
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):out[p+n.name]=n;walk(n.body,p+n.name+'.')
 walk(ast.parse(raw).body);return out
scope={'spacr/io.py':['MissingTEMReader','read_dm','read_mrc','read_tem_image','tiff_pixel_size_nm','image_pixel_size_nm'],'spacr/qt/mask_engine.py':['pixel_size_nm','read_image','save_mask'],'spacr/qt/make_masks_demo.py':['tem_example_folder','download_tem_example','load_the_tem_test_data'],'spacr/qt/screens/make_masks.py':['MaskScreen._on_contrast_preset','MaskScreen._show_test_data_menu']};sourceproof={}
for p,names in scope.items():
 raw=(r/p).read_bytes();prior=subprocess.check_output(['git','show',old+':'+p]);a,b=nodes(raw),nodes(prior);sourceproof[p]={'source_sha256':sha(raw),'owning':{}}
 for name in names:
  if name not in a:continue
  x,y=ast.dump(a[name]),ast.dump(b[name]);assert x==y,(p,name);sourceproof[p]['owning'][name]={'AST_sha256':sha(x.encode()),'identical_hosted':True}
files=['tests/test_690_tem_readers_and_pixel_size.py','tests/qt/test_690_make_masks_tem.py'];tests={};passes={p:{} for p in files};fail=[]
with Path('/mnt/wd4tb/scratch/ci-38699600639-current-triage-20261015/failed.log').open(errors='replace') as f:
 for line in f:
  for p in files:
   if p not in line:continue
   m=re.search(r'(PASSED|FAILED) ('+re.escape(p)+r'::\S+)',line)
   if m:
    if m[1]=='PASSED':passes[p][m[2]]=passes[p].get(m[2],0)+1
    else:fail.append(m[2])
for p in files:
 raw=(r/p).read_bytes();assert raw==subprocess.check_output(['git','show',old+':'+p]);tests[p]={'source_sha256':sha(raw),'byte_identical_hosted':True,'assertions':len([n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.Assert)]),'hosted_unique_PASS':len(passes[p]),'PASS_nodes':passes[p]}
assert not fail
sys.path.insert(0,str(r/'tools'));import build_i18n_catalogs as normal;import build_documentation_i18n as api
normal.canonical_sources=lambda:(_ for _ in ()).throw(RuntimeError('No full runtime collector'))
langs=['sv','de','es','zh_CN','fr','pt','hi','ko','is'];first=json.loads((r/'docs/i18n/reviewed/runtime/sv/2026-10-11-n690-make-masks-tem.json').read_text());sources={x['source'] for x in first['records']};assert len(sources)==11;runtime={}
with tempfile.TemporaryDirectory(dir=s,prefix='eleven-exact-runtime-inputs-') as tmp:
 normal.REVIEWED_RUNTIME_DIR=Path(tmp);normal.canonical_sources=lambda:{'setting_labels':{},'setting_tooltips':{},'ui':sources}
 for lang in langs:
  p=r/f'docs/i18n/reviewed/runtime/{lang}/2026-10-11-n690-make-masks-tem.json';q=Path(tmp)/lang;q.mkdir();(q/'owned.json').write_bytes(p.read_bytes());normal.reviewed_runtime_translations.cache_clear();reviewed=normal.reviewed_runtime_translations(lang)
  tr=ast.parse((r/f'spacr/qt/i18n_catalogs/{lang}.py').read_text());t={n.targets[0].id:ast.literal_eval(n.value) for n in tr.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'UI','SOURCE_HASHES'}};checks=[]
  for source in sources:
   value=t['UI'][source];assert t['SOURCE_HASHES'][('UI',source)]==sha(source.encode());assert reviewed[source]==value;assert normal._translation_candidate_valid(source,value,lang,force=normal._looks_translatable(source));checks.append({'source':source,'target':value,'target_sha256':sha(value.encode())})
  runtime[lang]={'normal_scoped_review_compiled_candidate_PASS':True,'review_file_sha256':sha(p.read_bytes()),'checks':checks}
normal.canonical_sources=lambda:(_ for _ in ()).throw(RuntimeError('No full runtime collector'))
arrival=json.loads((r/'tests/data/release_contracts/690_tem_api_arrivals_2026-10-11.json').read_text());docs={}
for key in arrival:
 module,name=key.rsplit('.',1);p=r/(module.replace('.','/')+'.py');node=nodes(p.read_bytes())[name];docs[key]=api._autoapi_class_doc(node) if isinstance(node,ast.ClassDef) else api._clean_doc(node)
english=json.loads((r/'docs/source/_static/i18n/api/en.json').read_text())['symbols'];issues=[];apichecks={}
for key,doc in docs.items():assert english[key]['text']==doc,key
for lang in langs:
 compiled=json.loads((r/f'docs/source/_static/i18n/api/{lang}.json').read_text())['symbols'];review=api.reviewed_api_block_translations(docs,lang);apichecks[lang]={}
 for key,doc in docs.items():
  row=compiled[key];pins=row['source_sha256']==api._source_hash(doc) and row['source_blocks_sha256']==api._source_block_hashes(doc) and row['translation_source_blocks_sha256']==api._translation_source_block_hashes(doc);assert pins,(lang,key)
  sb,_=api.translatable_blocks(doc);tb,_=api.translatable_blocks(row['text']);assert len(sb)==len(tb);checks=[]
  for i,(a,b) in enumerate(zip(sb,tb)):
   reviewed=review.get(a)==b;good=reviewed or (api._api_block_valid(a,b,lang) and api._api_block_valid(api._api_translation_source(a),b,lang));checks.append({'index':i,'source_sha256':sha(a.encode()),'target_sha256':sha(b.encode()),'exact_review':reviewed,'normal_selected_block_gate_PASS':good})
   if not good:issues.append({'language':lang,'symbol':key,'index':i,'source':a,'target':b})
  apichecks[lang][key]={'source_context_block_pins_current':pins,'blocks':checks}
handoff=r/'features/HANDOFF.md';text=handoff.read_text();clause='- the lessons for N690 TEM and N673/N692 FIB-SEM are DROPPED: the maintainer never asked for them;';assert clause in text
proof={'scope':'N690 exact originalcontract/current source/owning existingchecks +11runtime and10APIarrivals, read-only metadata; no tests/models/network/collectors/generators/source edits.','owning_source':sourceproof,'test_evidence':tests,'hosted_FAIL_nodes':fail,'runtime11_nine_locale_normal_scoped_review_compiled_checks':runtime,'API10_nine_locale_source_block_and_normal_selected_gate_checks':apichecks,'API_actual_problems':issues,'tutorial_authority':{'handoff_sha256':sha(handoff.read_bytes()),'actual_clause':clause,'dated_heading':'2026-10-12 18:40 EDT — WS → Home','published_provenance_commit':'d44a9c6a65d','later_inventory':'2026-10-13 18:54 EDT — full existing84lessonremake; no newTEM/FIBSEMroute included','disposition':'Explicit cancellation supersedes Oct11item tutorialpending; no tutorial publication claimed orresurrected.'},'data_and_F1_existing_evidence':{'original_item':'features/new/690_make_masks_for_tem_images.txt','builder':'tools/build_tem_example_dataset.py','actual_registry':'spacr.example_archives EXAMPLE_SETS tem; repo einarolafsson/spacr-example-tem DATASET;31,508,480bytes markersmanifest/images/groundtruth, sameofflinecachemenu/CLI','accepted_real_CPU_raw_proof':'features/data/691_current_whole_item_reconciliation_2026-10-15/source-history.json','original_required_report':'TenexactselectedTEMfields TP142 FP8 FN53 F1.823 Dice.842; originalcpsamF1.467 comparison retained initem. No metrics transferredtootherdataset/checkpoint or rerun.','existing_raw_inference_source_binding':'features/data/691_current_whole_item_reconciliation_2026-10-15/original-science-source-binding.json','limits':'Historical actual CPUdata/model evidence retained, no new livecache/dataset/weights scan/inference/native acceptance.'}}
(s/'proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'current_owning_AST':True,'hosted_PASS':{p:v['hosted_unique_PASS'] for p,v in tests.items()},'runtime_checks':99,'API_symbols':len(docs),'API_problems':issues,'tutorial':'explicitlyDROPPED bymaintainer viaWS'}),flush=True)
