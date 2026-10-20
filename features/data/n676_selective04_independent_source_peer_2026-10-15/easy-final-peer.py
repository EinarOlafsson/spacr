import ast,hashlib,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
r=Path('/mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015'); out=Path(__file__).parent; packet='features/data/n676_selective04_source_adoption_2026-10-15/'
def git(*a): return subprocess.check_output(['git','-C',str(r),*a])
def dump(x): return ast.dump(x,include_attributes=False)
head=git('rev-parse','HEAD').decode().strip(); assert head.startswith('f53d6dce5d'); assert not git('status','--porcelain')
pins=json.loads(git('show',head+':'+packet+'final-source-pins.json')); paths=git('ls-tree','-r','--name-only',head,'features/data/n676_platform_installers_selective_completed_2026-10-15').decode().splitlines(); verified=[]
for row in pins['files']:
 data=git('show',head+':'+row['file']); assert hashlib.sha256(data).hexdigest()==row['sha256']; assert (r/row['file']).read_bytes()==data
 if row.get('owner_snapshot_exact'):
  matches=[p for p in paths if p.endswith('/source/'+row['file']) and git('show',head+':'+p)==data]; assert matches,row['file']; row=dict(row,independently_matching_owner_snapshots=matches)
 verified.append(row)
f='tools/tutorials/native_live_timing.py'; old=ast.parse(git('show',pins['base']+':'+f)); new=ast.parse(git('show',head+':'+f)); helper=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name=='complete_native_duration'); new.body=[x for x in new.body if x is not helper and not(isinstance(x,ast.ImportFrom) and x.module=='fractions')]; assert dump(new)==dump(old)
p=git('show',head+':features/data/n676_complete_native_clips_fix_2026-10-14/e257c1f3c-complete-native-clips.patch').decode(); added='\n'.join(l[1:] for l in p.splitlines() if l.startswith('+') and not l.startswith('+++')); start=added.index('def complete_native_duration('); end=added.index('    return duration',start)+len('    return duration'); assert dump(ast.parse(added[start:end]).body[0])==dump(helper)
t='tools/tutorials/tests/test_native_live_timing.py'; oldt=ast.parse(git('show',pins['base']+':'+t)); newt=ast.parse(git('show',head+':'+t)); assert all(dump(a)==dump(b) for a,b in zip(oldt.body,newt.body)); assert len(newt.body)==len(oldt.body)+3
logs={}
for prefix in ['owning','affected']:
 xml=git('show',head+':'+packet+prefix+'-tests.xml'); cases=list(ET.fromstring(xml).iter('testcase')); fail=[x for x in cases if x.find('failure') is not None]; skip=[x for x in cases if x.find('skipped') is not None]; logs[prefix]={'cases':len(cases),'failed':len(fail),'skipped':len(skip),'xml_sha256':hashlib.sha256(xml).hexdigest(),'exit':git('show',head+':'+packet+prefix+'-tests.exit').decode().strip(),'log_sha256':hashlib.sha256(git('show',head+':'+packet+prefix+'-tests.log')).hexdigest()}
 if prefix=='owning': assert len(cases)==60 and len(fail)==25 and all('complete_native_duration' in x.find('failure').get('message','') for x in fail)
 else: assert len(cases)==37 and not fail and not skip
census=json.loads((out/'easy-local-dependency-census.json').read_text())
for row in census['changed_file_local_imports']+census['direct_existing_callers']:
 if row['symbol']=='complete_native_duration': assert any(isinstance(n,ast.FunctionDef) and n.name==row['symbol'] for n in ast.parse((r/row['target']).read_bytes()).body)
 else: assert row['exists_as_source_binding']
result={'verdict':'ACCEPT: bounded source adoption and staged finite evidence; no fresh browser/native tutorial acceptance','review_kind':'independent AI source and evidence technical review','author_final_commit':head,'base':pins['base'],'verified_source_files':verified,'proofs':{'six_owner_snapshot_bytes_independently_equal':True,'only_Fraction_and_exact_author_helper_added_to_timing_AST':True,'every_prior_timing_test_AST_preserved':True,'three_new_test_functions_expand_to_nine_duration_cases':True,'prior_42_literal_local_bindings_now_all_present':True,'no_other_literal_missing_dependencies_found':True},'actual_logs':logs,'census_prior_missing_only':census['missing'],'capture_source':'b929b7dc53465132322c3ca0b866f567de773b9f','player_source':'2427d57f17ad78d97cf81f18720c1cd9e0c7cc02','reviewer_proof_script_initial_attempt':'Diff-added-line extraction included a later indented patch fragment and refused AST parsing; corrected extraction stops at the exact helper return. No author code/test failure or runtime execution.','limits':['No tests, application imports, browser, native capture, encoding, rendering, model, network, upload or installation executed by reviewer.','Initial 35 passes/25 dependency failures and final affected 37 passes are separate runs; 32 prior unaffected passes were not replayed.','Historical 226 owner tests are not fresh Root execution; no whole tutorial or future publication certification.','Census covers literal local from-imports and direct Python callers, not dynamic imports or JS runtime resolution.']}
(out/'easy-final-independent-source-peer.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'verdict':result['verdict'],'head':head,'files':len(verified),'logs':logs}))
