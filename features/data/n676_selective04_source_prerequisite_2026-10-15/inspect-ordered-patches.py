import subprocess,re,json,hashlib,ast,pathlib
rev='083d21924284249cf1ad08b5bb1ff57ca9bbc1c2'
base='features/data/n676_platform_installers_selective_completed_2026-10-15/work-evidence/'
get=lambda path:subprocess.check_output(['git','show',rev+':'+path])
patches=[('source-handoff','0001-selective04-mixed-visual-retention.patch'),('source-handoff-playback-followup','0001-selective04-static-playback-clock-followup.patch'),('source-handoff-pending-seek-followup','0001-selective04-pending-visual-seek-followup.patch')]
tracked=set(subprocess.check_output(['git','ls-tree','-r','--name-only',rev],text=True).splitlines());state={};initial={};receipts=[]
for folder,filename in patches:
 raw=get(base+folder+'/'+filename);text=raw.decode();changes=[]
 for match in re.finditer(r'^diff --git a/(.*?) b/[^\n]+\n(.*?)(?=^diff --git|\Z)',text,re.M|re.S):
  name,body=match.groups();index=re.search(r'^index ([0-9a-f]+)\.\.([0-9a-f]+)',body,re.M);assert index
  if name not in state:state[name]=get(name) if name in tracked else b'';initial[name]=state[name]
  if name.endswith('verify_staged_lesson.py') and folder == 'source-handoff': state[name]=pathlib.Path('/mnt/wd4tb/scratch/work04-source-adoption-readonly-20261015/reconstructed-required-verifier.py').read_bytes()
  current=state[name].decode().splitlines(keepends=True);hunks=[]
  for h in re.finditer(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n(.*?)(?=^@@ |\Z)',body,re.M|re.S):
   lines=h.group(5).splitlines(keepends=True);valid=[]
   for line in lines:
    if line.startswith((' ','+','-')):
     if sum(x[0] in ' -' for x in valid)==int(h.group(2) if h.group(2) is not None else 1) and sum(x[0] in ' +' for x in valid)==int(h.group(4) if h.group(4) is not None else 1):break
     valid.append(line)
    elif line.startswith('\\'):continue
    elif line.strip():break
   before=[line[1:]for line in valid if line[0]in' -'];after=[line[1:]for line in valid if line[0]in' +']
   assert len(before)==(int(h.group(2)) if h.group(2) is not None else 1),(name,'old hunkcount')
   assert len(after)==(int(h.group(4)) if h.group(4) is not None else 1),(name,'new hunkcount')
   positions=[i for i in range(len(current)-len(before)+1)if current[i:i+len(before)]==before]if before else [0]
   assert len(positions)==1,(name,'nonunique/notapplicable',positions)
   pos=positions[0];current[pos:pos+len(before)]=after;hunks.append({'original_line':int(h.group(1)),'actual_line':pos+1,'old_lines':len(before),'new_lines':len(after)})
  state[name]=''.join(current).encode();changes.append({'file':name,'patch_old_blob':index.group(1),'patch_new_blob':index.group(2),'hunks':hunks,'result_sha256':hashlib.sha256(state[name]).hexdigest()})
 receipts.append({'folder':folder,'patch':filename,'sha256':hashlib.sha256(raw).hexdigest(),'files':changes})
results=[]
for name,data in state.items():
 ownerfolder=patches[-1][0] if name in ['tools/tutorials/authoring/web/app_v2.js','tools/tutorials/tests/test_player_preserve_visual_intervals.py']else patches[0][0]
 owner=get(base+ownerfolder+'/source/'+name);record={'file':name,'current_root_sha256':hashlib.sha256(initial[name]).hexdigest()if initial[name]else None,'planned_sha256':hashlib.sha256(data).hexdigest(),'owner_snapshot_sha256':hashlib.sha256(owner).hexdigest(),'matches_final_owner_snapshot':data==owner,'source_adoption_needed':initial[name]!=data}
 if name.endswith('verify_staged_lesson.py'):
  a=ast.parse(initial[name]);b=ast.parse(data);o=ast.parse(owner)
  names={'check_native_boundary_playback','check_complete_native_trace','check_complete_native_boundary_playback'}
  original_functions={n.name:ast.dump(n) for n in a.body if isinstance(n,ast.FunctionDef)}
  planned_functions={n.name:ast.dump(n) for n in b.body if isinstance(n,ast.FunctionDef)}
  assert all(planned_functions[k]==v for k,v in original_functions.items() if k not in names)
  ownerfuncs={n.name:ast.dump(n)for n in o.body if isinstance(n,ast.FunctionDef)}
  assert all(planned_functions[k]==ownerfuncs[k] for k in names)
  assert data==owner
  record['exact_changed_or_added_methods']=sorted(names);record['other_Root_function_ASTs_identical']=True;record['three_method_ASTs_equal_owner_snapshot']=True
  record['additional_module_delta']='Exact predecessor import sys and _ROOT/sys.path bootstrap; no other module-level AST change.'
 else:assert data==owner,name
 results.append(record)
publication=json.loads(get('features/data/n676_platform_installers_selective_completed_2026-10-15/final-publication-review.json'))
report={'status':'READ ONLY patch applicability/adoption plan; no source application or tests','root_pin':rev,'patches_in_required_order':receipts,'files':results,'source_identity_distinction':{'capture_and_stage':publication['capture_source_commit'],'final_player':publication['source_commit']},'plan':['Root/Work must approve the exact missing verifier predecessor delta before applying the three supplied patches.', 'Prerequisite changes only sys import/root path bootstrap, native interruption/dispatch checks, and two complete-native helpers; every other original Root function AST remains identical.', 'Apply the reviewed predecessor delta then exact K4/K5/K6 patches in order in a private integration branch. No whole snapshot replacement.', 'In-memory exact hunk application after prerequisite yields all six final owner snapshots byte-for-byte, including verifier099373 and JS/player2427.', 'Root assigns finite owning validation and independent final source peer separately; no tests or source application here.', 'Exclude additional87 media selector patch: CI owns the two media fixture selectors. No recording/render/encoder/media/network required.'],'limits':'Published selective04 assets are a separate media acceptance. Retained12visuals/legacyPerformance remain disclosed; no wholefresh/nativecomplete/humanlisten or livePagesdeployment claim. Existing owner tests224/225/226 are separate staged histories, not summed or rerun here.'}
s=pathlib.Path('/mnt/wd4tb/scratch/work04-source-adoption-readonly-20261015');s.mkdir(exist_ok=True);(s/'plan.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(results,indent=2))
