import ast,collections,csv,hashlib,io,json,pathlib,re,struct,subprocess,tarfile
root=pathlib.Path('/mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015')
out=pathlib.Path('/mnt/wd4tb/scratch/n692-whole-item-closeout-20261015')
current='095894ad308a27a855629922ffe854bb04e3ebd4'; hosted='8fc2aa5601847ccd8f7cff287c36c6a9ee00ba53'
def get(rev,p):return subprocess.check_output(['git','show',f'{rev}:{p}'],cwd=root)
def sha(b):return hashlib.sha256(b).hexdigest()
paths=['spacr/qt/mask_engine.py','spacr/qt/make_masks_demo.py','spacr/qt/screens/make_masks.py','spacr/example_archives.py','tools/build_fibsem_example_dataset.py','tests/qt/test_692_three_d_mask_drawing.py','tests/test_692_fibsem_example_dataset.py','tests/qt/test_673_make_masks_3d_in_container.py']
bindings={p:{'current_sha256':sha(get(current,p)),'hosted_sha256':sha(get(hosted,p)),'whole_file_identical':get(current,p)==get(hosted,p)} for p in paths}
for p in paths:
 if p not in ('spacr/qt/mask_engine.py','spacr/example_archives.py'):assert bindings[p]['whole_file_identical']
def normalized(rev,path,omit):
 t=ast.parse(get(rev,path))
 for n in ast.walk(t):
  if isinstance(n,(ast.ClassDef,ast.Module)):
   n.body=[a for a in n.body if not isinstance(a,(ast.FunctionDef,ast.AsyncFunctionDef)) or a.name not in omit]
 return ast.dump(t,include_attributes=False)
assert normalized(current,'spacr/qt/mask_engine.py',{'assign_objects'})==normalized(hosted,'spacr/qt/mask_engine.py',{'assign_objects'})
assert normalized(current,'spacr/example_archives.py',{'expand_measure_arrays'})==normalized(hosted,'spacr/example_archives.py',{'expand_measure_arrays'})
log=pathlib.Path('/mnt/wd4tb/scratch/ci-38699600639-current-triage-20261015/failed.log')
selected=[]; counts=collections.defaultdict(collections.Counter)
for i,line in enumerate(log.open(errors='replace'),1):
 m=re.search(r'(PASSED|FAILED|SKIPPED) (tests/(?:qt/)?test_(?:692_[^ :]+|673_make_masks_3d_in_container)\.py)::([^\s]+)',line)
 if m:
  job=line.split('\t')[0]; counts[(job,m[2])][m[1]]+=1
  selected.append({'line':i,'job':job,'state':m[1],'nodeid':m[2]+'::'+m[3],'raw':line.rstrip()})
assert not any(x['state']=='FAILED' for x in selected)
(out/'hosted-owning-results.json').write_text(json.dumps(selected,indent=2)+'\n')
archive=pathlib.Path('/mnt/wd4tb/scratch/n692/build/spacr-example-fibsem.tar')
h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
metadata=json.loads((out/'hf-dataset-metadata.json').read_text())
asset=next(a for a in metadata['siblings'] if a['rfilename']=='spacr-example-fibsem.tar')
assert archive.stat().st_size==asset['size'] and h.hexdigest()==asset['lfs']['sha256']
with tarfile.open(archive) as tar:
 names=tar.getnames(); assert names[-1]=='manifest.csv'
 manifest=list(csv.DictReader(io.StringIO(tar.extractfile('manifest.csv').read().decode())))
 assert len(manifest)==1
 row=manifest[0]
 assert row['shape_zyx']=='128x256x256' and row['mitochondria']=='43' and row['modality']=='FIB-SEM'
 for field,p in [('image_sha256',row['image']),('labels_sha256',row['ground_truth_mask'])]:
  raw=tar.extractfile(p).read(); endian='<' if raw[:2]==b'II' else '>'
  assert struct.unpack_from(endian+'H',raw,2)[0]==42
  offset=struct.unpack_from(endian+'I',raw,4)[0]; hh=hashlib.sha256(); pages=0
  while offset:
   n=struct.unpack_from(endian+'H',raw,offset)[0]; tags={}
   for i in range(n):
    tag,kind,count,value=struct.unpack_from(endian+'HHII',raw,offset+2+12*i)
    if kind in (3,4) and count==1: tags[tag]=value & 65535 if kind==3 else value
   assert tags[259]==1 and tags[256]==tags[257]==256
   assert tags[273]+tags[279]<=len(raw)
   hh.update(raw[tags[273]:tags[273]+tags[279]])
   offset=struct.unpack_from(endian+'I',raw,offset+2+12*n)[0];pages+=1
  assert pages==128 and hh.hexdigest()==row[field]
(out/'published-dataset-byte-proof.json').write_text(json.dumps({'dataset_revision':metadata['sha'],'asset_sha256':h.hexdigest(),'asset_bytes':archive.stat().st_size,'local_preserved_archive':str(archive),'manifest':manifest,'each_decoded_uncompressed_array_hash_matches_manifest':True,'hash_semantics':'build() stores image.tobytes/labels.tobytes SHA, not TIFF container SHA; stdlib TIFF strip reader verifies actual128page pixel payload; initial container-hash assertion was an audit-method error, preserved separately','no_dataset_download':'Only public small metadata and pinned README read back; existing preserved archive exactly matches current published LFS hash','official_licence_policy':'https://www.ebi.ac.uk/empiar/about/ states all EMPIAR data reusable under CC0; read2026-10-15','source_entry_record':'/mnt/wd4tb/scratch/n692/entry.json confirms EMPIAR-10982 FIB-SEM and HeLa150angstrom XY; Z15nm remains an explicit assumption'},indent=2)+'\n')
requirements=[
 {'requirement':'Open plain TIFF/OME/MRC with shape/axes/spacing and preserve saved labels','evidence':['test_plain_stack_round_trip_keeps_shape_labels_and_voxel_size','test_ome_and_mrc_volumes_keep_their_physical_voxel_size','test_save_keeps_shape_axes_and_voxel_size_and_ctrl_s_saves_the_volume'],'result':'All pass in coverage jobs; untraced Qt omitted mrcfile and skipped the one MRC test, explicitly not claimed there.'},
 {'requirement':'XY mouse brush/erase affects exactly shown slice; XZ/YZ side drawing writes correct voxels','evidence':['test_the_volume_opens_on_its_middle_slice_and_paints_with_the_mouse','test_drawing_on_side_views_interpolating_and_assigning','test_side_views_draw_with_the_brush_and_follow_the_main_tool'],'result':'Actual Qt mouse events and exact independently indexed voxel assertions pass, not merely direct backend invocation.'},
 {'requirement':'Interpolate only between drawn sections; assign disconnected 3D objects; history/IDs','evidence':['test_interpolation_fills_only_the_gap_between_drawn_slices','test_assign_objects_splits_unconnected_parts_into_new_ids','test_painting_protects_other_objects_and_undo_restores_voxels'],'result':'Exact arrays, independent labels/connectivity, undo/redo pass; current count-only correction additionally6PASS3.19s, parent accepted.'},
 {'requirement':'Publish small licence-checked FIB-SEM image+label example and build tool','evidence':['published-dataset-byte-proof.json','https://huggingface.co/datasets/einarolafsson/spacr-example-fibsem/tree/'+metadata['sha'],'https://www.ebi.ac.uk/empiar/about/'],'result':'Public immutable revision contains exact25,221,120-byte existing archive, image/label array hashes match manifest,128x256x256/43instances,CC0 card and attribution. Z spacing assumption explicit; no new biological validation.'},
 {'requirement':'Both Load test data entries open blank/with labels; never overwrite ground truth','evidence':['test_the_menu_offers_the_volume_alone_and_with_masks','test_with_masks_opens_the_labels_for_correction_and_keeps_the_truth','test_a_missing_set_is_downloaded_then_opened'],'result':'Real current MakeMasks widget/session; masks vs corrected_masks destinations and byte-for-byte original truth preservation asserted.'},
 {'requirement':'spacr-download fibsem registry/build/archive and mocked acquisition','evidence':['tests/test_692_fibsem_example_dataset.py','test_the_download_worker_targets_the_fibsem_repository'],'result':'All5 build/registry/load/CLI tests pass at minimumPython3.9,fast,andcoverage; real archive fixtures and mocked transport, no new live network download.'},
 {'requirement':'Current English/nine-locale runtime/API closure, no freeze hold','evidence':['features/data/692_foreground_count_and_publication_state_2026-10-15/public-source-api-proof.json','features/data/692_foreground_count_and_publication_state_2026-10-15/scoped-api-proof.json','features/data/692_foreground_count_and_publication_state_2026-10-15/prior-runtime-audit-source-reuse.json'],'result':'Existing parent normal8symbols/37blocks-per-locale333total receipt. Current GUI/demo exact owning bytes and engine only tr/prose-preserving count return; no new collector/generator repeated.'},
 {'requirement':'Dedicated N673/N692 FIB-SEM tutorial','evidence':['features/HANDOFF.md maintainer drop dated2026-10-12'],'result':'DROPPED by maintainer; not an acceptance prerequisite and not reopened.'}
]
for p in ['cf1de1f813','bcd58c3c78','8f7ac02e5b','b12440063b','c224c50279']:
 assert subprocess.run(['git','merge-base','--is-ancestor',p,current],cwd=root).returncode==0
receipt={'item':'N692','decision':'PROPOSE WHOLE ITEM COMPLETE AT ITS RECORDED SOFTWARE+EXAMPLE-DATA CONTRACT; root review required','source_revision':current,'hosted_evidence':{'run':38699600639,'source':hosted,'whole_run':'FAILED overall; owning N692/N673 outcomes below remain literal individual passes, not full-CI acceptance','raw_existing_log':str(log),'result_rows':'hosted-owning-results.json','groups':[{'job':j,'file':f,'counts':dict(c)} for (j,f),c in counts.items()]},'source_bindings':bindings,'source_difference_disposition':{'mask_engine':'Only assign_objects final return changed; parent609789 six-test source-bound receipt closes it','example_archives':'Only unrelated expand_measure_arrays changed; complete other module AST identical to hostedsource'},'requirements':requirements,'remaining_at_this_recorded_contract':[],'limits':['No new tests/imports/app interaction/model/GPU/capture run','No native Windows/macOS renderer signoff, native speaker, or biological/scientific/general segmentation quality certification','N673 broader native desktop/anchor requests and fullrelease/fullCI remain separately tracked; not added to N692 original HOW TO KNOW IT WORKED','Historical2706 broad pass claim not used as acceptance proof; selected literal hosted results and exact source preservation used instead','Public dataset pin/readback is current metadata+matching preserved archive, not fresh end-to-end network downloader test']}
(out/'whole-item-review.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':str(out/'whole-item-review.json'),'sha256':sha((out/'whole-item-review.json').read_bytes()),'decision':receipt['decision'],'requirements':len(requirements),'remaining':receipt['remaining_at_this_recorded_contract']},indent=2))
