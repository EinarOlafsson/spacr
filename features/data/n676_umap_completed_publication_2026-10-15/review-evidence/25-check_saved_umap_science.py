"""Read-only saved UMAP provenance checks; never imports spaCR or refits."""
import argparse, collections, csv, hashlib, json, math, pathlib, sqlite3, subprocess, sys
ROOT=pathlib.Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--stage',type=pathlib.Path,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
s=a.stage.resolve();c=s/'captures/umap'; alias='/tmp/spacr-tutorial-current/profile/.cache/spacr/example_data/'
def load(p):return json.loads(p.read_bytes())
def h(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()
def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':h(p)}
def physical(v):
 assert v.startswith(alias),v
 q=s/'example_data'/v[len(alias):];assert q.resolve().is_relative_to((s/'example_data').resolve());assert q.is_file();return q
prov=load(c/'provenance.json');host=load(s/'host-run.json');assert prov['commit']==a.source_commit and not prov['app_source_modified'];assert host['failure'] is None and host['app_exit_code']==host['recorder_rc']==0
settings=load(c/'configured_settings.json');assert settings==load(c/'settings_after_run.json')==load(c/'batch_settings.json')
expected={'row_limit':400,'tables':['cell','cytoplasm','nucleus','pathogen'],'random_seed':42,'n_jobs':2,'reduction_method':'umap','n_neighbors':15,'min_dist':.1,'clustering':'dbscan','remove_cluster_noise':False,'remove_highly_correlated':True,'log_data':False,'resnet_features':False,'gpu':False,'batch_correction':'none'}
for k,v in expected.items():assert settings[k]==v,(k,settings[k])
batch=load(c/'batch_acceptance.json');assert batch['accepted'] and batch['worker_ok'] and batch['worker_finished'] and batch['figure_count']>=1
runs=list((s/'app-state/runs').glob('*__umap/manifest.json'));assert len(runs)==1;m=load(runs[0]);assert m['status']=='success' and m['traceback'] is None
csvp=s/'example_data/plate1/results/embedding_results.csv';pdf=csvp.with_name('UMAP_embedding.pdf');db=s/'example_data/plate1/measurements/measurements.db'
inputs=m['input_hashes'];outputs=m['output_hashes'];dbalias=alias+'plate1/measurements/measurements.db';assert h(db)==inputs[dbalias]['sha256']
for f in [csvp,pdf]:
 key=alias+str(f.relative_to(s/'example_data'));assert h(f)==outputs[key]['sha256'] and f.stat().st_size==outputs[key]['size_bytes']
assert pdf.read_bytes().startswith(b'%PDF-')
with csvp.open(newline='') as f:r=csv.DictReader(f);rows=list(r);cols=r.fieldnames
assert len(rows)==400 and len({x['png_path'] for x in rows})==400
clusters=collections.Counter()
conn=sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True)
try:
 tables={x[0] for x in conn.execute("select name from sqlite_master where type='table'")};assert set(expected['tables'])<=tables
 png={x[0]:x[1] for x in conn.execute('select png_path,prcfo from png_list')}
 for row in rows:
  assert row['png_path'] in png
  assert conn.execute('select count(*) from cell where prcf=? and object_label=?',(row['prcf'],row['object_label'])).fetchone()[0]==1
  assert row['source_database']==pathlib.PurePosixPath(settings['src']).name
  cluster=float(row['cluster']);assert math.isfinite(cluster) and cluster.is_integer();clusters[str(int(cluster))]+=1
  crop=physical(row['png_path']);entry=inputs[row['png_path']];assert crop.stat().st_size==entry['size_bytes'] and h(crop)==entry['sha256']
finally:conn.close()
tour=load(c/'explorer_tour.json');assert tour['point_count']==400 and tour['settings_unchanged'] and tour['database_unchanged'];assert not any(tour[k] for k in ['annotations_written','search_started','new_embedding_requested','advanced_controls_toured']);assert tour['nested_routes']==[]
assert [r['point_index'] for r in tour['clicked_points']]==[0,200]
for picked in tour['clicked_points']:
 index=picked['point_index'];assert picked['database_path']==rows[index]['png_path'] and picked['database_path'] in picked['label']
profile=load(s/'capture-profile.json');assert profile['show_beta'] and not any(profile[k] for k in ['show_alpha','show_alpha_features','show_alpha_species'])
console='\n'.join(load(c/'batch_console_after_tour.json'));assert 'Embedding shape: (400, 2)' in console and 'Trained and fit reducer: umap on cpu' in console and '✓ Finished' in console
source=[]
for path in ['spacr/core.py','tools/tutorials/capture_umap.py','spacr/utils.py']:
 b=subprocess.check_output(['git','show',a.source_commit+':'+path],cwd=ROOT);source.append({'path':path,'git_commit':a.source_commit,'sha256':hashlib.sha256(b).hexdigest()})
 if path.endswith('capture_umap.py'):
  assert b'np.isfinite(explorer._embedding).all()' in b and b"len(explorer._embedding) != len(explorer._records)" in b and b"explorer._picked != index" in b
out={'schema_version':1,'lesson':'15_image_umap','scope':'saved-output and source-bound runtime invariants only; no refit','saved_science_checks_passed':True,'whole_capture_accepted':False,'native_capture_accepted':False,'capture_source_commit':a.source_commit,'actual_provenance_status':prov['status'],'actual_completed_capture':prov['completed_capture'],'actual_gate_failed':host['gate_failed'],'selected_rows':400,'all_400_unique_crop_paths_match_database_and_pre_run_hashes':True,'all_400_cell_identities_exist':True,'database_matches_pre_run_manifest_sha256':True,'output_csv_and_pdf_match_current_successful_run_manifest':True,'cluster_counts':dict(clusters),'settings_verified':expected,'runtime_2d_finite_embedding':{'console_shape':[400,2],'source_bound_completed_explorer_checks':'400 records aligned; finite nonempty embedding; real picks0/200 nonnull crop; normal tour returned','independently_recomputed_coordinates':False,'limitation':'Embedding coordinates are not persisted in embedding_results.csv. No independent coordinate-array arithmetic or UMAP refit was performed.'},'effective_workers':{'requested_n_jobs':2,'observed_console':'UMAP(n_jobs=1, random_state=42, verbose=True)','fixed_seed_overrides_parallelism_warning_preserved':True},'preprocessing_console':['Median-imputed 17459 missing values in 90 partially observed feature columns','Dropped0 all-missing columns','remove_highly_correlated=True; log_data=False; resnet_features=False; batch_correction=none'],'explorer_tour':tour,'profile_visibility':{k:profile[k] for k in ['show_beta','show_alpha','show_alpha_features','show_alpha_species']},'source_bindings':source,'evidence':[bind(f) for f in [s/'host-run.json',c/'provenance.json',s/'capture-profile.json',c/'configured_settings.json',c/'settings_after_run.json',c/'batch_settings.json',c/'batch_acceptance.json',c/'explorer_tour.json',c/'batch_console_after_tour.json',runs[0],db,csvp,pdf]],'limits':['Cluster labels are unsupervised assignments, not validated biological classes.','No independent reconstruction of feature preprocessing, fitted coordinates, clustering or projection quality.','Runtime assertions establish coordinate finiteness and crop selection only through pinned executed recorder/source and completed receipt.','Turn57 adapter-bypass failure remains; this review grants no native or whole-lesson acceptance.']}
a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'output':str(a.output),'sha256':h(a.output),'rows':400,'clusters':dict(clusters),'whole_capture_accepted':False}))
