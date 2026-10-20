from pathlib import Path
import json,pstats,traceback
p=Path('/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015/installed-cprofile-diagnostic');r={'profiling_complete':False}
try:
 calls=json.loads((p/'profile-calls.json').read_text());instrument=json.loads((p/'instrumentation.json').read_text());nav=json.loads((p/'profiled-navigation.json').read_text());assert instrument['import_hook_restored'] is True
 assert {'preferences-open','mask-build'}<=set(row['label'] for row in calls)
 assert all(row['thread_name']=='MainThread' for row in calls)
 assert not any('DIAGNOSTIC PROFILE OUTPUT ERROR' in run.get('worker',{}).get('stderr_tail','') for run in nav['runs'])
 for row in calls:
  file=p/row['pstats_file'];stats=pstats.Stats(str(file));assert stats.total_calls>0
  for order in ['cumulative','tottime']:assert (p/(file.stem+'-'+order+'.txt')).stat().st_size>0
 r.update(profiling_complete=True,profile_calls=calls,instrumentation=instrument,benchmark_passed=nav['passed'],benchmark_violations=nav['violations'])
except BaseException:r['error']=traceback.format_exc()
(p/'profiling-completeness.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if r['profiling_complete'] else 2)
