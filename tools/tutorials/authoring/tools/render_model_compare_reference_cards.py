"""Three truthful Model Compare references from the original saved outputs; no GUI/terminal simulation."""
from pathlib import Path
import argparse,csv,hashlib,json,math,shutil,sys
VISUALS=('gui_00_home','gui_02_model_parameters','api_00_inputs','api_02_actual_comparison','api_04_actual_comparison_figure','api_03_outputs')
CARDS=(VISUALS[2],VISUALS[3],VISUALS[5])
RESULTS={'comparison.csv':'586d2a9fb420921174dd160e54b11b3c8d002a2b6da1f8a1d202c13ecb097daa',
 'comparison.png':'5fcaf5839406b60fcd46bacf2f3c2f6f025dd82388a7c63ffb8972e4eba97fb2',
 'run.json':'7ac7491f13ed09e35e0cfc81e6c86571cdb35497f1d3bb97c9a735ba442a5d00'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def checked_results(folder):
 folder=Path(folder)
 if folder.is_symlink() or not folder.is_dir():raise ValueError('Require independent saved-result directory')
 for name,sha in RESULTS.items():
  path=folder/name
  if path.is_symlink() or not path.is_file() or digest(path)!=sha:raise ValueError('Original saved result changed: '+name)
 run=json.loads((folder/'run.json').read_text())
 if (run.get('accepted') is not True or run.get('source_unchanged') is not True
     or any(run.get(k) is not False for k in ('accuracy_validated','different_model_weights_compared','gui_workflow_completed','inference_performed'))
     or run.get('artifacts')!={k:v for k,v in RESULTS.items() if k!='run.json'}
     or set(run.get('sources',{}))!={'image.tif','batch.tif','preview.npy'}):raise ValueError('Saved comparison scope changed')
 with (folder/'comparison.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
 if len(rows)!=1:raise ValueError('Require the one original comparison row')
 row=rows[0];comparison=run['comparison']
 for key in ('n_objects_a','n_objects_b','n_matched','mean_matched_iou','ari'):
  value=float(row[key])
  if not math.isfinite(value) or value!=comparison[key]:raise ValueError('CSV and original comparison disagree: '+key)
 return comparison

def definitions(comparison):
 return {
 CARDS[0]:('Prepare the saved-mask example',[
  'Extract the linked example into a project folder.',
  'image.tif: the source image',
  'batch.tif and preview.npy: masks of the same field',
  'Keep the filenames; choose a separate results folder.']),
 CARDS[1]:('Run the comparison helper',[
  'python model_compare_example.py --source inputs --output comparison',
  f"Objects: A {comparison['n_objects_a']} · B {comparison['n_objects_b']} · Matched {comparison['n_matched']}",
  f"Mean matched IoU: {comparison['mean_matched_iou']:.3f} · Adjusted Rand index: {comparison['ari']:.3f}",
  'Compare the overlays with the source image.']),
 CARDS[2]:('Inspect and retain the saved outputs',[
  'comparison.csv: object counts, matches and overlap scores',
  f"A {comparison['n_objects_a']} · B {comparison['n_objects_b']} · Matched {comparison['n_matched']}",
  f"Mean matched IoU {comparison['mean_matched_iou']:.6f} · ARI {comparison['ari']:.6f}",
  'Keep comparison.png, the inputs and analysis settings.'])}

def checked_lesson(lesson):
 if lesson.get('id')!='21_model_compare' or [s.get('visual') for s in lesson.get('scenes',[])]!=list(VISUALS):raise ValueError('Require exact six-scene Model Compare source')

def validate_cards(folder,lesson,repo):
 from PIL import Image
 folder,repo=Path(folder),Path(repo);checked_lesson(lesson)
 if folder.is_symlink():raise ValueError('Reference folder must not be a symlink')
 comparison=checked_results(folder/'source-results');defs=definitions(comparison)
 path=folder/'cards.json'
 if path.is_symlink():raise ValueError('Card contract must not be a symlink')
 contract=json.loads(path.read_text())
 header=dict(schema=1,lesson=lesson['id'],english_sha256=identity(lesson),definitions_sha256=identity(defs),
  generator_sha256=digest(__file__),style_helper_sha256=digest(repo/'tools/tutorials/authoring/tools/render_install_keyframes.py'),
  result_sources=[dict(name=n,sha256=h) for n,h in sorted(RESULTS.items())])
 if {k:v for k,v in contract.items() if k!='cards'}!=header:raise ValueError('Reference source binding changed')
 rows=contract.get('cards',[])
 if len(rows)!=3 or [r.get('visual') for r in rows]!=list(CARDS):raise ValueError('Require exactly the three reviewed reference slots')
 for row in rows:
  visual=row['visual'];authored=next(s for s in lesson['scenes'] if s['visual']==visual);image=folder/(visual+'.png')
  expected=dict(visual=visual,image=visual+'.png',sha256=digest(image),kind='authored_reference_card',narration_sha256=identity(authored['narration']),
   references=header['result_sources'])
  if row!=expected or image.is_symlink():raise ValueError('Reference image, narration or saved-output binding changed')
  with Image.open(image) as frame:
   if frame.size!=(3840,2160) or frame.format!='PNG':raise ValueError('Reference must be a 4K PNG')
   frame.verify()
 return {r['visual']:r for r in rows}

def render(repo,lesson,results,destination):
 repo,results,destination=Path(repo),Path(results),Path(destination);checked_lesson(lesson)
 comparison=checked_results(results);defs=definitions(comparison)
 if destination.exists() or destination.is_symlink():raise FileExistsError('Preserve existing reference cards')
 sys.path.insert(0,str(repo/'tools/tutorials/authoring/tools'));import render_install_keyframes as r
 from PIL import ImageDraw
 destination.mkdir(parents=True);sources=destination/'source-results';sources.mkdir()
 for name in RESULTS:shutil.copy2(results/name,sources/name)
 checked_results(sources);cards=[]
 for visual,(title,lines) in defs.items():
  image=r.gradient_background();draw=ImageDraw.Draw(image)
  r.draw_text(draw,(150,100),'Model Compare · Saved-result reference',r.SUBTITLE,r.CYAN)
  r.draw_text(draw,(150,190),title,r.TITLE,r.TEXT)
  for index,line in enumerate(lines):
   if r.BODY.getlength(line)>r.scaled(1620):raise ValueError('Reference text exceeds safe width')
   r.draw_text(draw,(170,380+index*130),line,r.BODY,r.TEXT)
  r.draw_text(draw,(150,1000),'spaCR tutorial series',r.SMALL,r.MUTED)
  r.draw_text(draw,(1770,1000),'Reference · saved comparison',r.SMALL,r.MUTED,anchor='ra')
  path=destination/(visual+'.png');image.save(path)
  authored=next(s for s in lesson['scenes'] if s['visual']==visual)
  cards.append(dict(visual=visual,image=path.name,sha256=digest(path),kind='authored_reference_card',narration_sha256=identity(authored['narration']),references=[dict(name=n,sha256=h) for n,h in sorted(RESULTS.items())]))
 contract=dict(schema=1,lesson=lesson['id'],english_sha256=identity(lesson),definitions_sha256=identity(defs),generator_sha256=digest(__file__),style_helper_sha256=digest(repo/'tools/tutorials/authoring/tools/render_install_keyframes.py'),result_sources=[dict(name=n,sha256=h) for n,h in sorted(RESULTS.items())],cards=cards)
 (destination/'cards.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
 validate_cards(destination,lesson,repo)
 print('Three saved-result reference cards; no fresh comparison, capture or GUI claim',flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('repo','lesson','results','destination'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();render(a.repo,json.loads(a.lesson.read_text()),a.results,a.destination)
if __name__=='__main__':main()
