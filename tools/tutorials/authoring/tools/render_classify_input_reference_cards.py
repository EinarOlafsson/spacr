"""Three labelled Classify preparation references from actual saved inputs; no capture/run."""
from collections import Counter
from pathlib import Path
import argparse,hashlib,json,shutil,sys
LESSON='10_classify_cv'
VISUALS=('gui_19_setting_classifier_family','prepare_01_source_example','prepare_02_prepare_existing_split','prepare_03_exported_manifest','gui_18_source_folder_choice','gui_19_setting_generate_training_dataset','gui_19_setting_model_type','gui_19_setting_val_split','gui_19_setting_plot','gui_24_batch_figure','gui_23_batch_finished')
CARDS=VISUALS[1:4]
SOURCES={'commands.json':'9868717b3071392e41a9296c15ed5fb2e5e35ce06c48533b424917c0d89e7ccc','tutorial_input_manifest.json':'aa7a433936430aabf491b3f39c1db1c8095f45ea964c30f2c5fcfc58542688a8','scientific_acceptance.json':'ac1c885918e5da42063846ad1a7029433ee09ac5221d4e4d2d532859f2ef4af2'}
HELPER='0ae0a0e3d2b1353017ba96f57ecd74d924a969a680a13b290294058cdfb94768'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def checked_lesson(lesson):
 if lesson.get('id')!=LESSON or [s.get('visual') for s in lesson.get('scenes',[])]!=list(VISUALS):raise ValueError('Require exact Classify11 ordered scenes')
def checked_sources(folder):
 folder=Path(folder)
 if folder.is_symlink():raise ValueError('Independent source directory required')
 for name,h in SOURCES.items():
  p=folder/name
  if p.is_symlink() or not p.is_file() or digest(p)!=h:raise ValueError('Accepted preparation source changed: '+name)
 commands=read(folder/'commands.json');manifest=read(folder/'tutorial_input_manifest.json');proof=read(folder/'scientific_acceptance.json')
 if ([r.get('scene') for r in commands]!=['01_source_example','02_prepare_existing_split','03_exported_manifest'] or any(r.get('accepted') is not True or r.get('returncode')!=0 for r in commands)):raise ValueError('Original preparation commands must all pass')
 preparation=proof.get('preparation',{});inputs=preparation.get('inputs',{})
 if (proof.get('accepted') is not True or preparation.get('accepted') is not True or preparation.get('helper_sha256')!=HELPER or proof.get('native',{}).get('inputs')!=inputs or inputs.get('objects_checked')!=64 or inputs.get('sample_counts')!={'train':32,'test':32}):raise ValueError('Actual accepted split proof differs')
 if (manifest.get('accepted') is not True or manifest.get('source_database_sha256')!=inputs.get('source_database_sha256') or manifest.get('source_unchanged') is not True or manifest.get('model_started') is not False or manifest.get('class_values')!={'infected_1':1,'infected_2':2}):raise ValueError('Saved split input semantics differ')
 rows=manifest.get('records',[])
 wells={'train':[['plate1','r12','c1'],['plate1','r5','c2']],'test':[['plate1','r12','c2'],['plate1','r5','c1']]}
 if len(rows)!=64 or len({r['prcfo'] for r in rows})!=64 or any(manifest.get(k+'_wells')!=v for k,v in wells.items()) or inputs.get('actual_wells')!=wells:raise ValueError('Saved image count/well split differs')
 groups=Counter((r['split'],tuple(r['well']),r['class_name']) for r in rows)
 expected={(split,tuple(well),label):8 for split,items in wells.items() for well in items for label in ('infected_1','infected_2')}
 if groups!=expected or any(r['filename']!=r['prcfo']+'.png' for r in rows):raise ValueError('Require the actual 8 existing crops per class/well')
 if '"objects": 64' not in commands[1]['output'] or '"train_objects": 32' not in commands[1]['output'] or '"test_objects": 32' not in commands[1]['output']:raise ValueError('Original helper output differs')
 return manifest

def definitions():
 return {CARDS[0]:('Labelled example inputs','Dataset reference',[
  'Annotate → Load test data → Load',
  'Keep the plate folder: measurements database and PNG crops.',
  'Own data: Measure crops → assign classes in Annotate.']),
 CARDS[1]:('Prepare the example split','Command reference',[
  'Extract prepare_classify_split.py from the linked example ZIP.',
  'python prepare_classify_split.py',
  '    --source <plate-folder> --destination <new-folder>',
  'Example result: 64 labelled images · 32 training · 32 test.']),
 CARDS[2]:('Inspect the split and manifest','Dataset reference',[
  'train/infected_1/   train/infected_2/',
  'test/infected_1/    test/infected_2/',
  'tutorial_input_manifest.json: each image, class and well.',
  'Training: plate1 r12c1, r5c2 · Test: plate1 r12c2, r5c1.',
  'Keep every well entirely in training or test.'])}

def validate_cards(folder,lesson,repo):
 from PIL import Image
 folder,repo=Path(folder),Path(repo);checked_lesson(lesson)
 if digest(repo/'tools/tutorials/prepare_classify_split.py')!=HELPER:raise ValueError('Current linked split helper differs from accepted command')
 if folder.is_symlink() or (folder/'cards.json').is_symlink():raise ValueError('Independent card contract required')
 checked_sources(folder/'source-results');contract=read(folder/'cards.json')
 header=dict(schema=1,lesson=LESSON,english_sha256=identity(lesson),definitions_sha256=identity(definitions()),generator_sha256=digest(__file__),style_helper_sha256=digest(repo/'tools/tutorials/authoring/tools/render_install_keyframes.py'),source_hashes=SOURCES)
 if {k:v for k,v in contract.items() if k!='cards'}!=header:raise ValueError('Reference contract source/wording changed')
 rows=contract.get('cards',[])
 if len(rows)!=3 or [r.get('visual') for r in rows]!=list(CARDS):raise ValueError('Only three reviewed Classify card slots')
 for row in rows:
  p=folder/(row['visual']+'.png');scene=next(s for s in lesson['scenes'] if s['visual']==row['visual'])
  if row!=dict(visual=scene['visual'],image=p.name,sha256=digest(p),kind='authored_reference_card',narration_sha256=identity(scene['narration'])) or p.is_symlink():raise ValueError('Reference image/narration changed')
  with Image.open(p) as image:
   if image.size!=(3840,2160) or image.format!='PNG':raise ValueError('Require 4K reference PNG')
   image.verify()
 return {r['visual']:r for r in rows}

def render(repo,lesson,commands,split_manifest,acceptance,destination):
 repo,destination=Path(repo),Path(destination);checked_lesson(lesson)
 if digest(repo/'tools/tutorials/prepare_classify_split.py')!=HELPER:raise ValueError('Current linked split helper differs from accepted command')
 sources={'commands.json':Path(commands),'tutorial_input_manifest.json':Path(split_manifest),'scientific_acceptance.json':Path(acceptance)}
 for name,path in sources.items():
  if path.is_symlink() or digest(path)!=SOURCES[name]:raise ValueError('Original accepted source bytes differ: '+name)
 if destination.exists() or destination.is_symlink():raise FileExistsError('Preserve earlier cards')
 sys.path.insert(0,str(repo/'tools/tutorials/authoring/tools'));import render_install_keyframes as r
 from PIL import ImageDraw
 destination.mkdir(parents=True);saved=destination/'source-results';saved.mkdir()
 for name in SOURCES:shutil.copy2(sources[name],saved/name)
 checked_sources(saved);cards=[]
 for visual,(title,label,lines) in definitions().items():
  image=r.gradient_background();draw=ImageDraw.Draw(image)
  r.draw_text(draw,(150,100),'Classify · '+label,r.SUBTITLE,r.CYAN);r.draw_text(draw,(150,190),title,r.TITLE,r.TEXT)
  for n,line in enumerate(lines):
   if r.BODY.getlength(line)>r.scaled(1620):raise ValueError('Reference line exceeds normal safe width')
   r.draw_text(draw,(170,360+n*115),line,r.BODY,r.TEXT)
  r.draw_text(draw,(150,1000),'spaCR tutorial series',r.SMALL,r.MUTED);r.draw_text(draw,(1770,1000),label,r.SMALL,r.MUTED,anchor='ra')
  p=destination/(visual+'.png');image.save(p);scene=next(s for s in lesson['scenes'] if s['visual']==visual)
  cards.append(dict(visual=visual,image=p.name,sha256=digest(p),kind='authored_reference_card',narration_sha256=identity(scene['narration'])))
 contract=dict(schema=1,lesson=LESSON,english_sha256=identity(lesson),definitions_sha256=identity(definitions()),generator_sha256=digest(__file__),style_helper_sha256=digest(repo/'tools/tutorials/authoring/tools/render_install_keyframes.py'),source_hashes=SOURCES,cards=cards)
 (destination/'cards.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n');validate_cards(destination,lesson,repo)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('repo','lesson','commands','split-manifest','acceptance','destination'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();render(a.repo,read(a.lesson),a.commands,a.split_manifest,a.acceptance,a.destination)
if __name__=='__main__':main()
