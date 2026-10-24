"""Three authored instruction-card replacements; all original02 clocks stay intact."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,shutil,sys

LESSON='02_install_spacr'
VISUALS=('00_install_routes','pip_01_create_venv','pip_02_activate_venv','pip_03_install','pip_04_check',
 'pip_05_launch','pip_conda_01_create','pip_conda_02_install','conda_01_create','conda_02_plan',
 'conda_03_activate','docker_01_install','docker_02_image_check','docker_03_mount_and_run','docker_04_gpu',
 '06_current_home','pip_06_update','conda_04_update','pip_07_uninstall','pip_08_remove_environment','conda_05_uninstall')
CARDS={'pip_01_create_venv':'07_create_environment','pip_02_activate_venv':'08_activate_environment','pip_05_launch':'10_launch_commands'}
MASTER_SHA='331e12e22e6118223af0310f75d8ceb32fc4ce45409d12756331d087f02c7dae'
TIMING_SHA='714e5fe2c65e72dd1813ce123d4883354e0ab651c9caec95e0b547024916f847'
BOUNDS=((0,792),(792,1359),(1359,2227),(2227,2925),(2925,3338),(3338,3982),(3982,4674),
 (4674,5489),(5489,6236),(6236,6614),(6614,7248),(7248,8108),(8108,8858),(8858,9871),
 (9871,10771),(10771,12571),(12571,13319),(13319,14177),(14177,14905),(14905,15724),(15724,16614))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def checked_lesson(lesson):
 if lesson.get('id')!=LESSON or [s.get('visual') for s in lesson.get('scenes',[])]!=list(VISUALS):
  raise ValueError('Require exact ordered twenty-one Installation scenes')

def checked_source(original):
 if (original.get('lesson')!=LESSON or original.get('kind')!='native_visual_timing'
     or original.get('fps')!=30 or original.get('native_live_video') is not True
     or original.get('playback_speed')!=1 or original.get('looped_or_stretched') is not False
     or [s.get('visual') for s in original.get('scenes',[])]!=list(VISUALS)
     or original.get('coverage',{}).get('accepted') is not True or original['coverage'].get('gaps')!=[]):
  raise ValueError('Require the accepted original Installation source clock')
 for index,(scene,(start,end)) in enumerate(zip(original['scenes'],BOUNDS),1):
  if (type(scene.get('native_live_video')) is not bool or scene['native_live_video']!=(index==16)
      or type(scene.get('scene')) is not int or scene['scene']!=index or any(type(scene.get(k)) not in (int,float) or not math.isfinite(scene[k]) for k in ('speech_start','speech_end','scene_end','duration'))
      or any(abs(scene[k]-v)>1e-8 for k,v in [('speech_start',start/30),('speech_end',end/30),('scene_end',end/30),('duration',(end-start)/30)])):
   raise ValueError('Installation source index, original bounds or one-Home-native classification changed')
 if abs(original.get('total_duration',0)-16614/30)>1e-8:raise ValueError('Installation total source clock changed')

def definitions(repo):
 sys.path.insert(0,str(Path(repo)/'tools/tutorials'))
 import render_pip_instruction_cards as renderer
 chosen={name:(title,rows,note) for name,title,rows,note in renderer.instruction_cards('1.5.1.3') if name in CARDS.values()}
 if set(chosen)!=set(CARDS.values()):raise ValueError('Require only the three existing pip instruction definitions')
 return renderer,{visual:chosen[name] for visual,name in CARDS.items()}

def header(repo,lesson,rows):
 repo=Path(repo)
 return dict(schema=1,lesson=LESSON,english_sha256=identity(lesson),generator_sha256=digest(__file__),
  renderer_sha256=digest(repo/'tools/tutorials/render_pip_instruction_cards.py'),
  style_helper_sha256=digest(repo/'tools/tutorials/authoring/tools/render_install_keyframes.py'),
  definitions_sha256=identity(rows),source_master_sha256=MASTER_SHA,source_timing_sha256=TIMING_SHA)

def validate_cards(folder,lesson,repo):
 from PIL import Image
 folder=Path(folder);checked_lesson(lesson);_,rows=definitions(repo)
 expected_files={'cards.json','source-native-timings.json',*(v+'.png' for v in CARDS)}
 if (folder.is_symlink() or any(p.is_symlink() or not p.is_file() for p in folder.iterdir())
     or {p.name for p in folder.iterdir()}!=expected_files):raise ValueError('Independent exact three-card source directory required')
 if digest(folder/'source-native-timings.json')!=TIMING_SHA:raise ValueError('Original accepted Installation source clock bytes changed')
 checked_source(read(folder/'source-native-timings.json'))
 contract=read(folder/'cards.json')
 if {k:v for k,v in contract.items() if k!='cards'}!=header(repo,lesson,rows):raise ValueError('Installation card source/definition bindings changed')
 expected=[]
 for visual in CARDS:
  path=folder/(visual+'.png');authored=lesson['scenes'][VISUALS.index(visual)]
  with Image.open(path) as image:
   if image.size!=(3840,2160) or image.format!='PNG':raise ValueError('Require genuine full-size generated reference PNG')
   image.verify()
  expected.append(dict(visual=visual,image=path.name,sha256=digest(path),kind='authored_reference_card',
    narration_sha256=identity(authored['narration']),references=[dict(kind='unchanged_published_source',master_sha256=MASTER_SHA,timing_sha256=TIMING_SHA)]))
 if contract.get('cards')!=expected:raise ValueError('Installation card pixels, exact slots or narration changed')
 return {row['visual']:row for row in expected}

def render(repo,lesson,baseline,destination):
 sys.path.insert(0,str(Path(repo)/'tools/tutorials'))
 from published_native_reuse import _published_inputs
 repo,baseline,destination=Path(repo),Path(baseline),Path(destination);checked_lesson(lesson)
 baseline,original,bindings,_=_published_inputs(baseline,LESSON);checked_source(original)
 if bindings['master']['sha256']!=MASTER_SHA or bindings['timing']['sha256']!=TIMING_SHA:raise ValueError('Original Installation visual source changed')
 if destination.exists() or destination.is_symlink():raise FileExistsError('Preserve previous reference output')
 _,rows=definitions(repo)
 style_path=repo/'tools/tutorials/authoring/tools/render_install_keyframes.py'
 spec=importlib.util.spec_from_file_location('installation_reference_style',style_path);style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
 destination.mkdir(parents=True)
 shutil.copyfile(baseline/bindings['timing']['path'],destination/'source-native-timings.json')
 cards=[]
 for visual,(title,instructions,note) in rows.items():
  image,draw=style.canvas(title,'Installation and environment commands')
  draw.rounded_rectangle(style.box((150,190,1770,925)),radius=style.scaled(19),fill=style.PANEL,outline=style.PANEL_LINE,width=2)
  for index,(label,command) in enumerate(instructions):
   if draw.textlength(command,font=style.COMMAND)>style.scaled(1500) or draw.textlength(label,font=style.SMALL)>style.scaled(1500):raise ValueError('Instruction would be clipped')
   top=235+index*168;style.draw_text(draw,(200,top),label,style.SMALL,style.MUTED);style.draw_text(draw,(200,top+53),command,style.COMMAND,style.TEXT)
  if draw.textlength(note,font=style.SMALL)>style.scaled(1430):raise ValueError('Instruction note would be clipped')
  style.note(draw,note);path=destination/(visual+'.png');image.save(path)
  cards.append(dict(visual=visual,image=path.name,sha256=digest(path),kind='authored_reference_card',
   narration_sha256=identity(lesson['scenes'][VISUALS.index(visual)]['narration']),
   references=[dict(kind='unchanged_published_source',master_sha256=MASTER_SHA,timing_sha256=TIMING_SHA)]))
 contract=header(repo,lesson,rows);contract['cards']=cards
 (destination/'cards.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
 validate_cards(destination,lesson,repo)
 print('Three authored instruction references; unchanged commands/21 clocks; no capture, installation or synthesis',flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('repo','lesson','baseline','destination'):p.add_argument('--'+name,type=Path,required=True)
 args=p.parse_args();render(args.repo,read(args.lesson),args.baseline,args.destination)

if __name__=='__main__':main()
