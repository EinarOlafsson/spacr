"""Explicit maintenance-v1 references; preserve original04 clocks and real results."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,shutil,sys
LESSON='04_platform_installers'
REVISION='maintenance-v1'
VISUALS=('08_official_release','09_windows_guidance','10_macos_guidance','11_linux_commands','02_installer_backend','03_installed_versions','07_privacy_keep_off','14_current_nightly_performance','13_current_nightly_home','05_doctor','12_logs_and_versions','15_uninstall_windows','16_uninstall_macos','17_uninstall_linux')
CARDS=tuple(v for i,v in enumerate(VISUALS,1) if i not in (7,8,9,10))
BOUNDS=((0,592),(592,1170),(1170,1919),(1919,2760),(2760,3234),(3234,3606),(3606,4293),(4293,5290),(5290,6181),(6181,7060),(7060,7806),(7806,8500),(8500,9433),(9433,10237))
MASTER_SHA='7c93360d640bed781a0756675ff9cdbde2aaf6d907096b8e017e088d1019ec4a'
TIMING_SHA='3988e51266fc3df86a97619898d6f8494eb9d74bb94a2b83395216320ba1357f'
SOURCES={'verification-commands.json':'e5ef36773218ea4f8fdd8d86bfd6521d62bd3a72bc77fdb70fd2bc8863d177ca',
 'verification-provenance.json':'bf3db0c7c0ab6f1b3626fbd4d972b8cb873cbb3dda4e59f856c54697de239b7d',
 'accepted-provenance.json':'622151a59fe7ef83c65e5c3c7b1b190bb66c699ce0b3ab5ab3998c822b0d3475',
 '02_installer_backend.png':'73d08e6d0976be6aaac96356531ae96ef7ca6d9d351095da242df604c05d1cf9',
 '03_installed_versions.png':'bf56ef733510ce0d945636175c3bf26975f9d79d759b5bd38f13c1b3b21dd123'}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def checked_lesson(lesson):
 if lesson.get('id')!=LESSON or [s.get('visual') for s in lesson.get('scenes',[])]!=list(VISUALS):raise ValueError('Require exact fourteen Platform scenes')

def checked_source(clock):
 if (clock.get('lesson')!=LESSON or clock.get('kind')!='native_visual_timing' or clock.get('fps')!=30
     or clock.get('native_live_video') is not True or clock.get('playback_speed')!=1 or clock.get('looped_or_stretched') is not False
     or [s.get('visual') for s in clock.get('scenes',[])]!=list(VISUALS)
     or clock.get('coverage',{}).get('accepted') is not True or clock['coverage'].get('gaps')!=[]):raise ValueError('Accepted original Platform clock required')
 for i,(scene,(a,b)) in enumerate(zip(clock['scenes'],BOUNDS),1):
  if (type(scene.get('scene')) is not int or scene['scene']!=i or type(scene.get('native_live_video')) is not bool or scene['native_live_video']!=(i in (8,9))
      or any(type(scene.get(k)) not in (int,float) or not math.isfinite(scene[k]) or abs(scene[k]-v)>1e-8 for k,v in [('speech_start',a/30),('speech_end',b/30),('scene_end',b/30),('duration',(b-a)/30)])):raise ValueError('Platform original indexed bounds/native classification changed')
 if abs(clock.get('total_duration',0)-10237/30)>1e-8:raise ValueError('Platform total clock changed')

def checked_results(folder):
 folder=Path(folder)
 if folder.is_symlink():raise ValueError('Independent saved result sources required')
 for name,sha in SOURCES.items():
  p=folder/name
  if p.is_symlink() or not p.is_file() or digest(p)!=sha:raise ValueError('Original saved result bytes changed: '+name)
 proof=read(folder/'verification-provenance.json');accepted=read(folder/'accepted-provenance.json')
 if (proof.get('completed_capture') is not True or proof.get('verification_recorded_live') is not True
     or proof.get('application_source_modified') is not False or proof.get('installed_identity',{}).get('version')!='1.5.1.3'
     or accepted.get('completed_capture') is not True or accepted.get('original_evidence_preserved') is not True):raise ValueError('Original installation result provenance differs')
 commands=read(folder/'verification-commands.json');selected={v:[r for r in commands if r.get('scene')==v] for v in VISUALS[4:6]}
 for v,rows in selected.items():
  if len(rows)!=1 or rows[0].get('accepted') is not True or rows[0].get('returncode')!=0 or accepted['frame_origins'][v].get('image_sha256')!=SOURCES[v+'.png']:raise ValueError('Saved terminal result/source-image relationship changed')
 profile=json.loads(selected[VISUALS[4]][0]['output']);versions=selected[VISUALS[5]][0]['output']
 if (profile.get('requested_backend')!='cpu' or profile.get('active_backend')!='cpu' or profile.get('torch_version')!='2.14.1+cpu'
     or profile.get('cuda_available') is not False or profile.get('detected_accelerator')!='nvidia'
     or 'spaCR: 1.5.1.3\n' not in versions or 'Qt: 6.11.2\n' not in versions):raise ValueError('Actual example profile/installed versions changed')
 return profile,versions

def definitions(repo):
 sys.path.insert(0,str(Path(repo)/'tools/tutorials'));import render_installer_instruction_cards as normal
 originals={name:(title,rows,note) for name,title,rows,note in normal.installer_cards('1.5.1.3')}
 mapping={VISUALS[0]:'18_download_desktop',**{v:v for v in CARDS if v not in VISUALS[:1]+VISUALS[4:6]}}
 rows={v:originals[name] for v,name in mapping.items()}
 replacements={
  'Official release asset — source-reviewed guidance, not a Windows recording':'Windows online installer',
  'Current installer source enables automatic acceleration by default':'Automatic accelerator selection is enabled by default',
  'Official package for Intel and Apple silicon — not a macOS recording':'Universal package for Intel and Apple silicon'}
 notes={VISUALS[0]:'Choose the installer for your operating system; setup downloads its own Python runtime.',
  VISUALS[1]:'The online installer downloads its own Python runtime.',
  VISUALS[3]:'Use automatic accelerator selection, or choose CPU-only operation explicitly.',
  VISUALS[11]:'Close spaCR before uninstalling; keep projects and analysis results separately.',
  VISUALS[12]:'The per-user runtime is separate; remove it only if you also want to discard it.',
  VISUALS[13]:'Keep microscopy projects and analysis results outside the installation directory.'}
 for v,(title,instructions,note) in list(rows.items()):
  title={VISUALS[3]:'Linux x86-64: run the installer',VISUALS[10]:'Installation logs and diagnostics'}.get(v,title)
  rows[v]=(title,[(replacements.get(label,label),command) for label,command in instructions],notes.get(v,note))
 rows[VISUALS[4]]=('Check the selected compute backend',[
  ('Example installation profile','requested_backend: cpu  |  active_backend: cpu'),
  ('Installed runtime','PyTorch 2.14.1+cpu  |  cuda_available: false'),
  ('Hardware detected at installation','detected_accelerator: nvidia')],
  'Read install-profile.json to distinguish detected hardware from the active runtime backend.')
 rows[VISUALS[5]]=('Check installed versions',[
  ('Example installed package','spaCR: 1.5.1.3'),('Example installed Qt version','Qt: 6.11.2'),
  ('Run in your installed spaCR environment','python -c "import spacr; print(spacr.__version__)"')],
  'Keep the installed version with your analysis notes, especially before an update.')
 return {v:rows[v] for v in CARDS}

def header(repo,lesson,rows):
 repo=Path(repo)
 return dict(schema=1,lesson=LESSON,reference_revision=REVISION,english_sha256=identity(lesson),generator_sha256=digest(__file__),
  renderer_sha256=digest(repo/'tools/tutorials/render_installer_instruction_cards.py'),style_helper_sha256=digest(repo/'tools/tutorials/authoring/tools/render_install_keyframes.py'),
  definitions_sha256=identity(rows),source_master_sha256=MASTER_SHA,source_timing_sha256=TIMING_SHA,result_sources=SOURCES)

def validate_cards(folder,lesson,repo):
 from PIL import Image
 folder=Path(folder);checked_lesson(lesson);rows=definitions(repo)
 if (folder.is_symlink() or {p.name for p in folder.iterdir()}!={'cards.json','source-evidence',*(v+'.png' for v in CARDS)}
     or any(p.is_symlink() for p in folder.iterdir())):raise ValueError('Exact independent maintenance card directory required')
 evidence=folder/'source-evidence'
 if {p.name for p in evidence.iterdir()}!=set(SOURCES)|{'original-native-timings.json'}:raise ValueError('Exact original result/clock sources required')
 checked_results(evidence)
 if (evidence/'original-native-timings.json').is_symlink() or digest(evidence/'original-native-timings.json')!=TIMING_SHA:raise ValueError('Accepted Platform source clock bytes changed')
 checked_source(read(evidence/'original-native-timings.json'));contract=read(folder/'cards.json')
 if {k:v for k,v in contract.items() if k!='cards'}!=header(repo,lesson,rows):raise ValueError('Maintenance definition/source bindings changed')
 expected=[]
 for v in CARDS:
  path=folder/(v+'.png')
  with Image.open(path) as image:
   if image.size!=(3840,2160) or image.format!='PNG':raise ValueError('Full-size normal reference PNG required')
   image.verify()
  expected.append(dict(visual=v,image=path.name,sha256=digest(path),kind='authored_reference_card',narration_sha256=identity(lesson['scenes'][VISUALS.index(v)]['narration']),reference_revision=REVISION))
 if contract.get('cards')!=expected:raise ValueError('Maintenance pixels, exact slots or narration changed')
 return {r['visual']:r for r in expected}

def render(repo,lesson,baseline,verification,accepted_capture,destination):
 repo,baseline,verification,accepted_capture,destination=map(Path,(repo,baseline,verification,accepted_capture,destination))
 sys.path.insert(0,str(repo/'tools/tutorials'));from published_native_reuse import _published_inputs
 checked_lesson(lesson);baseline,clock,bindings,_=_published_inputs(baseline,LESSON);checked_source(clock)
 if bindings['master']['sha256']!=MASTER_SHA or bindings['timing']['sha256']!=TIMING_SHA:raise ValueError('Accepted published Platform source changed')
 if destination.exists() or destination.is_symlink():raise FileExistsError('Preserve old references and incomplete output')
 incoming={'verification-commands.json':verification/'commands.json','verification-provenance.json':verification/'provenance.json','accepted-provenance.json':accepted_capture/'provenance.json',**{v+'.png':accepted_capture/(v+'.png') for v in VISUALS[4:6]}}
 for name,path in incoming.items():
  if path.is_symlink() or digest(path)!=SOURCES[name]:raise ValueError('Original installation result bytes changed')
 rows=definitions(repo);stylepath=repo/'tools/tutorials/authoring/tools/render_install_keyframes.py'
 spec=importlib.util.spec_from_file_location('maintenance_reference_style',stylepath);style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
 destination.mkdir(parents=True);evidence=destination/'source-evidence';evidence.mkdir()
 for name,path in incoming.items():shutil.copyfile(path,evidence/name)
 shutil.copyfile(baseline/bindings['timing']['path'],evidence/'original-native-timings.json');checked_results(evidence)
 cards=[]
 for v,(title,instructions,note) in rows.items():
  image,draw=style.canvas(title,'Desktop installation and maintenance')
  draw.rounded_rectangle(style.box((150,190,1770,925)),radius=style.scaled(19),fill=style.PANEL,outline=style.PANEL_LINE,width=2)
  for i,(label,command) in enumerate(instructions):
   if draw.textlength(command,font=style.COMMAND)>style.scaled(1500) or draw.textlength(label,font=style.SMALL)>style.scaled(1500):raise ValueError('Reference line would be clipped')
   top=235+i*(136 if len(instructions)>3 else 168)
   if top+53+style.COMMAND.size/style.SCALE>810:raise ValueError('Reference line overlaps note')
   style.draw_text(draw,(200,top),label,style.SMALL,style.MUTED);style.draw_text(draw,(200,top+53),command,style.COMMAND,style.TEXT)
  if draw.textlength(note,font=style.SMALL)>style.scaled(1430):raise ValueError('Reference note would be clipped')
  style.note(draw,note);path=destination/(v+'.png');image.save(path)
  cards.append(dict(visual=v,image=path.name,sha256=digest(path),kind='authored_reference_card',narration_sha256=identity(lesson['scenes'][VISUALS.index(v)]['narration']),reference_revision=REVISION))
 contract=header(repo,lesson,rows);contract['cards']=cards
 (destination/'cards.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n');validate_cards(destination,lesson,repo)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('repo','lesson','baseline','verification','accepted-capture','destination'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();render(a.repo,read(a.lesson),a.baseline,a.verification,a.accepted_capture,a.destination)
if __name__=='__main__':main()
