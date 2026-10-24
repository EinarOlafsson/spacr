"""Indexed CLI references from accepted published examples, without executing commands."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
LESSON='06_api'
KIND='published_legacy_api_picture_references'
VISUALS=('01_version','02_list','03_describe','05_report_api','06_dry_run','03_describe')
CARDS=(1,3,4,5,6)
PUBLISHED={
 'master':'090735b9b233339cbff894391e723b5c5376b0387c22a1a2634160733e398aeb',
 'clock':'f73e32b649bc474fae3c0b74dcc6be5a99711cb7d855bbb739f27c0540a18977',
 'web':'d633e3c4a9d69ea0e5af7a44ee7ee06c5446250925e9fbe08d661f5bf9b10d7b'}
SOURCE_FRAMES={1:'312cf79d989801a8c15ed7372e1411194a1ac6937f850dee88f028df6c8b445e',3:'1703125f2b3cde9d701f3e6de266bc974636d39cc039ba2a5c46bef38af5f977',4:'ede79efff1e39ccda38120a1a7b1c50df718c3700100877dcadff87efc611246',5:'b0cbc14d7b18f4f747010231b41b2eeb67b463031ec5217721d4ae601721682f',6:'1703125f2b3cde9d701f3e6de266bc974636d39cc039ba2a5c46bef38af5f977'}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def checked_lesson(lesson):
 if lesson.get('id')!=LESSON or [s.get('visual') for s in lesson.get('scenes',[])]!=list(VISUALS):raise ValueError('Require exact six indexed API chapters')
def filename(index):return f'scene-{index:03}--{VISUALS[index-1]}.png'
def checked_published(source):
 visuals=source.get('original_visuals',[])
 if (source.get('lesson')!=LESSON or source.get('kind')!=KIND or len(visuals)!=3
     or visuals[0].get('sha256')!=PUBLISHED['master'] or visuals[1].get('sha256')!=PUBLISHED['web']
     or source.get('original_clock',{}).get('sha256')!=PUBLISHED['clock']):raise ValueError('Require actual reviewed API master/web/original clock')
def checked_frames(folder):
 from PIL import Image
 folder=Path(folder)
 if folder.is_symlink() or not folder.is_dir():raise ValueError('Independent saved chapter frames required')
 for index,h in SOURCE_FRAMES.items():
  p=folder/f'chapter-{index:03}.png'
  if p.is_symlink() or not p.is_file() or digest(p)!=h:raise ValueError('Reviewed published chapter frame changed')
  with Image.open(p) as image:
   if image.format!='PNG':raise ValueError('Require saved published PNG')
   image.verify()
def definitions():
 return {
  1:('Check the installed version','Command and example output',[
   'Activate the Python environment containing spaCR.',
   "python -c 'import spacr; print(spacr.__version__)'",
   'Example output: 1.5.0.9',
   'Keep your installed version with the analysis notes.']),
  3:('Describe a batch command','Command and example reference',[
   'spacr-run --describe regression',
   'Function: spacr.ml.perform_regression(settings)',
   'Required settings: paired_data and dependent_variable.',
   'Pair phenotype score tables with guide count tables.',
   'Read the API reference for parameters and results.']),
  4:('Inspect a Python function','Command and example signature',[
   'from spacr.report import build_report',
   'import inspect; print(inspect.signature(build_report))',
   "(src: 'Any', out: 'Any', fmt: 'str' = 'html',",
   "    **kwargs: 'Any') -> 'List[Path]'",
   'Prepare the arguments, then use the function in your script.']),
  5:('Validate settings before execution','Command and example result',[
   'spacr-run regression --settings <settings-file> --dry-run',
   'Example inputs: paired phenotype scores and guide counts.',
   'Read the validation messages and correct missing inputs.',
   'Example result: no problems found; settings are runnable.',
   'Example dry-run: perform_regression() was not called.',
   'In this example, nothing was executed.']),
  6:('Run the saved configuration','Command instructions',[
   'spacr-run regression --settings <settings-file>',
   'Use your saved settings and actual input paths.',
   'Follow the console until the command finishes.',
   'Inspect the outputs described in the Regression tutorial.',
   'Keep inputs, settings, version and console log with results.'])}
def header(lesson,source,repo):
 return dict(schema=1,kind=KIND,lesson=LESSON,english_sha256=identity(lesson),definitions_sha256=identity(definitions()),generator_sha256=digest(__file__),style_helper_sha256=digest(Path(repo)/'tools/tutorials/authoring/tools/render_install_keyframes.py'),published_source_sha256=identity(source),published_visual_hashes=PUBLISHED,source_frame_hashes={str(k):v for k,v in SOURCE_FRAMES.items()})
def validate_cards(folder,lesson,repo,source):
 from PIL import Image
 folder=Path(folder);checked_lesson(lesson);checked_published(source)
 if folder.is_symlink() or (folder/'cards.json').is_symlink():raise ValueError('Independent indexed card contract required')
 checked_frames(folder/'source-frames');contract=read(folder/'cards.json')
 if {k:v for k,v in contract.items() if k!='cards'}!=header(lesson,source,repo):raise ValueError('API reference source/wording changed')
 rows=contract.get('cards',[])
 if [r.get('source_scene') for r in rows]!=list(CARDS):raise ValueError('Only five exact indexed API slots')
 for index,row in zip(CARDS,rows):
  p=folder/filename(index);authored=lesson['scenes'][index-1]
  expected=dict(source_scene=index,visual=authored['visual'],image=p.name,sha256=digest(p),kind='authored_reference_card',narration_sha256=identity(authored['narration']))
  if row!=expected or p.is_symlink():raise ValueError('Indexed API image/narration changed')
  with Image.open(p) as image:
   if image.size!=(3840,2160) or image.format!='PNG':raise ValueError('Require normal 4K reference PNG')
   image.verify()
 return {r['source_scene']:r for r in rows}
def render(repo,lesson,source,frames,destination):
 repo,destination=Path(repo),Path(destination);checked_lesson(lesson);checked_published(source);checked_frames(frames)
 if destination.exists() or destination.is_symlink():raise FileExistsError('Preserve prior reference cards')
 sys.path.insert(0,str(repo/'tools/tutorials/authoring/tools'));import render_install_keyframes as style
 from PIL import ImageDraw
 destination.mkdir(parents=True);saved=destination/'source-frames';saved.mkdir()
 for index in CARDS:shutil.copy2(Path(frames)/f'chapter-{index:03}.png',saved/f'chapter-{index:03}.png')
 rows=[]
 for index,(title,label,lines) in definitions().items():
  image=style.gradient_background();draw=ImageDraw.Draw(image)
  style.draw_text(draw,(150,100),'Python API and batch commands',style.SUBTITLE,style.CYAN)
  style.draw_text(draw,(150,190),title,style.TITLE,style.TEXT)
  for n,line in enumerate(lines):
   if style.BODY.getlength(line)>style.scaled(1620):raise ValueError('Reference line exceeds normal safe width')
   style.draw_text(draw,(170,360+n*115),line,style.BODY,style.TEXT)
  style.draw_text(draw,(150,1000),'spaCR tutorial series',style.SMALL,style.MUTED)
  style.draw_text(draw,(1770,1000),label,style.SMALL,style.MUTED,anchor='ra')
  p=destination/filename(index);image.save(p);scene=lesson['scenes'][index-1]
  rows.append(dict(source_scene=index,visual=scene['visual'],image=p.name,sha256=digest(p),kind='authored_reference_card',narration_sha256=identity(scene['narration'])))
 (destination/'cards.json').write_text(json.dumps({**header(lesson,source,repo),'cards':rows},ensure_ascii=False,indent=2)+'\n')
 validate_cards(destination,lesson,repo,source)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('repo','stage','source-frames','destination'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.repo/'tools/tutorials'))
 from published_legacy_picture_mix import checked_source
 source,lesson,clock=checked_source(a.stage,LESSON)
 render(a.repo,lesson,source,a.source_frames,a.destination)
if __name__=='__main__':main()
