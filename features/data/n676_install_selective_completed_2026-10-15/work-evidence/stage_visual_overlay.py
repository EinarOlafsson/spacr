#!/usr/bin/env python3
"""Prepare/run only the approved scene7 command-reference overlay. Root launches."""
from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'tools/tutorials')]
sys.meta_path[:]=[f for f in sys.meta_path if '__editable__' not in (getattr(f,'__module__','') or type(f).__module__)]
import argparse,copy,hashlib,json,os,shutil,subprocess
W=Path(__file__).resolve().parent
N=Path('/media/carruthers/mnt3/claude/n676_stages')
BASE=N/'host/stage-23-r60-20261015/release-candidate-append-7ew0avek'
ORIGINAL=Path('/media/carruthers/mnt3/codex/scratch/tutorials-agent-20261008/tutorials-requested-live-r1')
LESSON='02_install_spacr';VISUAL='pip_conda_01_create';PROD=Path('production')/LESSON
CARD_SOURCE=W/'scene07-command-reference'
BASE_SHA='8065f31b9a758cd0a823efbeb8e631cbc740c1c88ca0ea21d7a41e08eab2dcd4'
STAGE=Path(os.environ.get('SELECTIVE_STAGE',str(N/'host/stage-02-selective-r2-20261015'))).resolve()
def read(p):return json.loads(p.read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def write_new(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def cp(src,dst):
 assert src.is_file() and not src.is_symlink();dst.parent.mkdir(parents=True,exist_ok=True)
 with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
 assert bind(src)['sha256']==bind(dst)['sha256'] and src.stat().st_size==dst.stat().st_size

def common():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU only'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==os.environ['SOURCE_COMMIT']
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(), 'Freeze clean source first'
 assert sha(BASE/'release-manifest.json')==BASE_SHA
 assert STAGE.is_relative_to(N/'host') and not STAGE.is_relative_to(ROOT)

def stage():
 common();assert not STAGE.exists(), 'Preserve existing stage'
 original=ORIGINAL/PROD;baseline=BASE/'media_host'/LESSON
 timing=read(baseline/'video/native-live-timings.json');assert sha(original/'scenes.json')==timing['visual_spec_sha256']
 assert (original/'video/native-live-timings.json').read_bytes()==(baseline/'video/native-live-timings.json').read_bytes()
 assert sha(original/'video'/f'{LESSON}_silent.mp4')==sha(baseline/'video'/f'{LESSON}_silent.mp4')
 assert len(timing['scenes'])==17 and round(timing['total_duration']*30)==12901
 assert round(timing['scenes'][6]['speech_start']*30)==4420 and round(timing['scenes'][6]['duration']*30)==692
 assert timing['scenes'][6]['visual']==VISUAL and timing['scenes'][6]['native_live_video'] is False
 assert not timing.get('complete_native_clips',False)
 records={row['path']:row for row in read(BASE/'release-manifest.json')['files']}
 card_source=CARD_SOURCE/VISUAL;card_source=card_source.with_suffix('.png')
 card_provenance=CARD_SOURCE/'provenance.json';card_frames=read(CARD_SOURCE/'frames.json');assert set(card_frames)=={VISUAL} and sha(card_source)==card_frames[VISUAL]['sha256']
 proof=read(card_provenance);assert proof['author_source_sha256']==sha(ROOT/'tools/tutorials/render_installation_route_cards.py') and proof['style_source_sha256']==sha(ROOT/'tools/tutorials/authoring/tools/render_install_keyframes.py')
 assert 'reference' in proof['scope'].lower() and proof['published'] is False
 card_log=W/'scene07-card-render.log';assert card_log.read_text().rstrip().endswith('Terminal exit status: 0')
 card_acceptance=Path(os.environ['APPROVED_CARD_EVIDENCE']);assert sha(card_acceptance)==os.environ['APPROVED_CARD_SHA256'];approval=read(card_acceptance)
 assert approval['accepted'] is True and approval['image']==bind(card_source) and approval['generator_provenance']==bind(card_provenance) and approval['render_log']==bind(card_log)
 catalogs={p.name:read(p) for p in sorted((BASE/'web/catalog').glob('*.json'))};assert len(catalogs)==14
 english=read(ROOT/'tools/tutorials/lessons'/f'{LESSON}.json');published=next(x for x in catalogs['lessons_en.json']['lessons'] if x['id']==LESSON);canonical=copy.deepcopy(published);inventory=canonical.pop('narration_voices');assert canonical==english and sum(map(len,inventory.values()))==50
 specs={name:read(original/name) for name in ['visual.json','scenes.json']}
 inputs={}
 for spec in specs.values():
  for row in spec['scenes']:
   for rel in [row['image'],*[row['clip'][k] for k in ('video','receipt') if row.get('clip')]]:
    p=(original/rel).resolve();assert p.is_relative_to(ORIGINAL/'captures') and p.is_file();inputs[p]=STAGE/p.relative_to(ORIGINAL)
   assert sha(original/row['image'])==row['capture_sha256']
   if row.get('clip'):
    assert sha(original/row['clip']['video'])==row['clip']['sha256'] and sha(original/row['clip']['receipt'])==row['clip']['receipt_sha256']
 for lang in ['da','de','es','fr','hi','is','it','ja','ko','nb','pt-BR','sv','zh-CN']:
  p=original/f'review.{lang}.json';assert p.read_bytes()==(ROOT/'tools/tutorials/lessons/reviews'/f'{LESSON}.{lang}.json').read_bytes()
 # Read-only retention checks complete before writes; full originals remain untouched.
 STAGE.mkdir(parents=True)
 for name in catalogs:cp(BASE/'web/catalog'/name,STAGE/'catalog'/name)
 from retain_narration import retained_sources
 retained_sources(STAGE,ORIGINAL,BASE/'web/catalog',LESSON,inventory,require_staged=False)
 for lang,voices in inventory.items():
  for voice in voices:
   for suffix in ['.m4a','.json']:
    rel=Path('audio')/lang/(voice+suffix);record=records[(Path('media_host')/LESSON/rel).as_posix()]
    assert sha(baseline/rel)==record['sha256'] and (baseline/rel).stat().st_size==record['bytes']
    cp(baseline/rel,STAGE/PROD/rel)
 retained=retained_sources(STAGE,ORIGINAL,BASE/'web/catalog',LESSON,inventory,require_staged=True)
 for name in ['lesson.en.json','visual.json','scenes.json',*[f'review.{x}.json' for x in ['da','de','es','fr','hi','is','it','ja','ko','nb','pt-BR','sv','zh-CN']]]:cp(original/name,STAGE/PROD/name)
 for src,dst in inputs.items():cp(src,dst)
 carddir=STAGE/'captures/scene7_command_reference'
 for name in [VISUAL+'.png','frames.json','provenance.json']:cp(CARD_SOURCE/name,carddir/name)
 frames=read(carddir/'frames.json');assert set(frames)=={VISUAL};image=carddir/frames[VISUAL]['image'];assert sha(image)==frames[VISUAL]['sha256'];generator=bind(carddir/'provenance.json')
 new_origin={'kind':'reference_guidance','capture':str(carddir),'source_visual':VISUAL,'image_sha256':sha(image),'installed_identity':None,'representation':'generated_command_reference_not_terminal_output','generator_provenance':generator}
 old_origin=copy.deepcopy(specs['scenes.json']['capture_source']['frame_origins'][VISUAL])
 for name,spec in specs.items():
  spec['scenes'][6]['image']='../../captures/scene7_command_reference/'+image.name;spec['scenes'][6]['capture_sha256']=sha(image)
  spec['capture_source']['frame_origins'][VISUAL]=copy.deepcopy(new_origin)
  spec['capture_source']['completed_capture']=False;spec['capture_source']['frame_pixels_modified']=True
  (STAGE/PROD/name).write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
 write_new(STAGE/PROD/'retained-narration.json',retained)
 write_new(STAGE/PROD/'selective-stage-preparation.json',{'schema':1,'lesson':LESSON,'source_commit':os.environ['SOURCE_COMMIT'],'baseline_manifest':bind(BASE/'release-manifest.json'),'original_spec':bind(original/'scenes.json'),'original_timing':bind(baseline/'video/native-live-timings.json'),'original_master':bind(baseline/'video'/f'{LESSON}_silent.mp4'),'new_card':bind(image),'card_acceptance':bind(card_acceptance),'card_render_log':bind(card_log),'original_origin':old_origin,'new_origin':new_origin,'audio_sidecar_files_retained':100,'catalogs_retained':14,'native_Home_source_assets_unchanged':True,'scope':'Only generated scene7 command reference, not an installation or fresh native capture.'})
 print('Private scene7 stage ready; no master rendered',flush=True)

def overlay():
 common();assert (STAGE/PROD/'selective-stage-preparation.json').is_file()
 from retained_visual_refresh import overlay_command
 prep=read(STAGE/PROD/'selective-stage-preparation.json');root=STAGE/PROD;output=root/'video'/f'{LESSON}_silent.mp4';assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
 master=Path(prep['original_master']['path']);card=Path(prep['new_card']['path']);assert bind(master)==prep['original_master'] and bind(card)==prep['new_card']
 command=overlay_command(master,card,output,4420,692,12901,30)
 log=root/'video/scene7-overlay.log'
 with log.open('x') as f:
  rc=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode;f.write('\nTerminal exit status: '+str(rc)+'\n')
 assert rc==0,'Preserve failed render and log; never manufacture successful proof'
 renderlog=root/'video/scene7-overlay-command.json';write_new(renderlog,{'command':command,'returncode':rc,'output_log':bind(log)})
 receipt={'schema':1,'kind':'retained_visual_refresh','lesson':LESSON,'baseline_root':str(BASE),'original_root':str(ORIGINAL/PROD),'baseline_manifest':bind(BASE/'release-manifest.json'),'publication_receipt':bind(BASE/'publication-receipt.json'),'original_spec':prep['original_spec'],'original_timing':prep['original_timing'],'original_master':prep['original_master'],'new_spec':bind(root/'scenes.json'),'new_master':bind(output),'render_log':bind(renderlog),'replacements':[{'scene':7,'first_frame':4420,'frame_count':692,'original_image_sha256':read(ORIGINAL/PROD/'scenes.json')['scenes'][6]['capture_sha256'],'new_image_sha256':sha(card),'original_origin':prep['original_origin'],'new_origin':prep['new_origin']}]}
 receiptpath=root/'retained-visual-refresh.json';write_new(receiptpath,receipt)
 timing=copy.deepcopy(read(Path(prep['original_timing']['path'])));timing['visual_spec_sha256']=receipt['new_spec']['sha256'];timing['retained_visual_refresh']={'receipt_sha256':sha(receiptpath),'baseline_manifest_sha256':BASE_SHA}
 write_new(root/'video/native-live-timings.json',timing)
 print(json.dumps({'receipt':bind(receiptpath),'timing':bind(root/'video/native-live-timings.json'),'master':bind(output)}),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['stage','overlay']);args=parser.parse_args();{'stage':stage,'overlay':overlay}[args.phase]()
