#!/usr/bin/env python3
"""Stage accepted full-native Home + actual doctor reference; root runs after review."""
from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'tools/tutorials')]
sys.meta_path[:]=[f for f in sys.meta_path if '__editable__' not in (getattr(f,'__module__','') or type(f).__module__)]
import argparse,copy,hashlib,json,os,shutil,subprocess
W=Path(__file__).resolve().parent;N=Path('/media/carruthers/mnt3/claude/n676_stages')
BASE=N/'host/stage-02-selective-r2-20261015/release-candidate-append-lxmwr53e';ORIGINAL=N/'host/stage-04-r23'
STAGE=N/'host/stage-04-selective-20261015';LESSON='04_platform_installers';PROD=Path('production')/LESSON
BASE_SHA='98d88a909b3eeaf14f829297d965461e62a07dff7dfd25ce50d2c2bc15a25439'
LANGS=['da','de','es','fr','hi','is','it','ja','ko','nb','pt-BR','sv','zh-CN']
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def checked(b):
 p=Path(b['path']);assert p.is_absolute() and not p.is_symlink() and bind(p)==b;return p

def write_new(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def cp(a,b):
 assert not a.is_symlink() and a.is_file();b.parent.mkdir(parents=True,exist_ok=True)
 with a.open('rb') as f,b.open('xb') as g:shutil.copyfileobj(f,g)
 assert a.stat().st_size==b.stat().st_size and sha(a)==sha(b)
def common():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==os.environ['SOURCE_COMMIT']
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
 assert sha(BASE/'release-manifest.json')==BASE_SHA
 assert sha(W/'root-component-decision-v2.json')=='e6cd05d93c47999c2cdfdd7dadbd4ed9f7f64941fa8ca106845b3febd0d2456c'
 assert sha(W/'root-native-duration-policy.json')=='ff59b6019f8a39aebc8013c7227decc395a67202ee86d31154485e03e85abd9b'

def stage():
 common();assert not STAGE.exists(), 'Preserve any previous private stage'
 from retained_visual_refresh import _checked_home
 original=ORIGINAL/PROD;baseline=BASE/'media_host'/LESSON
 old=read(original/'scenes.json');timing=read(baseline/'video/native-live-timings.json')
 assert sha(original/'scenes.json')==timing['visual_spec_sha256']=='f2aa93a88325b83a8bd854d99b65021e9705fe5ac03e8f13659c8be1a52242d7'
 assert sha(original/'video/native-live-timings.json')==sha(baseline/'video/native-live-timings.json')=='1d966853b48b82e6b6f0440e7a5936d70764d1edb82748ef1eee9de10ec79f63'
 assert sha(original/'video'/f'{LESSON}_silent.mp4')==sha(baseline/'video'/f'{LESSON}_silent.mp4')=='71fc4905d51039d5ccef673c0c7c6030f47c4b377ba670a899766d8b62a28e41'
 assert len(old['scenes'])==len(timing['scenes'])==14 and round(timing['total_duration']*30)==10007
 # Future actual root native acceptance is mandatory and never authored here.
 ap=Path(os.environ['ACCEPTED_NATIVE_EVIDENCE']).resolve();assert sha(ap)==os.environ['ACCEPTED_NATIVE_SHA256'];approval=read(ap)
 assert approval['accepted'] is True and approval['source_scene']=='13_current_nightly_home' and approval['source_commit']==os.environ['SOURCE_COMMIT'] and approval['requested_minimum_seconds']==28
 raw=checked(approval['video']).parent;assert raw in [N/'host/i04-home-r1/captures/home',N/'host/i04-home-r1-b/captures/home']
 name='13_current_nightly_home';capture_stage=raw.parent.parent
 loc={'video':raw/(name+'.mp4'),'native_video':raw/(name+'.native.mp4'),'receipt':raw/(name+'.capture.json'),'frames_manifest':raw/'frames.json','gate':raw/(name+'.gate.json'),'facts':raw/(name+'.host.json'),'pointer':raw/(name+'.pointer.json'),'poster':raw/(name+'.png'),'host_run':capture_stage/'host-run.json','provenance':raw/'provenance.json','cleanup':capture_stage/'gnome-host/cleanup.json','profile':capture_stage/'capture-profile.json','exact_source':capture_stage/'exact-source.json'}
 native={'capture_root':str(raw),'source_commit':approval['source_commit'],**{k:bind(p) for k,p in loc.items()}}
 count=len(read(loc['receipt'])['frames']);assert type(count) is int and count>=840 and approval['actual_native_frames']==count
 native_row={'scene':9,'kind':'strict_full_native_clip','first_frame':5111,'frame_count':count,'original_image_sha256':old['scenes'][8]['capture_sha256'],'new_image_sha256':sha(loc['poster']),'original_origin':copy.deepcopy(old['capture_source']['frame_origins'][name]),'new_video_sha256':sha(loc['video']),'new_receipt_sha256':sha(loc['receipt']),'native':native,'native_visual_acceptance':bind(ap)}
 raw_clip={'video':str(loc['video']),'sha256':sha(loc['video']),'receipt':str(loc['receipt']),'receipt_sha256':sha(loc['receipt']),'duration':count/30}
 native_row['new_origin']=_checked_home(native_row,loc['poster'],raw_clip,Path('/'))
 card_accept=W/'scene10-doctor-card-visual-acceptance.json';assert sha(card_accept)=='0c1efb1d98a00577558ba114c104cb19fddaadfc8c785372ee49d77d5577aee2';cardapproval=read(card_accept);assert cardapproval['accepted'] is True
 image=checked(cardapproval['image']);card_source=image.parent;provenance=checked(cardapproval['generator_provenance']);generator=read(provenance)
 assert generator['author_source_sha256']==sha(ROOT/'tools/tutorials/render_installation_route_cards.py') and generator['style_source_sha256']==sha(ROOT/'tools/tutorials/authoring/tools/render_install_keyframes.py')
 render=read(checked(cardapproval['owning_render_receipt']));assert render['terminal_exit_status']==0
 history=Path(generator['verification_capture'])/'commands.json';assert history.is_file()
 catalogs={p.name:read(p) for p in (BASE/'web/catalog').glob('*.json')};assert len(catalogs)==14
 english=read(ROOT/'tools/tutorials/lessons'/f'{LESSON}.json');pub=next(x for x in catalogs['lessons_en.json']['lessons'] if x['id']==LESSON);canonical=copy.deepcopy(pub);inventory=canonical.pop('narration_voices');assert canonical==english and sum(map(len,inventory.values()))==50
 specs={n:read(original/n) for n in ['scenes.json','visual.json']};inputs={}
 for spec in specs.values():
  for row in spec['scenes']:
   assert sha(original/row['image'])==row['capture_sha256']
   for rel in [row['image'],*[row['clip'][k] for k in ['video','receipt'] if row.get('clip')]]:
    src=(original/rel).resolve();assert src.is_relative_to(ORIGINAL/'captures') and not src.is_symlink();inputs[src]=STAGE/src.relative_to(ORIGINAL)
   if row.get('clip'):
    assert sha(original/row['clip']['video'])==row['clip']['sha256'] and sha(original/row['clip']['receipt'])==row['clip']['receipt_sha256']
 for lang in LANGS:assert (original/f'review.{lang}.json').read_bytes()==(ROOT/'tools/tutorials/lessons/reviews'/f'{LESSON}.{lang}.json').read_bytes()
 assert read(original/'lesson.en.json')==english
 # Real capture/card/source prechecks complete. Never overwrite originals or an earlier stage.
 STAGE.mkdir()
 for n in catalogs:cp(BASE/'web/catalog'/n,STAGE/'catalog'/n)
 from retain_narration import retained_sources
 retained_sources(STAGE,ORIGINAL,BASE/'web/catalog',LESSON,inventory,require_staged=False)
 records={x['path']:x for x in read(BASE/'release-manifest.json')['files']}
 for lang,voices in inventory.items():
  for voice in voices:
   for suffix in ['.m4a','.json']:
    rel=Path('audio')/lang/(voice+suffix);record=records[(Path('media_host')/LESSON/rel).as_posix()];src=baseline/rel
    assert sha(src)==record['sha256'] and src.stat().st_size==record['bytes'];cp(src,STAGE/PROD/rel)
 retained=retained_sources(STAGE,ORIGINAL,BASE/'web/catalog',LESSON,inventory,require_staged=True)
 for n in ['lesson.en.json',*[f'review.{lang}.json' for lang in LANGS]]:cp(original/n,STAGE/PROD/n)
 for src,dst in inputs.items():cp(src,dst)
 fresh=STAGE/'captures/fresh_home'
 for key in ['video','receipt','poster']:cp(loc[key],fresh/loc[key].name)
 carddir=STAGE/'captures/doctor_command_reference'
 for n in [image.name,'frames.json','provenance.json']:cp(card_source/n,carddir/n)
 card=carddir/image.name
 card_origin={'kind':'reference_guidance','capture':str(carddir),'source_visual':'05_doctor','image_sha256':sha(card),'installed_identity':None,'representation':'generated_command_reference_not_terminal_output','generator_provenance':bind(carddir/'provenance.json')}
 doctor_row={'scene':10,'kind':'generated_command_reference_not_terminal_output','first_frame':5111+count,'frame_count':879,'original_image_sha256':old['scenes'][9]['capture_sha256'],'new_image_sha256':sha(card),'original_origin':copy.deepcopy(old['capture_source']['frame_origins']['05_doctor']),'new_origin':card_origin,'card_visual_acceptance':bind(card_accept),'historical_doctor':bind(history)}
 for n,spec in specs.items():
  spec['scenes'][8].update(image='../../captures/fresh_home/'+loc['poster'].name,capture_sha256=sha(loc['poster']),clip={'video':'../../captures/fresh_home/'+loc['video'].name,'sha256':sha(loc['video']),'receipt':'../../captures/fresh_home/'+loc['receipt'].name,'receipt_sha256':sha(loc['receipt']),'duration':count/30})
  spec['scenes'][9].update(image='../../captures/doctor_command_reference/'+card.name,capture_sha256=sha(card))
  spec['capture_source']['frame_origins'][name]=copy.deepcopy(native_row['new_origin']);spec['capture_source']['frame_origins']['05_doctor']=copy.deepcopy(card_origin)
  spec['capture_source'].update(completed_capture=False,frame_pixels_modified=True);write_new(STAGE/PROD/n,spec)
 write_new(STAGE/PROD/'retained-narration.json',retained)
 write_new(STAGE/PROD/'selective-stage-preparation.json',{'schema':2,'lesson':LESSON,'source_commit':os.environ['SOURCE_COMMIT'],'baseline_manifest':bind(BASE/'release-manifest.json'),'original_spec':bind(original/'scenes.json'),'original_timing':bind(baseline/'video/native-live-timings.json'),'original_master':bind(baseline/'video'/f'{LESSON}_silent.mp4'),'home':bind(fresh/loc['video'].name),'card':bind(card),'native_frames':count,'replacements':[native_row,doctor_row],'scope':'Only two approved visuals; whole actual nativeHome and generated reference, no new installation/doctor execution.'})
 print('Accepted-source mixed stage prepared; no master rendered',flush=True)

def render():
 common()
 from retained_visual_refresh import mixed_overlay_command,mixed_retained_timing
 root=STAGE/PROD;prep=read(root/'selective-stage-preparation.json');assert prep['source_commit']==os.environ['SOURCE_COMMIT']
 master=checked(prep['original_master']);home=checked(prep['home']);card=checked(prep['card']);count=prep['native_frames'];output=root/'video'/f'{LESSON}_silent.mp4';assert not output.exists();output.parent.mkdir(exist_ok=True)
 command=mixed_overlay_command(master,home,card,output,count);log=root/'video/mixed-render.log'
 with log.open('x') as f:
  rc=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode;f.write('\nTerminal exit status: '+str(rc)+'\n')
 assert rc==0,'Preserve failed render/log; no successful proof emitted'
 renderlog=root/'video/mixed-render-command.json';write_new(renderlog,{'command':command,'returncode':rc,'output_log':bind(log)})
 receipt={'schema':2,'kind':'retained_visual_refresh','lesson':LESSON,'baseline_root':str(BASE),'original_root':str(ORIGINAL/PROD),'policy':bind(W/'root-native-duration-policy.json'),'baseline_manifest':prep['baseline_manifest'],'publication_receipt':bind(BASE/'publication-receipt.json'),'original_spec':prep['original_spec'],'original_timing':prep['original_timing'],'original_master':prep['original_master'],'new_spec':bind(root/'scenes.json'),'new_master':bind(output),'render_log':bind(renderlog),'replacements':prep['replacements']}
 rp=root/'retained-visual-refresh.json';write_new(rp,receipt)
 new=mixed_retained_timing(read(checked(prep['original_timing'])),receipt['new_spec']['sha256'],sha(rp),BASE_SHA,receipt['replacements'][0]['new_video_sha256'],receipt['replacements'][0]['new_receipt_sha256'],count)
 write_new(root/'video/native-live-timings.json',new)
 print(json.dumps({'receipt':bind(rp),'timing':bind(root/'video/native-live-timings.json'),'master':bind(output),'whole_native_frames':count,'total_frames':9167+count}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['stage','render']);a=p.parse_args();{'stage':stage,'render':render}[a.phase]()
