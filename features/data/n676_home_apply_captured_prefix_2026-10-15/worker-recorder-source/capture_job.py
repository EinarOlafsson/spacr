"""Owned native phase, never called during preparation. Runs only the approved single-clip comparison."""
from pathlib import Path
import sys,os,argparse,subprocess,json,hashlib,signal,time
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--revision');p.add_argument('--cleanup-only',action='store_true');p.add_argument('--check-only',action='store_true');args=p.parse_args()
R=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014');W=Path(__file__).resolve().parent;S=args.stage.resolve()
sys.path[:0]=[str(R),str(R/'tools/tutorials')]
assert S.parent==Path('/media/carruthers/mnt3/claude/n676_stages/host') and S.name=='lesson05-worker-apply-r1'
if args.cleanup_only:
    from gnome_capture_host import terminate_marked
    survivors=terminate_marked(str(S/'gnome-host')) if (S/'gnome-host').is_dir() else []
    if S.is_dir():(S/'owner-cleanup.json').write_text(json.dumps({'survivors_after_cleanup':survivors,'scope':'Existing stage-marked owned cleanup; not a diagnostic success'},indent=2)+'\n')
    raise SystemExit(3 if survivors else 0)
assert args.revision and len(args.revision)==40 and subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==args.revision
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True).strip()
assert args.revision == '87ffcb60cab0eca8772cb1ffaccc12f3f4f48be9'
approval_path=Path(os.environ['SPACR_WORKER_APPLY_APPROVAL_PATH']).resolve()
approval_bytes=approval_path.read_bytes()
assert hashlib.sha256(approval_bytes).hexdigest()==os.environ['SPACR_WORKER_APPLY_APPROVAL_SHA256']
approval=json.loads(approval_bytes)
required={'schema':1,'source_commit':args.revision,'root_launch_approved':True,'recording_comparison_authorized':True,'only_scene':'13h_preferences_apply','execution_timeout_seconds':300,'cleanup_grace_seconds':30,'automatic_retry':False}
assert all(type(approval[k]) is type(v) and approval[k]==v for k,v in required.items())
assert not S.exists()
if args.check_only:
    print('Source and explicit owner supersession/root approval verified; no diagnostic launched.')
    raise SystemExit(0)

assert os.environ.get('DISPLAY') not in (':0',':0.0')
S.mkdir(exist_ok=False)
for name in ['profile','app-state','config/home','state/home','cache','logs','mpl','tmp','example_data']:(S/name).mkdir(parents=True,exist_ok=True)
env=dict(os.environ)
for key in ['QT_OPENGL','LIBGL_ALWAYS_SOFTWARE','MESA_LOADER_DRIVER_OVERRIDE','GALLIUM_DRIVER','__EGL_VENDOR_LIBRARY_FILENAMES','__GLX_VENDOR_LIBRARY_NAME','GLCONTEXT_DEVICE_INDEX','SPACR_COMPOSITE_CARDS']:
    env.pop(key,None)
env.update(HOME=str(S/'profile'),SPACR_HOME=str(S/'app-state'),XDG_CONFIG_HOME=str(S/'config/home'),XDG_STATE_HOME=str(S/'state/home'),XDG_CACHE_HOME=str(S/'cache'),SPACR_LOG_DIR=str(S/'logs'),SPACR_NEWS_CACHE=str(S/'cache/news'),MPLCONFIGDIR=str(S/'mpl'),TMPDIR=str(S/'tmp'),QT_QPA_PLATFORM='xcb',QT_SCALE_FACTOR='1',QT_AUTO_SCREEN_SCALE_FACTOR='0',QT_FONT_DPI='96',SPACR_LANGUAGE='en',SPACR_NO_SPLASH='1',SPACR_TUTORIAL_CACHE_ISOLATED='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')

peer=Path('/media/carruthers/mnt3/claude/n676_stages/work/lesson05-r59/apply-diagnosis/Home-native-diagnostic-proposal-source-peer.json');peer_sha=hashlib.sha256(peer.read_bytes()).hexdigest();d=json.loads(peer.read_text());assert d['all11_hook_bodies_and_source_segments_identical'] is True
# Recorded original five app bodies are the compatibility anchor; never restamp.
loaded=[]
for row in d['five_files']:
    path=R/row['path'];h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==row['D746_sha256'];loaded.append({'path':row['path'],'sha256':h})
(S/'source-identity.json').write_text(json.dumps({'checkout_commit':args.revision,'original_capture_app_source':'d7461e54ed5325997f6ff1328b90bd2818329f4d','five_app_files':loaded,'hook_peer_sha256':peer_sha,'original_capture_manifest_sha256':d['source_manifest']['sha256'],'original_capture_profile_sha256':d['profile']['sha256'],'limits':'Later checkout may add only owner-reviewed non-app changes. No hook or original source identity restamped.'},indent=2)+'\n')
# The same owning isolated profile helper as recording; no settings shadow or CPU backend override.
prepare=[sys.executable,str(R/'tools/tutorials/capture_host_mode.py'),'--prepare-profile','--font-scale','1','--out',str(S/'capture-profile.json')]
with (S/'logs/profile-private.log').open('x') as f:subprocess.run(prepare,cwd=R,env={**env,'QT_QPA_PLATFORM':'offscreen'},stdout=f,stderr=subprocess.STDOUT,check=True)
profile=json.loads((S/'capture-profile.json').read_text())
known={'theme':'dark','field':'data_art_impulse_lens','field_density':.25,'popup_backdrop':'off','pane_opacity_pct':60,'font_scale':1.0,'spaceout':False,'show_alpha':False,'show_alpha_features':False,'show_alpha_species':False,'anonymous_reports_enabled':False,'anonymous_report_notice_due':False}
assert all(profile[k]==v for k,v in known.items())

# Native rendering policy is identical to capture: NVIDIA compositor/default app renderer.
command=[sys.executable,str(R/'tools/tutorials/gnome_capture_host.py'),'--stage',str(S/'gnome-host'),'--size','2560x1440','--refresh','60','--settle','3','--',sys.executable,str(W/'capture_driver.py'),'--stage',str(S),'--repo',str(R),'--peer',str(peer),'--peer-sha256',peer_sha]
signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
with (S/'logs/native-private.log').open('x') as f:code=subprocess.run(command,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=False).returncode
assert all(hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'] for row in loaded)
raise SystemExit(code)
