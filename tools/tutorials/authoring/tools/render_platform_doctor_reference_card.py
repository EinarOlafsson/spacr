"""Render one CPU diagnostic reference from unchanged accepted historical results."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
VISUALS=('08_official_release','09_windows_guidance','10_macos_guidance','11_linux_commands','02_installer_backend','03_installed_versions','07_privacy_keep_off','14_current_nightly_performance','13_current_nightly_home','05_doctor','12_logs_and_versions','15_uninstall_windows','16_uninstall_macos','17_uninstall_linux')
CARD='05_doctor'
MASTER_SHA='eb353f10bd81dbcba6f7f76761ccc286de7cc6d1575dc3155a45bec54d486d40'
SOURCE_HASHES={'accepted-scene-spec.json':'b925219b304ba1fa615aeeca98a5a169150b5e6018a24becf28e9fbca66e958e',
 'accepted-card.png':'af843fc43126a768f6690d41756a037287602a064dfb8c8b306e7e666cbaed1a',
 'verification-provenance.json':'bf3db0c7c0ab6f1b3626fbd4d972b8cb873cbb3dda4e59f856c54697de239b7d',
 'verification-commands.json':'e5ef36773218ea4f8fdd8d86bfd6521d62bd3a72bc77fdb70fd2bc8863d177ca'}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def checked_sources(paths):
 if set(paths)!=set(SOURCE_HASHES):raise ValueError('Require original accepted card/spec and actual diagnostic evidence')
 for name,path in paths.items():
  path=Path(path)
  if path.is_symlink() or not path.is_file() or digest(path)!=SOURCE_HASHES[name]:raise ValueError('Accepted Doctor source changed: '+name)
 spec=json.loads(Path(paths['accepted-scene-spec.json']).read_text());frame=spec['scenes'][9]
 origin=spec['capture_source']['frame_origins'][CARD]
 originals=[r for r in spec['capture_source']['source_captures'] if r['capture']==origin['capture']]
 if (frame['capture_sha256']!=SOURCE_HASHES['accepted-card.png'] or origin['image_sha256']!=frame['capture_sha256']
     or origin['kind']!='reference_guidance' or len(originals)!=1
     or originals[0]['provenance']['verification_sha256']!=SOURCE_HASHES['verification-provenance.json']):raise ValueError('Original generated card lineage differs')
 proof=json.loads(Path(paths['verification-provenance.json']).read_text());commands=json.loads(Path(paths['verification-commands.json']).read_text())
 rows=[r for r in commands if r.get('scene')==CARD]
 if (proof.get('completed_capture') is not True or proof.get('verification_recorded_live') is not True
     or proof.get('doctor_gpu_allocation_probe_requested') is not False
     or proof.get('installed_identity',{}).get('version')!='1.5.1.3' or len(rows)!=1):raise ValueError('Actual historical Doctor verification missing')
 row=rows[0];text=row.get('output','')
 if (row.get('command')!=['spacr-doctor','--no-gpu-probe'] or row.get('accepted') is not True or row.get('returncode')!=0
     or not all(part in text for part in ('13 passed, 1 warnings, 0 failed, 0 errored, 4 skipped','missing pylibCZIrw','torch currently uses cpu','no GPU probe was run'))):raise ValueError('Actual Doctor command/results differ')
 return proof

def checked_lesson(lesson):
 if lesson.get('id')!='04_platform_installers' or [s.get('visual') for s in lesson.get('scenes',[])]!=list(VISUALS):raise ValueError('Require exact current fourteen-scene Platform Installers')

def validate_cards(folder,lesson,repo):
 from PIL import Image
 folder,repo=Path(folder),Path(repo);checked_lesson(lesson)
 if folder.is_symlink():raise ValueError('Independent reference directory required')
 if (folder/'source-evidence').is_symlink() or not (folder/'source-evidence').is_dir():raise ValueError('Independent saved source evidence required')
 for name in ('cards.json','frames.json','provenance.json'):
  if (folder/name).is_symlink():raise ValueError('Reference metadata must not be a symlink')
 checked_sources({name:folder/'source-evidence'/name for name in SOURCE_HASHES})
 contract=json.loads((folder/'cards.json').read_text())
 refs=[dict(name=n,sha256=h) for n,h in sorted(SOURCE_HASHES.items())]
 header=dict(schema=1,lesson=lesson['id'],english_sha256=identity(lesson),generator_sha256=digest(__file__),renderer_sha256=digest(repo/'tools/tutorials/render_installation_route_cards.py'),style_helper_sha256=digest(repo/'tools/tutorials/authoring/tools/render_install_keyframes.py'),source_master_sha256=MASTER_SHA,source_evidence=refs)
 if {k:v for k,v in contract.items() if k!='cards'}!=header:raise ValueError('Doctor reference source bindings differ')
 expected_image=folder/(CARD+'.png');authored=lesson['scenes'][9]
 expected=dict(visual=CARD,image=expected_image.name,sha256=digest(expected_image),kind='authored_reference_card',narration_sha256=identity(authored['narration']),references=refs)
 if contract.get('cards')!=[expected] or expected_image.is_symlink():raise ValueError('Doctor reference image/narration/source differs')
 frames=json.loads((folder/'frames.json').read_text());generated=json.loads((folder/'provenance.json').read_text())
 if (frames!={CARD:dict(image=CARD+'.png',sha256=expected['sha256'],buttons=[],kind='generated_command_reference_not_terminal_output')}
     or generated['verification_sha256']!=SOURCE_HASHES['verification-provenance.json']
     or generated['author_source_sha256']!=header['renderer_sha256']
     or generated['style_source_sha256']!=header['style_helper_sha256']):raise ValueError('Normal generated-card provenance differs')
 with Image.open(expected_image) as frame:
  if frame.size!=(3840,2160) or frame.format!='PNG':raise ValueError('Doctor card must be a full4K PNG')
  frame.verify()
 return {CARD:expected}

def render(repo,lesson,verification,accepted_spec,accepted_card,destination):
 import shutil
 repo,verification,destination=Path(repo),Path(verification),Path(destination);checked_lesson(lesson)
 sources={'accepted-scene-spec.json':Path(accepted_spec),'accepted-card.png':Path(accepted_card),
          'verification-provenance.json':verification/'provenance.json','verification-commands.json':verification/'commands.json'}
 checked_sources(sources)
 if destination.exists() or destination.is_symlink():raise FileExistsError('Preserve existing reference output')
 sys.path.insert(0,str(repo/'tools/tutorials'))
 tool=repo/'tools/tutorials/render_installation_route_cards.py'
 spec=importlib.util.spec_from_file_location('normal_doctor_reference_renderer',tool);renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
 expected=('05_doctor','spaCR Doctor: CPU diagnostic example',[
  ('Run from the installed spaCR environment','spacr-doctor --no-gpu-probe'),
  ('Example diagnostic summary','13 passed | 1 warning | 0 failed | 0 errored | 4 skipped'),
  ('Backend and optional file-format reader','CPU passed | GPU probe skipped | Optional CZI reader missing')],
  'CPU diagnostic example: GPU probe skipped. Install the CZI extra only if you need CZI files.')
 if renderer.selected_cards([CARD])!=[expected]:raise ValueError('Reviewed Doctor wording/values differ')
 before=sys.argv
 try:
  sys.argv=[str(tool),'--verification',str(verification),'--destination',str(destination),'--only',CARD];renderer.main()
 finally:sys.argv=before
 evidence=destination/'source-evidence';evidence.mkdir()
 for name,path in sources.items():shutil.copy2(path,evidence/name)
 checked_sources({name:evidence/name for name in sources})
 refs=[dict(name=n,sha256=h) for n,h in sorted(SOURCE_HASHES.items())]
 card=dict(visual=CARD,image=CARD+'.png',sha256=digest(destination/(CARD+'.png')),kind='authored_reference_card',narration_sha256=identity(lesson['scenes'][9]['narration']),references=refs)
 contract=dict(schema=1,lesson=lesson['id'],english_sha256=identity(lesson),generator_sha256=digest(__file__),renderer_sha256=digest(tool),style_helper_sha256=digest(repo/'tools/tutorials/authoring/tools/render_install_keyframes.py'),source_master_sha256=MASTER_SHA,source_evidence=refs,cards=[card])
 (destination/'cards.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n');validate_cards(destination,lesson,repo)
 print('One historical CPU diagnostic reference; original command/results retained, no new Doctor run',flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('repo','lesson','verification','accepted-spec','accepted-card','destination'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();render(a.repo,json.loads(a.lesson.read_text()),a.verification,a.accepted_spec,a.accepted_card,a.destination)
if __name__=='__main__':main()
