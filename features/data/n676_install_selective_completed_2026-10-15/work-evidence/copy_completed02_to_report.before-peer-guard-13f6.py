#!/usr/bin/env python3
"""Root-run checked generated-docs/packet copy; never fetch/rebase/stage/commit/push."""
from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import argparse,hashlib,json,shutil,subprocess
W=Path(__file__).resolve().parent
REPORT=Path('/media/carruthers/mnt3/claude/spacr-lesson20-report-20261013')
REL=Path('docs/source/_extra/tutorials')
PACKET_REL=Path('features/data/n676_install_selective_completed_2026-10-15')
PEER_REL=Path('features/data/n676_install_selective_completed_2026-10-15-peer.json')
K3='14ecf9d1054e15194bb7b9ff01779af061fae875'
def git(repo,*args):return subprocess.check_output(['git',*args],cwd=repo)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def inventory(folder):
 assert folder.is_dir() and not folder.is_symlink()
 result={}
 for p in sorted(folder.rglob('*')):
  assert not p.is_symlink(), 'Refuse all symlinks, including dotfiles'
  if p.is_file():result[p.relative_to(folder).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
  else:assert p.is_dir(), 'Refuse special files'
 return result
def checked_copy(src,dst,row):
 assert not src.is_symlink() and src.stat().st_size==row['bytes'] and sha(src)==row['sha256']
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 assert dst.stat().st_size==row['bytes'] and sha(dst)==row['sha256']
def ignored(paths):
 code=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],cwd=REPORT,input=b'\0'.join(p.encode() for p in paths)+b'\0',stdout=subprocess.PIPE)
 assert code.returncode in (0,1)
 return [p.decode() for p in code.stdout.split(b'\0') if p]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--report-head',required=True);p.add_argument('--origin-head',required=True,help='Actual origin/nightly after root final fetch/rebase')
 p.add_argument('--home-base',required=True,help='Prior reviewed REPORT base for Home path drift comparison')
 p.add_argument('--source-head',required=True,help='Actual ROOT K3 or final docs/checkpoint-only P')
 p.add_argument('--packet-manifest-sha256',required=True)
 p.add_argument('--packet-peer',type=Path,required=True);p.add_argument('--packet-peer-sha256',required=True)
 p.add_argument('--final-review',type=Path,required=True);p.add_argument('--final-review-sha256',required=True)
 p.add_argument('--guard-path',action='append',default=[],help='Optional additional exact progress path root wants checked against fresh origin')
 a=p.parse_args()
 assert git(REPORT,'rev-parse','HEAD').decode().strip()==a.report_head and not git(REPORT,'status','--porcelain').strip()
 assert git(REPORT,'rev-parse','origin/nightly').decode().strip()==a.origin_head
 subprocess.run(['git','merge-base','--is-ancestor',a.origin_head,a.report_head],cwd=REPORT,check=True)
 assert git(ROOT,'rev-parse','HEAD').decode().strip()==a.source_head
 changed=git(ROOT,'diff','--name-only',K3,a.source_head).decode().splitlines()
 assert all(x.startswith(str(REL)+'/') or x.startswith('tools/tutorials/release_candidate/') for x in changed), 'Final source changed implementation/canonical inputs'
 guards=[str(REL),str(PACKET_REL),str(PEER_REL),*a.guard_path]
 assert not git(REPORT,'diff','--name-only',a.home_base,a.origin_head,'--',*guards).strip(), 'Home changed scoped delivery paths; root must reconcile before copying'
 assert not (REPORT/PACKET_REL).exists() and not (REPORT/PEER_REL).exists()
 packet=W/'completed-publication-packet';pm=packet/'packet-manifest.json';assert sha(pm)==a.packet_manifest_sha256
 manifest=json.loads(pm.read_text());assert manifest['lesson']=='02_install_spacr' and manifest['source_commit']==K3
 assert manifest['counts']['checkpoint_snapshot']==148 and manifest['counts']['official_reviews_bound_in_export']==13
 assert sha(a.final_review)==a.final_review_sha256
 final=json.loads(a.final_review.read_text());assert final['lesson']=='02_install_spacr' and final['status']=='pass' and final['source_commit']==K3
 assert manifest['final_publication_review']==bind(a.final_review)
 assert sha(a.packet_peer)==a.packet_peer_sha256
 peer=json.loads(a.packet_peer.read_text());assert peer['status']=='pass'
 def binds_manifest(obj):
  if isinstance(obj,dict):
   if obj.get('sha256')==a.packet_manifest_sha256 and obj.get('path')==str(pm.resolve()):return True
   if obj.get('packet_manifest_sha256')==a.packet_manifest_sha256:return True
   return any(binds_manifest(v) for v in obj.values())
  if isinstance(obj,list):return any(binds_manifest(v) for v in obj)
  return False
 assert binds_manifest(peer), 'Actual independent packet peer must bind exact manifest'
 packet_files=inventory(packet);declared={}
 for row in manifest['files']:
  name=row['path'];rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts and rel.as_posix()==name and name not in declared
  declared[name]={'bytes':row['bytes'],'sha256':row['sha256']}
 assert set(packet_files)==set(declared)|{'packet-manifest.json'} and all(packet_files[k]==v for k,v in declared.items())
 source=ROOT/REL;target=REPORT/REL;source_files=inventory(source);target_files=inventory(target);assert source_files
 tracked={x.decode() for x in git(REPORT,'ls-files','-z','--',str(REL)).split(b'\0') if x}
 for options in [('--others','--exclude-standard'),('--others','--ignored','--exclude-standard')]:
  assert not git(REPORT,'ls-files',*options,'-z','--',str(REL)).strip(b'\0'), 'Refuse user/untracked/ignored generated destination files'
 assert all((REL/n).as_posix() in tracked for n in target_files)
 assert not ignored([(REL/n).as_posix() for n in source_files]), 'No ignored generated-doc file may be silently omitted'
 packet_destinations=[(PACKET_REL/n).as_posix() for n in packet_files]
 force_add=ignored(packet_destinations);assert all(n.endswith('.log') and n in packet_destinations for n in force_add), 'Only declared actual packet log paths may require explicit force-add'
 assert not ignored([str(PEER_REL)])
 obsolete=sorted(set(target_files)-set(source_files));archive=W/'report-copy-obsolete-generated-docs';receipt=W/'report-copy-readback.json'
 assert not archive.exists() and not receipt.exists()
 peer_before=bind(a.packet_peer);review_before=bind(a.final_review)
 # Only now mutate the narrowly approved REPORT destinations. Obsolete tracked docs move externally.
 for name in obsolete:
  dst=archive/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(target/name,dst)
  assert dst.stat().st_size==target_files[name]['bytes'] and sha(dst)==target_files[name]['sha256']
 for name,row in source_files.items():checked_copy(source/name,target/name,row)
 for name,row in packet_files.items():checked_copy(packet/name,REPORT/PACKET_REL/name,row)
 checked_copy(a.packet_peer,REPORT/PEER_REL,peer_before)
 assert inventory(source)==source_files and inventory(target)==source_files and inventory(packet)==packet_files and inventory(REPORT/PACKET_REL)==packet_files
 assert bind(a.packet_peer)==peer_before and bind(a.final_review)==review_before
 assert git(REPORT,'rev-parse','HEAD').decode().strip()==a.report_head and git(ROOT,'rev-parse','HEAD').decode().strip()==a.source_head
 actual_ignored=[x.decode() for x in git(REPORT,'ls-files','--others','--ignored','--exclude-standard','-z','--',str(PACKET_REL)).split(b'\0') if x]
 assert set(actual_ignored)==set(force_add), 'No silently omitted or undeclared ignored paths'
 report={'schema':1,'status':'checked_copy_completed_no_stage_commit_push','lesson':'02_install_spacr','report_head':a.report_head,'fetched_origin_head':a.origin_head,'home_base':a.home_base,'root_source_head':a.source_head,'packet_manifest_sha256':a.packet_manifest_sha256,'final_review':review_before,'packet_peer':peer_before,'adjacent_peer_path':str(REPORT/PEER_REL),'docs_files':source_files,'packet_files':packet_files,'obsolete_generated_archived':{n:target_files[n] for n in obsolete},'archive':str(archive),'force_add_whitelist':{'bound_packet_manifest_sha256':a.packet_manifest_sha256,'paths':sorted(force_add),'instruction':'Root must explicitly git add -f only these exact declared log paths; verify no declared file omitted before committing.'},'limits':['Copy only exact generated tutorial tree/new selective02 packet/adjacent actual peer; preserve Home updates and all other files.','No fetching, rebasing, staging, committing, pushing, remote verification or new acceptance performed.']}
 with receipt.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps(bind(receipt)))
if __name__=='__main__':main()
