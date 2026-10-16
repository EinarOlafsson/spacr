"""Export actual recorded06 link-metadata completion; never runs publication."""
from pathlib import Path
import sys
ROOT = Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path.insert(0, str(ROOT))
import argparse, copy, hashlib, json, re, shutil, subprocess

WORK = Path(__file__).resolve().parent
BASE = Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-05-selective-20261015/release-candidate-append-ln3oto4l')
BASE_SHA = 'ced803ebdd5e93a6ada3fead48a3c6595afd77ef05d104dc0bd43e87a210bd2a'
SOURCE = '5430c0ceabbd9cf99a4f40ea3030eefa89c2bd87'
PARENT = '8ab0b4f421baf9e6318c9f9f0d08ccd73241113b'
LESSON = '06_api'
def require(value, message):
    if not value: raise ValueError(message)
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def binding(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))
def checked(row):
    path = Path(row['path']); require(path.is_file() and not path.is_symlink() and binding(path) == row, 'Evidence binding differs')
    return path.resolve()
def records(rows):
    result = {r['path']: dict(bytes=r['bytes'], sha256=r['sha256']) for r in rows}
    require(len(result) == len(rows), 'Duplicate manifest paths'); return result
def safe(relative):
    path = Path(relative); require(not path.is_absolute() and '..' not in path.parts, 'Unsafe copy path'); return path
def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT)
def cases(report, manifest_sha):
    rows = report['ready_playback_cases']
    require(report['passed'] is True and report['manifest_sha256'] == manifest_sha and len(rows) == 84 and
            len({r['lesson'] for r in rows}) == 84 and all(r['passed'] is True for r in rows), 'Actual84 browser cases required')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'component-decision', 'final-review', 'media-reuse-proof', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--phase-log', action='append', required=True, help='role=path; roles build,candidate,placeholder,checkpoint,resume,pages,hosted,record')
    parser.add_argument('--evidence', type=Path, action='append', default=[])
    args = parser.parse_args(); candidate = args.candidate.resolve(strict=True); output = args.output.resolve()
    require(not output.exists() and not output.is_relative_to(candidate) and not output.is_relative_to(BASE), 'Output must be fresh and separate')
    require(git('rev-parse', 'HEAD').decode().strip() == SOURCE, 'Expected actual held06 source')
    subprocess.run(['git','diff','--quiet',SOURCE,'--','spacr'],cwd=ROOT,check=True)
    canonical_path = ROOT/'tools/tutorials/lessons/06_api.json'; canonical = read(canonical_path)
    reviews = sorted((ROOT/'tools/tutorials/lessons/reviews').glob('06_api.*.json')); require(len(reviews) == 13, 'Expected13 official reviews')
    changed = git('diff', '--name-only', PARENT, SOURCE).decode().splitlines()
    require(set(changed) == {str(p.relative_to(ROOT)) for p in reviews}, 'Only13 review metadata source changes allowed')
    require(git('show', PARENT+':'+str(canonical_path.relative_to(ROOT))) == canonical_path.read_bytes(), 'Canonical changed')
    normalized = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    for path in reviews:
        relative = str(path.relative_to(ROOT)); new = read(path); old = json.loads(git('show', PARENT+':'+relative))
        require(git('show', SOURCE+':'+relative) == path.read_bytes() and new['english_sha256'] == normalized, 'Current source review changed')
        old.pop('english_sha256'); new.pop('english_sha256'); require(old == new, 'Review text or other metadata changed')
    manifest_path = candidate/'release-manifest.json'; manifest = read(manifest_path); manifest_sha = digest(manifest_path)
    baseline = read(BASE/'release-manifest.json'); require(digest(BASE/'release-manifest.json') == BASE_SHA and manifest['baseline_manifest_sha256'] == BASE_SHA, 'Wrong C05 baseline')
    require(manifest['link_only_updates'] == [LESSON] and manifest['ready_lessons'] == manifest['routes'] == 84 and
            manifest['preserved_lessons'] == 84 and not manifest['refreshed_lessons'] and not manifest['appended_lessons'] and not manifest['withdrawn_lessons'], 'Wrong metadata-only scope')
    before, after = records(baseline['files']), records(manifest['files'])
    media = {p:r for p,r in after.items() if p.startswith('media_host/')}
    require(media == {p:r for p,r in before.items() if p.startswith('media_host/')} and len(media) == 5370, 'Media changed')
    total = sum(r['bytes'] for r in media.values()); require(total == 8807787387, 'Wrong retained media bytes')
    posters = {p:r for p,r in after.items() if p.startswith('web/production/') and p.endswith('/poster.jpg')}
    require(len(posters) == 84 and posters == {p:r for p,r in before.items() if p in posters}, 'Posters changed')
    audio = [p for p in media if p.startswith('media_host/06_api/audio/')]
    require(sum(p.endswith('.json') for p in audio) == sum(p.endswith('.m4a') for p in audio) == 27, 'Expected retained27 tracks')
    names = sorted(p.name for p in (BASE/'web/catalog').glob('*.json'))
    require(len(names) == 14 and names == sorted(p.name for p in (candidate/'web/catalog').glob('*.json')), 'Catalog set changed')
    for name in names:
        old, new = read(BASE/'web/catalog'/name)['lessons'], read(candidate/'web/catalog'/name)['lessons']
        require(len(old) == len(new) == len({x['id'] for x in new}) == 84 and [x['id'] for x in old] == [x['id'] for x in new], 'Catalog routes changed')
        for a,b in zip(old,new):
            a,b = copy.deepcopy(a),copy.deepcopy(b)
            if a['id'] == LESSON:
                require(b['scenes'][0]['related_lessons'] == ['02_install_spacr','04_platform_installers'], 'Wrong prerequisite links')
                a['scenes'][0].pop('related_lessons',None); b['scenes'][0].pop('related_lessons',None)
            require(a == b, 'Caption text, media metadata or other83 object changed')
    decision, final, reuse = read(args.component_decision), read(args.final_review), read(args.media_reuse_proof)
    require(decision['lesson'] == LESSON and decision['accepted'] is True and decision['source_commit'] == SOURCE and all(decision[k] == 'KEEP' for k in ('video','audio','captions')), 'Actual KEEP decision required')
    require(final['status'] == 'pass' and final['source_commit'] == SOURCE and Path(final['candidate']).resolve() == candidate and final['manifest'] == binding(manifest_path), 'Actual postrecord final review required')
    require(final['component_decision'] == binding(args.component_decision) and final['media_reuse_proof'] == binding(args.media_reuse_proof), 'Final review proof bindings differ')
    require(reuse['status'] == 'pass' and reuse['lesson'] == LESSON and Path(reuse['candidate']).resolve() == candidate and reuse['manifest'] == binding(manifest_path) and Path(reuse['baseline_candidate']).resolve() == BASE, 'Wrong media reuse proof')
    require(reuse['baseline_manifest'] == binding(BASE/'release-manifest.json') and reuse['baseline_publication_receipt'] == binding(BASE/'publication-receipt.json'), 'Wrong immutable baseline proof')
    require(reuse['media_records_equal'] is True and reuse['media_files'] == len(media) and reuse['media_bytes'] == total and reuse['fresh_upload'] is False and reuse['fresh_readback'] is False and reuse['media_map_sha256'] == hashlib.sha256(json.dumps(media,sort_keys=True,separators=(',',':')).encode()).hexdigest(), 'Reuse scope differs')
    old_publication = read(BASE/'publication-receipt.json'); old_readback = read(checked(reuse['reused_readback']))
    require(old_readback == old_publication['readback'], 'Original readback was restamped')
    source = ROOT/'tools/tutorials/release_candidate'; cp = read(source/'checkpoint.json'); publication = read(source/'publication-receipt.json'); hosted = read(source/'published-media-browser-checks.json')
    require(Path(cp['private_candidate']).resolve() == candidate and cp['manifest_sha256'] == publication['manifest_sha256'] == hosted['manifest_sha256'] == manifest_sha and digest(source/'release-manifest.json') == manifest_sha, 'Recorded candidate differs')
    require(cp['published'] is False and cp['release_hold'] is False and cp['media_uploaded'] is True and cp['pages_tree_ready'] is True and cp['browser_checks_complete'] is True, 'Actual predeploy record incomplete')
    require(publication == read(candidate/'publication-receipt.json') and hosted == read(candidate/'checks/published-media-browser-checks.json'), 'Recorded receipts differ')
    require(final['publication_receipt'] == binding(candidate/'publication-receipt.json') and final['recorded_checkpoint'] == binding(source/'checkpoint.json') and final['published_media_browser_checks'] == binding(source/'published-media-browser-checks.json'), 'Final record bindings differ')
    require(publication['readback'] == old_readback and publication['commit'] == reuse['media_commit'] == old_publication['commit'] and publication['branch'] == old_publication['branch'] and publication['tag'] == cp['media_revision']['tag'] == old_publication['tag'], 'Immutable reuse identity changed')
    rb = publication['readback']; require(rb['passed'] is True and rb['files_expected'] == rb['metadata_matched'] == rb['downloaded_sha256_matched'] == len(media) and rb['bytes_read_back'] == total and not rb['metadata_failures'] and not rb['download_failures'], 'Sealed original readback incomplete')
    require(re.fullmatch('[0-9a-f]{40}',publication['commit']) and cp['media_revision']['commit'] == publication['commit'] and cp['media_revision']['media_root'] == hosted['media_root'] == publication['media_root'], 'Wrong immutable media root')
    local = read(source/'candidate-browser-checks.json'); cases(local,manifest_sha); cases(hosted,manifest_sha)
    require(local == read(candidate/'checks/candidate-browser-checks.json'), 'Candidate84 receipt changed')
    mutation = read(source/'placeholder-mutation-checks.json')
    require(mutation == read(candidate/'checks/placeholder-mutation-checks.json') and mutation['passed'] is True and mutation['baseline_before_and_after_passed'] is True and mutation['source_files_modified'] is False and mutation['player_sha256'] == digest(candidate/'web/app_v2.js'), 'Placeholder evidence invalid')
    pages = ROOT/'docs/source/_extra/tutorials'; require(Path(hosted['pages_tree']).resolve() == pages and hosted['index_sha256'] == publication['pages']['index_sha256'] == digest(pages/'index.html'), 'Prepared Pages binding differs')
    require(set(re.findall(r'data-(?:audio|video4k|web)-root="([^"]+)"',(pages/'index.html').read_text())) == {publication['media_root']}, 'Pages pin differs')
    rows = records(cp['files']); require(len(rows) == 145, 'Expected145 recorded checkpoint rows')
    queue = []
    for relative,row in rows.items():
        path = source/safe(relative); require(path.resolve().is_relative_to(source) and not path.is_symlink() and path.is_file() and binding(path)['sha256'] == row['sha256'] and path.stat().st_size == row['bytes'], 'Checkpoint closure differs')
        if relative.startswith('web/'): require(after[relative] == row, 'Checkpoint web differs candidate')
        queue.append((path,'checkpoint/'+relative))
    for name in ('checkpoint.json','publication-receipt.json','published-media-browser-checks.json'):
        require(name not in rows,'Duplicate receipt'); queue.append((source/name,'checkpoint/'+name))
    for path in reviews: queue.append((path,'official-reviews/'+path.name))
    proof = [args.component_decision,args.final_review,args.media_reuse_proof,checked(reuse['reused_readback']),BASE/'publication-receipt.json',canonical_path,pages/'index.html',Path(__file__),WORK/'build_link_only.py',WORK/'run_tool.py',WORK/'review-hash-rebind-application-receipt.json']
    proof += [checked(row) for row in decision['assessments']] + [checked(final[k]) for k in ('preservation_peer','publication_reuse_peer')] + args.evidence
    phase = {}
    for value in args.phase_log:
        role,path = value.split('=',1); require(role not in phase,'Duplicate phase log'); phase[role] = Path(path)
    require(set(phase) == {'build','candidate','placeholder','checkpoint','resume','pages','hosted','record'}, 'Actual eight phase logs required')
    for path in phase.values(): require('Terminal exit status: 0' in path.read_text(),'Phase not terminal0')
    proof += list(phase.values()); proof = list(dict.fromkeys(Path(p).resolve(strict=True) for p in proof))
    for index,path in enumerate(proof):
        require(path.is_file() and not path.is_symlink() and path.suffix.lower() in {'.json','.py','.sh','.log','.txt','.md','.xml','.png','.jpg','.html'} and path.stat().st_size <= 5_000_000,'Only explicit small safe evidence allowed')
        queue.append((path,f'evidence/{index:03d}-{path.name}'))
    snapshots = [(p,rel,binding(p)) for p,rel in queue]; require(len({rel for _,rel,_ in snapshots}) == len(snapshots),'Duplicate output paths')
    output.mkdir(); exported = []
    for path,relative,original in snapshots:
        target = output/safe(relative); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
        require(binding(path) == original and binding(target)['sha256'] == original['sha256'] and target.stat().st_size == original['bytes'],'Source changed during copy')
        exported.append(dict(path=relative,full_path=str(target),bytes=original['bytes'],sha256=original['sha256'],source=original))
    require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} == {r['path'] for r in exported},'Packet is not closed')
    result = dict(schema=1,status='recorded_metadata_completion_export',lesson=LESSON,source_commit=SOURCE,candidate=str(candidate),manifest=binding(manifest_path),checkpoint_files=148,official_reviews=13,retained_voice_tracks=27,media_files=len(media),media_bytes=total,media_commit=publication['commit'],fresh_capture=False,fresh_upload=False,fresh_readback=False,human_listening_signoff=False,native_speaker_signoff=False,deployment_pending=True,files=exported)
    (output/'packet-manifest.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(binding(output/'packet-manifest.json')))

if __name__ == '__main__': main()
