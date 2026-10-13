"""Copy completed narration between private stages after normal audio verification.

Both stages must contain the same authored source in every spoken language.
The source is retained, copied files are byte checked, and a receipt names all
fifty tracks. This command never renders speech or publishes media.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
from build_appended_candidate import verify_tracks
from build_release_candidate import copy_checked
from stage_lesson import REPO, read, write
from audit_staged_catalogs import CATALOGS


def authored(value):
    value=deepcopy(value)
    value.pop('narration_voices',None)
    return value


def adopt(source,target,identities,receipt):
    import sys
    sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
    from render_all_voices import LANGUAGES
    source=source.resolve();target=target.resolve()
    if source==target or receipt.exists():
        raise ValueError('Use distinct private stages and a new receipt path')
    catalogs={name:read(source/'catalog'/name) for name in CATALOGS}
    targets={name:read(target/'catalog'/name) for name in CATALOGS}
    expected={(language,voice) for language,(_,voices) in LANGUAGES.items() for voice in voices}
    verified=[]
    for identity in identities:
        for name in CATALOGS:
            left=next(row for row in catalogs[name]['lessons'] if row['id']==identity)
            right=next(row for row in targets[name]['lessons'] if row['id']==identity)
            if authored(left)!=authored(right):
                raise ValueError(f'Authored narration source differs: {identity}/{name}')
        english=next(row for row in catalogs['lessons_en.json']['lessons'] if row['id']==identity)
        canonical=read(REPO/'tools/tutorials/lessons'/f'{identity}.json')
        if authored(english)!=canonical:
            raise ValueError(f'Canonical English source differs: {identity}')
        inventory={(path.parent.name,path.stem) for path in
                   (source/'production'/identity/'audio').glob('*/*.m4a')}
        if inventory!=expected:
            raise ValueError(f'All fifty narration tracks must be completed: {identity}')
        declared,records=verify_tracks(source,canonical,catalogs)
        verified.append({'lesson':identity,'voices':declared,'tracks':records})
    copies=[]
    for lesson in verified:
        for track in lesson['tracks']:
            for suffix,key in (('.m4a','audio_sha256'),('.json','timing_sha256')):
                relative=Path('production')/lesson['lesson']/'audio'/track['language']/(track['voice']+suffix)
                copy_checked(source/relative,target/relative,copies,target,track[key])
    write(receipt,{'scope':'Private source-bound narration adoption; no publication',
                   'source_stage':str(source),'target_stage':str(target),
                   'normal_audio_verification_passed':True,'lessons':verified,'files':copies})
    return verified


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--lessons',required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    result=adopt(args.source,args.target,args.lessons.split(','),args.receipt)
    print(f"Copied {sum(len(row['tracks']) for row in result)} normally verified source-bound tracks")
    return 0


if __name__=='__main__':
    raise SystemExit(main())
