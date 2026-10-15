from pathlib import Path
import hashlib,json,gzip,shutil,subprocess,xml.etree.ElementTree as ET
root=Path('/mnt/wd4tb/spacr-worktrees/codex-n698-public-backend-integration-20261015')
scratch=Path('/mnt/wd4tb/scratch/n698-web-localization-20261015')
evidence=root/'features/data/698_web_runtime_locales_2026-10-15'
for name in ('runtime','callbacks','audit-arrivals'):
    assert (scratch/(name+'.exit')).read_text().strip()=='0',name
    with gzip.open(evidence/(name+'.log.gz'),'wb') as output:output.write((scratch/(name+'.log')).read_bytes())
    shutil.copy2(scratch/(name+'.exit'),evidence/(name+'.exit'))
for name in ('scoped-runtime.py','runtime-receipt.json','audit-arrivals.py','audit-arrivals-receipt.json','closure-run.sh','runtime-run.sh'):
    shutil.copy2(scratch/name,evidence/name)
with gzip.open(evidence/'callbacks.xml.gz','wb') as output:output.write((scratch/'callbacks.xml').read_bytes())
xml=ET.parse(scratch/'callbacks.xml');cases=list(xml.iter('testcase'));assert not list(xml.iter('failure')) and not list(xml.iter('error')) and not list(xml.iter('skipped'))
callbacks=[case for case in cases if 'test_web_mask_runtime_locales_n698' in case.attrib.get('classname','')];assert len(callbacks)==27,len(callbacks)
before=json.loads((evidence/'source-body-before.json').read_text());after={}
for name,old in before['files'].items():
    digest=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if name=='tests/test_web_mask_runtime_locales_n698.py':
        repair=json.loads((evidence/'test-import-repair.json').read_text());assert digest==repair['actual_repaired_test_sha256']
        after[name]={'sha256':digest,'exact_frozen_body':False,'exact_test_only_qualified_import_repair':True}
    else:
        assert digest==old['current_sha256'],name
        after[name]={'sha256':digest,'exact_frozen_body':True}
(evidence/'source-body-after.json').write_text(json.dumps({'frozen_source_pin':before['frozen_source_pin'],'post_actual_writer_and_owning_checks':True,'files':after},indent=2)+'\n')
writer=json.loads((scratch/'runtime-receipt.json').read_text());assert len(writer['locales'])==9 and all(row['owned_reviewed_rows']==178 for row in writer['locales'].values())
audit=json.loads((scratch/'audit-arrivals-receipt.json').read_text());assert audit['normal_full_nine_runtime_audit_exit']==0
receipt={'private_source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'frozen_source_pin':before['frozen_source_pin'],'source_bodies_verified':len(after),'new_ui_sources':178,'reviewed_language_pairs':1602,'independent_review':'All9 exact arrays independently read by parent/installer peers;41 terminology/precision refinements then3 Korean protected-literal followups, all exact approvals archived; no native/human signoff','normal_writer':writer,'writer_audit_languages':0,'actual_separate_full_runtime_audit_languages':9,'actual_localized_callback_and_zip_cases':len(callbacks),'combined_owning_cases':len(cases),'final_owning_errors_failures_skips':0,'frozen_runtime_and_owner_bodies_exact':20,'test_body_change':'Exact package-qualified fixture import repair only','audit_lifecycle':'First wrapper stop left separately capped audit child running; it actually returned internal audit0 and refresh counts, preserved intermediate JSON, no wrapper exit available and stdout overwritten. Final healthy end-to-end chain intentionally completed one redundant metadata pass; no callbacks repeated after44PASS. Lifecycle/raw failures archived, no reconstructed stdout.','audit_and_arrivals':audit,'limitations':['Graph controller is stubbed; real refusals execute before inference; no model scientific acceptance from these checks','Real ZIP packagers preserve original upload/member hashes and unavailable semantics','No hosted/freeGPU/browserWebGPU/shared-site/deployment acceptance','Parent N698 remains partial; no final fullCI/release claim']}
(evidence/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
