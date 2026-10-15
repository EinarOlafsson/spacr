from pathlib import Path
import json,hashlib,gzip,sys,copy,shutil
root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/home-b-f561-classifier-ui-20261015');out=root/'features/data/561_classifier_ui_translation_inputs_2026-10-15';sys.path.insert(0,str(root/'tools'));import build_documentation_i18n as a;import build_guide_i18n as g
sha=lambda s:hashlib.sha256(s.encode() if isinstance(s,str) else s).hexdigest()
f=scratch/'targets/asia.json';old_file_sha256=sha(f.read_bytes());data=json.loads(f.read_text());before=copy.deepcopy(data);old=data['ko']['api']['3'];assert 'GUI 뷰을' in old;new=old.replace('GUI 뷰을','GUI 뷰를');data['ko']['api']['3']=new
plan=json.loads((scratch/'translation-obligations.json').read_text());row=plan['api'][3];assert a._reviewed_api_block_valid(row['source'],new,'ko');assert a._reviewed_api_block_valid(row['context'],new,'ko');assert not g.inline_markup_problems(new)
f.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
review=root/'docs/i18n/reviewed/api/ko/2026-10-15-f561-classifier-ui-api-home-b.json';payload=json.loads(review.read_text());targets=[x for x in payload['records'] if x['label']==row['label']];assert len(targets)==1 and targets[0]['translation']==old;targets[0]['translation']=new;review.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
follow={'language':'ko','label':row['label'],'old_target':old,'old_target_sha256':sha(old),'new_target':new,'new_target_sha256':sha(new),'source_sha256':row['source_sha256'],'context_sha256':row['context_sha256'],'reason':'Independent reviewer correction: GUI 뷰 ends in a vowel and takes object particle 를, not 을.','normal_source_context_checks':2,'actual_inline_markup':'PASS','all_other395_targets_unchanged':True,'old_asia_target_file_sha256':old_file_sha256,'new_asia_target_file_sha256':sha(f.read_bytes())}
for lang,parts in before.items():
 for domain,vals in parts.items():
  for index,target in vals.items():
   if (lang,domain,index)!=('ko','api','3'):assert data[lang][domain][index]==target
(out/'korean-particle-followup.json').write_text(json.dumps(follow,ensure_ascii=False,indent=2)+'\n');(scratch/'korean-particle-followup.json').write_text(json.dumps(follow,ensure_ascii=False,indent=2)+'\n')
(out/'asia-targets.json.gz').write_bytes(gzip.compress(f.read_bytes(),mtime=0))
receipt=json.loads((out/'receipt.json').read_text());receipt['new_review_files'][str(review.relative_to(root))]=sha(review.read_bytes());receipt['final_target_followup']='korean-particle-followup.json; one target revalidated after the full initial input-only gates, no replay of unchanged checks.';(out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
shutil.copyfile(scratch/'korean-followup.py',out/'korean-followup.py')
print(json.dumps({'target_sha256':sha(new),'review_file_sha256':sha(review.read_bytes()),'asia_sha256':sha(f.read_bytes())}))
