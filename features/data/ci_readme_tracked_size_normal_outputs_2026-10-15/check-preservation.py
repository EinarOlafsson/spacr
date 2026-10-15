import hashlib,json,subprocess,sys
from pathlib import Path
r=Path.cwd();o=Path('/mnt/wd4tb/scratch/n43-readme-size-maintenance-20261015/run');plan=json.loads((r/'features/data/ci_readme_tracked_size_inputs_2026-10-15/input-review-and-plan.json').read_text());head='6b842e0964c0749040f01bb590606cb197a51d95'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==head
for path,digest in plan['final_input_tool_test_pins'].items():assert hashlib.sha256((r/path).read_bytes()).hexdigest()==digest,path
allowed=set()
for lang in ['en','sv','de','es','zh_CN','fr','pt','hi','ko','is']:
 allowed.add('README.rst' if lang=='en' else f'docs/i18n/readme/README.{lang}.rst')
 allowed.update({f'docs/readme_images/{lang}/install-from-source-light-2.svg',f'docs/readme_images/{lang}/install-from-source-light-2.desktop.svg',f'docs/readme_text/{lang}/install-from-source-light-2.md'})
changed=set(subprocess.check_output(['git','diff','--name-only'],text=True).splitlines());assert changed<=allowed,('Unexpected generated/source drift',sorted(changed-allowed))
for name in changed:
 if name.endswith(('.rst','.md')):
  before=subprocess.check_output(['git','show',head+':'+name]);after=(r/name).read_bytes();assert after.replace(b'4657',b'3416').replace(b'2026-10-15',b'2026-10-08')==before,('Unrelated README/text edits',name)
receipt={'phase':sys.argv[1],'input_head':head,'frozen_input_tool_test_pins':len(plan['final_input_tool_test_pins']),'all_pins_unchanged':True,'only_allowed_normal_panel_outputs_changed':sorted(changed),'all_other_tracked_source_review_panel_ID_link_files_unchanged':True,'README_and_readable_MD_inverse_size_date_restores_entire_bytes':True,'network_attempts':len((o/'network-refusals.jsonl').read_text().splitlines()) if (o/'network-refusals.jsonl').exists() else 0}
assert receipt['network_attempts']==0;(o/(sys.argv[1]+'-preservation.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
