from pathlib import Path
import hashlib,json,subprocess
repo=Path.cwd(); site=Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-site-20261014'); out=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
sgit=lambda *args:subprocess.check_output(['git','-C',str(site),*args])
old='e33c81e244cfbad30e95a7657c1ad6c3f547ac22'; current=sgit('rev-parse','HEAD').decode().strip()
assert current=='652f833c36b7eef0f503465031038f2a7ab89aec'
paths=['src/assets/js/fx/field.js','data/models.json','data/showcase.json','data/papers.json','tools/models/collect.py','tools/releases.mjs','tools/check_release.mjs','.github/workflows/deploy.yml']
unchanged={}
for p in paths:
 a=sgit('show',f'{old}:{p}'); b=sgit('show',f'{current}:{p}'); assert a==b,p
 unchanged[p]={'sha256':sha(b),'exact_since_deployed_source':old}
fonts=[p for p in sgit('ls-files','src/assets/fonts').decode().splitlines() if p.endswith('.woff2')]
for p in fonts:
 a=sgit('show',f'{old}:{p}');b=sgit('show',f'{current}:{p}');assert a==b,p
 unchanged[p]={'sha256':sha(b),'exact_since_deployed_source':old}
items={}
for n,path in [('N686','features/new/686_a_free_project_website_with_rich_effects.txt'),('N687','features/new/687_website_v2_flow_theme_model_showcase_and_release_sync.txt')]:
 data=(repo/path).read_bytes(); items[n]={'path':path,'sha256':sha(data),'dated_notes_used':['2026-10-10 initial build/deployment and later maintainer changes','2026-10-11 rounds 3–8','2026-10-12 current project/public naming changes','2026-10-14 release/startup/reduced-motion repairs and deployment']}
prior={}
for kind in ('release_asset_sync','startup_loading','reduced_motion'):
 path=f'features/data/687_{kind}_2026-10-14/receipt.json'; data=subprocess.check_output(['git','show',f'HEAD:{path}']);prior[path]=sha(data)
clauses=[
 {'clause':1,'requirement':'spaCR flow/advection theme, pointer response, pause/reduced-motion/CSS fallback','basis':['N687 2026-10-10 build note and original91 browser checks','2026-10-14 reduced-motion23 and startup19 checks','2026-10-15 six production-context Node and four real software-browser checks'],'current_source':'field.js exact since deployed e33; fx.js only independently reviewed early software fallback change, preserving explicit always and accelerated behavior'},
 {'clause':2,'requirement':'self-hosted Open Sans400 headings/300 body and AA contrast in both themes','basis':['N687 2026-10-10 computed style/contrast and four-width/two-theme review','unchanged font assets and global font rules'],'current_source':'CSS changes since e33 only isolated N698 mask controls; global h1–h4 and body declarations unchanged','limits':'No new human readability or physical-display claim'},
 {'clause':3,'requirement':'real Hoechst/cell-mask cross-channel and plaque examples with labels/model/version/counts/source','basis':['2026-10-10 real published-weight CPU model outputs and four cross-channel enumeration','2026-10-11 plaque-v4 replacement; actual100 vs98 reference, F1.960'],'current_source':'data/models.json, data/showcase.json and model collector byte-identical since deployed e33; outputs not regenerated or inferred'},
 {'clause':4,'requirement':'verified DOI papers and refreshed candidates with curated allowlist','basis':['2026-10-10 six full-text verifications, later maintainer removal of cite-only cp_measure','subsequent published five-paper allowlist'],'current_source':'data/papers.json byte-identical since deployed e33; no new scientific/citation inference'},
 {'clause':5,'requirement':'plain factual site copy','basis':['original N687 editorial rewrite and dated accepted followups','N698504 independently reviewed additions and source peer'],'limits':'Technical review, not human read-aloud usability certification'},
 {'clause':6,'requirement':'four viewport sizes/both themes and mobile performance>=90/accessibility>=95','basis':['historical four-size/two-theme layout acceptance retained for unchanged layout','2026-10-15 current immutable public audit38755788985 all15 cold samples; no best-sample selection'],'current_evidence':'features/data/687_hosted_mobile_after_fallback_2026-10-15/receipt.json','scope':'five primary public routes including locked protected-page entrances; historical decrypted functional checks remain historical, not new decrypted performance measurements'},
 {'clause':7,'requirement':'automatic release sync by hourly poll, optional dispatch, measured dry-run latency','basis':['2026-10-10 real dry-run63s and unchanged hourly:23/daily06:17 workflow','2026-10-14 real eleven release-asset/source tests and deployed manifest validation','2026-10-15 successful normal production38755493027'],'current_source':'deploy.yml/releases.mjs/check_release.mjs exact since e33; no new release or faster-than-cron promise'},
 {'clause':8,'requirement':'prominent badge plus plain Website line in ten READMEs, Website metadata and docs landing link','basis':['Website badges already first; previously reviewed2026-10-10 exact nine translations reused','current assigned raw-header source fix plus normal generation/metadata verification'],'status':'PENDING final current generation/checks and coordinator publication; no whole-item completion before that'}
]
result={'status':'WHOLE_N687_CLOSEOUT_PROPOSAL_PENDING_FINAL_README_CHECKS_AND_PUBLICATION','site_source':current,'public_revision':'fe6f4f46a10f04c3133848874b9d1d87049dc4ac','item_sources':items,'unchanged_source_bindings':unchanged,'prior_receipt_hashes':prior,'N687_original_eight_clauses':clauses,'N686_remaining_original_requirement':'60fps on a mid-range physical laptop; no such current acceptance exists. Hosted Lighthouse and software-renderer functional proof do not substitute for it.','N687_physical_frame_rate_requirement':False,'scope_boundary':'No paid hosting, new release/full CI, protected-page decrypt, model rerun, tutorial modification, human/native-speaker or physical-device acceptance is claimed. Parent independently reviews and owns final item state.'}
(out/'whole-item-proposal.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'unchanged_bound_files':len(unchanged),'clauses':len(clauses)}))
