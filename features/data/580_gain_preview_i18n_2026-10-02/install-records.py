import hashlib,json
from pathlib import Path
p=Path(__file__).parent;r=Path.cwd();sources=json.loads((p/'sources.json').read_text());gates=json.loads((p/'candidate-gates.json').read_text())
assert gates['targets']==180 and not gates['issues']
for lang in ['sv','de','es','zh_CN','pt','hi','ko','is','fr']:
 draft=p/f'{lang}-draft.json';final=p/f'{lang}-final.json';targets=json.loads(final.read_text());assert set(targets)==set(sources)
 author='root' if lang=='sv' else 'shared_ui' if lang in ['zh_CN','hi'] else 'mask_editor' if lang in ['ko','is'] else 'merge_interfaces'
 peer=p/('sv-peer-review.json' if lang=='sv' else 'applied-peer-corrections.json')
 payload={'schema':1,'language':lang,'review_kind':'Direct Codex AI technical translation with independent Codex AI peer review; no human/native-speaker signoff.','draft_author':author,'technical_reviewer':'shared_ui' if lang=='sv' else 'root','draft_sha256':hashlib.sha256(draft.read_bytes()).hexdigest(),'peer_review_sha256':hashlib.sha256(peer.read_bytes()).hexdigest(),'review':'All twenty new gain-preview captions reviewed for multiplicative intensity factors, microscope fields versus assay plates, reference wells, camera offset, source-folder isolation, first plate in name order, unchanged images/measurements, valid settings and cancellation that discards results while an existing file scan safely finishes. Root module-name and terminology corrections are recorded separately from immutable drafts. Standard context normalization and release gates are unchanged.','records':[{'table':'ui','key':s,'source':s,'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'translation':targets[s]} for s in sources]}
 (r/f'docs/i18n/reviewed/runtime/{lang}/2026-10-02-calibration-gain-preview.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print('Wrote180source-bound independently AI-peer-reviewed target records;9locales')
