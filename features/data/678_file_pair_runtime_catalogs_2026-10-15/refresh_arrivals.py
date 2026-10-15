from pathlib import Path
import hashlib,json,sys
root=Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-coordinator-20261014')
sys.path[:0]=[str(root/'tools'),str(root)]
import build_i18n_catalogs as builder
import i18n_caption_arrivals as arrivals
before=arrivals.load_arrival_contract()
baseline=json.dumps(before['baseline'],sort_keys=True,ensure_ascii=False).encode()
canonical=builder.canonical_sources
sources=canonical()
builder.canonical_sources=lambda:sources
result=arrivals.refresh_arrival_contract()
after=arrivals.load_arrival_contract()
assert json.dumps(after['baseline'],sort_keys=True,ensure_ascii=False).encode()==baseline
assert canonical()==sources,'Canonical sources changed during refresh'
receipt={'normal_tool':'i18n_caption_arrivals.refresh_arrival_contract','canonical_cache':'One frozen normal canonical source result reused by review lookups; original collector compared again after refresh. No validator modified.','result':result,'baseline_count':len(before['baseline']),'baseline_sha256':hashlib.sha256(baseline).hexdigest(),'baseline_values_unchanged':True,'fixture_sha256':hashlib.sha256(arrivals.FIXTURE.read_bytes()).hexdigest()}
Path('/mnt/wd4tb/scratch/home-b-file-pair-arrivals-20261015/receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,sort_keys=True),flush=True)
