import sys,json
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-n698-public-backend-integration-20261015');sys.path[:0]=[str(root/'tools'),str(root)]
import build_i18n_catalogs as b
original=b.canonical_sources;sources=original();b.canonical_sources=lambda:sources
assert original()==sources,'Canonical source changed before audit'
audit_exit=b.audit(sources,tuple(b.MODEL_SPECS));print(json.dumps({'normal_full_nine_runtime_audit_exit':audit_exit}),flush=True)
import i18n_caption_arrivals as arrivals
result=arrivals.refresh_arrival_contract();print(json.dumps({'normal_arrivals':result}),flush=True)
assert original()==sources,'Canonical source changed after audit/refresh'
Path('/mnt/wd4tb/scratch/n698-web-localization-20261015/audit-arrivals-receipt.json').write_text(json.dumps({'mode':'One actual canonical snapshot reused for ordinary loaders only, original collector rechecked before/after; all normal review/hash/candidate/arrival gates retained','normal_full_nine_runtime_audit_exit':audit_exit,'normal_arrivals':result},indent=2)+'\n')
raise SystemExit(audit_exit)
