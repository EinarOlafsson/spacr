import sys,json
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-f562-direct-table-identity-20261015');sys.path[:0]=[str(root/'tools'),str(root)]
import build_i18n_catalogs as b
sources=b.canonical_sources();audit_exit=b.audit(sources,tuple(b.MODEL_SPECS));print(json.dumps({'normal_full_nine_runtime_audit_exit':audit_exit}),flush=True)
import i18n_caption_arrivals as arrivals
result=arrivals.refresh_arrival_contract();print(json.dumps({'normal_arrivals':result}),flush=True)
Path('/mnt/wd4tb/scratch/f562-direct-table-identity-20261015/audit-arrivals-receipt.json').write_text(json.dumps({'normal_full_nine_runtime_audit_exit':audit_exit,'normal_arrivals':result},indent=2)+'\n')
raise SystemExit(audit_exit)
