import functools,hashlib,json,sys,time
from pathlib import Path
root=Path.cwd();stage=Path(__file__).parent;sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as b
b._simplify_chinese_prose=functools.lru_cache(maxsize=65536)(b._simplify_chinese_prose)
paths=list((root/'spacr/qt/i18n_catalogs').glob('*.py'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p.relative_to(root)):sha(p) for p in paths};start=time.monotonic()
sources=b.canonical_sources();status=b.audit(sources,tuple(b.MODEL_SPECS))
assert all(sha(root/p)==h for p,h in before.items())
(stage/'runtime-audit-result.json').write_text(json.dumps({'status':status,'elapsed_seconds':round(time.monotonic()-start,2),'source_counts':{k:len(v) for k,v in sources.items() if hasattr(v,'__len__')},'unchanged_catalogs':before},indent=2)+'\n')
sys.exit(status)
