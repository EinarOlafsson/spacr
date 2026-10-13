import functools,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
sys.path.insert(0,str(Path.cwd()/'tools'))
import build_i18n_catalogs as builder
builder._simplify_chinese_prose=functools.lru_cache(maxsize=65536)(builder._simplify_chinese_prose)
builder.SECONDARY_MODEL_FOLDER='/tmp/spacr-translation-no-7b-fallback'
sys.argv=['build_i18n_catalogs.py','--repair-invalid-only','--device','cpu','--threads','1','--batch-size','8']
raise SystemExit(builder.main())
