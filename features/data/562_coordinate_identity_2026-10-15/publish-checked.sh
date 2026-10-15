#!/bin/bash
set -euo pipefail
checked_worktree=$1
checked_receipt=$2
checked_message=$3
shift 3
cd "$checked_worktree"
python - "$checked_receipt" <<'PY'
from pathlib import Path
import hashlib,json,subprocess,sys
receipt=json.loads(Path(sys.argv[1]).read_text())
for directory,expected in receipt['validated_tree_ids'].items():
    actual=subprocess.check_output(['git','rev-parse','HEAD:'+directory],text=True).strip()
    if actual!=expected:raise SystemExit('Publication input tree differs: '+directory)
for name,expected in receipt['artifact_sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise SystemExit('Publication artifact differs: '+name)
PY
git fetch origin nightly
git merge-base --is-ancestor origin/nightly HEAD
git diff --cached --quiet
git add -- "$checked_receipt" "$@"
git commit -m "$checked_message"
git push origin HEAD:nightly
