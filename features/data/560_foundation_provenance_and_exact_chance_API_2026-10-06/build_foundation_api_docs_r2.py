from pathlib import Path
from sphinx.cmd.build import build_main
out = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/foundation-api-docs-all-r1')
assert out.is_dir()
raise SystemExit(build_main(['-W', '-E', '-b', 'html', '--keep-going', 'docs/source', str(out)]))
