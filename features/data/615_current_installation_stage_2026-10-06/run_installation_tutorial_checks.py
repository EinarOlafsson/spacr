from pathlib import Path
import sys

sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(Path.cwd()))
import spacr
assert Path(spacr.__file__).resolve() == Path.cwd() / 'spacr/__init__.py'
import pytest
raise SystemExit(pytest.main(['-q', 'tools/tutorials/tests/test_translation_review.py',
                             'tools/tutorials/tests/test_no_alpha_feature_tutorials.py']))
