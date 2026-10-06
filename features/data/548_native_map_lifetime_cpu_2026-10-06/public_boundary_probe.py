"""Audit SCN's exact API additions without changing any production guard or pin."""
import dataclasses
import hashlib
import json
from pathlib import Path
from tests import test_docstring_correctness as D

rows = list(D._public_callables())
expected = {'spacr.convert.scn_to_rgb8', 'spacr.convert.read_scn',
            'spacr.qt.mask_engine.read_image'}
added = [row for row in rows if row.symbol in expected]
assert len(added) == 3
original = D._public_callables
D._public_callables = lambda: (row for row in rows if row.symbol not in expected)
D.test_public_callable_inventory_is_source_derived_not_docstring_derived()
D._public_callables = original
root=Path('/mnt/wd4tb/scratch/f548-io-map-lifetime-20261006')
receipt = {'source_root': str(Path(D.__file__).resolve().parents[1]),
           'io_sha256': hashlib.sha256(Path('spacr/io.py').read_bytes()).hexdigest(),
           'test_sha256': hashlib.sha256(Path(D.__file__).read_bytes()).hexdigest(),
           'exact_additions': [dataclasses.asdict(row) for row in added],
           'all_prior_inventory_signature_bucket_digest_pins_restored': True,
           'guard_changed': False}
(root/'public-boundary-scn-delta.json').write_text(json.dumps(receipt,default=lambda value: sorted(value),indent=2)+'\n')
print('Exactly three SCN arrivals; removing only these in a scratch audit restores every prior inventory pin/digest. Production guard unchanged.',flush=True)
