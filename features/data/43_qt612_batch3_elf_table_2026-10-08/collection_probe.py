"""Stop after actual pytest collection so an owned ELF core can be inspected."""

import json
import os
import signal
import sys
from pathlib import Path


def pytest_collection_finish(session):
    """Write bounded process facts, then stop before any test executes."""
    facts = {
        'pid': os.getpid(),
        'executable': os.path.realpath(sys.executable),
        'maps': sum(1 for _ in Path('/proc/self/maps').open()),
        'items': len(session.items),
        'node_files': sorted({item.nodeid.split('::', 1)[0] for item in session.items}),
    }
    target = Path('/mnt/wd4tb/scratch/native-qt612-7a-collection-20261008/collection.json')
    target.write_text(json.dumps(facts, indent=2) + '\n')
    print('owned_collection_probe=' + json.dumps({key: facts[key] for key in ('pid', 'executable', 'maps', 'items')}), flush=True)
    os.kill(os.getpid(), signal.SIGUSR1)
