from pathlib import Path
import sys
R=Path("/media/carruthers/mnt3/claude/spacr-player-readiness-port-20261015")
sys.path[:0]=[str(R),str(R/"tools"),str(R/"tools/tutorials/tests")]
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
