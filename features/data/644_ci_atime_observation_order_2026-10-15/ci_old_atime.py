import os
import pytest

@pytest.fixture(autouse=True)
def age_disposable_cache_for_observation(request):
    if request.node.originalname != "test_cache_mutation_refuses_active_worker_then_succeeds_after_finalization":
        return
    folder, metadata = request.getfixturevalue("cache")
    for path in (folder / "kept.json", metadata):
        before = path.stat()
        os.utime(path, ns=(1, before.st_mtime_ns))
        assert path.stat().st_atime_ns == 1
