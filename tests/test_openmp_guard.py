"""The duplicate-OpenMP guard.

Defends the containment for the 2026-08-14 classify segfault: a crash report
caught one OpenMP call chain crossing two libomp images (xgboost's barrier code
calling torch's ``__kmp_suspend_initialize_thread``, SIGSEGV at 0x580). The
guard sets the *calling thread's* OpenMP thread count to one, on every resident
runtime, while spaCR trains — a one-thread parallel region is serialized by
libomp and never builds the worker pool the crash needs.

Measured end-to-end through ``spacr.ml.ml_analysis`` on a worker thread, in a
process with all three runtimes mapped, counting threads parked in
``__kmp_launch_worker`` / ``__kmp_fork_barrier`` afterwards:

    no guard ............................. 19
    region clamp only .................... 10  (all in xgboost's runtime)
    region clamp + guarded joblib ........  0

The estimator's own ``n_jobs``/``nthread`` was measured to change nothing at
all (10 parked either way), which is why it is not what this guards.

What these tests actually pin down, in order of what would hurt most if it
regressed:

* a failing probe must NOT change the caller's thread count — the guard
  protects a run and must never be able to end one (INVARIANTS §10);
* one runtime, or none, must leave the request untouched;
* the escape hatch must work, because a user who has aligned their runtimes
  should not pay for this forever.
"""

import os

import pytest

from spacr import openmp_guard


@pytest.fixture(autouse=True)
def _reset_warning_latch():
    """The 'said it once' latch is module state and leaks between tests."""
    openmp_guard._WARNED = False
    yield
    openmp_guard._WARNED = False


@pytest.fixture
def clamping_platform(monkeypatch):
    """Pin the platform, so these assertions mean the same thing on Linux CI.

    The guard only clamps where the crash is documented (macOS).
    Without this the clamp tests would pass on a developer's Mac and fail in
    CI, which is the least useful way for a test to be wrong.
    """
    monkeypatch.setattr(openmp_guard.sys, "platform", "darwin")


class TestResidentRuntimes:
    def test_returns_realpaths_without_duplicates(self, monkeypatch, tmp_path):
        real = tmp_path / "libomp.dylib"
        real.write_bytes(b"")
        link = tmp_path / "libomp-link.dylib"
        link.symlink_to(real)

        monkeypatch.setattr(openmp_guard.sys, "platform", "darwin")
        monkeypatch.setattr(
            openmp_guard,
            "_macos_images",
            lambda: [str(real), str(link), "/usr/lib/libSystem.B.dylib"],
        )

        # realpath, because tmp_path is under a symlinked /var on macOS.
        assert openmp_guard.resident_openmp_runtimes() == [
            os.path.realpath(real)
        ]

    def test_two_distinct_files_are_two_runtimes(self, monkeypatch, tmp_path):
        """Byte-identical builds at two paths still have two sets of globals."""
        first = tmp_path / "torch" / "libomp.dylib"
        second = tmp_path / "brew" / "libomp.dylib"
        for path in (first, second):
            path.parent.mkdir(parents=True)
            path.write_bytes(b"identical")

        monkeypatch.setattr(openmp_guard.sys, "platform", "darwin")
        monkeypatch.setattr(
            openmp_guard, "_macos_images", lambda: [str(first), str(second)]
        )

        assert openmp_guard.resident_openmp_runtimes() == sorted(
            [os.path.realpath(first), os.path.realpath(second)]
        )

    def test_recognises_gnu_and_intel_runtimes(self, monkeypatch):
        monkeypatch.setattr(openmp_guard.sys, "platform", "linux")
        monkeypatch.setattr(
            openmp_guard,
            "_linux_images",
            lambda: [
                "/usr/lib/libgomp.so.1",
                "/opt/intel/libiomp5.so",
                "/usr/lib/libc.so.6",
            ],
        )

        found = openmp_guard.resident_openmp_runtimes()

        assert found == ["/opt/intel/libiomp5.so", "/usr/lib/libgomp.so.1"]

    def test_unknown_platform_reports_nothing(self, monkeypatch):
        monkeypatch.setattr(openmp_guard.sys, "platform", "win32")
        assert openmp_guard.resident_openmp_runtimes() == []

    def test_probe_failure_reports_nothing(self, monkeypatch):
        monkeypatch.setattr(openmp_guard.sys, "platform", "darwin")
        monkeypatch.setattr(
            openmp_guard,
            "_macos_images",
            lambda: (_ for _ in ()).throw(OSError("dyld went away")),
        )
        assert openmp_guard.resident_openmp_runtimes() == []

    def test_reads_the_real_process(self):
        """Not a mock: whatever this interpreter has loaded must parse."""
        found = openmp_guard.resident_openmp_runtimes()
        assert isinstance(found, list)
        assert all(isinstance(path, str) for path in found)
        assert found == sorted(set(found))


class TestSingleThreadedOpenmp:
    @staticmethod
    def _fake_runtime(monkeypatch, maxima=10):
        """A stand-in libomp whose ICV we can watch being set and restored."""
        class Fake:
            def __init__(self):
                self.value = maxima
                self.history = []

            def omp_get_max_threads(self):
                return self.value

            def omp_set_num_threads(self, n):
                self.value = n
                self.history.append(n)

        fakes = {}

        def handle(path):
            if path not in fakes:
                fakes[path] = Fake()
            return fakes[path]

        monkeypatch.setattr(openmp_guard, "_handle", handle)
        return handle

    def test_single_runtime_does_not_touch_anything(
        self, monkeypatch, clamping_platform
    ):
        handle = self._fake_runtime(monkeypatch)
        monkeypatch.setattr(
            openmp_guard, "resident_openmp_runtimes", lambda: ["/a/libomp.dylib"]
        )
        with openmp_guard.single_threaded_openmp("x"):
            pass
        assert handle("/a/libomp.dylib").history == []

    def test_no_runtime_does_not_touch_anything(
        self, monkeypatch, clamping_platform
    ):
        handle = self._fake_runtime(monkeypatch)
        monkeypatch.setattr(openmp_guard, "resident_openmp_runtimes", list)
        with openmp_guard.single_threaded_openmp("x"):
            pass
        assert handle("/a/libomp.dylib").history == []

    def test_two_runtimes_clamp_to_one_and_restore(
        self, monkeypatch, clamping_platform
    ):
        """The measured fix: one thread, then the previous value back."""
        handle = self._fake_runtime(monkeypatch, maxima=10)
        monkeypatch.setattr(
            openmp_guard,
            "resident_openmp_runtimes",
            lambda: ["/a/libomp.dylib", "/b/libomp.dylib"],
        )
        with openmp_guard.single_threaded_openmp("XGBoost"):
            assert handle("/a/libomp.dylib").value == 1
            assert handle("/b/libomp.dylib").value == 1
        # Every runtime clamped, every runtime put back where it was.
        for path in ("/a/libomp.dylib", "/b/libomp.dylib"):
            assert handle(path).value == 10
            assert handle(path).history == [1, 10]

    def test_works_as_a_decorator(self, monkeypatch, clamping_platform):
        handle = self._fake_runtime(monkeypatch, maxima=8)
        monkeypatch.setattr(
            openmp_guard,
            "resident_openmp_runtimes",
            lambda: ["/a/libomp.dylib", "/b/libomp.dylib"],
        )

        @openmp_guard.single_threaded_openmp("training")
        def train():
            return handle("/a/libomp.dylib").value

        assert train() == 1
        assert handle("/a/libomp.dylib").value == 8

    def test_decorator_keeps_the_wrapped_signature(self, monkeypatch):
        """`spacr.settings` introspects pipeline callables (settings.py:308)."""
        import inspect

        @openmp_guard.single_threaded_openmp("training")
        def ml_like(settings, verbose=False):
            return settings

        params = list(inspect.signature(ml_like).parameters)
        assert params == ["settings", "verbose"]
        assert ml_like.__name__ == "ml_like"

    def test_restores_even_when_the_body_raises(
        self, monkeypatch, clamping_platform
    ):
        handle = self._fake_runtime(monkeypatch, maxima=10)
        monkeypatch.setattr(
            openmp_guard,
            "resident_openmp_runtimes",
            lambda: ["/a/libomp.dylib", "/b/libomp.dylib"],
        )
        with pytest.raises(ValueError):
            with openmp_guard.single_threaded_openmp("x"):
                raise ValueError("boom")
        assert handle("/a/libomp.dylib").value == 10

    def test_linux_reports_but_does_not_clamp(self, monkeypatch):
        """No evidence of the fault there, and the cluster is core-bound."""
        handle = self._fake_runtime(monkeypatch)
        monkeypatch.setattr(openmp_guard.sys, "platform", "linux")
        monkeypatch.delenv("SPACR_OPENMP_GUARD", raising=False)
        monkeypatch.setattr(
            openmp_guard,
            "resident_openmp_runtimes",
            lambda: ["/a/libomp.so", "/b/libgomp.so.1"],
        )

        with openmp_guard.single_threaded_openmp("x"):
            pass
        assert handle("/a/libomp.so").history == []
        assert openmp_guard.openmp_runtime_is_duplicated() is True

    def test_forcing_the_guard_on_clamps_off_darwin(self, monkeypatch):
        handle = self._fake_runtime(monkeypatch)
        monkeypatch.setattr(openmp_guard.sys, "platform", "linux")
        monkeypatch.setenv("SPACR_OPENMP_GUARD", "on")
        monkeypatch.setattr(
            openmp_guard,
            "resident_openmp_runtimes",
            lambda: ["/a/libomp.so", "/b/libgomp.so.1"],
        )

        with openmp_guard.single_threaded_openmp("x"):
            assert handle("/a/libomp.so").value == 1

    def test_a_broken_probe_costs_the_caller_nothing(
        self, monkeypatch, clamping_platform
    ):
        """The guard must never be the reason a run dies (INVARIANTS §10)."""
        monkeypatch.setattr(
            openmp_guard,
            "resident_openmp_runtimes",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with openmp_guard.single_threaded_openmp("x"):
            pass  # must not raise

    def test_a_runtime_without_the_symbols_is_skipped(
        self, monkeypatch, clamping_platform
    ):
        """One unusable image must not cost the clamp on the others."""
        inner = self._fake_runtime(monkeypatch, maxima=10)
        good = inner("/b/libomp.dylib")

        def handle(path):
            if path.endswith("bad.dylib"):
                raise AttributeError("no omp_set_num_threads")
            return inner(path)

        monkeypatch.setattr(openmp_guard, "_handle", handle)
        monkeypatch.setattr(
            openmp_guard,
            "resident_openmp_runtimes",
            lambda: ["/a/bad.dylib", "/b/libomp.dylib"],
        )
        with openmp_guard.single_threaded_openmp("x"):
            assert good.value == 1
        assert good.value == 10

    def test_explains_itself_once_and_names_the_runtimes(
        self, monkeypatch, capsys, clamping_platform
    ):
        self._fake_runtime(monkeypatch)
        monkeypatch.setattr(
            openmp_guard,
            "resident_openmp_runtimes",
            lambda: ["/torch/libomp.dylib", "/brew/libomp.dylib"],
        )
        with openmp_guard.single_threaded_openmp("XGBoost"):
            pass
        first = capsys.readouterr().out
        with openmp_guard.single_threaded_openmp("XGBoost"):
            pass
        second = capsys.readouterr().out

        assert "XGBoost" in first
        assert "/torch/libomp.dylib" in first
        assert "/brew/libomp.dylib" in first
        assert "SPACR_OPENMP_GUARD" in first
        assert second == ""

    def test_the_escape_hatch_disables_the_clamp(
        self, monkeypatch, clamping_platform
    ):
        handle = self._fake_runtime(monkeypatch)
        monkeypatch.setattr(
            openmp_guard,
            "resident_openmp_runtimes",
            lambda: ["/a/libomp.dylib", "/b/libomp.dylib"],
        )
        monkeypatch.setenv("SPACR_OPENMP_GUARD", "off")

        with openmp_guard.single_threaded_openmp("x"):
            pass
        assert handle("/a/libomp.dylib").history == []
        assert openmp_guard.openmp_runtime_is_duplicated() is False


class TestGuardedNJobs:
    """joblib workers are new threads and do not inherit the per-thread clamp.

    Measured on ml_analysis: 19 threads parked in __kmp_launch_worker with no
    guard, 10 with the region clamp alone (all of them in Homebrew's libomp,
    i.e. xgboost's runtime — the one that crashed), 0 once joblib is kept on
    the calling thread too.
    """

    def test_clamps_when_duplicated(self, monkeypatch, clamping_platform):
        monkeypatch.setattr(
            openmp_guard,
            "resident_openmp_runtimes",
            lambda: ["/a/libomp.dylib", "/b/libomp.dylib"],
        )
        assert openmp_guard.guarded_n_jobs(-1) == 1

    def test_leaves_the_request_alone_when_single(
        self, monkeypatch, clamping_platform
    ):
        monkeypatch.setattr(
            openmp_guard, "resident_openmp_runtimes", lambda: ["/a/libomp.dylib"]
        )
        assert openmp_guard.guarded_n_jobs(-1) == -1

    def test_off_platform_leaves_the_request_alone(self, monkeypatch):
        monkeypatch.setattr(openmp_guard.sys, "platform", "linux")
        monkeypatch.delenv("SPACR_OPENMP_GUARD", raising=False)
        monkeypatch.setattr(
            openmp_guard,
            "resident_openmp_runtimes",
            lambda: ["/a/libomp.so", "/b/libgomp.so.1"],
        )
        assert openmp_guard.guarded_n_jobs(-1) == -1

    def test_a_broken_probe_costs_the_caller_nothing(
        self, monkeypatch, clamping_platform
    ):
        monkeypatch.setattr(
            openmp_guard,
            "resident_openmp_runtimes",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert openmp_guard.guarded_n_jobs(-1) == -1

    @pytest.mark.parametrize("value", ["0", "off", "OFF", "false", "No"])
    def test_escape_hatch_spellings(self, monkeypatch, value):
        monkeypatch.setenv("SPACR_OPENMP_GUARD", value)
        assert openmp_guard._guard_disabled() is True

    def test_unset_escape_hatch_keeps_the_guard_on(self, monkeypatch):
        monkeypatch.delenv("SPACR_OPENMP_GUARD", raising=False)
        assert openmp_guard._guard_disabled() is False


class TestDuplicationPredicate:
    def test_true_only_above_one(self, monkeypatch):
        monkeypatch.delenv("SPACR_OPENMP_GUARD", raising=False)
        for runtimes, expected in (
            ([], False),
            (["/a/libomp.dylib"], False),
            (["/a/libomp.dylib", "/b/libomp.dylib"], True),
        ):
            monkeypatch.setattr(
                openmp_guard, "resident_openmp_runtimes", lambda r=runtimes: r
            )
            assert openmp_guard.openmp_runtime_is_duplicated() is expected


class TestCallSitesAreWired:
    """The guard is worthless if the estimators do not go through it.

    Checked as source text rather than by fitting a model: a real fit here
    would need xgboost, and on a machine with the duplicate runtimes the test
    itself would be the thing that segfaults.
    """

    @pytest.mark.parametrize(
        "relative, needle",
        [
            ("spacr/ml.py", "@single_threaded_openmp('classical ML training')"),
            ("spacr/ml.py", "guarded_n_jobs(n_jobs, 'permutation importance')"),
            ("spacr/timelapse.py", "@single_threaded_openmp('XGBoost infection QC')"),
        ],
    )
    def test_call_site_uses_the_guard(self, relative, needle):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        source = open(os.path.join(root, relative), encoding="utf-8").read()
        assert needle in source, f"{relative} no longer routes through the guard"
