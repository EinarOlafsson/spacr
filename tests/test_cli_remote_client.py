"""Table-driven tests of the ``spacr-remote`` client (:func:`spacr.cli_remote.main`).

Every case drives ``main`` with a real argument vector against a real on-disk
profile/job store redirected by ``SPACR_REMOTE_STATE_DIR`` into a fresh folder,
and the ``command`` backend runs real local programs (``/bin/echo`` and a small
shell script).  Persistence, argument parsing, backend dispatch and exit codes
are not stubbed; only the cluster probe and the allocation request (which would
need a Slurm login node) and the watch loop's own sleep are replaced.

One general test per command family: the profile commands, the submit
refusals, every command run against a submitted job, the exit-code mapping of
``main``, and the read-only resources/allocate pair.
"""
from __future__ import annotations

import json
import os
import stat
import textwrap
import types

import pytest

from spacr import cli_remote
from spacr.remote_execution import ExecutionProfile, ProfileStore, RemoteJob

pytestmark = pytest.mark.posix_only

OK, RUNTIME, USAGE = cli_remote.EXIT_OK, cli_remote.EXIT_RUNTIME, cli_remote.EXIT_USAGE
PROFILE_HEADER = "NAME                 BACKEND   HOST\n"
JOB_HEADER = "JOB           STATUS      MODULE              PROFILE             REMOTE ID\n"
NO_LOG_COMMAND = ("This command profile has no log command. Add one with "
                  "{external_id} in the execution profile.\n")


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _fresh_state(tmp_path, monkeypatch, name):
    """Point the profile/job stores at a new folder for one case."""
    root = tmp_path / name
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("SPACR_REMOTE_STATE_DIR", str(root / "state"))
    settings = root / "settings.json"
    settings.write_text(json.dumps({"src": str(root)}), encoding="utf-8")
    return root, settings


def _run(capsys, argv):
    """Run ``main`` and return ``(exit code, stdout, stderr)``."""
    code = cli_remote.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _command_profile_argv(name="cloud", *, status="/bin/echo COMPLETED {external_id}",
                          log="/bin/echo tail-of-{external_id}"):
    """The ``profile add`` vector of a ``command`` profile built on real programs."""
    argv = ["profile", "add", name, "--backend", "command",
            "--submit-command", "/bin/echo JOBID=4242 for {settings}",
            "--status-command", status,
            "--cancel-command", "/bin/echo cancelling {external_id}",
            "--job-id-pattern", r"JOBID=(\d+)"]
    return argv + (["--log-command", log] if log else [])


def _status_script(folder):
    """An executable status command that reports RUNNING once, then COMPLETED."""
    script = folder / "status.sh"
    script.write_text(textwrap.dedent("""\
        #!/bin/sh
        counter="$0.count"
        n=$(cat "$counter" 2>/dev/null || echo 0)
        n=$((n+1))
        echo "$n" > "$counter"
        if [ "$n" -ge 2 ]; then echo COMPLETED; else echo RUNNING; fi
    """), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _row(job_id, status, module="measure", profile="cloud", remote="4242"):
    """The exact job-table row ``spacr-remote`` prints."""
    return f"{job_id[:12]:12}  {status:10}  {module:18}  {profile:18}  {remote}\n"


# ---------------------------------------------------------------------------
# profile list / add / delete
# ---------------------------------------------------------------------------

SLURM_FLAGS = ["profile", "add", "hpc", "--backend", "slurm", "--host", "login01",
               "--workdir", "/work/spacr", "--local-root", "/data",
               "--remote-root", "/mnt/data", "--runner", "spacr-run",
               "--scheduler-options", "--partition gpu --time 4:00:00",
               "--poll-seconds", "30"]

# Each case is a script of steps (argv, exit code, exact stdout or a check of
# the parsed JSON, exact stderr), run against one fresh store.
PROFILE_CASES = [
    ("an empty store lists no profiles", [
        (["profile", "list"], OK, PROFILE_HEADER + "(no execution profiles)\n", ""),
    ]),
    ("an ssh profile lists its backend and host", [
        (["profile", "add", "workstation", "--backend", "ssh", "--host", "gpubox",
          "--workdir", "/scratch/spacr"], OK,
         "Saved ssh execution profile 'workstation'.\n", ""),
        (["profile", "list"], OK,
         PROFILE_HEADER + f"{'workstation':20} {'ssh':9} gpubox\n", ""),
    ]),
    ("a hostless command profile reads as local/custom", [
        (_command_profile_argv(), OK, "Saved command execution profile 'cloud'.\n", ""),
        (["profile", "list"], OK,
         PROFILE_HEADER + f"{'cloud':20} {'command':9} local/custom\n", ""),
        (["profile", "list", "--json"], OK,
         lambda saved: ([p["name"] for p in saved], saved[0]["backend"],
                        saved[0]["job_id_pattern"], saved[0]["host"])
         == (["cloud"], "command", r"JOBID=(\d+)", ""), ""),
    ]),
    ("every add flag reaches the saved record", [
        (SLURM_FLAGS, OK, "Saved slurm execution profile 'hpc'.\n", ""),
        (["profile", "list", "--json"], OK,
         lambda saved: {k: saved[0][k] for k in (
             "name", "backend", "host", "workdir", "local_root", "remote_root",
             "runner", "scheduler_options", "poll_seconds")} == {
             "name": "hpc", "backend": "slurm", "host": "login01",
             "workdir": "/work/spacr", "local_root": "/data",
             "remote_root": "/mnt/data", "runner": "spacr-run",
             "scheduler_options": "--partition gpu --time 4:00:00",
             "poll_seconds": 30}, ""),
    ]),
    ("delete removes the profile", [
        (_command_profile_argv(), OK, "Saved command execution profile 'cloud'.\n", ""),
        (["profile", "delete", "cloud"], OK, "Deleted execution profile 'cloud'.\n", ""),
        (["profile", "list"], OK, PROFILE_HEADER + "(no execution profiles)\n", ""),
    ]),
    ("deleting a profile that never existed is a usage error", [
        (["profile", "delete", "ghost"], USAGE, "",
         "error: Execution profile 'ghost' does not exist.\n"),
    ]),
    ("an invalid profile is refused rather than saved", [
        (["profile", "add", "broken", "--backend", "ssh"], USAGE, "",
         "error: An SSH workstation needs a host.\n"),
        (["profile", "list"], OK, PROFILE_HEADER + "(no execution profiles)\n", ""),
    ]),
]


def test_profile_commands_print_exact_tables_and_refusals(subtests, tmp_path,
                                                          monkeypatch, capsys):
    """``profile list/add/delete`` print exact tables, records and refusals."""
    for index, (case_id, steps) in enumerate(PROFILE_CASES):
        with subtests.test(msg=case_id):
            _fresh_state(tmp_path, monkeypatch, f"profile{index}")
            for argv, code, out, err in steps:
                got = _run(capsys, argv)
                assert (got[0], got[2]) == (code, err), argv
                if callable(out):
                    assert out(json.loads(got[1])), got[1]
                else:
                    assert got[1] == out, argv


# ---------------------------------------------------------------------------
# submit refusals
# ---------------------------------------------------------------------------

SUBMIT_REFUSALS = [
    ("an unknown module points at the module list",
     lambda root, s: ["submit", "not_a_module", "--settings", str(s), "--profile", "cloud"],
     lambda root: "error: Unknown spaCR module 'not_a_module'; use `spacr-run --list`.\n"),
    ("a missing settings file is one sentence, not a traceback",
     lambda root, s: ["submit", "measure", "--settings", str(root / "nope.json"),
                      "--profile", "cloud"],
     lambda root: (f"error: settings file not found: {root / 'nope.json'}\n"
                   "  Check the path, and that the share holding it is mounted on this node.\n")),
    ("a bad --set override is refused locally",
     lambda root, s: ["submit", "measure", "--settings", str(s), "--profile", "cloud",
                      "--set", "this_key_does_not_exist=1"],
     lambda root: ("error: --set this_key_does_not_exist=... names a setting that does not "
                   "exist for module 'measure': 'this_key_does_not_exist'.\n"
                   "  Run 'spacr-run --describe measure' to list the settings this module "
                   "accepts.\n")),
    ("an unknown profile is resolved before the job is recorded",
     lambda root, s: ["submit", "measure", "--settings", str(s), "--profile", "ghost"],
     lambda root: "error: Execution profile 'ghost' does not exist.\n"),
]


def test_submit_refusals_never_record_a_job(subtests, tmp_path, monkeypatch, capsys):
    """Every refused submission exits 2 with its exact sentence and leaves no job."""
    for index, (case_id, argv, message) in enumerate(SUBMIT_REFUSALS):
        with subtests.test(msg=case_id):
            root, settings = _fresh_state(tmp_path, monkeypatch, f"refusal{index}")
            assert _run(capsys, _command_profile_argv())[0] == OK
            assert _run(capsys, argv(root, settings)) == (USAGE, "", message(root))
            assert _run(capsys, ["list", "--json"]) == (OK, "[]\n", "")
            assert _run(capsys, ["list"]) == (OK, JOB_HEADER + "(no distributed jobs)\n", "")


# ---------------------------------------------------------------------------
# list / status / cancel / logs / watch on a submitted job
# ---------------------------------------------------------------------------

def _listed(capsys):
    """The parsed ``list --json`` job records."""
    code, out, err = _run(capsys, ["list", "--json"])
    assert (code, err) == (OK, "")
    return json.loads(out)


# (case, status command or "script", log command, argv tail, exit code,
#  stdout builder (job id -> exact text or (prefix-free) line check), stderr,
#  follow-up check of the stored records or None)
JOB_CASES = [
    ("list prints one compact row", None, None, ["list"], OK,
     lambda j: JOB_HEADER + _row(j, "queued"), "", None),
    ("list --json records the queued job", None, None, ["list", "--json"], OK,
     None, "", lambda jobs, root: (
         [(j["module"], j["external_id"], j["status"]) for j in jobs] == [("measure", "4242", "queued")]
         and os.path.exists(jobs[0]["settings_path"])
         and jobs[0]["settings_path"].startswith(str(root / "state")))),
    ("list --refresh polls the active job", None, None, ["list", "--refresh"], OK,
     lambda j: JOB_HEADER + _row(j, "success"), "", None),
    ("status --logs prints the row and the tail", None, None, ["status", "{job}", "--logs"], OK,
     lambda j: JOB_HEADER + _row(j, "success") + "\nLog tail:\ntail-of-4242\n\n", "", None),
    ("status without a log command prints no tail", None, "", ["status", "{job}"], OK,
     lambda j: JOB_HEADER + _row(j, "success"), "", None),
    ("status --json prints one record and no table", None, None, ["status", "{job}", "--json"], OK,
     lambda j: (lambda rec: (rec["job_id"], rec["status"]) == (j, "success")), "", None),
    ("a failed job makes status exit nonzero", "/bin/echo FAILED {external_id}", None,
     ["status", "{job}"], RUNTIME, lambda j: JOB_HEADER + _row(j, "failed"), "", None),
    ("status of an unknown job is a usage error", None, None, ["status", "deadbeef"], USAGE,
     lambda j: "", "error: Remote job 'deadbeef' was not found.\n", None),
    ("cancel prints and persists the new state", None, None, ["cancel", "{job}"], OK,
     lambda j: f"{j}: cancelled\n", "",
     lambda jobs, root: [job["status"] for job in jobs] == ["cancelled"]),
    ("logs prints exactly what the log command produced", None, None,
     ["logs", "{job}", "--lines", "5"], OK, lambda j: "tail-of-4242\n\n", "", None),
    ("logs without a log command explains how to add one", None, "", ["logs", "{job}"], OK,
     lambda j: NO_LOG_COMMAND, "", None),
    ("watch stops on the first terminal poll", None, None, ["watch", "{job}"], OK,
     lambda j: (lambda lines: [ln.split()[-1] for ln in lines] == ["success"]), "", None),
    ("watch of a failed job exits nonzero", "/bin/echo FAILED {external_id}", None,
     ["watch", "{job}"], RUNTIME,
     lambda j: (lambda lines: [ln.split()[-1] for ln in lines] == ["failed"]), "", None),
    ("watch --logs prints the tail once it finishes", None, None,
     ["watch", "{job}", "--logs", "--lines", "10"], OK,
     lambda j: (lambda lines: [ln.split()[-1] for ln in lines] == ["success", "tail-of-4242"]),
     "", None),
    ("watch polls a running job until it is terminal", "script", None,
     ["watch", "{job}", "--interval", "1"], OK,
     lambda j: (lambda lines: [ln.split()[-1] for ln in lines] == ["running", "success"]),
     "", None),
]


def test_every_command_on_a_submitted_job(subtests, tmp_path, monkeypatch, capsys):
    """A job submitted through the ``command`` backend answers every job command exactly."""
    for index, (case_id, status, log, tail, code, out, err, stored) in enumerate(JOB_CASES):
        with subtests.test(msg=case_id):
            root, settings = _fresh_state(tmp_path, monkeypatch, f"job{index}")
            if status == "script":
                status = f"{_status_script(root)} " + "{external_id}"
            profile_argv = _command_profile_argv(
                status=status or "/bin/echo COMPLETED {external_id}",
                log="/bin/echo tail-of-{external_id}" if log is None else log)
            assert _run(capsys, profile_argv)[0] == OK
            got = _run(capsys, ["submit", "measure", "--settings", str(settings),
                                "--profile", "cloud"])
            assert got[0] == OK and got[2] == ""
            job = got[1].split(" as ", 1)[1].split(" ", 1)[0]
            assert got[1] == f"Submitted measure as {job} (cloud/4242).\n"
            assert len(job) == 32

            # Only the watch loop's own sleeps are recorded: patching the
            # global time.sleep would also catch subprocess's wait back-off.
            slept = []
            clock = types.SimpleNamespace(**vars(cli_remote.time))
            clock.sleep = slept.append
            with monkeypatch.context() as patch:
                patch.setattr(cli_remote, "time", clock)
                result = _run(capsys, [part.replace("{job}", job) for part in tail])
            assert (result[0], result[2]) == (code, err)
            if out is None:
                pass
            elif tail[0] == "watch":
                assert out(job)([ln for ln in result[1].splitlines() if ln]), result[1]
            elif "--json" in tail:
                assert out(job)(json.loads(result[1])), result[1]
            else:
                assert result[1] == out(job)
            if stored is not None:
                assert stored(_listed(capsys), root)
            if tail[0] == "watch":
                # one poll interval elapsed only for the job that was running,
                # floored at two seconds however small --interval was
                assert slept == ([2] if status and "status.sh" in status else [])


# ---------------------------------------------------------------------------
# main's exit-code mapping
# ---------------------------------------------------------------------------

def test_main_maps_interrupts_and_missing_commands_to_exit_codes(subtests, tmp_path,
                                                                 monkeypatch, capsys):
    """Ctrl-C is exit 1 with one word; no subcommand is argparse's exit 2."""
    _fresh_state(tmp_path, monkeypatch, "main")
    with subtests.test(msg="an interrupt is reported rather than a traceback"):
        def interrupt(args):
            raise KeyboardInterrupt

        with monkeypatch.context() as patch:
            patch.setattr(cli_remote, "_cmd_list", interrupt)
            assert _run(capsys, ["list"]) == (RUNTIME, "", "interrupted\n")
    with subtests.test(msg="a missing command is an argparse usage error"):
        with pytest.raises(SystemExit) as excinfo:
            cli_remote.main([])
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert err.startswith("usage: spacr-remote [-h]")
        assert err.endswith("spacr-remote: error: the following arguments are required: command\n")


# ---------------------------------------------------------------------------
# resources / allocate
# ---------------------------------------------------------------------------

CLUSTER = {"host": "fake-host", "user": "fake-user", "scheduler": "slurm",
           "partitions": [{"partition": "cpu", "default": True, "nodes": 1, "cpus": 4,
                           "mem_mb": 8192, "gpus": 0, "gpu_types": [],
                           "time_limit": "1:00:00"}],
           "accounts": [{"account": "fake-account", "partition": "cpu", "qos": ["normal"]}]}
CPU_PROBLEM = "8 CPUs per task were requested, but nodes in cpu have at most 4."


def test_resources_and_allocate_use_the_saved_profile_read_only(subtests, tmp_path,
                                                                monkeypatch, capsys):
    """``resources`` reports the probed cluster and its problems; ``allocate`` names the holder."""
    _fresh_state(tmp_path, monkeypatch, "resources")
    profile = ExecutionProfile("cluster", "slurm", host="fake-host", workdir="/work",
                               resources={"cpus": "8"})
    store = ProfileStore()
    store.save(profile)
    before = store.path.read_bytes()
    monkeypatch.setattr(cli_remote, "_probe_cluster",
                        lambda chosen: CLUSTER if chosen == profile else None)
    holder = RemoteJob("abcdef123456", "allocation", "cluster", "slurm", external_id="1234")
    requested = []
    monkeypatch.setattr(cli_remote, "_start_allocation",
                        lambda manager, name: requested.append(name) or holder)

    with subtests.test(msg="resources --json"):
        code, out, err = _run(capsys, ["resources", "-p", "cluster", "--json"])
        assert (code, err) == (OK, "")
        assert json.loads(out) == {"cluster": CLUSTER, "problems": [CPU_PROBLEM]}
    with subtests.test(msg="resources as text"):
        assert _run(capsys, ["resources", "-p", "cluster"]) == (
            OK,
            "fake-host as fake-user; scheduler: slurm\n"
            "PARTITION            NODES  CPUS  MEM_GB  GPUS  TIME_LIMIT\n"
            "cpu*                     1     4       8     0  1:00:00\n"
            "account fake-account partition cpu qos normal\n",
            f"warning: {CPU_PROBLEM}\n")
    with subtests.test(msg="allocate"):
        assert _run(capsys, ["allocate", "-p", "cluster"]) == (
            OK,
            "Requested allocation abcdef123456 (Slurm 1234). When it is running: "
            "spacr-remote submit MODULE -s SETTINGS -p cluster --allocation abcdef123456\n",
            "")
        assert requested == ["cluster"]
    with subtests.test(msg="the profile store is untouched"):
        assert store.path.read_bytes() == before
