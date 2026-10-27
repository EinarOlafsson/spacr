"""Item 732: sign-in methods, editable job scripts, resources, login-node
refusal, allocations, log following and the cluster System probe.

Every remote is fake: a queued command runner stands in for ssh and the
scheduler, and user names, hosts, accounts and paths are invented. Where a
piece of shell matters (the login-node guard, the log-offset reader, the
exit-127 runner check), it is run for real with the local ``sh``/``bash``
against stub programs, so the test checks what a shell does with it.
"""
from __future__ import annotations

import json
import base64
import os
import stat
import subprocess
import time
from collections import deque
from dataclasses import replace

import pytest

from spacr import cli_remote
from spacr.remote_execution import (
    _PROFILE_PRESETS,
    _SCRIPT_PRESETS,
    CommandResult,
    ExecutionProfile,
    JobStore,
    ProfileStore,
    RemoteExecutionError,
    RemoteJobManager,
    _auth_mode,
    _exit_hint,
    _job_script_text,
    _log_chunk,
    _login_guard,
    _parse_cluster_probe,
    _parse_lsf_status,
    _parse_pbs_status,
    _preview_job_script,
    _remote_argv,
    _resource_problems,
    _run_command,
    _scheduler_directives,
    _signin_command,
    _start_allocation,
    _submit_job,
)

FAKE_USER = "ada_fake"
FAKE_ACCOUNT = "labfake0"


class QueueRunner:
    """Answer each command with the next scripted result and record it."""

    def __init__(self, *results):
        self.results = deque(results)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        result = self.results.popleft()
        if isinstance(result, Exception):
            raise result
        return result


def _manager(tmp_path, profile, runner):
    profiles = ProfileStore(tmp_path / "profiles.json")
    profiles.save(profile)
    return RemoteJobManager(profiles, JobStore(tmp_path / "jobs.json"), runner)


def _slurm(**changes):
    profile = ExecutionProfile("fakelakes", "slurm", host="fakelakes", workdir=f"/projects/{FAKE_USER}/work")
    return replace(profile, **changes).validate()


# ---------------------------------------------------------------- sign-in

def test_openssh_is_the_unchanged_default():
    profile = ExecutionProfile("w", "ssh", host="fake-ws", workdir="/w")
    assert _auth_mode(profile) == "openssh"
    assert _remote_argv(profile, ["hostname"]) == ["ssh", "fake-ws", "hostname"]


def test_a_profile_saved_before_sign_in_methods_keeps_interactive_sign_in():
    profile = ExecutionProfile.from_dict({"name": "old", "backend": "slurm", "host": "h",
                                          "workdir": "/w", "ssh_interactive": True})
    assert _auth_mode(profile) == "interactive"
    assert _remote_argv(profile, ["hostname"])[0] == "__spacr_interactive_ssh__"


def test_interactive_sign_in_routes_through_the_in_spacr_session():
    profile = _slurm(auth_mode="interactive", ssh_interactive=True)
    argv = _remote_argv(profile, ["sinfo"])
    assert argv[0] == "__spacr_interactive_ssh__"
    assert json.loads(argv[1])["host"] == "fakelakes"
    assert argv[2] == "sinfo"


def test_kerberos_sign_in_asks_openssh_for_gssapi():
    argv = _remote_argv(_slurm(auth_mode="gssapi"), ["hostname"])
    assert argv[0] == "ssh"
    assert "GSSAPIAuthentication=yes" in argv
    assert "GSSAPIDelegateCredentials=yes" in argv
    assert argv[-2:] == ["fakelakes", "hostname"]


def test_custom_ssh_command_places_host_and_command():
    placed = _slurm(auth_mode="custom", ssh_command="corp-ssh --via jump.example.org {host} -- {command}")
    assert _remote_argv(placed, ["squeue", "-h"]) == [
        "corp-ssh", "--via", "jump.example.org", "fakelakes", "--", "squeue -h"]
    appended = _slurm(auth_mode="custom", ssh_command="ssh -J jump.example.org {host}")
    assert _remote_argv(appended, ["hostname"]) == ["ssh", "-J", "jump.example.org", "fakelakes", "hostname"]


@pytest.mark.parametrize("command", ["sshpass -p hunter2 ssh {host}", "ssh -o Password=x {host}", ""])
def test_custom_ssh_command_may_not_carry_a_password_or_be_empty(command):
    with pytest.raises(RemoteExecutionError):
        _slurm(auth_mode="custom", ssh_command=command)


def test_unknown_sign_in_method_is_refused():
    with pytest.raises(RemoteExecutionError, match="sign-in method"):
        _slurm(auth_mode="carrier-pigeon")


def test_controlmaster_without_controlpath_gets_a_private_socket(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_REMOTE_STATE_DIR", str(tmp_path))
    profile = _slurm(ssh_options={"controlmaster": "auto", "controlpersist": "4h"})
    argv = _remote_argv(profile, ["hostname"])
    paths = [item for item in argv if item.startswith("ControlPath=")]
    assert len(paths) == 1 and paths[0].endswith("cm-%C")
    assert "ControlMaster=auto" in argv and "ControlPersist=4h" in argv
    explicit = _slurm(ssh_options={"controlmaster": "auto", "controlpath": "/tmp/fake-%C"})
    assert [item for item in _remote_argv(explicit, ["x"]) if item.startswith("ControlPath=")] == [
        "ControlPath=/tmp/fake-%C"]


def test_terminal_sign_in_command_opens_a_reusable_master(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_REMOTE_STATE_DIR", str(tmp_path))
    command = _signin_command(_slurm())
    assert command.startswith("ssh ")
    assert "ControlMaster=auto" in command and "ControlPersist=4h" in command
    assert command.endswith("-fN fakelakes")


# ---------------------------------------------------------------- templates

def test_the_template_decides_the_command_and_nothing_appends_settings(tmp_path):
    template = "#!/bin/bash -l\n{env_setup}\nmy-wrapper {module} {settings_path} --threads ${SLURM_CPUS_PER_TASK:-1}\n"
    profile = _slurm(job_script=template, env_setup="module load fakeenv", direct_run=False)
    runner = QueueRunner(CommandResult(0), CommandResult(0, "4711\n"))
    job = _submit_job(_manager(tmp_path, profile, runner), "mask", {"src": "/data"}, "fakelakes")
    script = runner.calls[1][1]["input_text"]
    assert script == job.job_script
    assert f"my-wrapper mask /projects/{FAKE_USER}/work/spacr-jobs/{job.job_id}/settings.json --threads" in script
    assert "--settings" not in script
    assert "${SLURM_CPUS_PER_TASK:-1}" in script
    assert "module load fakeenv" in script
    assert (tmp_path / "jobs" / job.job_id / "job.sh").read_text() == script


def test_the_default_script_still_builds_the_spacr_run_line_with_settings():
    script = _preview_job_script(_slurm(), "measure")
    assert "exec spacr-run measure --settings /projects/ada_fake/work/spacr-jobs/preview/settings.json" in script


def test_an_unknown_placeholder_is_refused_when_the_profile_is_saved():
    with pytest.raises(RemoteExecutionError, match="setings_path"):
        _slurm(job_script="spacr-run {module} {setings_path}\n")


def test_values_with_spaces_are_quoted_and_code_placeholders_are_not():
    profile = _slurm(workdir="/projects/ada_fake/my work", env_setup="conda activate spacr")
    script = _preview_job_script(profile, "mask")
    assert "cd '/projects/ada_fake/my work'" in script
    assert "\nconda activate spacr\n" in script


# ---------------------------------------------------------------- resources

GPU_RESOURCES = {"partition": "gpu", "account": FAKE_ACCOUNT, "gpus": "1", "gpu_type": "v100",
                 "cpus": "20", "mem": "90G", "time": "48:00:00", "mail_type": "END,FAIL",
                 "mail_user": "ada@example.org", "extra": "--constraint=fake\n#SBATCH --exclusive"}


def test_resources_become_sbatch_headers_ahead_of_the_guard_and_the_commands():
    script = _preview_job_script(_slurm(resources=GPU_RESOURCES), "mask")
    lines = script.splitlines()
    for header in ("#SBATCH --partition=gpu", f"#SBATCH --account={FAKE_ACCOUNT}", "#SBATCH --gpus=v100:1",
                   "#SBATCH --cpus-per-task=20", "#SBATCH --mem=90G", "#SBATCH --time=48:00:00",
                   "#SBATCH --mail-type=END,FAIL", "#SBATCH --mail-user=ada@example.org",
                   "#SBATCH --constraint=fake", "#SBATCH --exclusive"):
        assert header in lines
    last_header = max(index for index, line in enumerate(lines) if line.startswith("#SBATCH"))
    first_command = next(index for index, line in enumerate(lines) if line and not line.startswith("#"))
    assert last_header < first_command
    assert lines[first_command].startswith("spacr_host=")


def test_pbs_and_lsf_directives_carry_the_same_request():
    pbs = _scheduler_directives(replace(_slurm(resources=GPU_RESOURCES), backend="pbs"), "spacr-mask-1", "/w/log")
    assert "#PBS -q gpu" in pbs and f"#PBS -A {FAKE_ACCOUNT}" in pbs
    assert "#PBS -l select=1:ncpus=20:ngpus=1:mem=92160mb" in pbs
    assert "#PBS -l walltime=48:00:00" in pbs and "#PBS -m ea" in pbs
    lsf = _scheduler_directives(replace(_slurm(resources=GPU_RESOURCES), backend="lsf"), "spacr-mask-1", "/w/log")
    assert "#BSUB -q gpu" in lsf and "#BSUB -n 20" in lsf
    assert '#BSUB -gpu "num=1:gmodel=v100"' in lsf and "#BSUB -W 48:00" in lsf


@pytest.mark.parametrize("key,value", [("mem", "90G; rm -rf ~"), ("gpus", "two"), ("partition", "gpu standard"),
                                       ("account", "$(id)"), ("time", "forever"), ("colour", "blue")])
def test_resources_that_could_become_more_than_one_value_are_refused(key, value):
    with pytest.raises(RemoteExecutionError):
        _slurm(resources={key: value})


# ---------------------------------------------------------------- login nodes

def test_a_direct_run_aimed_at_a_login_host_is_refused_before_anything_is_sent(tmp_path):
    profile = ExecutionProfile("lake", "ssh", host=f"{FAKE_USER}@cl-login2.example.edu", workdir="/w")
    runner = QueueRunner()
    with pytest.raises(RemoteExecutionError, match="login node"):
        _submit_job(_manager(tmp_path, profile, runner), "mask", {"src": "/d"}, "lake")
    assert runner.calls == []


def test_declared_login_host_patterns_match_the_ssh_host_name(tmp_path):
    profile = ExecutionProfile("lake", "ssh", host="lake", workdir="/w", login_hosts="fakelakes*",
                               ssh_options={"hostname": "fakelakes.example.edu"})
    with pytest.raises(RemoteExecutionError, match="fakelakes.example.edu"):
        _submit_job(_manager(tmp_path, profile, QueueRunner()), "mask", {"src": "/d"}, "lake")


def test_the_explicit_direct_run_opt_in_skips_the_refusal_and_the_guard(tmp_path):
    profile = ExecutionProfile("lake", "ssh", host="cl-login2", workdir="/w", direct_run=True)
    runner = QueueRunner(CommandResult(0), CommandResult(0, "321\n"))
    job = _submit_job(_manager(tmp_path, profile, runner), "mask", {"src": "/d"}, "lake")
    assert job.status == "running"
    assert "spaCR refused" not in runner.calls[1][0][-1]
    assert "spaCR refused" not in job.job_script


def test_login_host_patterns_cannot_smuggle_shell():
    with pytest.raises(RemoteExecutionError, match="Login host pattern"):
        _slurm(login_hosts="login*) rm -rf ~;;")


def _stub(folder, name, output=""):
    path = folder / name
    path.write_text(f"#!/bin/sh\necho {output}\n" if output else "#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.mark.parametrize("host,scheduler,job_env,refused", [
    ("cl-login2", False, {}, True),
    ("cl-login2", True, {"SLURM_JOB_ID": "1234"}, False),
    ("gpu-node7", True, {}, True),
    ("gpu-node7", True, {"PBS_JOBID": "77.fake"}, False),
    ("lab-workstation", False, {}, False),
])
def test_the_remote_guard_refuses_login_nodes_in_a_real_shell(tmp_path, host, scheduler, job_env, refused):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir, "hostname", host)
    if scheduler:
        _stub(bin_dir, "sbatch")
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", **job_env}
    script = _login_guard(_slurm()) + "echo started\n"
    result = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True, timeout=30)
    if refused:
        assert result.returncode == 97
        assert "spaCR refused to start on " + host in result.stderr
        assert "started" not in result.stdout
    else:
        assert result.returncode == 0 and "started" in result.stdout


def test_the_ssh_launch_runs_the_guard_before_starting_anything(tmp_path):
    profile = ExecutionProfile("ws", "ssh", host="fake-ws", workdir="/w")
    refusal = CommandResult(97, "", "spaCR refused to start on cl-login2: it is a cluster login node.")
    runner = QueueRunner(CommandResult(0), refusal)
    with pytest.raises(RemoteExecutionError, match="refused to start"):
        _submit_job(_manager(tmp_path, profile, runner), "mask", {"src": "/d"}, "ws")
    launch = runner.calls[1][0][-1]
    assert launch.index("spacr_host=") < launch.index(") >")
    [job] = JobStore(tmp_path / "jobs.json").list()
    assert job.status == "failed"


# ---------------------------------------------------------------- allocations

def test_an_allocation_is_requested_with_the_profile_resources(tmp_path):
    profile = _slurm(resources={"gpus": "1", "cpus": "8", "time": "02:00:00"})
    runner = QueueRunner(CommandResult(0), CommandResult(0, "5550\n"))
    manager = _manager(tmp_path, profile, runner)
    holder = _start_allocation(manager, "fakelakes")
    assert holder.module == "allocation" and holder.external_id == "5550" and holder.status == "queued"
    argv, kwargs = runner.calls[1]
    assert "sbatch --parsable" in argv[-1]
    assert "#SBATCH --gpus=1" in kwargs["input_text"] and "#SBATCH --cpus-per-task=8" in kwargs["input_text"]
    assert "sleep infinity" in kwargs["input_text"]


def test_allocations_need_a_slurm_profile(tmp_path):
    profile = ExecutionProfile("pbs", "pbs", host="fake-pbs", workdir="/w")
    with pytest.raises(RemoteExecutionError, match="Slurm"):
        _start_allocation(_manager(tmp_path, profile, QueueRunner()), "pbs")


def test_jobs_run_inside_a_running_allocation_as_srun_steps(tmp_path):
    profile = _slurm(resources={"gpus": "1", "cpus": "8"})
    runner = QueueRunner(CommandResult(0), CommandResult(0, "5550\n"),
                         CommandResult(0, "PENDING\n"))
    manager = _manager(tmp_path, profile, runner)
    holder = _start_allocation(manager, "fakelakes")
    manager.refresh(holder.job_id, include_logs=False)
    with pytest.raises(RemoteExecutionError, match="wait until it is running"):
        _submit_job(manager, "mask", {"src": "/d"}, "fakelakes", holder.job_id)
    runner.results.extend([CommandResult(0, "RUNNING\n"), CommandResult(0), CommandResult(0, "8080\n")])
    manager.refresh(holder.job_id, include_logs=False)
    job = _submit_job(manager, "mask", {"src": "/d"}, "fakelakes", holder.job_id[:10])
    assert job.allocation_id == "5550" and job.status == "running"
    launch = runner.calls[-1][0][-1]
    assert "srun --jobid=5550 --overlap --ntasks=1" in launch
    assert "--cpus-per-task=8" in launch and "--gpus=1" in launch
    assert not any("sbatch" in call[0][-1] for call in runner.calls[3:])
    runner.results.extend([CommandResult(0, "EXIT:0\n"), CommandResult(0)])
    assert manager.refresh(job.job_id, include_logs=False).status == "success"
    assert "exit-code" in runner.calls[-1][0][-1] and "squeue -h -j 5550" in runner.calls[-1][0][-1]


def test_cancelling_a_job_inside_an_allocation_cancels_only_its_step(tmp_path):
    profile = _slurm()
    runner = QueueRunner(CommandResult(0), CommandResult(0, "5550\n"), CommandResult(0, "RUNNING\n"),
                         CommandResult(0), CommandResult(0, "8080\n"), CommandResult(0))
    manager = _manager(tmp_path, profile, runner)
    holder = _start_allocation(manager, "fakelakes")
    manager.refresh(holder.job_id, include_logs=False)
    job = _submit_job(manager, "mask", {"src": "/d"}, "fakelakes", holder.job_id)
    assert manager.cancel(job.job_id).status == "cancelled"
    script = runner.calls[-1][0][-1]
    assert "squeue -h -s -j 5550" in script and f"spacr-mask-{job.job_id[:8]}" in script
    assert "scancel 5550" not in script


# ---------------------------------------------------------------- logs

def test_log_chunks_continue_from_the_reported_offset(tmp_path):
    profile = _slurm()
    runner = QueueRunner(CommandResult(0), CommandResult(0, "61\n"),
                         CommandResult(0, "0\n" + base64.b64encode(b"line one\nli").decode()),
                         CommandResult(0, "11\n" + base64.b64encode(b"ne two\n").decode()))
    manager = _manager(tmp_path, profile, runner)
    job = _submit_job(manager, "mask", {"src": "/d"}, "fakelakes")
    assert _log_chunk(manager, job.job_id, 0) == ("line one\nli", 11)
    assert _log_chunk(manager, job.job_id, 11) == ("ne two\n", 18)
    assert "start=11" in runner.calls[-1][0][-1]


def test_the_log_reader_is_exact_in_a_real_shell(tmp_path):
    log = tmp_path / "job.log"
    profile = ExecutionProfile("local", "slurm", workdir=str(tmp_path))
    manager = _manager(tmp_path, profile, _run_command)
    record = manager.jobs.path
    from spacr.remote_execution import RemoteJob
    job = RemoteJob("abc123", "mask", "local", "slurm", log_reference=str(log), profile=profile.to_dict())
    JobStore(record).save(job)
    assert _log_chunk(manager, "abc123", 0) == ("", 0)
    log.write_text("first\n")
    assert _log_chunk(manager, "abc123", 0) == ("first\n", 6)
    with log.open("a") as handle:
        handle.write("second\nthird\n")
    assert _log_chunk(manager, "abc123", 6, limit=7) == ("second\n", 13)
    assert _log_chunk(manager, "abc123", 13) == ("third\n", 19)
    log.write_text("new\n")
    assert _log_chunk(manager, "abc123", 19) == ("new\n", 4)


# ---------------------------------------------------------------- System panel

PROBE = f"""##host
cl-login2.example.edu
##user
{FAKE_USER}
##cpus
36
##mem_mb
191000
##gpu
##sinfo
standard*|up|14-00:00:00|300|36|180000|(null)
gpu|up|14-00:00:00|20|40|370000|gpu:v100:2(S:0-1)
gpu|up|14-00:00:00|4|64|1000000|gpu:a100:4
largemem|up|7-00:00:00|8|36|1500000|(null)
##assoc
{FAKE_ACCOUNT}|gpu|normal,priority
{FAKE_ACCOUNT}||normal
##pbs
##lsf
##end
"""


def test_the_probe_reads_partitions_nodes_gpus_and_accounts():
    cluster = _parse_cluster_probe(PROBE)
    assert cluster["host"] == "cl-login2.example.edu" and cluster["user"] == FAKE_USER
    assert cluster["scheduler"] == "slurm" and cluster["cpus"] == 36
    standard = cluster["partitions"][0]
    assert standard["partition"] == "standard" and standard["default"] and standard["gpus"] == 0
    gpu = [row for row in cluster["partitions"] if row["partition"] == "gpu"]
    assert [(row["gpus"], row["gpu_types"], row["cpus"]) for row in gpu] == [(2, ["v100"], 40), (4, ["a100"], 64)]
    assert cluster["accounts"][0] == {"account": FAKE_ACCOUNT, "partition": "gpu", "qos": ["normal", "priority"]}


def test_resources_are_checked_against_what_the_cluster_reported():
    cluster = _parse_cluster_probe(PROBE)
    fits = {"partition": "gpu", "gpus": "4", "gpu_type": "a100", "cpus": "40", "mem": "300G",
            "time": "2-00:00:00", "account": FAKE_ACCOUNT, "qos": "normal"}
    assert _resource_problems(fits, cluster) == []
    problems = _resource_problems({"partition": "gpu", "gpus": "8", "cpus": "128", "mem": "2T",
                                   "time": "20-00:00:00", "gpu_type": "h100",
                                   "account": "notmine", "qos": "gold"}, cluster)
    text = "\n".join(problems)
    for fragment in ("8 GPUs", "128 CPUs", "MB of memory", "wall time", "h100", "notmine", "gold"):
        assert fragment in text
    assert "is not on this cluster" in _resource_problems({"partition": "nosuch"}, cluster)[0]
    assert "36 CPUs" not in "\n".join(_resource_problems({"cpus": "36"}, cluster))


def test_pbs_and_lsf_states_are_read():
    assert _parse_pbs_status("Job Id: 1.fake\n    job_state = R\n") == ("running", None)
    assert _parse_pbs_status("    job_state = F\n    Exit_status = 127\n") == ("failed", 127)
    assert _parse_pbs_status("    job_state = C\n    exit_status = 0\n") == ("success", 0)
    assert _parse_lsf_status("EXIT 127") == ("failed", 127)
    assert _parse_lsf_status("PEND -") == ("pending", None)
    assert _parse_lsf_status("DONE -") == ("success", 0)


def test_cli_resources_prints_the_partitions_and_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPACR_REMOTE_STATE_DIR", str(tmp_path))
    ProfileStore().save(_slurm(resources={"gpus": "9", "partition": "gpu"}))
    monkeypatch.setattr("spacr.remote_execution._run_command", lambda argv, **kw: CommandResult(0, PROBE))
    monkeypatch.setattr(cli_remote, "_probe_cluster",
                        lambda profile: _parse_cluster_probe(PROBE))
    assert cli_remote.main(["resources", "-p", "fakelakes"]) == 0
    out = capsys.readouterr()
    assert "standard*" in out.out and FAKE_ACCOUNT in out.out
    assert "9 GPUs" in out.err
    assert json.loads((tmp_path / "clusters.json").read_text())["fakelakes"]["scheduler"] == "slurm"


def test_cli_profile_add_takes_resources_environment_and_sign_in(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPACR_REMOTE_STATE_DIR", str(tmp_path))
    assert cli_remote.main(["profile", "add", "fl", "--backend", "slurm", "--host", "fl",
                            "--workdir", "/w", "--auth-mode", "gssapi", "--resource", "gpus=1",
                            "--resource", "time=01:00:00", "--env-setup", "module load mamba"]) == 0
    saved = ProfileStore().get("fl")
    assert saved.auth_mode == "gssapi" and saved.resources == {"gpus": "1", "time": "01:00:00"}
    assert cli_remote.main(["script", "mask", "-p", "fl"]) == 0
    script = capsys.readouterr().out
    assert "#SBATCH --gpus=1" in script and "module load mamba" in script


# ---------------------------------------------------------------- exit 127

def test_the_great_lakes_style_preset_sets_up_the_environment_before_spacr_runs():
    preset = _PROFILE_PRESETS["slurm_module_mamba"]
    profile = _slurm(env_setup=preset["env_setup"], resources=preset["resources"],
                     login_hosts=preset["login_hosts"])
    script = _preview_job_script(profile, "mask")
    run_line = script.index("exec spacr-run mask --settings")
    assert script.index("module load mamba") < script.index("mamba activate spacr") < run_line
    assert script.index("command -v spacr-run") < run_line
    assert "#SBATCH --gpus=1" in script and "#SBATCH --cpus-per-task=20" in script
    assert script.startswith("#!/bin/bash -l\n")


def test_a_missing_runner_ends_with_127_and_says_how_to_fix_it(tmp_path):
    profile = ExecutionProfile("ws", "ssh", host="fake-ws", workdir=str(tmp_path), direct_run=True,
                               runner="spacr-run-not-installed-fake")
    script = _preview_job_script(profile, "mask")
    path = tmp_path / "job.sh"
    path.write_text(script)
    result = subprocess.run(["bash", str(path)], capture_output=True, text=True, timeout=30,
                            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert result.returncode == 127
    assert "was not found on PATH (exit 127)" in result.stderr
    assert "module load mamba" in result.stderr


def test_a_job_that_exits_127_is_explained(tmp_path):
    runner = QueueRunner(CommandResult(0), CommandResult(0, "88\n"), CommandResult(0, ""),
                         CommandResult(0, "FAILED|127:0\n"))
    manager = _manager(tmp_path, _slurm(), runner)
    job = _submit_job(manager, "mask", {"src": "/d"}, "fakelakes")
    failed = manager.refresh(job.job_id, include_logs=False)
    assert failed.status == "failed" and failed.exit_code == 127
    assert "Exit code 127" in failed.error and "mamba activate spacr" in failed.error
    assert "login node" in _exit_hint(97, _slurm())
    assert _exit_hint(1, _slurm()) == ""


def test_every_script_preset_renders():
    for backend in ("ssh", "slurm", "pbs", "lsf"):
        profile = replace(_slurm(), backend=backend, job_script=_SCRIPT_PRESETS[backend]).validate()
        job_text = _preview_job_script(profile, "mask")
        assert "exec spacr-run mask --settings" in job_text
        assert "{" + "settings_path}" not in job_text


def test_the_rendered_guard_follows_direct_run(tmp_path):
    from spacr.remote_execution import RemoteJob
    job = RemoteJob("j1", "mask", "p", "ssh")
    guarded = _job_script_text(_slurm(), job, "/w/j", "/w/j/s.json", "/w/j/log")
    assert "spacr_host=" in guarded
    plain = _job_script_text(ExecutionProfile("p", "ssh", host="h", workdir="/w", direct_run=True),
                             job, "/w/j", "/w/j/s.json", "/w/j/log")
    assert "spacr_host=" not in plain


@pytest.mark.parametrize('option', ['PreferredAuthentications=password', 'PasswordAuthentication=yes'])
def test_custom_ssh_password_authentication_options_are_not_secrets(option):
    profile = _slurm(auth_mode='custom', ssh_command=f'ssh -o {option} {{host}}')
    assert _remote_argv(profile, ['hostname']) == ['ssh', '-o', option, 'fakelakes', 'hostname']


@pytest.mark.parametrize('field,value', [('host', 'another-cluster'), ('ssh_options', {'user': 'another-user'}),
                                        ('auth_mode', 'gssapi')])
def test_a_saved_allocation_cannot_cross_an_edited_profile_connection(tmp_path, field, value):
    from spacr.remote_execution import RemoteJob
    profile = _slurm()
    runner = QueueRunner()
    manager = _manager(tmp_path, profile, runner)
    manager.jobs.save(RemoteJob('holder', 'allocation', profile.name, 'slurm', status='running',
                               external_id='5550', profile=profile.to_dict()))
    manager.profiles.save(replace(profile, **{field: value}))
    with pytest.raises(RemoteExecutionError, match='original connection'):
        _submit_job(manager, 'mask', {'src': '/d'}, profile.name, 'holder')
    assert runner.calls == []


def test_resource_edits_do_not_invalidate_an_allocations_connection(tmp_path):
    from spacr.remote_execution import RemoteJob
    profile = _slurm()
    runner = QueueRunner(CommandResult(0), CommandResult(0, '8080\n'))
    manager = _manager(tmp_path, profile, runner)
    manager.jobs.save(RemoteJob('holder', 'allocation', profile.name, 'slurm', status='running',
                               external_id='5550', profile=profile.to_dict()))
    manager.profiles.save(replace(profile, resources={'cpus': '2'}))
    job = _submit_job(manager, 'mask', {'src': '/d'}, profile.name, 'holder')
    assert job.allocation_id == '5550' and '--cpus-per-task=2' in runner.calls[-1][0][-1]


def test_cancelling_a_waiting_allocation_launcher_preserves_unrelated_processes(tmp_path, monkeypatch):
    from spacr.remote_execution import RemoteJob, _SlurmBackend
    folder = tmp_path / 'bin'
    folder.mkdir()
    ready, unwanted = tmp_path / 'ready', tmp_path / 'unwanted-compute'
    (folder / 'srun').write_text(f'#!/bin/sh\nprintf ready > {ready}\nsleep 30\nprintf bad > {unwanted}\n')
    (folder / 'srun').chmod(0o700)
    _stub(folder, 'squeue', '5550.1 unrelated-step')
    _stub(folder, 'scancel')
    monkeypatch.setenv('PATH', str(folder) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('COLUMNS', '80')
    profile = ExecutionProfile('local', 'slurm', workdir=str(tmp_path))
    job = RemoteJob('a1b2c3d4', 'mask', 'local', 'slurm', allocation_id='5550')
    backend = _SlurmBackend(_run_command)
    unrelated = subprocess.Popen(['sleep', '30'], start_new_session=True)
    try:
        backend.submit(profile, job, '{}')
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), 'The actual waiting launcher must start before cancellation'
        backend.cancel(profile, job)
        assert job.status == 'cancelled'
        assert unrelated.poll() is None
        assert not unwanted.exists()
        assert (tmp_path / 'spacr-jobs' / job.job_id / 'launcher-group').read_text().strip() == job.external_id
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)
        if job.external_id:
            try:
                os.killpg(int(job.external_id), 9)
            except ProcessLookupError:
                pass


def test_allocation_cancellation_refuses_a_recycled_unrelated_process_group(tmp_path):
    from spacr.remote_execution import RemoteJob, _SlurmBackend
    unrelated = subprocess.Popen(['sleep', '30'], start_new_session=True)
    job_dir = tmp_path / 'job'
    job_dir.mkdir()
    (job_dir / 'launcher-group').write_text(str(unrelated.pid))
    job = RemoteJob('a1b2c3d4', 'mask', 'local', 'slurm', status='running', allocation_id='5550',
                    external_id=str(unrelated.pid), remote_job_dir=str(job_dir))
    profile = ExecutionProfile('local', 'slurm', workdir=str(tmp_path))
    try:
        with pytest.raises(RemoteExecutionError, match='ownership changed'):
            _SlurmBackend(_run_command).cancel(profile, job)
        assert job.status == 'running' and unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_allocation_cancellation_does_not_report_success_when_scheduler_query_fails(tmp_path, monkeypatch):
    from spacr.remote_execution import RemoteJob, _SlurmBackend
    folder = tmp_path / 'bin'
    folder.mkdir()
    (folder / 'squeue').write_text('#!/bin/sh\nexit 1\n')
    (folder / 'squeue').chmod(0o700)
    monkeypatch.setenv('PATH', str(folder) + os.pathsep + os.environ['PATH'])
    job = RemoteJob('abc123', 'mask', 'local', 'slurm', status='running', allocation_id='5550',
                    external_id='8080', remote_job_dir=str(tmp_path))
    profile = ExecutionProfile('local', 'slurm', workdir=str(tmp_path))
    with pytest.raises(RemoteExecutionError):
        _SlurmBackend(_run_command).cancel(profile, job)
    assert job.status == 'running'


@pytest.mark.parametrize('state,expected', [('CANCELLED', 'cancelled'), ('TIMEOUT', 'failed'), ('COMPLETED', 'failed')])
def test_step_without_exit_file_uses_the_terminal_holder_accounting(tmp_path, state, expected):
    from spacr.remote_execution import RemoteJob
    profile = _slurm()
    runner = QueueRunner(CommandResult(0, 'UNKNOWN\n'), CommandResult(0, ''),
                         CommandResult(0, f'{state}|0:0\n'))
    manager = _manager(tmp_path, profile, runner)
    job = RemoteJob('abc123', 'mask', profile.name, 'slurm', status='running', allocation_id='5550',
                    external_id='8080', remote_job_dir='/work/job', profile=profile.to_dict())
    manager.jobs.save(job)
    assert manager.refresh(job.job_id, include_logs=False).status == expected
    assert '5550' in runner.calls[-1][0][-1] and 'sacct' in runner.calls[-1][0][-1]


def test_log_chunks_preserve_utf8_crlf_and_actual_byte_offsets(tmp_path):
    from spacr.remote_execution import RemoteJob
    log = tmp_path / 'unicode.log'
    payload = 'abcé漢字\r\n'.encode()
    log.write_bytes(payload)
    profile = ExecutionProfile('local', 'slurm', workdir=str(tmp_path))
    manager = _manager(tmp_path, profile, _run_command)
    manager.jobs.save(RemoteJob('abc123', 'mask', profile.name, 'slurm', log_reference=str(log),
                               profile=profile.to_dict()))
    offset, pieces = 0, []
    for _ in range(len(payload)):
        text, new = _log_chunk(manager, 'abc123', offset, limit=1)
        pieces.append(text)
        if new == offset:
            break
        offset = new
    assert ''.join(pieces).encode() == payload
    assert offset == len(payload)
    assert _log_chunk(manager, 'abc123', offset, limit=4) == ('', offset)
    log.write_bytes('新\n'.encode())
    assert _log_chunk(manager, 'abc123', offset, limit=4) == ('新\n', 4)


@pytest.mark.parametrize('offset,limit', [(-1, 4), (1.5, 4), (True, 4), (0, 0), (0, -1), (0, True)])
def test_invalid_log_offsets_and_limits_fail_before_running_commands(tmp_path, offset, limit):
    profile = _slurm()
    runner = QueueRunner()
    manager = _manager(tmp_path, profile, runner)
    with pytest.raises(RemoteExecutionError):
        _log_chunk(manager, 'unused', offset, limit)
    assert runner.calls == []


def test_terminal_cli_follow_drains_all_chunks_without_inserting_newlines(tmp_path, monkeypatch, capsys):
    from spacr.remote_execution import RemoteJob
    monkeypatch.setenv('SPACR_REMOTE_STATE_DIR', str(tmp_path))
    profile = _slurm()
    ProfileStore().save(profile)
    JobStore().save(RemoteJob('abc123', 'mask', profile.name, 'slurm', status='success', external_id='5550',
                             log_reference='/work/log', profile=profile.to_dict()))
    wire = [CommandResult(0, '0\n' + base64.b64encode(b'a' * 262144).decode()),
            CommandResult(0, '262144\n' + base64.b64encode('é\n'.encode()).decode()),
            CommandResult(0, '262147\n')]
    runner = QueueRunner(*wire)
    manager = RemoteJobManager(runner=runner)
    monkeypatch.setattr(cli_remote, 'RemoteJobManager', lambda: manager)
    assert cli_remote.main(['watch', 'abc123', '--follow']) == 0
    output = capsys.readouterr()
    assert output.out == 'a' * 262144 + 'é\n'
    assert len(runner.calls) == 3


def test_partial_utf8_waits_while_active_and_finalizes_only_at_terminal_eof(tmp_path):
    from spacr.remote_execution import RemoteJob
    log = tmp_path / 'partial.log'
    log.write_bytes(b'\xe2\x82')
    profile = ExecutionProfile('local', 'slurm', workdir=str(tmp_path))
    manager = _manager(tmp_path, profile, _run_command)
    job = RemoteJob('abc123', 'mask', profile.name, 'slurm', status='running',
                    log_reference=str(log), profile=profile.to_dict())
    manager.jobs.save(job)
    assert _log_chunk(manager, job.job_id, 0, limit=4) == ('', 0)
    with log.open('ab') as handle:
        handle.write(b'\xac')
    assert _log_chunk(manager, job.job_id, 0, limit=4) == ('€', 3)
    log.write_bytes(b'ab\xe2\x82')
    job.status = 'success'
    manager.jobs.save(job)
    assert _log_chunk(manager, job.job_id, 0, limit=4) == ('ab', 2)
    assert _log_chunk(manager, job.job_id, 2, limit=4) == ('\ufffd', 4)
    assert _log_chunk(manager, job.job_id, 4, limit=4) == ('', 4)


@pytest.mark.parametrize('backend,prefix', [('slurm', '#SBATCH'), ('pbs', '#PBS'), ('lsf', '#BSUB')])
def test_scheduler_output_paths_preserve_spaces_apostrophes_and_original_command_arguments(tmp_path, backend, prefix):
    import shlex
    from spacr.remote_execution import RemoteJob
    workdir = "/work/ada's shared images"
    profile = replace(_slurm(workdir=workdir, resources={'cpus': '2'}), backend=backend)
    script = _preview_job_script(profile, 'mask')
    lines = [line for line in script.splitlines() if line.startswith(prefix + ' ')]
    output = next(line[len(prefix) + 1:] for line in lines if '--output=' in line or line.startswith(prefix + ' -o '))
    arguments = shlex.split(output)
    expected = workdir + '/spacr-jobs/preview/job.log'
    if backend == 'pbs':
        assert output == "-o '/work/ada'\"'\"'s shared images/spacr-jobs/preview/job.log'"
    else:
        assert output == ('--output=' if backend == 'slurm' else '-o ') + '"' + expected + '"'
    assert arguments == (['--output=' + expected] if backend == 'slurm' else ['-o', expected])
    cd = next(line for line in script.splitlines() if line.startswith('cd '))
    assert shlex.split(cd) == ['cd', workdir]
    runner = QueueRunner(CommandResult(0), CommandResult(0, '5550' if backend != 'lsf' else 'Job <5550>'))
    manager = _manager(tmp_path, profile, runner)
    job = _submit_job(manager, 'mask', {'src': '/data'}, profile.name)
    assert job.external_id == '5550'
    if backend == 'slurm':
        remote = shlex.split(runner.calls[-1][0][-1])
        assert remote[remote.index('--output') + 1] == workdir + f'/spacr-jobs/{job.job_id}/job.log'
    assert ('--cpus-per-task=2' if backend == 'slurm' else 'ncpus=2' if backend == 'pbs' else '-n 2') in job.job_script
