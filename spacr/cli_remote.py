"""``spacr-remote`` command-line client for distributed spaCR jobs.

Profiles and jobs are shared with the Qt Distributed Jobs screen.  The client
contains no Qt imports and is suitable for login nodes and SSH-only sessions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Iterable, Optional, Sequence

from .remote_execution import (
    _AUTH_MODES,
    ACTIVE_STATES,
    BACKENDS,
    ExecutionProfile,
    ProfileStore,
    RemoteExecutionError,
    RemoteJob,
    RemoteJobManager,
    _log_chunk,
    _preview_job_script,
    _probe_cluster,
    _resource_problems,
    _save_cluster_cache,
    _start_allocation,
    _submit_job,
)

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2


def _profile_from_args(args: argparse.Namespace) -> ExecutionProfile:
    """Build a validated profile from the ``profile add`` arguments."""
    return ExecutionProfile(
        name=args.name,
        backend=args.backend,
        host=args.host or "",
        workdir=args.workdir or "",
        local_root=args.local_root or "",
        remote_root=args.remote_root or "",
        runner=args.runner or "spacr-run",
        scheduler_options=args.scheduler_options or "",
        submit_command=args.submit_command or "",
        status_command=args.status_command or "",
        cancel_command=args.cancel_command or "",
        log_command=args.log_command or "",
        job_id_pattern=args.job_id_pattern or "",
        poll_seconds=args.poll_seconds,
        auth_mode=args.auth_mode or "",
        ssh_command=args.ssh_command or "",
        resources=_resources_from_args(args.resource or []),
        env_setup=args.env_setup or "",
        job_script=_read_text(args.job_script_file) if args.job_script_file else "",
        login_hosts=args.login_hosts or "",
        direct_run=bool(args.direct_run),
    ).validate()


def _resources_from_args(pairs: Sequence[str]) -> dict:
    """Parse repeated ``--resource KEY=VALUE`` arguments into a mapping.

    :param pairs: the raw arguments.
    :returns: key to value.
    :raises RemoteExecutionError: for an argument without ``=``.
    """
    resources = {}
    for pair in pairs:
        key, separator, value = str(pair).partition("=")
        if not separator or not key.strip():
            raise RemoteExecutionError(f"Resource {pair!r} must be KEY=VALUE, for example gpus=1.")
        resources[key.strip()] = value.strip()
    return resources


def _read_text(path: str) -> str:
    """Read a job-script template file.

    :param path: the file.
    :returns: its text.
    :raises RemoteExecutionError: if it cannot be read.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise RemoteExecutionError(f"Could not read {path}: {exc}") from exc


def _job_row(job: RemoteJob) -> str:
    """Format one compact terminal table row."""
    external = job.external_id or "—"
    error = f"  {job.error}" if job.error else ""
    return (
        f"{job.job_id[:12]:12}  {job.status:10}  {job.module:18}  "
        f"{job.profile_name:18}  {external}{error}"
    )


def _print_jobs(jobs: Iterable[RemoteJob], *, as_json: bool = False) -> None:
    """Print job records as JSON or a human-readable table."""
    rows = list(jobs)
    if as_json:
        print(json.dumps([job.to_dict() for job in rows], indent=2,
                         sort_keys=True))
        return
    print("JOB           STATUS      MODULE              PROFILE             REMOTE ID")
    for job in rows:
        print(_job_row(job))
    if not rows:
        print("(no distributed jobs)")


def _cmd_profile(args: argparse.Namespace) -> int:
    """Execute profile list/add/delete."""
    store = ProfileStore()
    if args.profile_command == "list":
        profiles = store.list()
        if args.json:
            print(json.dumps([item.to_dict() for item in profiles], indent=2,
                             sort_keys=True))
        else:
            print("NAME                 BACKEND   HOST")
            for profile in profiles:
                print(
                    f"{profile.name:20} {profile.backend:9} "
                    f"{profile.host or 'local/custom'}"
                )
            if not profiles:
                print("(no execution profiles)")
        return EXIT_OK
    if args.profile_command == "add":
        profile = _profile_from_args(args)
        store.save(profile)
        print(f"Saved {profile.backend} execution profile {profile.name!r}.")
        return EXIT_OK
    if args.profile_command == "delete":
        if not store.delete(args.name):
            raise RemoteExecutionError(
                f"Execution profile {args.name!r} does not exist."
            )
        print(f"Deleted execution profile {args.name!r}.")
        return EXIT_OK
    raise RemoteExecutionError("Choose profile list, add or delete.")


def _cmd_submit(args: argparse.Namespace) -> int:
    """Resolve a normal spaCR settings file and submit it."""
    from .cli import SettingsError, resolve_module, resolve_settings

    module = resolve_module(args.module)
    if module is None:
        raise RemoteExecutionError(
            f"Unknown spaCR module {args.module!r}; use `spacr-run --list`."
        )
    try:
        settings = resolve_settings(module, args.settings, args.set or [])
    except SettingsError as exc:
        raise RemoteExecutionError(str(exc)) from exc
    job = _submit_job(RemoteJobManager(), module.key, settings, args.profile, args.allocation or "")
    print(
        f"Submitted {module.key} as {job.job_id} "
        f"({job.profile_name}/{job.external_id})."
    )
    return EXIT_OK


def _cmd_list(args: argparse.Namespace) -> int:
    """List stored jobs, optionally polling active ones first."""
    manager = RemoteJobManager()
    jobs = (
        manager.refresh_all(include_logs=False)
        if args.refresh else manager.jobs.list()
    )
    _print_jobs(jobs, as_json=args.json)
    return EXIT_OK


def _cmd_status(args: argparse.Namespace) -> int:
    """Poll and print one job."""
    manager = RemoteJobManager()
    job = manager.refresh(args.job, include_logs=args.logs)
    if args.json:
        print(json.dumps(job.to_dict(), indent=2, sort_keys=True))
    else:
        _print_jobs([job])
        if job.log_tail:
            print("\nLog tail:\n" + job.log_tail)
    return EXIT_OK if job.status != "failed" else EXIT_RUNTIME


def _cmd_cancel(args: argparse.Namespace) -> int:
    """Cancel one active job."""
    job = RemoteJobManager().cancel(args.job)
    print(f"{job.job_id}: {job.status}")
    return EXIT_OK


def _cmd_logs(args: argparse.Namespace) -> int:
    """Print the latest remote log tail."""
    print(RemoteJobManager().logs(args.job, args.lines))
    return EXIT_OK


def _cmd_watch(args: argparse.Namespace) -> int:
    """Poll one job until it reaches a terminal state, optionally following its log."""
    manager = RemoteJobManager()
    offset = 0
    while True:
        job = manager.refresh(args.job, include_logs=False)
        previous_offset = offset
        text = ""
        if args.follow:
            text, offset = _log_chunk(manager, job.job_id, offset)
            if text:
                print(text, end="", flush=True)
        print(f"{job.updated_utc}  {job.status}", file=sys.stderr if args.follow else sys.stdout, flush=True)
        if job.status not in ACTIVE_STATES and job.status != "unknown":
            if args.follow and (text or offset != previous_offset):
                continue
            if job.error:
                print(job.error, file=sys.stderr)
            if args.logs:
                print(manager.logs(job.job_id, args.lines))
            return EXIT_OK if job.status == "success" else EXIT_RUNTIME
        time.sleep(max(2, args.interval))


def _cmd_script(args: argparse.Namespace) -> int:
    """Print the job script a submission through a profile would send."""
    profile = ProfileStore().get(args.profile)
    print(_preview_job_script(profile, args.module), end="")
    return EXIT_OK


def _cmd_resources(args: argparse.Namespace) -> int:
    """Read what the profile can reach on the cluster, read-only, and check its resources."""
    profile = ProfileStore().get(args.profile)
    cluster = _probe_cluster(profile)
    _save_cluster_cache(profile.name, cluster)
    problems = _resource_problems(profile.resources, cluster)
    if args.json:
        print(json.dumps({"cluster": cluster, "problems": problems}, indent=2, sort_keys=True))
        return EXIT_OK
    print(f"{cluster['host']} as {cluster['user']}; scheduler: {cluster['scheduler'] or 'none'}")
    print("PARTITION            NODES  CPUS  MEM_GB  GPUS  TIME_LIMIT")
    for row in cluster["partitions"]:
        print(f"{row['partition'] + ('*' if row['default'] else ''):20} {row['nodes']:5}  {row['cpus']:4}  "
              f"{round(row['mem_mb'] / 1024):6}  {row['gpus']:4}  {row['time_limit']}")
    for row in cluster["accounts"]:
        print(f"account {row['account']} partition {row['partition'] or '-'} qos {','.join(row['qos']) or '-'}")
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    return EXIT_OK


def _cmd_allocate(args: argparse.Namespace) -> int:
    """Request a Slurm allocation that later jobs can run inside."""
    job = _start_allocation(RemoteJobManager(), args.profile)
    print(f"Requested allocation {job.job_id} (Slurm {job.external_id}). "
          f"When it is running: spacr-remote submit MODULE -s SETTINGS -p {job.profile_name} "
          f"--allocation {job.job_id[:12]}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for the persistent distributed-job client."""
    parser = argparse.ArgumentParser(
        prog="spacr-remote",
        description=(
            "Submit spaCR modules to an SSH workstation, Slurm, or a "
            "configured cloud/HPC CLI and monitor them locally."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile", help="Manage execution profiles.")
    profile_sub = profile.add_subparsers(
        dest="profile_command", required=True
    )
    profile_list = profile_sub.add_parser("list", help="List profiles.")
    profile_list.add_argument("--json", action="store_true")
    profile_list.set_defaults(handler=_cmd_profile)

    profile_add = profile_sub.add_parser(
        "add", help="Create or replace a profile."
    )
    profile_add.add_argument("name")
    profile_add.add_argument(
        "--backend", choices=BACKENDS, required=True
    )
    profile_add.add_argument("--host", default="")
    profile_add.add_argument("--workdir", default="")
    profile_add.add_argument("--local-root", default="")
    profile_add.add_argument("--remote-root", default="")
    profile_add.add_argument("--runner", default="spacr-run")
    profile_add.add_argument("--scheduler-options", default="")
    profile_add.add_argument("--submit-command", default="")
    profile_add.add_argument("--status-command", default="")
    profile_add.add_argument("--cancel-command", default="")
    profile_add.add_argument("--log-command", default="")
    profile_add.add_argument("--job-id-pattern", default="")
    profile_add.add_argument("--poll-seconds", type=int, default=10)
    profile_add.add_argument("--auth-mode", choices=_AUTH_MODES, default="")
    profile_add.add_argument("--ssh-command", default="")
    profile_add.add_argument("--resource", action="append", default=[], metavar="KEY=VALUE")
    profile_add.add_argument("--env-setup", default="")
    profile_add.add_argument("--job-script-file", default="")
    profile_add.add_argument("--login-hosts", default="")
    profile_add.add_argument("--direct-run", action="store_true")
    profile_add.set_defaults(handler=_cmd_profile)

    profile_delete = profile_sub.add_parser("delete", help="Delete a profile.")
    profile_delete.add_argument("name")
    profile_delete.set_defaults(handler=_cmd_profile)

    submit = sub.add_parser("submit", help="Submit a spaCR settings file.")
    submit.add_argument("module")
    submit.add_argument("--settings", "-s", required=True)
    submit.add_argument("--profile", "-p", required=True)
    submit.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE"
    )
    submit.add_argument("--allocation", default="")
    submit.set_defaults(handler=_cmd_submit)

    script = sub.add_parser("script", help="Print the job script a submission would send.")
    script.add_argument("module")
    script.add_argument("--profile", "-p", required=True)
    script.set_defaults(handler=_cmd_script)

    resources = sub.add_parser("resources", help="Read what a profile can use on its cluster.")
    resources.add_argument("--profile", "-p", required=True)
    resources.add_argument("--json", action="store_true")
    resources.set_defaults(handler=_cmd_resources)

    allocate = sub.add_parser("allocate", help="Request a Slurm allocation to run jobs inside.")
    allocate.add_argument("--profile", "-p", required=True)
    allocate.set_defaults(handler=_cmd_allocate)

    listing = sub.add_parser("list", help="List persistent jobs.")
    listing.add_argument("--refresh", action="store_true")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(handler=_cmd_list)

    status = sub.add_parser("status", help="Poll one job.")
    status.add_argument("job")
    status.add_argument("--logs", action="store_true")
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=_cmd_status)

    cancel = sub.add_parser("cancel", help="Cancel one job.")
    cancel.add_argument("job")
    cancel.set_defaults(handler=_cmd_cancel)

    logs = sub.add_parser("logs", help="Read a remote log tail.")
    logs.add_argument("job")
    logs.add_argument("--lines", type=int, default=200)
    logs.set_defaults(handler=_cmd_logs)

    watch = sub.add_parser("watch", help="Poll until a job finishes.")
    watch.add_argument("job")
    watch.add_argument("--interval", type=int, default=10)
    watch.add_argument("--logs", action="store_true")
    watch.add_argument("--lines", type=int, default=200)
    watch.add_argument("--follow", action="store_true")
    watch.set_defaults(handler=_cmd_watch)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line client and convert known failures to exit codes."""
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        return int(args.handler(args))
    except RemoteExecutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())
