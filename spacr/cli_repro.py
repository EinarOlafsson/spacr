"""
``spacr repro`` — replay a pipeline run from its recorded journal.

Every pipeline invocation writes a run journal (see
:mod:`spacr.run_journal`) containing the exact settings + environment
that produced a result. This CLI re-runs those settings, opens a
FRESH journal folder, and reports whether the outcome matches.

Usage::

    spacr repro ~/.spacr/runs/2026-07-23_143507_ab12cd34__mask
    spacr repro ~/.spacr/runs/2026-07-23_143507_ab12cd34__mask --dry
    spacr repro ~/.spacr/runs/2026-07-23_143507_ab12cd34__mask --show

* ``--dry``  prints the resolved settings + which pipeline entry
  will run; doesn't invoke it.
* ``--show`` prints the manifest + settings; doesn't invoke it.
* ``--export snakemake|nextflow --out DIR`` writes the run as a
  workflow that runs the same module once per plate with ``spacr-run``,
  locally, on a cluster, or inside the spaCR Docker/Apptainer image.

Exit codes:
  0  — replay ran to completion (regardless of scientific outcome)
  1  — replay raised (journal captures the traceback for triage)
  2  — bad input (missing run folder / unresolvable app_key)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .run_journal import load_run_settings, open_run, runs_root


def _print_manifest(run_dir: Path) -> None:
    """Print the recorded run summary and optional model hashes.

    :param run_dir: Run-journal directory containing ``manifest.json``.
    :returns: ``None``; the formatted summary is written to standard output.
    """
    m = json.loads((run_dir / "manifest.json").read_text())
    print(f"run:       {run_dir.name}")
    print(f"app:       {m.get('app_key')}")
    print(f"status:    {m.get('status')}")
    print(f"start:     {m.get('start_utc')}")
    print(f"elapsed:   {m.get('elapsed_s')}s")
    print(f"n_settings:{m.get('n_settings')}")
    env = m.get("env", {})
    print(f"spacr:     {env.get('spacr')} (git {env.get('spacr_git')})")
    print(f"python:    {env.get('python')}  torch {env.get('torch')}  "
          f"cellpose {env.get('cellpose')}")
    if m.get("model_hashes"):
        print("models:")
        for k, v in m["model_hashes"].items():
            print(f"  {k}: {v}")


def _print_settings(settings: dict) -> None:
    """Print settings as deterministic, key-sorted rows.

    :param settings: Setting names and values to display.
    :returns: ``None``; the rows are written to standard output.
    """
    print("settings:")
    for k, v in sorted(settings.items()):
        print(f"  {k:32s} = {v!r}")


def _resolve_pipeline(app_key: str):
    """Return the pipeline callable for ``app_key`` or ``None``."""
    try:
        from .qt.bridge import resolve_pipeline_entry
        return resolve_pipeline_entry(app_key)
    except Exception:
        return None


_WORKFLOW_IMAGE = "ghcr.io/einarolafsson/spacr"
_WORKFLOW_ENGINES = ("snakemake", "nextflow")


def _workflow_image(manifest: Dict[str, Any]) -> str:
    """Return the CPU container image for the spaCR version that ran.

    A released version (digits and dots only) pins its own tag; any other
    version falls back to the rolling ``:cpu`` tag.
    """
    version = str((manifest.get("env") or {}).get("spacr") or "")
    if re.fullmatch(r"\d+(\.\d+)+", version):
        return f"{_WORKFLOW_IMAGE}:{version}-cpu"
    return f"{_WORKFLOW_IMAGE}:cpu"


def _workflow_plates(settings: Dict[str, Any],
                     plates: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Split recorded settings into one settings dict per plate.

    Each entry of ``src`` (a path or a list of paths) becomes its own job
    with ``src`` set to that one plate. ``plates`` replaces the recorded
    plates. Settings with no ``src`` stay one job named ``run``. When there
    is more than one job, explicit ``dst`` and ``dst_root`` output folders
    gain the unique plate id as a subfolder, so jobs do not overwrite each
    other's outputs. Single-job destinations and unset folders stay unchanged.

    :param settings: recorded settings; the input dictionary is not changed.
    :param plates: optional replacement source folders.
    :returns: ``{plate_id: settings}``, ids unique and safe as file names.
    """
    src = settings.get("src")
    if plates:
        sources = [str(p) for p in plates]
    elif isinstance(src, (list, tuple)):
        sources = [str(p) for p in src if str(p).strip()]
    elif isinstance(src, str) and src.strip():
        sources = [src]
    else:
        sources = []
    if not sources:
        return {"run": dict(settings)}
    jobs: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                      Path(source.rstrip("/\\")).name).strip("._") or "plate"
        plate_id, n = stem, 2
        while plate_id in jobs:
            plate_id, n = f"{stem}_{n}", n + 1
        jobs[plate_id] = {**settings, "src": source}
        if len(sources) > 1:
            for key in ("dst", "dst_root"):
                destination = settings.get(key)
                if isinstance(destination, str) and destination:
                    jobs[plate_id][key] = str(Path(destination) / plate_id)
    return jobs


def _snakefile(module: str, image: str, run_name: str) -> str:
    """Return the Snakefile text that runs ``module`` once per settings file."""
    return f"""# spaCR workflow exported from run {run_name}.
#
# One job per plate: every settings/<plate>.json is one `spacr-run {module}`.
# Add a plate by copying a settings file and changing its "src".
#
#   snakemake --cores 4                          # this machine
#   snakemake --cores 4 --use-apptainer          # inside {image}
#   snakemake --executor slurm --jobs 20 --use-apptainer   # a cluster
#
# Inside a container the data folders must be visible at the same paths,
# e.g. --apptainer-args "--bind /data". config.yaml holds the image and the
# spacr-run command.

configfile: "config.yaml"

PLATES = glob_wildcards("settings/{{plate}}.json").plate


rule all:
    input:
        expand("done/{{plate}}.ok", plate=PLATES)


rule spacr_{module}:
    input:
        "settings/{{plate}}.json"
    output:
        touch("done/{{plate}}.ok")
    log:
        "logs/{{plate}}.log"
    container:
        config["image"]
    params:
        spacr_run=config["spacr_run"]
    threads: config.get("threads", 1)
    shell:
        "{{params.spacr_run}} {module} --settings {{input}} > {{log}} 2>&1"
"""


def _nextflow_main(module: str, run_name: str) -> str:
    """Return the Nextflow DSL2 ``main.nf`` that runs ``module`` per plate."""
    return f"""#!/usr/bin/env nextflow
// spaCR workflow exported from run {run_name}.
//
// One task per plate: every settings/<plate>.json is one `spacr-run {module}`.
// Add a plate by copying a settings file and changing its "src".
//
//   nextflow run main.nf                              // this machine
//   nextflow run main.nf -profile apptainer           // inside the image
//   nextflow run main.nf -profile slurm,apptainer     // a cluster
//
// Inside a container the data folders must be visible at the same paths:
// add e.g. -v /data:/data to docker.runOptions in nextflow.config.

nextflow.enable.dsl = 2

process SPACR_{module.upper()} {{
    tag "${{plate}}"
    publishDir "${{params.outdir}}/logs", mode: 'copy', pattern: '*.log'

    input:
    tuple val(plate), path(settings)

    output:
    tuple val(plate), path("${{plate}}.log")

    script:
    \"\"\"
    ${{params.spacr_run}} {module} --settings ${{settings}} > ${{plate}}.log 2>&1
    \"\"\"
}}

workflow {{
    channel
        .fromPath("${{params.settings_dir}}/*.json")
        .map {{ f -> tuple(f.baseName, f) }}
        | SPACR_{module.upper()}
}}
"""


def _nextflow_config(image: str) -> str:
    """Return ``nextflow.config`` with local, container and SLURM profiles."""
    return f"""params {{
    settings_dir = "${{projectDir}}/settings"
    outdir       = "${{projectDir}}/results"
    spacr_run    = "spacr-run"
    image        = "{image}"
}}

process {{
    cpus      = 1
    container = params.image
}}

profiles {{
    docker {{
        docker.enabled    = true
        docker.runOptions = '-u $(id -u):$(id -g)'
    }}
    apptainer {{
        apptainer.enabled    = true
        apptainer.autoMounts = true
    }}
    slurm {{
        process.executor = 'slurm'
    }}
}}
"""


def _export_workflow(run_dir: Any, out_dir: Any, engine: str = "snakemake",
                     plates: Optional[List[str]] = None,
                     image: Optional[str] = None,
                     spacr_run: str = "spacr-run") -> Path:
    """Write a recorded run as a Snakemake or Nextflow workflow.

    The workflow runs the run's module once per plate with ``spacr-run`` and
    the recorded settings, one ``settings/<plate>.json`` each. Multiple jobs
    receive unique subfolders of explicit ``dst`` or ``dst_root`` folders;
    other settings and single-job destinations are preserved.

    :param run_dir: run-journal folder (or its name under the runs root).
    :param out_dir: folder to write the workflow into; created if missing.
    :param engine: ``"snakemake"`` or ``"nextflow"``.
    :param plates: plate folders to run instead of the recorded ``src``.
    :param image: container image; defaults to the CPU image of the
        spaCR version that ran.
    :param spacr_run: command that starts ``spacr-run`` on the nodes.
    :returns: the workflow's main file (``Snakefile`` or ``main.nf``).
    :raises ValueError: for an unknown engine, a folder that is not a run,
        or a module that cannot run headless.
    """
    from .cli import resolve_module

    if engine not in _WORKFLOW_ENGINES:
        raise ValueError(f"engine must be one of {_WORKFLOW_ENGINES}, "
                         f"not {engine!r}")
    run_dir = Path(run_dir)
    if not run_dir.exists() and (runs_root() / run_dir.name).exists():
        run_dir = runs_root() / run_dir.name
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{run_dir} is not a run folder (no manifest.json)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    module = resolve_module(str(manifest.get("app_key") or ""))
    if module is None:
        raise ValueError(f"module {manifest.get('app_key')!r} cannot run "
                         f"headless, so it cannot be exported")
    settings = load_run_settings(run_dir)
    image = image or _workflow_image(manifest)
    out = Path(out_dir)
    (out / "settings").mkdir(parents=True, exist_ok=True)
    for plate_id, plate_settings in _workflow_plates(settings, plates).items():
        (out / "settings" / f"{plate_id}.json").write_text(
            json.dumps(plate_settings, indent=2, default=str), encoding="utf-8")
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if engine == "snakemake":
        (out / "config.yaml").write_text(
            f"image: {json.dumps('docker://' + image)}\n"
            f"spacr_run: {json.dumps(spacr_run)}\nthreads: 1\n",
            encoding="utf-8")
        main = out / "Snakefile"
        main.write_text(_snakefile(module.key, image, run_dir.name),
                        encoding="utf-8")
    else:
        config = _nextflow_config(image)
        if spacr_run != "spacr-run":
            config = config.replace('spacr_run    = "spacr-run"',
                                    f"spacr_run    = {json.dumps(spacr_run)}")
        (out / "nextflow.config").write_text(config, encoding="utf-8")
        main = out / "main.nf"
        main.write_text(_nextflow_main(module.key, run_dir.name),
                        encoding="utf-8")
    return main


def main(argv=None) -> int:
    """CLI entry point wired as the ``spacr-repro`` console script.

    :param argv: optional argv list; defaults to ``sys.argv[1:]``.
    :returns: process exit code.
    """
    p = argparse.ArgumentParser(
        prog="spacr repro",
        description="Replay a spaCR pipeline run from its recorded "
                    "journal folder.",
    )
    p.add_argument("run_dir",
                    help="Path to a folder under ~/.spacr/runs/, or the "
                         "folder's basename.")
    p.add_argument("--dry", action="store_true",
                    help="Print resolved settings + app; don't run.")
    p.add_argument("--show", action="store_true",
                    help="Print manifest + settings; don't run.")
    p.add_argument("--export", choices=_WORKFLOW_ENGINES,
                    help="Write the run as a Snakemake or Nextflow workflow "
                         "into --out instead of running it.")
    p.add_argument("--out", metavar="DIR",
                    help="Folder for --export.")
    p.add_argument("--plates", nargs="+", metavar="FOLDER",
                    help="With --export: plate folders to run instead of "
                         "the recorded src.")
    p.add_argument("--image", metavar="IMAGE",
                    help="With --export: container image; default is the "
                         "CPU image of the spaCR version that ran.")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        candidate = runs_root() / args.run_dir
        if candidate.exists():
            run_dir = candidate
        else:
            print(f"error: no such run folder: {args.run_dir}",
                    file=sys.stderr)
            return 2

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"error: {run_dir} is not a valid run folder "
                f"(no manifest.json)", file=sys.stderr)
        return 2

    if args.export:
        if not args.out:
            print("error: --export needs --out DIR", file=sys.stderr)
            return 2
        try:
            main_file = _export_workflow(run_dir, args.out, args.export,
                                         plates=args.plates, image=args.image)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"wrote {main_file}")
        return 0

    manifest = json.loads(manifest_path.read_text())
    app_key = manifest.get("app_key")
    settings = load_run_settings(run_dir)

    if args.show:
        _print_manifest(run_dir)
        print()
        _print_settings(settings)
        return 0

    entry = _resolve_pipeline(app_key)
    if entry is None:
        print(f"error: no pipeline entry for app_key={app_key!r}",
                file=sys.stderr)
        return 2

    if args.dry:
        print(f"would run: {entry.__module__}.{entry.__name__}(settings)")
        _print_settings(settings)
        return 0

    from .figure_font import _open_sans_is_the_default

    print(f"replaying {app_key} — this opens a NEW run journal folder.")
    with _open_sans_is_the_default(), open_run(app_key, settings) as run:
        try:
            entry(settings)
            run.set_status("success")
        except Exception as e:
            run.set_status("failed")
            print(f"replay raised: {type(e).__name__}: {e}",
                    file=sys.stderr)
            return 1
    print(f"done → {run.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
