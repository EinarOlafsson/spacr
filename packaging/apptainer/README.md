# spaCR under Apptainer / Singularity

The Docker images in `packaging/docker/`, repackaged as single `.sif` files
for clusters where Docker is not allowed and Apptainer (or SingularityCE) is.
Nothing is decided twice: `spacr.def` starts from the CPU or CUDA Docker image
and changes only what has to change when the same image runs without a
daemon, without root, and with the user's real home mounted.

## Build

From the repository root, as an ordinary user (Apptainer 1.2 or newer, or
SingularityCE 4.0 or newer, for `--build-arg`):

```bash
# from the published Docker images
apptainer build spacr-cpu.sif packaging/apptainer/spacr.def
apptainer build --build-arg IMAGE=ghcr.io/einarolafsson/spacr:cuda \
    spacr-cuda.sif packaging/apptainer/spacr.def

# from an image you built with packaging/docker/Dockerfile.cpu
apptainer build --build-arg BOOTSTRAP=docker-daemon \
    --build-arg IMAGE=spacr:cpu spacr-cpu.sif packaging/apptainer/spacr.def
```

Pin a release by tag (`IMAGE=ghcr.io/einarolafsson/spacr:1.5.1.0-cuda12.4`)
rather than `cpu` / `cuda` when results have to be reproducible.

The build runs the definition's `%test`, which is the Docker images' own smoke
test: one real `external_masks` run, no model, no network. If a login node has
no user namespaces, build on a workstation and copy the `.sif` over; it is one
file.

On a machine where `/etc/subuid` lists the user but `newuidmap` is not
installed, add `--ignore-subuid` to `apptainer build`.

## Run

```bash
apptainer run spacr-cpu.sif                                  # list modules
apptainer run spacr-cpu.sif spacr-run mask -s mask.csv
apptainer run --nv spacr-cuda.sif spacr-doctor               # GPU
apptainer exec spacr-cpu.sif python /opt/spacr/smoke_pipeline.py --mask
```

`--nv` is Apptainer's `--gpus all`: it binds the host NVIDIA driver in. The
CUDA image needs a host driver of 550 or newer, as under Docker.

**What differs from Docker, and why.**

| | Docker | Apptainer |
|---|---|---|
| user | `spacr` (UID 1000) or `--user` | always you |
| `HOME` | inside the image | your real home, bind-mounted |
| models | a mounted `/models`, linked into `HOME` | `~/.cellpose/models` in your home |
| data | `-v` mounts | `$HOME`, `$PWD` and `/tmp` by default; `--bind` the rest |
| image | writable layer, discarded | read-only |
| PID 1 | tini | none needed |

Because `HOME` is your real home, the entrypoint links nothing into it under
Apptainer: a link from `~/.cellpose/models` to a `/models` that exists only
inside the image would break Cellpose on the host afterwards. Models you have
already downloaded in your home are simply found. To use a lab-wide model
folder instead, bind it read-only:

```bash
apptainer run --nv --bind /shared/cellpose_models:/models:ro spacr-cuda.sif ...
```

and Cellpose loads from it (the Model Zoo still lists your home's folders).

Your host environment passes through by default. `PYTHONHOME` is cleared, but
a host `PYTHONPATH` from a conda or module setup is not; if an import inside
the container resolves to a host package, run with `--cleanenv`.

## Batch runs

`spacr_slurm.sh` is a Slurm array job: one plate per task, Mask then Measure,
one pair of settings files for every plate, workers capped at the CPUs Slurm
granted. Copy it, edit the variables at the top, and submit with
`sbatch --array=...`. For a CPU partition drop `--gres` and `--nv`.

Every run writes its reproducibility manifest to `~/.spacr/runs/` in your
home, as it does outside a container.

## Checked

Built unprivileged with Apptainer 1.5.4 on Ubuntu (no root, no setuid,
`--ignore-subuid`) from the local CPU and CUDA Docker images; `%test` passed
in both, and `smoke_pipeline.py --mask` passed on the CPU in both. See
`features/data/586_apptainer_smoke_2026-09-27.json`, which also records the
`--nv` run.

Nothing is published yet. `features/data/586_apptainer_images_workflow.yml`
is a proposed release workflow for the maintainer; until it is adopted, build
the `.sif` yourself as above.
