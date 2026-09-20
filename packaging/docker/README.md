# spaCR container images

Two images, built from this checkout, published to GHCR on each release tag
by `.github/workflows/docker-images.yml`.

| File | Tag | Base |
|---|---|---|
| `Dockerfile.cpu` | `ghcr.io/einarolafsson/spacr:<version>`, `:<version>-cpu`, `:cpu`, `:latest` | `python:3.12-slim-bookworm` |
| `Dockerfile.cuda` | `ghcr.io/einarolafsson/spacr:<version>-cuda12.4`, `:cuda12.4`, `:cuda` | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` |

The user-facing instructions are the **Container images** section of
`docs/source/installer_guide.rst`. This file is the record of *why* the images
are shaped the way they are, so that a future change is made against the
reasons rather than around them.

## Build and check

Always from the repository root — the checkout is the build context:

```bash
docker build -f packaging/docker/Dockerfile.cpu  -t spacr:cpu  .
docker build -f packaging/docker/Dockerfile.cuda -t spacr:cuda .

docker run --rm spacr:cpu spacr --version
docker run --rm spacr:cpu id -u                          # must not be 0
docker run --rm spacr:cpu python3 /opt/spacr/smoke_pipeline.py
```

Those three commands are exactly what the release workflow runs before it
pushes anything.

## The decisions, and what they cost

**The image is the CLI and the pipelines. The GUI is a documented Linux-only
extra.** A containerised Qt application needs the host's display socket passed
in, which works on Linux, needs a third-party X server on macOS and Windows,
and is a support burden out of proportion to the benefit. The part users
cannot install is the CUDA stack, not the window. The Qt runtime libraries are
in the image anyway — PySide6 is a core dependency of the package, so it is
installed whether or not anyone opens a window, and ~60 MB of `libxcb*` is
what makes the documented extra actually work instead of being a paragraph
that fails.

**Two tags, CPU and CUDA 12.4, never one image with everything.** The CPU
image works on any host with no driver at all and is the one a reviewer, a CI
job or a laptop should use. CUDA 12.4 is a statement about the *host*: the
driver is on the host and cannot be containerised away, so the floor is what
the image promises, and it is 550+.

**Models are not baked in.** A cpsam checkpoint is ~1.2 GB and goes stale
between releases, and the Model Zoo exists to fetch them. Mount `/models`.
`entrypoint.sh` then makes that folder *both* `CELLPOSE_LOCAL_MODELS_PATH`
and the paths `spacr/model_zoo.py` resolves through `Path.home()` — it reads
`~/.cellpose/models` and `~/.spacr/models` directly, which no environment
variable reaches, so the entrypoint symlinks them. Setting only the variable
would leave the zoo listing an empty folder.

**Data is mounted, never copied in.** There is no `COPY` of anything under
`/data` and there never should be.

**SAMCell and DINOCell are not in the images.** They pin PyTorch versions that
conflict with spaCR's and with each other. An image that cannot resolve is not
an image; the Model Zoo's isolated-environment install is the answer inside a
running container.

**Non-root, with a settable UID.** Every output file a container writes to a
mounted folder is owned by the UID inside it. An image that runs as root hands
a scientist a results folder they need `sudo` to delete, and that is the single
most common complaint about scientific Docker images. The image's own user is
`spacr` (UID 1000 by default, `--build-arg SPACR_UID=` to change it), and
`docker run --user "$(id -u):$(id -g)"` is supported: that UID has no passwd
entry and therefore no writable home, so `entrypoint.sh` relocates `HOME` and
every cache under it. The workflow asserts both — the default UID is not 0,
and `--user 4242:4242` still runs.

**Two stages.** The build stage carries `build-essential`, because a handful of
dependencies still build from an sdist; the runtime stage carries none of it.
One self-contained virtualenv at `/opt/spacr/venv` is all that crosses.

**No BuildKit-only directives.** `RUN --mount=type=cache` would make a local
rebuild skip the downloads, but the legacy builder — which is what ships
without the `buildx` component, and what these were proved on — fails on the
directive rather than ignoring it. Two tags that only build on some
installations of Docker is not a Docker image.

**torch is installed first, from PyTorch's own index.** A plain
`pip install torch` on Linux resolves to the CUDA build and drags ~2.5 GB of
`nvidia-*-cu12` wheels into an image with no driver to use them. Installing
torch from `…/whl/cpu` (or `…/whl/cu124`) before the package means the spaCR
install finds the requirement satisfied and never reaches the default index
for it.

**The compiled bytecode stays in the image, and the first build proved why.**
It ended with `find $VIRTUAL_ENV -name '*.pyc' -delete`, which reads as
housekeeping. `/opt/spacr/venv` is root-owned and the process runs as `spacr`,
so an image without `.pyc` recompiles what it imports on *every* run and can
never keep the result. A/B inside one container, compiling in place between
the halves:

| | no bytecode | with bytecode |
|---|---|---|
| `import spacr.measure, torch, cellpose` | 6224 / 6030 / 6346 ms | 3084 / 3075 / 3110 ms |
| `spacr-run --list` | 510 ms | 278 ms |
| image | 3.61 GB | 4.0 GB |

11% of the image for half the start-up of every run. Note also that pip
compiles on install whatever `PYTHONDONTWRITEBYTECODE` says — 493 `.pyc`
either way in a throwaway container — so it was the deletion and not the
variable.

**The CPU image still carries one NVIDIA wheel, and that is not a leak.**
`nvidia-nccl-cu13` is 252 MB and `xgboost` declares it on Linux x86-64 for its
own distributed GPU support. torch is *not* the culprit: the build log shows
`torch 2.14.0+cpu` already satisfied when the package installs. Swapping in
the separate `xgboost-cpu` distribution would drop it, and would also mean the
image no longer satisfies `xgboost>=2.0.3,<4` as setup.py declares it — an
image whose dependency set quietly differs from the package is a worse problem
than 5% of its size. Measured, left alone, written down.

**The image is built from the checkout, not from PyPI.** An image tagged with
a spaCR version should contain that source, and a release-tag build that
installed from PyPI would race the PyPI upload and could publish the previous
release under the new tag. The workflow additionally refuses to build when the
tag name and `setup.py`'s `VERSION` disagree.

## Files

| File | What it is |
|---|---|
| `Dockerfile.cpu` | The CPU image. |
| `Dockerfile.cuda` | The CUDA 12.4 image. |
| `entrypoint.sh` | Writable `HOME`, caches under it, `/models` wired to both readers. |
| `smoke_pipeline.py` | One real pipeline run on a synthetic field: no model, no GPU, no network. |
| `../../.dockerignore` | Keeps the 1.2 GB checkout out of the build context. |
