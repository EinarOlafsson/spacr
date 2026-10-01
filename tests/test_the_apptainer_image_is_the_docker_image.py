"""The Apptainer definition, asserted off the files rather than off a build.

A build takes minutes and gigabytes and needs Apptainer, which CI does not
have, so what is pinned here is the contract a reviewer would otherwise
re-derive by reading the definition beside the Docker files:

* the SIF is built FROM the Docker images, so torch, CUDA, the venv and the
  non-root user are decided once, in packaging/docker/;
* the runscript goes through the entrypoint but not through tini, which
  under Apptainer is not PID 1;
* under Apptainer the entrypoint writes nothing into HOME, which there is the
  user's real home on the cluster;
* the build's own %test is the Docker images' smoke test, and the smoke test
  can run the Mask + Measure job the HPC images are accepted on.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APPTAINER_DIR = REPO_ROOT / "packaging" / "apptainer"
DOCKER_DIR = REPO_ROOT / "packaging" / "docker"
DEFINITION = APPTAINER_DIR / "spacr.def"
DOCKER_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-images.yml"


def _sections() -> dict[str, str]:
    """The definition split into its header and its ``%`` sections."""
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for line in DEFINITION.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^%(\w+)", line)
        if match:
            current = match.group(1)
            sections[current] = []
            continue
        sections[current].append(line)
    return {name: "\n".join(lines) for name, lines in sections.items()}


def _header() -> dict[str, str]:
    """The ``Key: value`` lines of the header, comments dropped."""
    out = {}
    for line in _sections()["header"].splitlines():
        if line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _arguments() -> dict[str, str]:
    """The ``%arguments`` defaults."""
    out = {}
    for line in _sections()["arguments"].splitlines():
        if "=" in line:
            key, value = line.strip().split("=", 1)
            out[key] = value
    return out


def test_the_sif_is_built_from_the_docker_image():
    """Bootstrap and source are build arguments defaulting to the registry."""
    header = _header()
    assert header["Bootstrap"] == "{{ BOOTSTRAP }}"
    assert header["From"] == "{{ IMAGE }}"
    defaults = _arguments()
    assert defaults["BOOTSTRAP"] == "docker"
    repository, tag = defaults["IMAGE"].rsplit(":", 1)
    assert repository == "ghcr.io/einarolafsson/spacr"
    assert f"$IMAGE:{tag}" in DOCKER_WORKFLOW.read_text(encoding="utf-8"), (
        f"the default tag {tag!r} is not one the Docker workflow publishes."
    )


def test_every_file_the_definition_copies_exists():
    """``%files`` paths resolve against the repository root."""
    pairs = [line.split() for line in _sections()["files"].splitlines()
             if line.strip()]
    assert pairs
    for source, destination in pairs:
        assert (REPO_ROOT / source).is_file(), source
        assert destination.startswith("/opt/spacr/")


def test_the_runscript_uses_the_entrypoint_but_not_tini():
    """Apptainer shares the host PID namespace, so tini would only warn."""
    runscript = _sections()["runscript"]
    assert "exec /opt/spacr/entrypoint.sh" in runscript
    assert "tini" not in runscript
    assert "spacr-run --list" in runscript


def test_the_build_runs_the_smoke_test():
    """A SIF that imports but cannot measure fails at build time."""
    test = _sections()["test"]
    assert "smoke_pipeline.py" in test
    for line in filter(str.strip, test.splitlines()):
        assert line.strip().startswith("/opt/spacr/entrypoint.sh "), (
            "the build's %test runs with HOME on the read-only image; "
            "without the entrypoint's writable-HOME fallback the run journal "
            "cannot be created and the smoke test fails."
        )


def test_no_model_or_data_is_baked_into_the_sif():
    """Models are found in the user's home or a bound /models, never copied."""
    text = DEFINITION.read_text(encoding="utf-8")
    copied = _sections()["files"]
    for forbidden in ("cpsam", ".cellpose", "/models", "/data", "wget", "curl"):
        assert forbidden not in copied, forbidden
    assert "models" in _sections()["help"]
    assert text.count("From:") == 1


def test_the_definition_needs_no_root_step():
    """No %post: unprivileged builds run it under a glibc-sensitive fakeroot."""
    assert "post" not in _sections()
    for name in ("entrypoint.sh", "smoke_pipeline.py"):
        assert os.access(DOCKER_DIR / name, os.X_OK), (
            f"{name} must keep its executable bit; %files copies the mode "
            "and there is no %post to chmod it."
        )


@pytest.mark.parametrize("section", ["environment", "runscript", "test"])
def test_each_script_section_is_posix_sh(section, tmp_path):
    """Apptainer runs these with /bin/sh; a bashism fails only on the node."""
    script = tmp_path / f"{section}.sh"
    script.write_text(_sections()[section] + "\n", encoding="utf-8")
    shell = shutil.which("dash") or shutil.which("sh")
    completed = subprocess.run([shell, "-n", str(script)],
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def _entrypoint(tmp_path: Path, **extra: str) -> tuple[Path, str]:
    """Run the entrypoint around ``env`` with a fresh HOME, return both."""
    home = tmp_path / "home"
    home.mkdir()
    environment = {
        key: value for key, value in os.environ.items()
        if key not in ("CELLPOSE_LOCAL_MODELS_PATH", "APPTAINER_CONTAINER",
                       "SINGULARITY_CONTAINER")
    }
    environment.update(HOME=str(home), SPACR_CONTAINER_QUIET="1", **extra)
    completed = subprocess.run(
        ["sh", str(DOCKER_DIR / "entrypoint.sh"), "env"],
        env=environment, capture_output=True, text=True, check=True)
    return home, completed.stdout


@pytest.mark.parametrize("variable", ["APPTAINER_CONTAINER",
                                      "SINGULARITY_CONTAINER"])
def test_under_apptainer_nothing_is_linked_into_the_real_home(
        variable, tmp_path):
    """A link to an image-only /models would outlive the container."""
    home, output = _entrypoint(tmp_path, **{variable: "/opt/spacr.sif"})
    assert not (home / ".cellpose").exists()
    assert not (home / ".spacr").exists()
    if not os.path.ismount("/models"):
        assert "CELLPOSE_LOCAL_MODELS_PATH=" not in output


def test_the_entrypoint_hands_a_bound_models_folder_to_cellpose():
    """Only a folder actually mounted on /models counts as the model folder."""
    entrypoint = (DOCKER_DIR / "entrypoint.sh").read_text(encoding="utf-8")
    environment = _sections()["environment"]
    for text in (entrypoint, environment):
        assert "' /models ' /proc/self/mountinfo" in text
        assert "CELLPOSE_LOCAL_MODELS_PATH" in text


def test_the_image_libraries_come_before_the_nv_bound_host_libraries(tmp_path):
    """--nv binds a host libEGL that can need a newer glibc than the image's.

    The image's library directory is prepended so its own libglvnd copies win
    over /.singularity.d/libs, while existing entries are kept after it.
    """
    script = tmp_path / "env.sh"
    script.write_text(_sections()["environment"]
                      + '\nprintf %s "$LD_LIBRARY_PATH"\n', encoding="utf-8")
    shell = shutil.which("dash") or shutil.which("sh")
    result = subprocess.run(
        [shell, str(script)], capture_output=True, text=True, check=True,
        env={"PATH": os.environ.get("PATH", ""),
             "LD_LIBRARY_PATH": "/usr/local/nvidia/lib"})
    entries = result.stdout.split(":")
    assert entries[-1] == "/usr/local/nvidia/lib"
    libdirs = [d for d in ("/usr/lib/x86_64-linux-gnu",
                           "/usr/lib/aarch64-linux-gnu") if os.path.isdir(d)]
    assert set(entries[:-1]) == set(libdirs)


def test_the_slurm_recipe_is_a_valid_array_job():
    """The batch recipe parses, uses the GPU flag and caps the workers."""
    script = APPTAINER_DIR / "spacr_slurm.sh"
    text = script.read_text(encoding="utf-8")
    completed = subprocess.run(["bash", "-n", str(script)],
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "--nv" in text
    assert "n_jobs=${SLURM_CPUS_PER_TASK" in text
    for module in ("run mask", "run measure"):
        assert module in text
    assert os.access(script, os.X_OK)


def test_the_smoke_script_can_run_mask_and_measure():
    """``--mask`` segments and measures, and requires every synthetic cell."""
    text = (DOCKER_DIR / "smoke_pipeline.py").read_text(encoding="utf-8")
    assert '"--mask"' in text
    assert 'zip(("mask", "measure")' in text
    assert "count == len(SYNTHETIC_CELLS)" in text
