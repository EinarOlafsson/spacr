"""The two container images, asserted off the files rather than off a build.

Item 425. Building either image takes tens of minutes and several gigabytes,
so a test that builds one is a test nobody runs. What can be asserted cheaply
is every property a reviewer would otherwise have to re-derive by reading
three files at once, and each of these is a failure that has happened to a
real scientific image somewhere:

* the image runs as a non-root user, so a results folder on a mounted volume
  does not come back owned by root;
* no model checkpoint and no data is baked in;
* the release workflow smoke-tests each image BEFORE it pushes it, never
  after, and pushes nothing when the checks fail;
* the workflow does not fire on every push, which is what would make a
  multi-gigabyte build a tax on every commit;
* `.dockerignore` does not exclude a file `pip install .` needs — an
  allow-list mistake there fails the build with a message about the wrong
  thing.

The smoke script's own module-level imports are pinned to the standard
library for the same reason `spacr.cli` pins its: it has to be able to run
and report before anything heavy is importable.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO_ROOT / "packaging" / "docker"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-images.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
GUIDE = REPO_ROOT / "docs" / "source" / "installer_guide.rst"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"

VARIANTS = ("cpu", "cuda")


def _dockerfile(variant: str) -> str:
    """The text of one Dockerfile."""
    return (DOCKER_DIR / f"Dockerfile.{variant}").read_text(encoding="utf-8")


def _instructions(text: str) -> list[tuple[str, str]]:
    """Every Dockerfile instruction as ``(verb, rest)``, continuations joined.

    Comments and blank lines are dropped, and a line ending in a backslash is
    joined to the next one, so a rule spelled across six lines is matched the
    same as a rule spelled across one.
    """
    joined: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        joined.append(buffer + line)
        buffer = ""
    if buffer:
        joined.append(buffer)
    out = []
    for line in joined:
        parts = line.split(None, 1)
        out.append((parts[0].upper(), parts[1] if len(parts) > 1 else ""))
    return out


@pytest.fixture(scope="module")
def workflow() -> dict:
    """The parsed release workflow.

    PyYAML follows YAML 1.1, where a bare ``on:`` key is the boolean ``True``.
    Both spellings are read, as ``tests/test_ci_concurrency.py`` does.
    """
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    document["_triggers"] = document.get("on", document.get(True))
    return document


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_image_runs_as_somebody_who_is_not_root(variant):
    """The last USER is a named non-root account.

    Without this, every file a pipeline writes into a mounted results folder
    comes back owned by root on the host.
    """
    users = [rest.strip() for verb, rest in _instructions(_dockerfile(variant))
             if verb == "USER"]

    assert users, f"Dockerfile.{variant} never switches away from root"
    assert users[-1] == "spacr", (
        f"Dockerfile.{variant} ends as {users[-1]!r}; the image must run as "
        f"the non-root 'spacr' account."
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_uid_can_be_chosen_at_build_time(variant):
    """A UID that cannot be changed is a UID that will not match the host."""
    text = _dockerfile(variant)

    assert re.search(r"^ARG SPACR_UID=", text, re.M), (
        f"Dockerfile.{variant} has no SPACR_UID build argument, so a user "
        f"whose UID is not 1000 cannot bake their own."
    )
    assert "${SPACR_UID}" in text


@pytest.mark.parametrize("variant", VARIANTS)
def test_no_model_and_no_data_is_baked_into_the_image(variant):
    """Only the virtualenv, the entrypoint and the smoke script are copied in.

    A checkpoint in the image is ~1.2 GB that goes stale at the next release,
    and a dataset in the image is somebody's data in a public registry.
    """
    copied = [rest for verb, rest in _instructions(_dockerfile(variant))
              if verb in {"COPY", "ADD"}]
    sources = []
    for rest in copied:
        words = [word for word in rest.split() if not word.startswith("--")]
        sources.extend(words[:-1])

    assert sources, f"Dockerfile.{variant} copies nothing at all"
    for source in sources:
        assert not source.endswith((".pt", ".pth", ".npy", ".tif", ".db")), (
            f"Dockerfile.{variant} copies {source!r} into the image; models "
            f"and data are mounted, never baked in."
        )
        assert source in {
            "/opt/spacr/venv",
            "packaging/docker/entrypoint.sh",
            "packaging/docker/smoke_pipeline.py",
            "/src",
            ".",
        }, (
            f"Dockerfile.{variant} copies {source!r}; if that is deliberate, "
            f"say so here and in packaging/docker/README.md."
        )


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_mounted_models_folder_reaches_both_readers(variant):
    """``/models`` exists, and the entrypoint wires it to both readers.

    Cellpose reads ``CELLPOSE_LOCAL_MODELS_PATH``; ``spacr.model_zoo`` resolves
    ``Path.home()/'.cellpose'/'models'`` directly and no variable reaches that.
    Wiring only one of them leaves the other looking at an empty folder.
    """
    assert "/models" in _dockerfile(variant)

    entrypoint = (DOCKER_DIR / "entrypoint.sh").read_text(encoding="utf-8")
    assert "CELLPOSE_LOCAL_MODELS_PATH" in entrypoint
    assert ".cellpose/models" in entrypoint
    assert ".spacr/models" in entrypoint


def test_the_entrypoint_gives_a_foreign_uid_a_writable_home():
    """``--user "$(id -u):$(id -g)"`` has no passwd entry and no home.

    The documentation tells a user on a shared filesystem to pass exactly
    that, so the entrypoint has to cope with it rather than failing six frames
    inside matplotlib's font cache.
    """
    entrypoint = (DOCKER_DIR / "entrypoint.sh").read_text(encoding="utf-8")

    assert entrypoint.startswith("#!/bin/sh"), (
        "the entrypoint must be POSIX sh: the runtime stage installs no bash "
        "guarantee beyond the base image's."
    )
    assert "MPLCONFIGDIR" in entrypoint
    assert '-w "$HOME"' in entrypoint, (
        "nothing tests whether HOME is writable, so the fallback below can "
        "never be reached."
    )
    assert "/tmp/spacr-home" in entrypoint, (
        "no fallback home; a container run with --user cannot write to "
        "/home/spacr."
    )
    assert "export HOME" in entrypoint, (
        "a reassigned HOME that is not exported is invisible to the Python "
        "process that needs it."
    )
    assert entrypoint.rstrip().endswith('exec "$@"'), (
        "the entrypoint must exec the command, or signals never reach the "
        "pipeline and Ctrl-C leaves it running."
    )


def test_the_entrypoint_replaces_a_runtime_dir_that_is_not_there():
    """A host ``XDG_RUNTIME_DIR`` does not exist inside the container.

    Every X11-in-Docker recipe tells a user to pass it in, and Qt then says
    "XDG_RUNTIME_DIR points to non-existing path" on every start. Defaulting
    an unset variable is not enough: the broken case is a variable that is
    *set* to a path the container does not have.
    """
    entrypoint = (DOCKER_DIR / "entrypoint.sh").read_text(encoding="utf-8")

    assert "XDG_RUNTIME_DIR" in entrypoint, (
        "nothing normalises XDG_RUNTIME_DIR, so a documented `docker run` "
        "prints a Qt warning before the window appears."
    )
    assert '-d "${XDG_RUNTIME_DIR:-}"' in entrypoint, (
        "the entrypoint accepts whatever XDG_RUNTIME_DIR names without "
        "checking that it is a directory in this container."
    )
    assert "export XDG_RUNTIME_DIR" in entrypoint
    assert 'chmod 700 "$XDG_RUNTIME_DIR"' in entrypoint, (
        "Qt checks the mode as well as the path."
    )

    guide = GUIDE.read_text(encoding="utf-8")
    assert "-e XDG_RUNTIME_DIR " not in guide, (
        "the guide still tells the user to pass the host's runtime directory "
        "in, which is the thing that produces the warning."
    )


def test_the_workflow_does_not_fire_on_every_push(workflow):
    """A release, a hand-pushed tag, a deliberate dispatch. Nothing else.

    Each image is multi-gigabyte. Building both on every commit spends an hour
    of runner time to learn what `tests` already guards.
    """
    triggers = workflow["_triggers"]

    assert set(triggers) == {"push", "workflow_call", "workflow_dispatch"}, (
        f"the docker workflow is triggered by {sorted(triggers)}"
    )
    assert "branches" not in triggers["push"], (
        "the docker workflow would build both images on every push to a "
        "branch."
    )
    assert triggers["push"]["tags"] == ["v*"]


def test_a_release_publishes_an_image_without_relying_on_the_tag_push():
    """The release CALLS this workflow. A release tag cannot start it.

    `release.yml` creates the tag in its last job with `git push origin
    "$RELEASE_TAG"`, using the credentials `actions/checkout` persists -- the
    default GITHUB_TOKEN. GitHub starts no workflow run from an event pushed
    with that token, so a `push: tags` trigger on its own would sit idle
    through every release while the installer guide told users an image had
    been published. The repository's own answer to this is `workflow_call`
    (`online-installers.yml`), and that is what this asserts.
    """
    release = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))

    callers = {
        name: job
        for name, job in release["jobs"].items()
        if str(job.get("uses", "")).endswith("docker-images.yml")
    }
    assert list(callers) == ["container-images"], (
        "release.yml does not call docker-images.yml, so the only way an "
        f"image reaches GHCR is by hand; its jobs are {list(release['jobs'])}."
    )

    job = callers["container-images"]
    assert job["with"]["publish"] is True
    assert job["permissions"]["packages"] == "write", (
        "a called workflow cannot be granted more than the calling job holds, "
        "and release.yml's own permissions do not include packages: write, so "
        "every `docker push` would be denied."
    )
    assert "installers" in job["needs"], (
        "the images must be built from the commit the release tags, which is "
        "the one the installers job commits."
    )
    assert "needs.installers.outputs.release_commit" in job["with"]["source_ref"]
    assert "needs.bump.outputs.version" in job["with"]["release_version"]


def test_the_called_run_builds_the_commit_it_was_given():
    """Both jobs check out `inputs.source_ref`, never a bare `github.sha`.

    Inside a called workflow `github.sha` is the CALLER's commit: for a
    release, the push to main that started it -- one commit before the version
    bump and several before the installers. A checkout without the input would
    build the previous version and tag it as the new one.
    """
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    for name, job in workflow["jobs"].items():
        checkouts = [
            step for step in job["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout")
        ]
        assert checkouts, f"job {name} checks nothing out"
        for step in checkouts:
            ref = str(step.get("with", {}).get("ref", ""))
            assert "inputs.source_ref" in ref, (
                f"{name} checks out {ref or 'the default ref'}, which for a "
                f"called run is the caller's commit."
            )


def test_which_run_this_is_never_comes_from_the_event_name(workflow):
    """The mode is an input. `github.event_name` is the caller's event.

    1.5.0.5 shipped with `online-installers`' collect job gated on
    `github.event_name == 'workflow_call'`, a condition that is never true
    inside a called workflow. It was skipped on every release, and the release
    was tagged with an empty SHA. The same context here would compare `main`
    against setup.py's VERSION and fail every release.
    """
    plan = [
        step for step in workflow["jobs"]["plan"]["steps"]
        if step.get("id") == "plan"
    ][0]["run"]

    assert 'if [ -n "${RELEASE_VERSION:-}" ]; then' in plan, (
        "nothing distinguishes a called run from a tag push by its inputs."
    )
    assert plan.index("mode=release") < plan.index('EVENT_NAME" = "push"'), (
        "the event name is read before the release input, so a release -- "
        "which reports the caller's `push` -- takes the tag branch."
    )
    assert "RELEASE_VERSION: ${{ inputs.release_version }}" in WORKFLOW.read_text(
        encoding="utf-8")


def test_the_guide_only_promises_an_image_while_the_release_builds_one():
    """The user-facing claim and the wiring are one assertion, not two.

    The installer guide tells a reader to `docker pull` a tag by version. That
    sentence is true exactly as long as a release still builds and pushes the
    image, so it is asserted here rather than left to be discovered by a user
    whose pull returns "manifest unknown".
    """
    guide = GUIDE.read_text(encoding="utf-8")
    if "published to the GitHub Container Registry" not in guide:
        pytest.skip("the guide no longer promises published images")

    release = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    assert any(
        str(job.get("uses", "")).endswith("docker-images.yml")
        for job in release["jobs"].values()
    ), (
        "the installer guide says images are published as part of every "
        "release, but no release job builds one."
    )


def test_the_workflow_queues_rather_than_cancelling(workflow):
    """A release build killed by the next tag leaves a release with no image."""
    assert workflow["concurrency"]["cancel-in-progress"] is False


def test_every_image_is_smoke_tested_before_it_is_pushed(workflow):
    """The three checks run first, and the push step is last.

    An image published before it is checked is worse than no image: the user
    finds out an hour into their own data.
    """
    steps = workflow["jobs"]["image"]["steps"]
    names = [str(step.get("name", "")) for step in steps]

    smoke = [index for index, name in enumerate(names)
             if name.startswith("Smoke")]
    pushes = [index for index, step in enumerate(steps)
              if "docker push" in str(step.get("run", ""))]

    assert len(smoke) >= 3, (
        f"only {len(smoke)} smoke steps: the version, the UID and one real "
        f"pipeline run are all required."
    )
    assert pushes, "the workflow never pushes anything"
    assert min(pushes) > max(smoke), (
        "a push step runs before a smoke step; the image would be published "
        "and then checked."
    )


def test_the_push_is_conditional_and_the_smoke_tests_are_not(workflow):
    """Nothing is pushed unless the plan job says so, and checks always run."""
    steps = workflow["jobs"]["image"]["steps"]

    for step in steps:
        name = str(step.get("name", ""))
        run = str(step.get("run", ""))
        if "docker push" in run or "docker login" in run:
            assert "needs.plan.outputs.push" in str(step.get("if", "")), (
                f"step {name!r} publishes without checking the plan's push "
                f"decision."
            )
        if name.startswith("Smoke"):
            assert "if" not in step, (
                f"smoke step {name!r} is conditional; a check that can be "
                f"skipped is not a gate."
            )


def test_the_workflow_builds_the_dockerfiles_that_exist(workflow):
    """The variants the matrix can produce each have a Dockerfile."""
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'packaging/docker/Dockerfile.$VARIANT' in text
    for variant in VARIANTS:
        assert (DOCKER_DIR / f"Dockerfile.{variant}").is_file()
        assert f'"{variant}"' in text or f"'{variant}'" in text


def test_the_image_and_its_tag_cannot_disagree_about_the_version(workflow):
    """A tag pushed at the wrong commit must fail the build, not publish."""
    plan = workflow["jobs"]["plan"]["steps"][-1]["run"]

    assert 'VERSION = "([^"]+)"' in plan, (
        "the plan step no longer reads the version out of setup.py"
    )
    assert "setup.py says" in plan, (
        "nothing compares the tag name to setup.py's VERSION, so a tag at the "
        "wrong commit publishes an image labelled with a version it does not "
        "contain."
    )


def test_the_cuda_image_states_the_floor_it_promises():
    """CUDA 12.4 and a 550+ host driver, in the file and in the guide.

    The driver is on the host and cannot be containerised away, so the floor
    is the promise; a reader who cannot find it assumes there is none.
    """
    text = _dockerfile("cuda")

    assert "12.4" in text
    assert "550" in text, "the Dockerfile does not name the host driver floor"

    guide = (REPO_ROOT / "docs" / "source" / "installer_guide.rst").read_text(
        encoding="utf-8")
    assert "cuda12.4" in guide
    assert "550" in guide


def test_the_build_context_still_holds_what_pip_install_needs():
    """`.dockerignore` must not exclude a file the wheel build opens.

    setup.py opens README.rst, pyproject.toml carries the metadata, and
    setup.py's `data_files` installs packaging/linux's desktop entry and its
    icons on Linux. Excluding one of those fails the build with a message
    about the wrong thing.
    """
    patterns = [line.strip()
                for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]

    needed = ("README.rst", "pyproject.toml", "setup.py", "MANIFEST.in",
              "requirements.txt", "spacr/", "packaging/")
    for required in needed:
        assert required not in patterns, (
            f".dockerignore excludes {required!r}, which `pip install .` reads."
        )
    # And the four trees that make the difference are excluded.
    for heavy in ("docs/", "tools/", "tests/", ".git"):
        assert heavy in patterns, (
            f".dockerignore no longer excludes {heavy!r}; the build context "
            f"goes back to 1.2 GB."
        )


def test_every_shell_step_in_the_workflow_parses(workflow):
    """`bash -n` over each `run:` block.

    A workflow's shell is not checked by anything until it runs, and this one
    runs on a release tag -- the worst moment to learn that a heredoc lost its
    indentation or a `for` lost its `done`. GitHub's expression placeholders
    are replaced with a literal first, because `${{ ... }}` is not shell.
    """
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash on this machine")

    blocks = []
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            if "run" in step:
                blocks.append(
                    (f"{job_name}/{step.get('name', step.get('id', '?'))}",
                     str(step["run"])))

    assert blocks, "the workflow runs no shell at all"
    for where, script in blocks:
        neutral = re.sub(r"\$\{\{[^}]*\}\}", "EXPRESSION", script)
        result = subprocess.run([bash, "-n"], input=neutral, text=True,
                                capture_output=True, check=False)
        assert result.returncode == 0, (
            f"{where} is not valid shell:\n{result.stderr}"
        )


def test_the_smoke_script_imports_nothing_heavy_at_module_scope():
    """It must be able to run and report before torch is importable.

    Same rule `spacr.cli` keeps: everything heavy is imported inside the
    function that needs it, so a broken scientific stack produces a named
    failure rather than a traceback before the first line of output.
    """
    script = DOCKER_DIR / "smoke_pipeline.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))

    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top_level.add((node.module or "").split(".")[0])

    heavy = sorted(name for name in top_level
                   if name and name not in sys.stdlib_module_names)
    assert not heavy, (
        f"smoke_pipeline.py imports {heavy} at module scope; import them "
        f"inside the function that uses them."
    )


def test_the_smoke_script_asserts_that_objects_were_measured():
    """A database that exists and holds nothing is a failed run, not a pass."""
    text = (DOCKER_DIR / "smoke_pipeline.py").read_text(encoding="utf-8")

    assert "REQUIRED_TABLES" in text
    assert "count > 0" in text, (
        "the smoke script checks that tables exist but not that they hold a "
        "measured object; an empty measurements.db would pass."
    )
