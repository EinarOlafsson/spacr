import subprocess
import tarfile
import zipfile
from pathlib import Path
wheel = next(Path('/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015/artifacts').glob("*.whl"))
sdist = next(Path('/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015/artifacts').glob("*.tar.gz"))
tracked = set(subprocess.check_output(
    ["git", "ls-files", "spacr"], text=True).splitlines())
pruned_roots = (
    "spacr/resources/home/",
    "spacr/resources/icons/backup_icons/",
)
excluded_files = {
    "spacr/resources/.DS_Store",
    "spacr/resources/icons/.DS_Store",
    "spacr/resources/icons/loading_spinner_1.gif",
    "spacr/resources/icons/loading_spinner_logo.gif",
    "spacr/resources/icons/loading_spinner_logo (1).gif",
    "spacr/resources/icons/loading_spinner_logo",
}
def is_runtime_python(name):
    return name.endswith(".py") and not name.startswith(pruned_roots)
def is_runtime_resource(name):
    if (name in excluded_files
            or name.startswith(pruned_roots)
            or name.startswith("spacr/resources/models/")):
        return False
    return (
        (name.startswith("spacr/resources/data/")
         and name.endswith(".csv"))
        or name.startswith("spacr/resources/icons/")
        or name.startswith("spacr/resources/font/")
        or name.startswith("spacr/resources/setting_animations/")
        or (name.startswith("spacr/resources/themes/")
            and name.endswith(".jpg"))
    )
required = {
    name for name in tracked
    if is_runtime_python(name) or is_runtime_resource(name)
}
with zipfile.ZipFile(wheel) as archive:
    wheel_names = set(archive.namelist())
with tarfile.open(sdist) as archive:
    sdist_names = {
        name.split("/", 1)[1]
        for name in archive.getnames()
        if "/" in name
    }
forbidden_roots = pruned_roots + ("spacr/resources/models/",)
for label, names in (("wheel", wheel_names), ("sdist", sdist_names)):
    missing = sorted(required - names)
    forbidden = sorted(
        name for name in names
        if name in excluded_files or name.startswith(forbidden_roots)
    )
    assert not missing, f"{label} omits runtime files: {missing}"
    assert not forbidden, f"{label} ships excluded files: {forbidden}"
    print(f"{label}: {len(required)} required runtime files present")
