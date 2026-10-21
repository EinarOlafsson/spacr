"""Shared stdlib catalog projection for Python and frozen release builds."""

def project_runtime_catalogs(root):
    """Omit development provenance from copied distribution catalogs only.

    Repository catalogs keep the fields used by generation audits. Runtime
    translations and source hashes are independent of these literal fields.
    Replacing staged files also preserves sources hard-linked by sdist.
    Already projected catalogs need no change when building from an sdist.
    """
    import ast
    import os
    from pathlib import Path
    import tempfile

    folder = Path(root) / "spacr" / "qt" / "i18n_catalogs"
    fields = {"MODEL", "LICENSE", "SECONDARY_MODEL", "SECONDARY_LICENSE",
              "NORMALIZER"}
    for language in ("de", "es", "fr", "hi", "is", "ko", "pt", "sv", "zh_CN"):
        path = folder / (language + ".py")
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines(keepends=True)
        removed = set()
        for node in ast.parse(source).body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in fields):
                continue
            if (not isinstance(node.value, ast.Constant)
                    or not isinstance(node.value.value, str)
                    or node.lineno != node.end_lineno
                    or lines[node.lineno - 1].strip() != ast.get_source_segment(source, node)):
                raise ValueError("Expected literal catalog provenance: " + str(path))
            removed.update(range(node.lineno - 1, node.end_lineno))
        if not removed:
            continue
        projected = "".join(line for index, line in enumerate(lines)
                            if index not in removed)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="", dir=path.parent,
                    prefix="." + path.name + ".", delete=False) as stream:
                temporary = stream.name
                stream.write(projected)
            os.chmod(temporary, path.stat().st_mode)
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)



def stage_runtime_catalogs(root, destination):
    """Copy the complete catalog package into an owned build directory.

    The caller owns the destination and its lifetime. Source files, including
    the loader and canonical English catalog, remain unchanged.
    """
    from pathlib import Path
    import shutil

    relative = Path("spacr") / "qt" / "i18n_catalogs"
    target = Path(destination) / relative
    shutil.copytree(Path(root) / relative, target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    project_runtime_catalogs(destination)
    return target.parent
