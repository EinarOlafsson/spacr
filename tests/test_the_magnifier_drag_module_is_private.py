"""The internal stroke module must not create catalog keys for hidden pages."""

import shutil
import zlib

from tests.test_built_nested_helper_api import ROOT, _build_site, builder


def test_the_drag_module_extractor_boundary_matches_real_sphinx(tmp_path, monkeypatch):
    root = tmp_path / "source"
    package = root / "spacr"
    qt = package / "qt"
    qt.mkdir(parents=True)
    for directory in (package, qt):
        (directory / "__init__.py").write_text('"""Public package."""\n')
    for name in ("api", "core", "measure", "deep_spacr", "sequencing", "ml", "artifacts", "settings"):
        (package / f"{name}.py").write_text(f'"""Public {name} module."""\n')
    (package / "example.py").write_text(
        '"""Public example."""\ndef entry():\n    """Visible callable."""\n')
    shutil.copyfile(ROOT / "spacr/qt/_magnifier_drag.py", qt / "_magnifier_drag.py")

    output = _build_site(root, (), "drag-private")
    payload = (output / "objects.inv").read_bytes().split(b"\n", 4)[4]
    objects = {line.split()[0] for line in zlib.decompress(payload).decode().splitlines()}
    assert "spacr.example.entry" in objects
    assert not (output / "api/spacr/qt/_magnifier_drag/index.html").exists()
    assert not any(key.startswith("spacr.qt._magnifier_drag") for key in objects)

    monkeypatch.setattr(builder, "ROOT", root)
    monkeypatch.setattr(builder, "API_DOC_ALIASES", {})
    documents = builder.public_docstrings()
    assert documents["spacr.example.entry"] == "Visible callable."
    assert not any(key.startswith("spacr.qt._magnifier_drag") for key in documents)
