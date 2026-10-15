"""Backend runtime captions enter catalogs only through their declared owners."""
from importlib import import_module
from pathlib import Path

import pytest


@pytest.fixture
def extract_backend_sources(tmp_path, monkeypatch):
    """Run the production extractor over a small source tree, without catalogs."""
    tools_dir = str(Path(__file__).resolve().parents[1] / "tools")
    monkeypatch.syspath_prepend(tools_dir)
    builder = import_module("build_i18n_catalogs")
    source_root = tmp_path / "spacr"
    (source_root / "qt").mkdir(parents=True)
    for filename in (
        "model_compare.py", "embeddings.py", "plot.py", "model_zoo.py",
        "install_cleanup.py", "settings.py", "object.py", "remote_execution.py",
        "attribution.py",
    ):
        (source_root / filename).write_text("", encoding="utf-8")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "_indirect_runtime_ui_sources", set)

    def extract(filename, source):
        (source_root / filename).write_text(source, encoding="utf-8")
        return set(builder.extract_static_ui_sources())

    return extract


def test_model_download_caption_is_extracted_from_shared_acquisition_owner(
    extract_backend_sources,
):
    sources = extract_backend_sources("model_zoo.py", '''
def _ensure_catalogue_model_file(entry, *, download=True):
    """Internal acquisition documentation is not a runtime caption."""
    raise ModelUnreadable(tr(
        "Automatic model downloads are disabled by SPACR_MODEL_DOWNLOAD. "
        "Copy verified model {model} (sha256 {sha256}) to {path}.",
        model=entry.key, sha256=entry.sha256, path=str(cached)))
''')
    assert sources == {
        "Automatic model downloads are disabled by SPACR_MODEL_DOWNLOAD. "
        "Copy verified model {model} (sha256 {sha256}) to {path}."
    }


def test_model_download_scope_excludes_other_functions_and_metadata(
    extract_backend_sources,
):
    sources = extract_backend_sources("model_zoo.py", '''
METADATA = tr("This metadata is outside the runtime owner.")
def _ensure_model_file(requested, *, download=True):
    return tr("The former wrapper must not become an extra runtime owner.")
def unrelated_model_backend():
    return tr("Other backend diagnostics remain outside this extraction scope.")
class DifferentOwner:
    def _ensure_catalogue_model_file(self):
        return tr("An unrelated method with the same name must not be scanned.")
def _ensure_catalogue_model_file(entry, *, download=True):
    raise RuntimeError("An untranslated backend detail must not be inventoried.")
''')
    assert sources == set()


def test_install_cleanup_cli_literals_enter_runtime_sources(
    extract_backend_sources,
):
    captions = (
        "remove only this recognized installation, preserving its data",
        "require an interpreter outside the selected root",
        "restore one unchanged retained private POSIX runtime",
        "previous desktop process that must have exited",
        "copy the private managed Python outside this installation",
        "Uninstall preparation stopped: {error}",
        "Runtime recovery stopped: {error}",
        "Uninstall requires a helper and interpreter outside the selected installation.",
        "Targeted removal requires one recognized canonical installation and cannot purge user data.",
    )
    source = 'def _main():\n'
    source += '    parser.add_argument("--private-action", dest="private_action")\n'
    source += ''.join(f'    tr({caption!r})\n' for caption in captions)
    source += '    raise RuntimeError("Internal cleanup detail is not translated.")\n'
    assert extract_backend_sources("install_cleanup.py", source) == set(captions)


@pytest.mark.parametrize('owner, indices', (
    ('_cf_preserved_channels', (0, 1)),
    ('_cf_preserve_reference_channels', (2, 3)),
    ('_counterfactual_sequences', (4, 5)),
    ('_counterfactual_report', (4, 5)),
))
def test_counterfactual_reference_diagnostics_reach_the_runtime_inventory(
    extract_backend_sources, owner, indices,
):
    captions = (
        "Preserved channels must be a list of valid zero-based whole channel indices.",
        "Select each preserved channel only once.",
        "Reference and edited crops must have matching floating NCHW shapes, dtype and device.",
        "Reference and edited crops must contain finite pixels.",
        "Reference preservation requires NCHW crop tensors.",
        "Reference crops must contain finite pixels.",
    )
    selected = {captions[index] for index in indices}
    source = f'def {owner}():\n' + ''.join(
        f'    tr({caption!r})\n' for caption in sorted(selected))
    assert extract_backend_sources('attribution.py', source) == selected


def test_counterfactual_scope_excludes_unrelated_and_untranslated_diagnostics(
    extract_backend_sources,
):
    sources = extract_backend_sources('attribution.py', '''
tr("A module-level diagnostic outside the reference-channel owners.")
def unrelated_attribution_method():
    return tr("An unrelated backend diagnostic must not enter the catalog.")
class OtherOwner:
    def _counterfactual_report(self):
        return tr("A method sharing an owner name must not enter the catalog.")
def _cf_preserved_channels():
    """Internal reference-channel documentation is not a runtime caption."""
    raise ValueError("An old non-translated diagnostic remains outside the inventory.")
''')
    assert sources == set()
