"""The README's hardware table agrees with the resolver that produced it.

Instruction 323. The table says what each hardware configuration
accelerates, and it is GENERATED from
``spacr.accelerator.capabilities()`` with each backend's probe faked --
the same trick the resolver's own tests use to exercise 19 backends on a
machine that has one.

Generated rather than typed because a typed table is wrong within a day.
Instruction 321 has the precedent: the module grid carried a hand-written
copy of Home's layout and went stale the first time Home changed.

The table also has to be RIGHT in a way a regeneration cannot check --
that its three colours mean what the legend says. Those are the
assertions below with reasons attached.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.rst"
GENERATED = ROOT / "docs" / "source" / "_generated" / "hardware_table.rst"

GREEN, PURPLE, RED = "\U0001F7E2", "\U0001F7E3", "\U0001F534"


@pytest.fixture(scope="module")
def generator():
    """The generator module, loaded from packaging/ by path."""
    spec = importlib.util.spec_from_file_location(
        "_readme_visuals", ROOT / "packaging" / "generate_readme_visuals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def table(generator):
    return generator._hardware_table()


def test_the_readme_holds_what_the_generator_produces(generator, table):
    """Regenerating changes nothing. If it does, somebody typed.

    This is the check that keeps the table honest with no effort: it
    fails the moment the resolver's answers change and the generator has
    not been re-run.
    """
    readme = README.read_text(encoding="utf-8")
    block = readme.partition(generator.HARDWARE_BEGIN)[2].partition(
        generator.HARDWARE_END)[0]
    drawn = generator.build_readme_images.convert_generated(
        table, "en", "hardware", render=False)
    assert block.strip() == drawn.strip(), (
        "README.rst disagrees with the generated hardware table; run "
        "python packaging/generate_readme_visuals.py")
    # THE TABLE IS AN IMAGE since 2026-10-12, and its alt text is the table
    # read out row by row, so every row is still in the README's words.
    headers = {"Hardware"}
    for line in re.findall(r"^   \* - (.+)$", table, re.M):
        if line not in headers:
            assert f"{line}: " in block, line


def test_the_generated_file_matches_too(table):
    assert GENERATED.read_text(encoding="utf-8").strip() == table.strip()


def _row(table_text, label):
    """The cells of one hardware row, in column order."""
    body = table_text[table_text.index(f"* - {label}"):]
    body = body[:body.index("* - ", 4)] if "* - " in body[4:] else body
    return re.findall(r"^     - (.+)$", body, re.M)


def test_cuda_is_green_across_the_board(table):
    """The only configuration with years behind it."""
    assert _row(table, "NVIDIA (CUDA)") == [f"{GREEN} GPU"] * 3 + [f"{PURPLE} GPU"]


def test_the_new_backends_are_purple_not_green(table):
    """Implemented is not the same claim as exercised.

    Metal on an Intel Mac was measured -- 444.5 s to 3.2 s -- but on one
    machine, one day old. ROCm, XPU and Apple Silicon Metal were measured
    on nothing, because no such hardware was available. Painting them
    green because the code exists is what the beta tier prevents.
    """
    for label in ("AMD on Linux (ROCm)", "AMD in an Intel Mac (Metal)",
                  "Apple Silicon (Metal)", "Intel Arc/Xe (XPU)"):
        cells = _row(table, label)
        assert cells[0] == f"{PURPLE} GPU", label
        assert cells[1] == f"{PURPLE} GPU", label


def test_umap_is_red_on_every_gpu_that_is_not_cuda(table):
    """cuML ships for CUDA only, so there is nothing to wait for.

    RED, not "CPU in green": on a machine that HAS a GPU, a task that
    cannot use it is unsupported on that hardware. This is the cell the
    table exists for.
    """
    for label in ("AMD on Linux (ROCm)", "AMD in an Intel Mac (Metal)",
                  "Apple Silicon (Metal)", "Intel Arc/Xe (XPU)"):
        assert _row(table, label)[2] == f"{RED} CPU", label


def test_the_no_gpu_row_is_green_everywhere(table):
    """THE MOST DANGEROUS CELL IN THE TABLE, if it were red.

    Its cells say CPU because that is what they use. Red would say spaCR
    does not support running without a GPU -- false, and the most
    damaging claim this table could make. Every task runs and every
    result is identical; only the clock changes.
    """
    assert _row(table, "No GPU") == [f"{GREEN} CPU"] * 4


def test_the_intel_mac_row_stands_on_its_own(table):
    """It must not be merged into either neighbour.

    That grouping is the mistake instruction 319's own backend table
    made -- Metal filed under Apple Silicon, AMD under ROCm -- and since
    ROCm has no macOS build, the configuration that actually works was
    named nowhere. It hid a 139x speedup until somebody measured it.
    """
    text = README.read_text(encoding="utf-8")
    assert "AMD in an Intel Mac (Metal)" in text
    assert "Apple Silicon (Metal)" in text


def test_the_legend_names_all_three_marks(table):
    """A legend describing a mark the table never uses, or a mark with no
    legend entry, is worse than no legend."""
    used = set(re.findall(f"[{GREEN}{PURPLE}{RED}]", table))
    legend = table[table.index("supported (stable)") - 5:]
    for mark in used:
        assert mark in legend, f"{mark} is used but not in the legend"
    for mark in (GREEN, PURPLE, RED):
        assert mark in used, f"{mark} is in the legend but never used"


def test_a_column_that_lost_its_capability_row_fails_loudly(generator,
                                                            monkeypatch):
    """Renaming a task in `capabilities()` must not empty a column.

    Silently blank cells are the failure mode of deriving from another
    module's strings, so the generator refuses instead.
    """
    monkeypatch.setattr(generator, "HARDWARE_TASKS",
                        (("Nonsense", "NoSuchTaskPrefix"),))
    with pytest.raises(ValueError, match="diverged"):
        generator._hardware_table()


@pytest.mark.parametrize("kind,device", [("mps", "mps"), ("cuda", "cuda")])
def test_detected_but_unusable_gpu_never_claims_gpu_acceleration(generator, monkeypatch, kind, device):
    from spacr import accelerator

    kwargs = dict(kind=kind, device=device, label="Detected GPU", detected=True, usable=False)
    found = accelerator.Accelerator(**kwargs)
    assert found.is_cuda is (kind == "cuda")
    assert found.is_gpu is False
    monkeypatch.setattr(accelerator, "_opengl_likely", lambda: False)
    capabilities = {task: (accelerated, detail)
                    for task, accelerated, detail in accelerator.capabilities(found)}
    accelerated, detail = capabilities["UMAP / t-SNE / clustering"]
    assert accelerated is False
    assert detail.startswith("CPU")
    monkeypatch.setattr(generator, "HARDWARE_ROWS", (
        ("Detected GPU, unavailable runtime", "beta", kwargs),
    ))
    assert _row(generator._hardware_table(), "Detected GPU, unavailable runtime") == [
        f"{RED} CPU"] * 3 + [f"{PURPLE} GPU"]


@pytest.mark.parametrize("language", ("en", "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr"))
def test_every_readme_has_current_hardware_table_and_sourced_cpu_context(generator, table, language):
    tool = generator.build_readme_images
    localized = table if language == "en" else generator.localize_hardware_markup(table, language)
    readme = tool.readme_path(language).read_text(encoding="utf-8")
    block = readme.partition(generator.HARDWARE_BEGIN)[2].partition(generator.HARDWARE_END)[0]
    assert block.strip() == tool.convert_generated(localized, language, "hardware", render=False).strip()
    output = tool.compose(language, render=False)
    blocks = tool.parse_rst(localized)
    assert [b["kind"] for b in blocks] == ["table", "paragraph"]
    assert blocks[0]["rows"][0] == ["Hardware", "Cellpose 4", "Torch", "UMAP / clustering", "3D"]
    assert len(blocks[0]["rows"]) == 7
    source = tool.load_source(language)
    hardware = next(row for row in source
                    if row.get("kind") == "generated" and row.get("name") == "hardware")
    assert len(hardware["details"]) == 3
    assert hardware["details"][0].startswith("Auto:")
    details = [{"kind": "paragraph", "text": text} for text in hardware["details"]]
    page = output.pages["generated-hardware-1"]
    assert page == (tool.readable_page(blocks, language, generated=True).rstrip() +
                    "\n\n" + tool.readable_page(details, language))
    assert "generated-hardware-2" not in output.pages
    for term in ("444.5 s", "3.2 s", "256×256", "Core i5-10500", "Radeon Pro 5300",
                 "torch 2.2.2", "Python 3.12.11", "macOS 26.5.2", "Python 3.9",
                 "2.28", "Vulkan", "Direct3D 12", "OpenGL", "Metal", "wgpu"):
        assert term in page
    target = "features/new/319_use_non_nvidia_accelerators_where_they_exist.txt"
    assert (ROOT / target).is_file()
    assert "../../../" + target in page
    assert "444.5 s" not in readme and "True 3D" not in block
    assert not any(row.get("kind") == "paragraph" and
                   "spacr-download" in row.get("text", "") for row in source)
    assert tool.yaml_to_blocks(tool.blocks_to_yaml(source)) == source


def test_3d_is_a_single_beta_gpu_column_with_supported_cpu_fallback(table):
    assert table.count(".. list-table::") == 1
    assert table.count("     - 3D\n") == 1
    for label in ("NVIDIA (CUDA)", "AMD on Linux (ROCm)",
                  "AMD in an Intel Mac (Metal)", "Apple Silicon (Metal)",
                  "Intel Arc/Xe (XPU)"):
        assert _row(table, label)[3] == f"{PURPLE} GPU"
    assert _row(table, "No GPU")[3] == f"{GREEN} CPU"


def test_readable_3d_details_follow_the_actual_backend_order(generator):
    source = generator.build_readme_images.load_source("en")
    details = next(row["details"] for row in source if row.get("name") == "hardware")
    from spacr.qt.screens.make_masks import VOLUME_BACKENDS, volume_backend_chain

    for system in ("linux", "win32", "darwin"):
        chain = volume_backend_chain("auto", offered=VOLUME_BACKENDS, platform=system) + ("cpu",)
        names = {"vulkan": "Vulkan", "d3d12": "Direct3D 12", "metal": "Metal",
                 "opengl": "OpenGL", "cpu": "CPU"}
        assert " → ".join(names[name] for name in chain) in details[0]
    assert "historical example" in details[2]
    assert "not a speed guarantee" in details[2]
