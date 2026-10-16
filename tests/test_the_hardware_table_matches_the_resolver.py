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
    headers = {generator.HARDWARE_3D_HEADER[0], "Hardware"}
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
    assert _row(table, "NVIDIA (CUDA)") == [f"{GREEN} GPU"] * 3


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
    assert _row(table, "No GPU") == [f"{GREEN} CPU"] * 3


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
        f"{RED} CPU"] * 3


@pytest.mark.parametrize("language", ("en", "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr"))
def test_every_readme_has_current_hardware_table_and_sourced_cpu_context(generator, table, language):
    tool = generator.build_readme_images
    localized = table if language == "en" else generator.localize_hardware_markup(table, language)
    readme = tool.readme_path(language).read_text(encoding="utf-8")
    block = readme.partition(generator.HARDWARE_BEGIN)[2].partition(generator.HARDWARE_END)[0]
    assert block.strip() == tool.convert_generated(localized, language, "hardware", render=False).strip()
    output = tool.compose(language, render=False)
    blocks = tool.parse_rst(localized)
    second = [i for i, b in enumerate(blocks) if b["kind"] == "table"][1]
    assert output.pages["generated-hardware-1"] == tool.readable_page(
        blocks[:second], language, generated=True)
    assert output.pages["generated-hardware-2"] == tool.readable_page(
        blocks[second:], language, generated=True)
    source = tool.load_source(language)
    position = next(i for i, row in enumerate(source)
                    if row.get("kind") == "generated" and row.get("name") == "hardware")
    context = source[position + 1]["text"]
    for term in ("CPU execution is supported", "444.5 s", "3.2 s", "256×256",
                 "Core i5-10500", "Radeon Pro 5300", "torch 2.2.2", "historical example",
                 "usable backend"):
        assert term in context
    target = "features/new/319_use_non_nvidia_accelerators_where_they_exist.txt"
    assert (ROOT / target).is_file() and target in context
    page = next(value for value in output.pages.values() if "444.5 s" in value)
    assert "../../../" + target in page
    assert "444.5 s" in readme


def test_the_3d_table_names_each_platform_api_then_opengl_then_cpu(generator):
    """True 3D: wgpu's native API first, then OpenGL, then the CPU.

    Vulkan on Linux, Vulkan or Direct3D 12 on Windows, Metal on macOS; an
    install without wgpu (Python 3.9, other processors, old glibc, or the
    installer's opt-out) starts at OpenGL.
    """
    rows = dict(generator.HARDWARE_3D_ROWS)
    assert rows["Linux (x86-64, ARM64)"].startswith("Vulkan (wgpu)")
    assert rows["Windows (x64, ARM64)"].startswith(
        "Vulkan / Direct3D 12 (wgpu)")
    assert rows["macOS (Intel, Apple silicon)"].startswith("Metal (wgpu)")
    assert rows["Without wgpu"].startswith("OpenGL, CPU")
    assert "Python 3.9" in rows["Without wgpu"]
    assert "2.28" in rows["Without wgpu"]
    for cell in rows.values():
        assert cell.split(" (")[0].endswith("OpenGL, CPU") or \
            cell.endswith("OpenGL, CPU")
    table = generator._hardware_table()
    assert "   :widths: 32 58" in table


@pytest.mark.parametrize("language", ("sv", "de", "es", "zh_CN", "pt", "hi",
                                      "ko", "is", "fr"))
def test_the_3d_table_words_are_translated_and_product_names_kept(
        generator, table, language):
    localized = generator.localize_hardware_markup(table, language)
    for source in ("3D rendering", "True 3D, tried in order", "Without wgpu",
                   "other processors"):
        assert source not in localized, (language, source)
    for name in ("Vulkan", "Metal", "Direct3D 12", "OpenGL", "wgpu",
                 "Python 3.9", "2.28", "True 3D"):
        assert name in localized, (language, name)
