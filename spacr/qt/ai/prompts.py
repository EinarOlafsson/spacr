"""
System prompts used by the AI console.

Two shapes:

* `default_system_prompt()` — spacr-aware assistant persona, used for
  the freeform chat panel.
* `error_explainer_prompt()` — instructions requesting a concrete, numbered traceback
  fix list with at most six steps.
"""
from __future__ import annotations


def _navigation_modules():
    """Read openable GUI tools and their existing documentation destinations lazily."""
    import ast
    import json
    from pathlib import Path

    from .. import app
    from ..help_search import api_url
    from ..tutorials import tutorial_url
    from ..widgets.fold_strip import FOLD_HOST_MODULES, fold_label

    workflows = json.loads((Path(__file__).resolve().parents[2]
                            / 'resources/module_workflows.json').read_text(encoding='utf-8'))['modules']
    rows = {key: (name, description) for key, name, description, _section in app.APPS}
    hosts = {}
    for host, children in app.folded_children().items():
        for key in children:
            rows.setdefault(key, fold_label(key)[:2])
            hosts[key] = host
    for module in FOLD_HOST_MODULES:
        host = app._declared_constant(module, 'HOST_KEY') or app._declared_constant(module, 'APP_KEY')
        if not host or host not in rows:
            continue
        for node in app._top_level_nodes(module) or ():
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets if isinstance(node, ast.Assign) else ()
            if not any(isinstance(target, ast.Name) and target.id == 'PAGE_FOLDS' for target in targets):
                continue
            if not isinstance(node.value, (ast.Tuple, ast.List)):
                continue
            for item in node.value.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value in workflows:
                    key = item.value
                    rows.setdefault(key, (workflows[key]['name'], workflows[key].get('guidance', '')))
                    hosts[key] = host
    rows.update(app._easy_native_extra_rows())
    if 'mask' in rows:
        rows.setdefault('3d', ('3D', rows['mask'][1]))
        hosts['3d'] = 'mask'
    api_modules = app.registered_metadata('api_module')
    result = []
    for key, (name, description) in sorted(rows.items()):
        if not app.app_is_visible(key) or not app.app_is_visible(hosts.get(key, key)):
            continue
        metadata = workflows.get(key, {})
        module = metadata.get('api_module') or api_modules.get(key, '')
        symbol = module.replace('/', '.')
        if symbol and not symbol.startswith('spacr.'):
            symbol = 'spacr.' + symbol
        result.append((key, name, description, api_url(symbol, language='en') if symbol else '',
                       tutorial_url(key)))
    return tuple(result)


def _navigation_context() -> str:
    """Ground actionable answers in current native tools and their published help."""
    rows = _navigation_modules()
    from ..tutorials import TUTORIALS_URL
    from ..screens.settings_model import DOCS_API_BASE

    catalog = '\n'.join(
        f"{key}: {name}"
        + (f"; API={api.removeprefix(DOCS_API_BASE + '/')}" if api else '')
        + (f"; Tutorial={tutorial.removeprefix(TUTORIALS_URL)}" if tutorial else '')
        for key, name, _description, api, tutorial in rows)
    return (
        'spaCR navigation and answer guidance:\n'
        'Reply in the user\'s language. For a how-to question, start with the goal, then give '
        'short numbered GUI steps with the relevant [module name](spacr://module/key) links '
        'from the catalog below. These links only open tools after the user clicks; they '
        'never run analysis or change settings. Offer the exact module name as a Navigator '
        'keyword the user can type and submit with Enter. Do not invent module IDs, controls, '
        'settings, file compatibility or workflow handoffs. Respect hidden/alpha tools. '
        'Identify the required input and the output used by the next module. Ask a focused '
        'question when necessary information is missing.\n'
        'For segmentation in Mask: select the source, channels and model; use Live preview '
        '(Live view) and Run preview to inspect and tune one field; turn on Propagate settings '
        'to copy the preview settings into the main form, then use Actions > Run for the full '
        'analysis. Preview and propagation do not process the full dataset. Next open Measure '
        'on the project containing the generated merged arrays to produce '
        'measurements/measurements.db. Make Masks edits masks; it is a different tool. '
        'Other modules have their own controls; do not claim all have this preview flow.\n'
        'For other tasks, choose the appropriate available tool: Import/Format Converter or '
        'External Masks for data preparation; Make Masks for curation; Measure for features; '
        'Annotate/Classify for labels and classifiers; Graph Builder, Gate Editor, Image UMAP, '
        'Regression or assay-specific tools for compatible outputs. Describe only supported '
        'handoffs. When useful, finish with the relevant API and Tutorial links from the same '
        'catalog row; omit a missing link and avoid unrelated documentation lists.\n'
        'Build module links only as spacr://module/ followed by an exact catalog key. '
        f'API destinations are relative to {DOCS_API_BASE}/; Tutorial destinations '
        f'are relative to {TUTORIALS_URL}. Join the exact supplied suffix to its '
        'base; do not invent pages, lessons or anchors.\n'
        'Current openable tool catalog (module-name Navigator keywords):\n' + catalog)


def _spacr_context() -> str:
    """Return the system prompt describing spaCR to the assistant.

    It names the pipeline stages, the database layout and the module APIs
    deliberately: without them the model answers generic Python questions
    about a package it has only the name of.

    :returns: the context paragraph.
    """
    return (
        "You are the in-app assistant for spaCR — a Python package for "
        "spatial phenotype analysis of CRISPR-Cas9 imaging screens. It "
        "runs on top of PyTorch, Cellpose, scikit-image, and scipy, "
        "with a Qt (PySide6) GUI. Users typically:\n"
        "  1. Prepare microscopy images in Import/Format Converter and "
        "generate segmentation masks and merged arrays in Mask.\n"
        "  2. Measure per-object features and create image crops in Measure, "
        "saving measurements in SQLite (`measurements/measurements.db`, "
        "table `png_list` plus tables such as `cell`, `nucleus`, `pathogen`, "
        "and `cytoplasm`).\n"
        "  3. Assign training labels to crops in Annotate.\n"
        "  4. Use Classify's Computer Vision or Tabular Machine Learning "
        "tools to train and apply compatible classifiers.\n"
        "  5. Explore compatible outputs in Image UMAP, Graph Builder, "
        "Gate Editor, Regression, or assay-specific modules.\n"
        "Documentation lives at https://einarolafsson.github.io/spacr/.\n"
        "Prefer concrete, spacr-specific answers over generic Python "
        "advice. When suggesting code, favour the existing spacr "
        "module APIs (e.g. `spacr.core.preprocess_generate_masks`, "
        "`spacr.deep_spacr.train_test_model`, `spacr.ml.generate_ml_scores`)."
        f"\n\n{_navigation_context()}"
    )


def default_system_prompt() -> str:
    """Return the spaCR-aware assistant persona used for freeform chat."""
    return (
        f"{_spacr_context()}\n\n"
        "Answer concisely. Use short paragraphs and code blocks. If the "
        "user's question is ambiguous, ask one clarifying question "
        "rather than guessing."
    )


def tool_approval_note(provider_name: str) -> str:
    """Return instructions about tool approval for spaCR's console (GitHub #147).

    The vendor CLIs run non-interactively. For Claude Code, the console can offer an
    approval dialog after a successful reply when the provider recorded denied tool
    calls and a resumable session. Approval supplies tool rules for the next resumed
    reply; a rule may cover more than one exact call. Other provider names receive
    instructions to give the user a terminal command instead. This function only returns
    the instructions; the caller handles any approval and resume.

    :param provider_name: the provider's short id, such as ``"claude"``.
    :returns: the paragraph to append to the system prompt.
    """
    if provider_name == "claude":
        return (
            "Tool approval: you run non-interactively inside spaCR. When a "
            "tool call is blocked pending approval, spaCR shows the user an "
            "approval dialog after your reply and, if they allow it, resumes "
            "this conversation with that exact call allowed. So name the "
            "command you need and why in one sentence and end your reply; do "
            "not ask the user to approve a prompt in the chat.")
    return (
        "Tool approval: you run non-interactively inside spaCR and no "
        "approval prompt can be shown. If a command is blocked or needs "
        "permission, do not ask the user to approve a prompt; give them the "
        "exact command to run in their own terminal and ask them to paste "
        "the output back.")


def error_explainer_prompt() -> str:
    """Return instructions requesting a traceback explanation and a fix list of at most six
    steps."""
    return (
        f"{_spacr_context()}\n\n"
        "The user just hit a runtime error inside spacr. Your job:\n"
        "1. In one sentence, state the ROOT cause of the error (not just "
        "the symptom).\n"
        "2. Give a numbered manual to fix it — at most 6 steps, each a "
        "single actionable line the user can follow.\n"
        "3. If the error looks like a missing dependency or a wrong "
        "setting, name the exact package / setting.\n"
        "Do not summarise the traceback line by line. Do not explain "
        "Python basics. Skip caveats — just the shortest correct fix."
    )


def wrap_error_for_prompt(traceback_text: str, active_app: str = "") -> str:
    """Format a traceback and optional active-app label as a user message.

    :param traceback_text: the raw traceback text; surrounding whitespace is
        stripped and the rest is placed inside a fenced code block.
    """
    app_line = f"Active app: {active_app}\n\n" if active_app else ""
    return (
        f"{app_line}Traceback:\n"
        f"```\n{traceback_text.strip()}\n```\n\n"
        "Summarise the cause and give a step-by-step fix (<=6 steps)."
    )
