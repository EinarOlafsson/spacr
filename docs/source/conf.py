"""Sphinx configuration for the spaCR API reference.

Theme: furo — auto-switches between light and dark modes based on
the reader's OS preference, has a clean sidebar for autoapi's per-
module tree, and renders reST docstrings without visual noise.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT / 'tools'))
import nested_helper_docs as _nested_helper_docs
import api_visibility as _api_visibility
from docs_version import source_version as _source_version
import build_guide_i18n as _guide_i18n

# Guides-only mode renders the translated user guides (and extracts their
# English messages). It skips AutoAPI, the tutorial media and the API
# catalogs: translated trees link to the English API and tutorial player.
# See tools/build_guide_i18n.py.
_guides_only = os.environ.get('SPACR_DOCS_GUIDES_ONLY', '') == '1'
if _guides_only:
    _source_api_projection = frozenset()
else:
    import build_documentation_i18n as _api_i18n
    _source_api_projection = _api_visibility.documented_tree(_api_i18n.public_docstrings())

sys.path.insert(0, os.path.abspath(
    os.path.join(__file__, '..', '..', 'spacr')
))

# -- Project metadata ------------------------------------------------------
project   = 'spaCR'
author    = 'Einar Birnir Olafsson'
copyright = f'2025-2026, {author}'

release = _source_version(_SOURCE_ROOT)
version = release

_source_branch = os.environ.get('GITHUB_REF_NAME', '').strip()
if _source_branch not in {'main', 'nightly'}:
    _source_branch = 'main'
docs_channel = 'nightly' if _source_branch == 'nightly' else 'stable'
_tutorial_root = os.path.join(
    os.path.dirname(__file__), '_extra', 'tutorials', 'production')
try:
    lesson_count = sum(
        os.path.isdir(os.path.join(_tutorial_root, name))
        for name in os.listdir(_tutorial_root)
    )
except OSError:
    lesson_count = 73
rst_epilog = (
    f'\n.. |spacr-version| replace:: {release}\n'
    f'.. |docs-channel| replace:: {docs_channel}\n'
    f'.. |lesson-count| replace:: {lesson_count}\n'
)

# -- General configuration -------------------------------------------------
extensions = [
    'sphinx.ext.napoleon',     # Google / NumPy / reST field lists
    'sphinx.ext.viewcode',     # [source] links on every symbol
    'sphinx.ext.intersphinx',  # cross-refs into stdlib / numpy / torch
    'sphinx_design',           # grid / card / tab directives on landing
    'autoapi.extension',       # AST-walk based auto reference
]
if _guides_only:
    extensions.remove('autoapi.extension')
    extensions.remove('sphinx.ext.viewcode')   # source pages live in English
    extensions.append('build_guide_i18n')
    html_copy_source = False

# -- Translated user guides (gettext) ----------------------------------------
# One catalog per page in docs/i18n/guides/<lang>/LC_MESSAGES/<page>.po. A
# message whose English changed no longer matches its msgid and renders in
# English; ``translation_progress_classes`` marks it ``untranslated`` so the
# stylesheet can flag it.
locale_dirs = ['../i18n/guides/']
gettext_compact = False
gettext_uuid = False
gettext_location = False
gettext_additional_targets = []
translation_progress_classes = True

suppress_warnings = ['misc.section', 'toc.not_included']
# `_generated/**` holds INCLUDE FRAGMENTS, not documents. Without this
# Sphinx parses each one twice -- once as a page in its own right and once
# where it is included -- which for settings_flow.rst means 1,058 duplicate
# `setting-flow-*` labels, every one of them a warning that `-W` makes
# fatal, and 18,000 lines resolved twice for a build that is already the
# slowest job in CI.
exclude_patterns = ['_autoapi_templates/**', '_generated/**']
if _guides_only:
    # ``api/`` may hold AutoAPI's kept sources from an English build; the
    # static exclusions keep the 161 MB API catalogs and the README deck out
    # of every translated tree (Sphinx applies exclude_patterns to
    # html_static_path too).
    exclude_patterns += ['api/**', 'i18n/**', 'deck/**',
                         *sorted(f'{page}.rst' for page in
                                 _guide_i18n.ENGLISH_ONLY_PAGES)]
default_role = 'py:obj'

intersphinx_mapping = {
    'python':      ('https://docs.python.org/3',                       None),
    'numpy':       ('https://numpy.org/doc/stable',                    None),
    'pandas':      ('https://pandas.pydata.org/docs',                  None),
    'scipy':       ('https://docs.scipy.org/doc/scipy',                None),
    'sklearn':     ('https://scikit-learn.org/stable',                 None),
    'torch':       ('https://pytorch.org/docs/stable',                 None),
    'matplotlib':  ('https://matplotlib.org/stable',                   None),
}

# -- Link check (``sphinx -b linkcheck``) ------------------------------------
# The tutorial player is copied into the build root from html_extra_path
# (docs/_build/extra_staged/tutorials), not from docs/source, so linkcheck's
# local-file test cannot see it. Every ``tutorials/`` link (the player itself,
# ``#lesson=<id>`` deep links from the API and workflow pages, and example
# downloads) resolves to /tutorials/ on each published channel.
# tests/test_module_workflow_map.py pins the relative depth of the API links;
# all 73 linked lesson ids were matched to lesson_catalog.js on 2026-09-30.
linkcheck_ignore = [
    r'^(\.\./)*tutorials/',
]
# Anchors that exist in the page a browser shows but not in the HTML that
# linkcheck downloads:
# * docs.pytorch.org/docs/stable/* is a JavaScript redirect stub to the
#   versioned tree (for example /docs/2.14/); the anchors are on that page.
# * GitHub renders README headings as ``user-content-<slug>`` ids and maps the
#   plain ``#<slug>`` fragment in JavaScript (Sphinx's own GitHub anchor rewrite
#   is disabled upstream, sphinx-doc/sphinx#9435).
linkcheck_anchors_ignore_for_url = [
    r'^https://(docs\.)?pytorch\.org/docs/stable/',
    r'^https://github\.com/EinarOlafsson/spacr/?$',
]

# Napoleon (Google / NumPy → reST bridge) — spaCR uses reST field
# lists natively but napoleon stays on so any legacy Args/Returns
# still render cleanly.
napoleon_google_docstring = True
napoleon_numpy_docstring  = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False

# -- AutoAPI ---------------------------------------------------------------
autoapi_type              = 'python'
autoapi_dirs              = [os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'spacr')
)]
autoapi_root              = 'api'
autoapi_add_toctree_entry = True
autoapi_template_dir      = '_autoapi_templates'
autoapi_options           = [
    'members',
    'show-inheritance',
    'show-module-summary',
]
# Imported objects already have a canonical page in the module that defines
# them. Re-emitting them under every importing module creates duplicate or
# non-package HTML ids that cannot be matched safely to the external docstring
# catalogs; aliases based on suffixes would be ambiguous.
autoapi_ignore            = [
    '*/tests/*',
    '*/qt/tutorial/*',
    # Asset-generation utilities are build inputs, not importable spaCR API.
    # Without this exclusion AutoAPI publishes them as top-level ``render``,
    # ``parts`` and similar modules instead of canonical ``spacr.*`` symbols.
    '*/resources/*/_generators/*',
    # Generated localization payloads are static data, not Python API. Their
    # source pages add tens of megabytes and expose no callable interface.
    '*/qt/i18n_catalogs/*',
]
autoapi_python_class_content = 'both'   # class docstring + __init__ docstring
# Keep the generated .rst. Every docutils complaint is reported against
# `api/spacr/<module>/index.rst:<line>`, and deleting those files turns the
# build log into line numbers with no file behind them -- the generated
# page is the only place the offending docstring appears as the parser saw
# it. The tree is regenerated from scratch on every build the workflow
# runs, which clears it first, so nothing stale survives.
autoapi_keep_files           = True
autoapi_member_order         = 'groupwise'   # attrs → methods, alphabetical inside
# A changed rollout must invalidate AutoAPI's otherwise unchanged source cache.
spacr_nested_helper_modules = tuple(sorted(_nested_helper_docs.ENABLED_MODULES))
spacr_explicit_api_modules = tuple(sorted(_api_visibility.EXPLICIT_MODULES))
spacr_source_api_projection = _source_api_projection
spacr_source_api_projection_sha256 = hashlib.sha256(
    '\n'.join(sorted(_source_api_projection)).encode('utf-8'),
).hexdigest()


def autoapi_prepare_jinja_env(env, *, app=None):
    import build_module_workflows
    build_module_workflows.prepare_jinja(env, root=_SOURCE_ROOT)
    _nested_helper_docs.prepare_jinja(
        env, root=_SOURCE_ROOT, ignore_patterns=autoapi_ignore,
    )
    if app is not None and app.config.spacr_source_api_projection:
        import build_documentation_i18n
        _api_visibility.prepare_jinja(
            env, root=_SOURCE_ROOT,
            documents=build_documentation_i18n.public_docstrings())

# -- HTML output — furo ----------------------------------------------------
html_theme      = 'furo'
html_title      = f'spaCR {release} documentation'
# THE PANELLED LOGO, not the bare mark. `logo_spacr.png` is pure white on
# transparent, and Sphinx serves one html_logo for both colour schemes --
# so against the light theme's white sidebar the bare mark is a blank
# space. `logo_spacr_docs.png` carries its own dark teal panel and reads
# in either scheme. Generated by packaging/generate_readme_visuals.py.
html_logo       = '_static/logo_spacr_docs.png'
# The favicon stays the bare mark: it is drawn at 16px against the
# browser's own tab strip, which is dark in both themes on every platform
# that ships one, and a panel at that size is mostly panel.
html_favicon    = '_static/logo_spacr.png'
templates_path  = ['_templates']
html_static_path = ['_static']
html_css_files  = ['custom.css']

# A content-derived version keeps browsers and intermediary caches from mixing
# API catalogs from different documentation builds.  The frontend also checks
# every localized source hash against the English manifest before rendering.
_api_catalog_hasher = hashlib.sha256()
_api_catalog_dir = os.path.join(
    os.path.dirname(__file__), '_static', 'i18n', 'api')
for _api_catalog_name in sorted(os.listdir(_api_catalog_dir)):
    if not _api_catalog_name.endswith('.json'):
        continue
    _api_catalog_hasher.update(_api_catalog_name.encode('utf-8'))
    with open(os.path.join(_api_catalog_dir, _api_catalog_name), 'rb') as _stream:
        for _chunk in iter(lambda: _stream.read(1024 * 1024), b''):
            _api_catalog_hasher.update(_chunk)
_api_catalog_version = _api_catalog_hasher.hexdigest()[:16]
_api_publication_language = os.environ.get('SPACR_DOCS_API_LANGUAGE', 'all')
if _api_publication_language not in ('all', 'english'):
    raise ValueError('SPACR_DOCS_API_LANGUAGE must be all or english')
html_js_files = [
    ('api_i18n.js', {
        'data-api-catalog-version': _api_catalog_version,
        'data-api-language': _api_publication_language,
        # Languages with translated user guides, served under /<lang>/.
        'data-guide-languages': ' '.join(_guide_i18n.catalog_languages()),
    }),
]

# -- Tutorial media --------------------------------------------------------
# `_extra` is 2,879 MiB, 93% of it one narration .m4a per lesson x language x
# voice. Copying it whole would put the built site multiples over the GitHub
# Pages 1 GB limit. tools/docs_media_budget.py stages a hardlinked subset --
# every lesson, every video, every poster, every caption, and no narration at
# all. That subset is 185 MiB, so the site lands at ~237 MiB.
#
# The narration is not dropped, it moved: all 54 voices are served from
# NARRATION_HOST, which is what lets voice_catalog.js keep offering every one
# of them. While the site carried the audio it could only afford one voice per
# language, so 27 of the 28 English voices were unreachable.
# SPACR_DOCS_FULL_AUDIO=1 still publishes the lot, which is how to build a
# copy of the docs that works with no access to the host. There is
# deliberately no fallback to the unfiltered tree: a staging failure that
# quietly republished 2,879 MiB is the thing this replaces.
#
# The site plays 1440p video, with the 4K masters offered as a quality option
# from the same host and linked per lesson on YouTube via youtube_links.js.
import importlib.util as _importlib_util
import pathlib as _pathlib

_budget_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'tools', 'docs_media_budget.py'))
_budget_spec = _importlib_util.spec_from_file_location(
    'spacr_docs_media_budget', _budget_path)
_budget = _importlib_util.module_from_spec(_budget_spec)
_budget_spec.loader.exec_module(_budget)

if _guides_only:
    html_extra_path = []
else:
    _voices = _budget.per_language_setting()
    _staged_extra = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '_build', 'extra_staged'))
    _budget.stage(_pathlib.Path(_staged_extra), per_language=_voices)
    print(_budget.report(per_language=_voices))

    html_extra_path = [_staged_extra]

html_theme_options = {
    # Auto-switching light/dark, with a manual toggle in the top bar
    'light_css_variables': {
        'color-brand-primary':  '#4A9EFF',
        'color-brand-content':  '#4A9EFF',
        'color-admonition-background': '#eef4ff',
        'font-stack':           '"Open Sans", ui-sans-serif, system-ui, -apple-system, sans-serif',
        'font-stack--monospace': 'JetBrains Mono, Consolas, monospace',
    },
    'dark_css_variables': {
        'color-brand-primary':  '#82b8ff',
        'color-brand-content':  '#82b8ff',
        'color-background-primary':   '#0d0d0d',
        'color-background-secondary': '#141414',
        'color-background-hover':     '#1c1c1c',
        'color-background-border':    '#262626',
        'color-foreground-primary':   '#e5e5e5',
        'color-foreground-secondary': '#c4c4c4',
        'color-admonition-background': '#141a24',
        'font-stack':           '"Open Sans", ui-sans-serif, system-ui, -apple-system, sans-serif',
    },
    'sidebar_hide_name': False,
    'navigation_with_keys': True,
    'source_repository':  'https://github.com/EinarOlafsson/spacr/',
    'source_branch':      _source_branch,
    'source_directory':   'docs/source/',
    'footer_icons': [
        {
            'name': 'Website',
            'url':  'https://einarolafsson.github.io/projects/spacr/',
            'html': '<svg stroke="currentColor" fill="none" stroke-width="1.6" viewBox="0 0 24 24" height="1.4em" width="1.4em" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3z"/></svg>',
            'class': '',
        },
        {
            'name': 'GitHub',
            'url':  'https://github.com/EinarOlafsson/spacr',
            'html': '',
            'class': 'fa-brands fa-solid fa-github fa-2x',
        },
    ],
}


def _skip_implementation_data(app, what, name, obj, skip, options):
    """Hide mutable module state while retaining documented constants."""
    source_policy = _api_visibility.source_page_policy(
        name, obj, getattr(app.config, 'spacr_source_api_projection', frozenset()))
    if source_policy is not None:
        return source_policy
    explicit_policy = _api_visibility.explicit_page_policy(name)
    if explicit_policy is not None:
        return explicit_policy
    helper_policy = _nested_helper_docs.helper_page_policy(what, name, obj, skip, options)
    if helper_policy is not None:
        return helper_policy
    if what in {'data', 'attribute'}:
        short_name = str(name).rsplit('.', 1)[-1]
        if not short_name.isupper():
            return True
    return None


# Nested-helper signatures keep their source annotations, which name classes
# imported into the helper's module. Sphinx cannot see those imports, so a
# short name defined in several modules is ambiguous. Point each such
# annotation at the class the module actually imports.
_HELPER_ANNOTATION_TARGETS = {
    ('spacr.flowview.collector', 'Node'): 'spacr.flowview.model.Node',
}


def _qualify_helper_annotations(app, doctree):
    from sphinx.addnodes import pending_xref
    for node in doctree.findall(pending_xref):
        if node.get('refdomain') != 'py':
            continue
        key = (node.get('py:module'), node.get('reftarget'))
        target = _HELPER_ANNOTATION_TARGETS.get(key)
        if target:
            node['reftarget'] = target
            node.attributes.pop('refspecific', None)


def setup(app):
    app.connect('doctree-read', _qualify_helper_annotations)
    app.add_config_value('spacr_nested_helper_modules', (), 'env')
    app.add_config_value('spacr_explicit_api_modules', (), 'env')
    app.add_config_value('spacr_source_api_projection', frozenset(), 'env')
    app.add_config_value('spacr_source_api_projection_sha256', '', 'env')
    if app.config.spacr_source_api_projection:
        from functools import partial
        app.config.autoapi_prepare_jinja_env = partial(
            autoapi_prepare_jinja_env, app=app)
    _nested_helper_docs.register_sphinx_directive(app)
    app.connect('autoapi-skip-member', _skip_implementation_data)
