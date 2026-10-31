Language & translation
======================

The spaCR Qt interface includes ten built-in languages:

* English
* Swedish (Svenska)
* German (Deutsch)
* Spanish (Español)
* Simplified Chinese / Mandarin (简体中文)
* Portuguese (Português)
* Hindi (हिन्दी)
* Korean (한국어)
* Icelandic (Íslenska)
* French (Français)

Open **spaCR → Preferences…**, choose a **Language** on the General tab and
press Save.
Existing windows and lazily opened module screens are translated immediately.
The selection is retained by ``QSettings`` for later launches.

What is translated
------------------

Language support covers the presentation layer of the Qt application:

* navigation, Preferences, common actions, tabs and section headings;
* the **AI** and **Live** controls, AI-provider setup, chat placeholders,
  streaming status and other chat chrome;
* spaCR-authored console notices such as run start, safe stop, completion,
  settings import and provider guidance;
* one-line descriptions for the built-in modules; and
* setting names, type hints, authored explanations, API-link captions,
  tooltips and accessible help text.

Changing the language while a chat is connecting or its Send button is in
Cancel mode preserves that state and any provider or path values. Only the
surrounding application wording changes.

Contextual help
---------------

Module and setting tooltips have complete external catalog entries in every
bundled language. Setting tooltips are assembled from separate semantic fields
so the setting name, type, authored explanation and API caption can be checked
independently. Format fields, code literals, scientific symbols, URLs and
option values are protected during catalog generation. A translation is
accepted only when those structural values survive; otherwise spaCR uses the
canonical English text.

The **API** link at the foot of each setting's tooltip keeps the exact
documentation URL in every language. Its caption and accessible name follow
the selected language, and changing language refreshes already-open settings
windows. Where visual help exists, the tooltip footer also offers an
**Animation** word that reveals the drawing beside the text. The scientific drawing itself is
language-neutral. See the
:doc:`setting animation gallery <setting_animations>` for every exact mapping.

Translation safety
------------------

Translation is presentation-only. Raw worker stdout, logs, tracebacks,
filenames, paths, setting values, database contents, annotations,
measurements, reports and saved results are never modified. User chat messages
and AI responses also remain exactly as written or returned.

The console distinguishes spaCR-authored interface notices from analytical
output. Notices may be presented in the selected language, while pipeline
lines and errors pass through unchanged. Template values such as a path,
provider name, function name or error detail are preserved even when the
surrounding notice is translated.

When a scientific or third-party term has no catalog entry, spaCR displays the
original English text rather than a substituted translation.

The environment variable ``SPACR_LANGUAGE`` can temporarily override the
saved preference, which is useful for screenshots and automated testing:

.. code-block:: bash

   SPACR_LANGUAGE=sv spacr
   SPACR_LANGUAGE=zh_CN spacr

Contributing translations
-------------------------

Compact reviewed chrome remains in :mod:`spacr.qt.i18n`. The full runtime
catalogs live in ``spacr/qt/i18n_catalogs``. Their setting and interface
records are bound to canonical English text by source hashes; setting records
use setting identifiers rather than English sentences as keys. Each language
file contains the exact current key set for
setting names, tooltips, category help, module summaries and extracted Qt
text. Tests require every locale to be non-empty and structurally complete.
New interface text always has a safe English fallback while a translation is
being reviewed.

The generator in ``tools/build_i18n_catalogs.py`` protects runtime fields and
applies the reviewed terminology table after translation. False friends such
as *screen*, *run*, *crop*, *mask*, *flow*, *plate* and *gate* are reviewed in
their software and microscopy context rather than as isolated words. Plugins
can provide exact translations through their translation metadata.

Review inputs and translated guides
-----------------------------------

Runtime and API review records are restored from the immutable revision in
``docs/translation_reviews_revision.json``. Run
``python tools/bootstrap_translation_reviews.py fetch`` to restore that
revision; ``python tools/bootstrap_translation_reviews.py check`` verifies
it using the existing cache without fetching. The bootstrap refuses to
overwrite locally modified or unpinned review records. Review metadata
identifies the recorded reviewer; AI technical review does not establish
native-speaker signoff.

Translated user guides use Sphinx gettext catalogs under
``docs/i18n/guides``. The guide-page language picker opens the corresponding
language tree; API translations instead load into the current page. Guide
paragraphs without a current, non-fuzzy translation remain in English.
``tools/build_guide_i18n.py`` extracts English templates, updates catalogs,
exports and imports translation worklists, and audits translations and review
labels. These guide catalogs are separate from the runtime and API catalogs.

API documentation and the project page
--------------------------------------

English docstrings remain beside their Python functions. Translated API text
is stored separately under ``docs/source/_static/i18n/api`` and keyed by the
fully qualified Python symbol plus a SHA-256 hash of the English source. The
documentation language picker loads these files on demand and checks them
against the English catalog. A failed load or freshness check restores the
English view. API language selection uses the page's ``lang`` query, a saved
browser choice, then a supported browser language; it is separate from the
Qt application's saved language. A changed English docstring also makes the
corresponding translation fail the freshness audit.

For a local API documentation build,
``python tools/build_documentation_i18n.py --assemble`` restores the separately
pinned ``api_memory`` catalogs from ``docs/media_revision.json``, applies the
source-bound review records and writes the English manifest. This assembles
existing translation memory without running a translation model.

Translated GitHub project pages live under ``docs/i18n/readme``. Their code,
commands, URLs, badges and language navigation are preserved exactly, while
the explanatory prose is localized. Model and license attribution is recorded
in ``docs/i18n/TRANSLATION_MODELS.md``.

Installers
----------

The Windows, macOS and Linux online installers share the ten locale resources
in ``packaging/i18n``. They select the operating system's UI language and fall
back to English; ``SPACR_INSTALL_LANGUAGE`` provides an explicit override.
Shell, PowerShell and NSIS resources are generated by
``packaging/i18n/render.py`` so translated messages never become duplicated
inside platform control flow.

Developers should send raw pipeline output through ``append_stdout`` or
``append_error`` and use ``append_notice`` only for a stable, spaCR-authored UI
template. Dynamic application chrome uses a stable source template so runtime
language changes cannot translate an earlier translation or overwrite a live
path/result label. Catalog tests enforce format-field
parity, and language tests assert that output and chat content remain
unchanged.

Related API
-----------

* :mod:`spacr.qt.i18n` — language catalogs and runtime widget retranslation.
* :mod:`spacr.qt.preferences` — persisted language selection and Preferences.
* :mod:`spacr.qt.i18n_module_summaries` — built-in module help lookup.
* :mod:`spacr.qt.screens.settings_model` — semantic setting tooltips and API links.
* :mod:`spacr.setting_animations` — exact setting-to-animation registry.
* :class:`spacr.qt.widgets.console_panel.ConsolePanel` — separate raw-output
  and localized-notice paths.
