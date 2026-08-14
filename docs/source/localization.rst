Localization
============

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

Select a language under **spaCR → Preferences → Language** and press Save.
Existing windows and lazily opened module screens are translated immediately.
The selection is retained by ``QSettings`` for later launches.

What is localized
-----------------

Localization covers the presentation layer of the Qt application:

* navigation, Preferences, common actions, tabs and section headings;
* the **AI** and **LIVE** controls, AI-provider setup, chat placeholders,
  streaming status and other chat chrome;
* spaCR-authored console notices such as run start, safe stop, completion,
  settings import and provider guidance;
* reviewed one-line descriptions for all built-in modules; and
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
option values are immutable during catalog generation. A translation is
accepted only when those structural values survive; otherwise spaCR uses the
canonical English text.

The teal API dots beside settings retain the exact documentation URL in every
language. Their hover captions and accessible names follow the selected
language, and changing language refreshes already-open settings windows.
Where visual help exists, the tooltip footer offers an **Animation** word that
reveals the drawing beside the text. The scientific drawing itself is
language-neutral. See the
:doc:`setting animation gallery <setting_animations>` for every exact mapping.

Translation safety
------------------

Localization is presentation-only. Raw worker stdout, logs, tracebacks,
filenames, paths, setting values, database contents, annotations,
measurements, reports and saved results are never modified. User chat messages
and AI responses also remain exactly as written or returned.

The console distinguishes spaCR-authored interface notices from analytical
output. Notices may be presented in the selected language, while pipeline
lines and errors pass through unchanged. Template values such as a path,
provider name, function name or error detail are preserved even when the
surrounding notice is translated.

When a scientific or third-party term has no catalog entry, spaCR displays the
original English text. This explicit fallback is preferable to guessing the
meaning of a technical control.

The environment variable ``SPACR_LANGUAGE`` can temporarily override the
saved preference, which is useful for screenshots and automated testing:

.. code-block:: bash

   SPACR_LANGUAGE=sv spacr
   SPACR_LANGUAGE=zh_CN spacr

Contributing translations
-------------------------

Compact, manually reviewed chrome remains in :mod:`spacr.qt.i18n`. The full
runtime catalogs live in ``spacr/qt/i18n_catalogs`` with English strings as
stable source keys. Each language file contains the exact current key set for
setting names, tooltips, category help, module summaries and extracted Qt
text. Tests require every locale to be non-empty and structurally complete.
New interface text always has a safe English fallback while a translation is
being reviewed.

The generator in ``tools/build_i18n_catalogs.py`` protects runtime fields and
applies the reviewed terminology table after translation. False friends such
as *screen*, *run*, *crop*, *mask*, *flow*, *plate* and *gate* are reviewed in
their software and microscopy context rather than as isolated words. Plugins
can provide exact translations through their translation metadata.

API documentation and the project page
--------------------------------------

English docstrings remain beside their Python functions. Translated API text
is stored separately under ``docs/source/_static/i18n/api`` and keyed by the
fully qualified Python symbol plus a SHA-256 hash of the English source. The
documentation language picker loads these files on demand; a changed English
docstring makes the corresponding translation fail the freshness audit
instead of silently displaying obsolete text.

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
path/result label. Catalog tests enforce translation width and format-field
parity, and localization tests assert that output and chat content remain
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
