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
* setting names, type hints, generic explanations, API-link captions and
  accessible help text.

Changing the language while a chat is connecting or its Send button is in
Cancel mode preserves that state and any provider or path values. Only the
surrounding application wording changes.

Contextual help
---------------

Module tooltips contain complete, reviewed translations in every bundled
language. Setting tooltips are assembled from separate semantic fields so the
setting name, type and API caption can be translated without rewriting a
scientific explanation word by word. A scientific tooltip body is translated
only when a complete reviewed translation exists; otherwise the canonical
English paragraph is shown. This prevents technically misleading
mixed-language help.

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

Catalogs live in :mod:`spacr.qt.i18n`. English strings are stable source keys;
each translation row contains Swedish, German, Spanish, Simplified Chinese,
Portuguese, Hindi, Korean, Icelandic and French in that order. Tests require
every core phrase, registered module name and registry section to be present
in all bundled catalogs. New interface text always has a safe English
fallback while a translation is being reviewed.

The longer built-in module descriptions live in the
``spacr.qt.i18n_module_summaries_*`` catalogs so fluent reviewers can inspect
complete scientific sentences independently of short button labels. Plugins
can provide exact translations through their translation metadata.

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
* :mod:`spacr.qt.i18n_module_summaries` — reviewed built-in module help.
* :mod:`spacr.qt.screens.settings_model` — semantic setting tooltips and API links.
* :mod:`spacr.setting_animations` — exact setting-to-animation registry.
* :class:`spacr.qt.widgets.console_panel.ConsolePanel` — separate raw-output
  and localized-notice paths.
