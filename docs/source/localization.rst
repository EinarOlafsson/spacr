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

Translation safety
------------------

Localization only changes static interface text. Paths, filenames, setting
values, database contents, annotations and console output are never modified.
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
