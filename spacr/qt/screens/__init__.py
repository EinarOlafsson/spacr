"""Screen widgets — one per spacr app.

NOTHING IS IMPORTED HERE. This package used to import nine screens at the top,
because a screen that owned its registry row had to be executed for the row to
exist, and this package is on the path of every screen the window builds — so
importing one screen imported nine, and with them pandas and everything under
it, before the window had drawn.

A row does not need the screen any more. :mod:`spacr.qt.app_catalog` declares
the key, the name, the sentence, the section, the stage and the NAME of the
screen factory, and :func:`spacr.qt.app_catalog.register_declared` registers
all of that without importing anything; the screen is imported the first time
somebody opens it. A screen that contributes a stylesheet block registers it
at that import, and :func:`spacr.qt.theme.register_widget_qss` puts it into the
live application sheet synchronously, before its first widget is constructed.
:data:`spacr.qt.theme.WIDGET_QSS_MODULES` remains the inventory used when an
exhaustive/static stylesheet is explicitly requested.

So: to add a screen, declare its row in ``app_catalog`` and — if it registers
a QSS block — name it in ``WIDGET_QSS_MODULES`` so exhaustive sheets remain
complete. Do not import it here merely for startup styling.
"""
