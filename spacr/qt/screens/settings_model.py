"""
Bridge between spacr's plain-python default settings and Qt form widgets.

The existing spacr GUI expresses settings as `{name: (widget_type, options,
default)}` triples via `spacr.gui_utils.convert_settings_dict_for_gui`.
Here we consume the same conversion output and materialize each entry as
a real Qt widget grouped into logical Section boxes based on
`spacr.settings.categories`.
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QLabel,
)


# ---------------------------------------------------------------------------
# Settings resolvers per app_key
# ---------------------------------------------------------------------------

def resolve_default_settings(app_key: str) -> Dict[str, Any]:
    """Return a fresh defaults dict for an app key, mirroring the Tk GUI
    dispatch in gui_core.setup_settings_panel."""
    from spacr.settings import (
        get_identify_masks_finetune_default_settings,
        set_default_analyze_screen,
        set_default_settings_preprocess_generate_masks,
        get_automated_motility_assay_default_settings,
        get_measure_crop_settings,
        deep_spacr_defaults,
        set_default_generate_barecode_mapping,
        set_default_umap_image_settings,
        get_map_barcodes_default_settings,
        get_analyze_recruitment_default_settings,
        get_check_cellpose_models_default_settings,
        get_analyze_plaque_settings,
        get_perform_regression_default_settings,
        get_train_cellpose_default_settings,
        get_default_generate_activation_map_settings,
    )
    if app_key == "mask":
        s = set_default_settings_preprocess_generate_masks(settings={})
        s = get_automated_motility_assay_default_settings(s)
        return s
    if app_key == "measure":
        return get_measure_crop_settings(settings={})
    if app_key == "classify":
        return deep_spacr_defaults(settings={})
    if app_key == "umap":
        return set_default_umap_image_settings(settings={})
    if app_key == "train_cellpose":
        return get_train_cellpose_default_settings(settings={})
    if app_key == "ml_analyze":
        return set_default_analyze_screen(settings={})
    if app_key == "cellpose_masks":
        return get_identify_masks_finetune_default_settings(settings={})
    if app_key == "cellpose_all":
        return get_check_cellpose_models_default_settings(settings={})
    if app_key == "map_barcodes":
        return set_default_generate_barecode_mapping(settings={})
    if app_key == "regression":
        return get_perform_regression_default_settings(settings={})
    if app_key == "recruitment":
        return get_analyze_recruitment_default_settings(settings={})
    if app_key == "activation":
        return get_default_generate_activation_map_settings(settings={})
    if app_key == "analyze_plaques":
        return get_analyze_plaque_settings(settings={})
    if app_key in ("annotate", "make_masks"):
        # These are interactive apps; return minimal placeholder.
        return {"src": "path to images"}
    return {"src": "path"}


def get_categories() -> Dict[str, List[str]]:
    """Return the {category_name: [setting keys]} mapping."""
    from spacr.settings import categories
    return categories


def get_tooltips() -> Dict[str, str]:
    """Return per-key tooltip text (spacr.settings.descriptions and .tooltips)."""
    tips: Dict[str, str] = {}
    try:
        from spacr.settings import descriptions, tooltips
    except Exception:
        return tips
    tips.update({k: v for k, v in descriptions.items() if isinstance(v, str)})
    tips.update({k: v for k, v in tooltips.items() if isinstance(v, str)})
    return tips


# ---------------------------------------------------------------------------
# API doc link per app
# ---------------------------------------------------------------------------

DOCS_BASE = "https://einarolafsson.github.io/spacr/index.html"


def api_docs_url(app_key: str) -> str:
    """Return the spacr documentation URL for a given app.

    The published docs don't yet split into per-function anchors, so
    we point every setting at the docs landing page. Users can search
    from there and don't hit 404s.
    """
    return DOCS_BASE


def format_tooltip(text: str, app_key: str) -> str:
    """Return an HTML tooltip body: description + a docs footer link.

    Kept minimal — Qt auto-detects rich text from the presence of any
    HTML tag, but complex containers/styles trip up some Qt builds and
    the tip renders empty. Plain <br> line breaks and a plain <a>
    footer render reliably on every platform we've tested.
    """
    body = (text or "").strip()
    url = api_docs_url(app_key)
    if body:
        return f'{body}<br><br><a href="{url}">API docs</a>'
    return f'See <a href="{url}">API docs</a>.'


def plain_tooltip(text: str, app_key: str) -> str:
    """Same content as `format_tooltip` but plain text — used by the
    hover-follows footer at the bottom of each AppScreen."""
    body = (text or "").strip()
    url = api_docs_url(app_key)
    if body:
        return f"{body}   ({url})"
    return f"See {url}"


# ---------------------------------------------------------------------------
# Widget factory
# ---------------------------------------------------------------------------

class _ListEdit(QLineEdit):
    """A QLineEdit that round-trips a Python list via repr()."""
    def get_value(self):
        text = self.text().strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except Exception:
            return text

    def set_value(self, v):
        self.setText(repr(v) if v is not None else "")


class _ScalarEdit(QLineEdit):
    def get_value(self):
        return self.text() or None

    def set_value(self, v):
        self.setText("" if v is None else str(v))


class SettingsWidgets:
    """Container for the Qt widgets bound to a settings dict.

    Instantiate with an `app_key`; call `.build_sections()` to get a list
    of (section_title, list_of_(label, widget)) tuples to feed into the
    Section widgets on a screen. `.collect()` returns the current settings
    dict after user edits."""

    def __init__(self, app_key: str, parent: Optional[QWidget] = None):
        self.app_key = app_key
        self._parent = parent
        self._defaults = resolve_default_settings(app_key)
        self._widgets: Dict[str, QWidget] = {}
        self._tooltips = get_tooltips()

    def build_sections(self) -> List[Tuple[str, List[Tuple[str, QWidget]]]]:
        """Group the settings by category and return one (title, rows)
        tuple per non-empty category, plus a trailing 'Other' section
        for anything not categorized."""
        from spacr.gui_utils import convert_settings_dict_for_gui
        variables = convert_settings_dict_for_gui(self._defaults)

        # Materialize a widget per key; attach a rich HTML tooltip that
        # ends with an "API docs →" link to the spacr docs.
        for key, meta in variables.items():
            kind, options, default = meta
            widget = self._widget_for(kind, options, default, key)
            if widget is not None:
                tip = format_tooltip(self._tooltips.get(key, ""), self.app_key)
                widget.setToolTip(tip)
                widget.setToolTipDuration(-1)  # respect system default (persistent)
                self._widgets[key] = widget

        # Bucket into sections.
        cats = get_categories()
        used_keys = set()
        sections: List[Tuple[str, List[Tuple[str, QWidget]]]] = []
        for cat_name, keys in cats.items():
            rows: List[Tuple[str, QWidget]] = []
            for k in keys:
                if k in self._widgets and k not in used_keys:
                    rows.append((self._label_for(k), self._widgets[k]))
                    used_keys.add(k)
            if rows:
                sections.append((cat_name, rows))

        # Trailing 'Other' for anything not in a category.
        remaining = [(self._label_for(k), self._widgets[k])
                     for k in self._widgets if k not in used_keys]
        if remaining:
            sections.append(("Other", remaining))

        return sections

    def tooltip_for(self, key: str) -> str:
        """Return the HTML-formatted tooltip for a given setting key."""
        return format_tooltip(self._tooltips.get(key, ""), self.app_key)

    def plain_tooltip_for(self, key: str) -> str:
        return plain_tooltip(self._tooltips.get(key, ""), self.app_key)

    def _label_for(self, key: str) -> str:
        return key.replace("_", " ").capitalize()

    def _widget_for(self, kind: str, options: Any, default: Any,
                    key: str) -> Optional[QWidget]:
        parent = self._parent
        if kind == "check":
            w = QCheckBox()
            w.setChecked(bool(default))
            return w
        if kind == "combo":
            w = QComboBox()
            for opt in (options or []):
                w.addItem("None" if opt is None else str(opt),
                          userData=opt)
            # Try to pre-select default
            for i in range(w.count()):
                if w.itemData(i) == default or w.itemText(i) == str(default):
                    w.setCurrentIndex(i)
                    break
            return w
        if kind == "entry":
            # Choose widget by inferred type from the DEFAULT value
            if isinstance(default, bool):
                w = QCheckBox()
                w.setChecked(default)
                return w
            if isinstance(default, int):
                w = QSpinBox()
                w.setRange(-1_000_000, 1_000_000)
                w.setValue(default)
                return w
            if isinstance(default, float):
                w = QDoubleSpinBox()
                w.setRange(-1e12, 1e12)
                w.setDecimals(6)
                w.setValue(default)
                return w
            if isinstance(default, list):
                w = _ListEdit()
                w.set_value(default)
                return w
            # Fallback — string or None
            w = _ScalarEdit()
            w.set_value(default)
            return w
        return None

    def collect(self) -> Dict[str, Any]:
        """Read all widgets and return the current settings dict."""
        out: Dict[str, Any] = {}
        for key, w in self._widgets.items():
            out[key] = self._read_widget(w)
        # Also carry over any defaults we didn't render (e.g. things not
        # in the categories map that convert_settings_dict_for_gui also
        # skipped).
        for k, v in self._defaults.items():
            out.setdefault(k, v)
        return out

    def _read_widget(self, w: QWidget) -> Any:
        if isinstance(w, QCheckBox):
            return bool(w.isChecked())
        if isinstance(w, QSpinBox):
            return int(w.value())
        if isinstance(w, QDoubleSpinBox):
            return float(w.value())
        if isinstance(w, QComboBox):
            return w.currentData() if w.currentData() is not None else w.currentText()
        if isinstance(w, _ListEdit):
            return w.get_value()
        if isinstance(w, _ScalarEdit):
            return w.get_value()
        if isinstance(w, QLineEdit):
            return w.text() or None
        return None
