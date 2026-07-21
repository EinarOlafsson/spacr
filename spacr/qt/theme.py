"""
Dark palette + QSS stylesheet for the spacr Qt GUI.

Single source of truth for every color, radius, and font size used by the
custom widgets and screens. Import `PALETTE` for programmatic access and
`stylesheet()` for the Qt StyleSheet string to hand to
`QApplication.setStyleSheet`.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Palette — kept aligned with the Tk gui_elements set_dark_style dict so
# switching between the two GUIs feels visually consistent.
# ---------------------------------------------------------------------------
PALETTE = {
    # Surfaces — pure black bg, subtle depth via layered near-blacks
    "bg":          "#000000",   # main window background
    "surface":     "#0d0e10",   # sidebar / panels — barely lifted from bg
    "surface_alt": "#161719",   # cards / grouped sections — one more step
    "surface_hi":  "#1f2124",   # hovered surfaces
    "border":      "#2a2d33",   # visible dividers
    "border_soft": "#1c1e22",   # hairline card borders
    # Text
    "fg":          "#ffffff",
    "fg_muted":    "#a1a6ad",   # secondary text (was #9ba0a6)
    "fg_dim":      "#6b6f76",   # disabled / hints
    # Accent
    "accent":      "#4A9EFF",   # primary interactive
    "accent_hi":   "#66B2FF",   # hover
    "accent_lo":   "#2F80D9",   # pressed
    "accent_soft": "#1e3550",   # accent-tinted surface (chips, highlights)
    # Status
    "success":     "#3fb950",
    "warning":     "#d29922",
    "error":       "#f85149",
    "info":        "#4A9EFF",
}


# ---------------------------------------------------------------------------
# Spacing / radius scale — 4/8-based, matches Tk gui_elements.
# ---------------------------------------------------------------------------
SPACING = {
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
    "xl": 24,
    "xxl": 32,
}

RADIUS = {
    "sm": 4,
    "md": 8,
    "lg": 12,
    "pill": 999,
}

FONT_SIZE = {
    "xs":      11,   # inline metadata, table cell suffixes
    "small":   12,   # captions, muted secondary text, form hints
    "body":    13,   # default body text
    "label":   13,   # form field labels
    "header":  15,   # card / section titles
    "subtitle":17,   # dialog headings, secondary display
    "title":   22,   # screen-level headings
    "display": 30,   # startup screen brand title
    "hero":    42,   # empty-state hero numerals
}

# Typography roles — pair size with weight + tracking + line-height
TYPOGRAPHY = {
    "display":   {"size": FONT_SIZE["display"],  "weight": 300, "tracking": "-0.4px", "line_height": "1.15"},
    "title":     {"size": FONT_SIZE["title"],    "weight": 500, "tracking": "-0.2px", "line_height": "1.2"},
    "subtitle":  {"size": FONT_SIZE["subtitle"], "weight": 500, "tracking": "-0.1px", "line_height": "1.25"},
    "header":    {"size": FONT_SIZE["header"],   "weight": 600, "tracking": "0px",    "line_height": "1.3"},
    "body":      {"size": FONT_SIZE["body"],     "weight": 400, "tracking": "0px",    "line_height": "1.45"},
    "small":     {"size": FONT_SIZE["small"],    "weight": 400, "tracking": "0px",    "line_height": "1.4"},
    "caption":   {"size": FONT_SIZE["xs"],       "weight": 500, "tracking": "0.6px",  "line_height": "1.4"},
    "hero":      {"size": FONT_SIZE["hero"],     "weight": 200, "tracking": "-0.5px", "line_height": "1.1"},
}


def apply_qpalette(app: QApplication) -> None:
    """Apply the palette to the QApplication so native controls (menu
    bars, tooltips, dialogs) match the QSS-styled widgets."""
    p = app.palette()
    p.setColor(QPalette.Window,          QColor(PALETTE["bg"]))
    p.setColor(QPalette.WindowText,      QColor(PALETTE["fg"]))
    p.setColor(QPalette.Base,            QColor(PALETTE["surface"]))
    p.setColor(QPalette.AlternateBase,   QColor(PALETTE["surface_alt"]))
    p.setColor(QPalette.ToolTipBase,     QColor(PALETTE["surface_alt"]))
    p.setColor(QPalette.ToolTipText,     QColor(PALETTE["fg"]))
    p.setColor(QPalette.Text,            QColor(PALETTE["fg"]))
    p.setColor(QPalette.Button,          QColor(PALETTE["surface"]))
    p.setColor(QPalette.ButtonText,      QColor(PALETTE["fg"]))
    p.setColor(QPalette.BrightText,      QColor(PALETTE["error"]))
    p.setColor(QPalette.Highlight,       QColor(PALETTE["accent"]))
    p.setColor(QPalette.HighlightedText, QColor(PALETTE["bg"]))
    p.setColor(QPalette.Link,            QColor(PALETTE["accent"]))
    p.setColor(QPalette.LinkVisited,     QColor(PALETTE["accent_lo"]))
    p.setColor(QPalette.PlaceholderText, QColor(PALETTE["fg_dim"]))
    p.setColor(QPalette.Mid,             QColor(PALETTE["border"]))
    p.setColor(QPalette.Midlight,        QColor(PALETTE["border_soft"]))
    p.setColor(QPalette.Dark,            QColor(PALETTE["surface_alt"]))
    p.setColor(QPalette.Shadow,          QColor("#000000"))
    app.setPalette(p)


def stylesheet() -> str:
    """Return the QSS string that styles every custom widget in the app."""
    P = PALETTE
    S = SPACING
    R = RADIUS
    F = FONT_SIZE
    return f"""
/* -----------------------------------------------------------------
 *  Base
 * ----------------------------------------------------------------- */
QWidget {{
    background-color: {P["bg"]};
    color: {P["fg"]};
    font-family: "Open Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: {F["body"]}px;
    outline: none;
}}
QMainWindow, QDialog {{
    background-color: {P["bg"]};
}}

/* -----------------------------------------------------------------
 *  Menu bar + menus
 * ----------------------------------------------------------------- */
QMenuBar {{
    background-color: {P["bg"]};
    color: {P["fg_muted"]};
    padding: {S["xs"]}px {S["sm"]}px;
    border-bottom: 1px solid {P["border_soft"]};
    font-size: {F["small"]}px;
}}
QMenuBar::item {{
    background: transparent;
    padding: {S["xs"]}px {S["sm"]}px;
    border-radius: {R["sm"]}px;
}}
QMenuBar::item:selected {{
    background: {P["surface"]};
    color: {P["fg"]};
}}
QMenu {{
    background-color: {P["surface_alt"]};
    color: {P["fg"]};
    border: 1px solid {P["border"]};
    border-radius: {R["md"]}px;
    padding: {S["xs"]}px;
}}
QMenu::item {{
    padding: {S["xs"]}px {S["md"]}px;
    border-radius: {R["sm"]}px;
    background: transparent;
}}
QMenu::item:selected {{
    background: {P["accent"]};
    color: {P["bg"]};
}}
QMenu::separator {{
    height: 1px;
    background: {P["border"]};
    margin: {S["xs"]}px {S["sm"]}px;
}}

/* -----------------------------------------------------------------
 *  Sidebar (main window navigation)
 * ----------------------------------------------------------------- */
#Sidebar {{
    background-color: {P["surface"]};
    border-right: 1px solid {P["border_soft"]};
}}
#SidebarTitle {{
    color: {P["accent"]};
    font-size: {F["title"]}px;
    font-weight: 600;
    padding: {S["md"]}px {S["md"]}px {S["sm"]}px;
    background: {P["surface"]};
}}
#SidebarSection {{
    color: {P["fg_dim"]};
    font-size: {F["xs"]}px;
    font-weight: 600;
    padding: {S["md"]}px {S["md"]}px {S["xs"]}px;
    text-transform: uppercase;
    letter-spacing: 1px;
    background: {P["surface"]};
}}
QPushButton#SidebarItem {{
    text-align: left;
    background: transparent;
    color: {P["fg_muted"]};
    padding: {S["sm"]}px {S["md"]}px;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0px;
    font-size: {F["body"]}px;
}}
QPushButton#SidebarItem:hover {{
    background: {P["surface_alt"]};
    color: {P["fg"]};
}}
QPushButton#SidebarItem:checked, QPushButton#SidebarItem[selected="true"] {{
    background: {P["surface_alt"]};
    color: {P["accent"]};
    border-left: 3px solid {P["accent"]};
}}

/* -----------------------------------------------------------------
 *  Cards / grouped sections
 * ----------------------------------------------------------------- */
QFrame#Card {{
    background-color: {P["surface"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["md"]}px;
}}
QFrame#Hero {{
    background-color: {P["surface"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["lg"]}px;
}}
QLabel#CardTitle {{
    color: {P["fg"]};
    font-size: {F["header"]}px;
    font-weight: 600;
    padding: 0px;
    background: transparent;
}}
QLabel#CardSubtitle {{
    color: {P["fg_muted"]};
    font-size: {F["small"]}px;
    background: transparent;
}}
QFrame#Divider {{
    background: {P["border"]};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}

/* -----------------------------------------------------------------
 *  Startup tiles
 * ----------------------------------------------------------------- */
QPushButton#Tile {{
    background-color: {P["surface"]};
    color: {P["fg"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["lg"]}px;
    padding: {S["md"]}px;
    font-size: {F["body"]}px;
    text-align: center;
    min-width: 96px;
    min-height: 96px;
}}
QPushButton#Tile:hover {{
    border: 1px solid {P["accent"]};
    background-color: {P["surface_alt"]};
    color: {P["accent"]};
}}
QPushButton#Tile:pressed {{
    background-color: {P["accent_lo"]};
    color: {P["bg"]};
}}
QLabel#TileCaption {{
    color: {P["fg"]};
    font-size: {F["body"]}px;
    font-weight: 500;
    background: transparent;
    padding-top: 4px;
}}

/* -----------------------------------------------------------------
 *  Typography helpers — pair each role with weight + tracking
 * ----------------------------------------------------------------- */
QLabel#Hero {{
    color: {P["fg"]};
    font-size: {F["hero"]}px;
    font-weight: 200;
    letter-spacing: -0.5px;
    background: transparent;
}}
QLabel#DisplayHeading {{
    color: {P["fg"]};
    font-size: {F["display"]}px;
    font-weight: 300;
    letter-spacing: -0.4px;
    background: transparent;
}}
QLabel#TitleHeading {{
    color: {P["fg"]};
    font-size: {F["title"]}px;
    font-weight: 500;
    letter-spacing: -0.2px;
    background: transparent;
}}
QLabel#Subtitle {{
    color: {P["fg_muted"]};
    font-size: {F["subtitle"]}px;
    font-weight: 400;
    background: transparent;
}}
QLabel#SubtitleSmall, QLabel#Muted {{
    color: {P["fg_muted"]};
    font-size: {F["small"]}px;
    background: transparent;
}}
QLabel#Caption {{
    color: {P["fg_dim"]};
    font-size: {F["xs"]}px;
    font-weight: 500;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    background: transparent;
}}
QLabel#SectionHeading {{
    color: {P["fg"]};
    font-size: {F["header"]}px;
    font-weight: 600;
    background: transparent;
}}

/* -----------------------------------------------------------------
 *  Buttons
 * ----------------------------------------------------------------- */
QPushButton {{
    background-color: {P["surface_alt"]};
    color: {P["fg"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["sm"]}px;
    padding: {S["sm"]}px {S["md"]}px;
    min-height: 22px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {P["surface_hi"]};
    border-color: {P["border"]};
    color: {P["fg"]};
}}
QPushButton:pressed {{
    background-color: {P["accent_lo"]};
    border-color: {P["accent_lo"]};
    color: {P["bg"]};
}}
QPushButton:checked {{
    background-color: {P["accent_soft"]};
    border-color: {P["accent"]};
    color: {P["accent"]};
}}
QPushButton:checked:hover {{
    background-color: {P["accent_soft"]};
    border-color: {P["accent_hi"]};
    color: {P["accent_hi"]};
}}
QPushButton:disabled {{
    color: {P["fg_dim"]};
    border-color: {P["border_soft"]};
    background-color: {P["surface"]};
}}
QPushButton#PrimaryButton {{
    background-color: {P["accent"]};
    color: {P["bg"]};
    border: none;
    font-weight: 600;
    padding: {S["sm"]}px {S["lg"]}px;
}}
QPushButton#PrimaryButton:hover {{
    background-color: {P["accent_hi"]};
}}
QPushButton#PrimaryButton:pressed {{
    background-color: {P["accent_lo"]};
}}
QPushButton#DangerButton {{
    background-color: transparent;
    color: {P["error"]};
    border: 1px solid {P["error"]};
    font-weight: 600;
    padding: {S["sm"]}px {S["lg"]}px;
}}
QPushButton#DangerButton:hover {{
    background-color: {P["error"]};
    color: {P["bg"]};
}}
QPushButton#GhostButton {{
    background-color: transparent;
    color: {P["fg_muted"]};
    border: none;
}}
QPushButton#GhostButton:hover {{
    color: {P["accent"]};
    background: transparent;
}}
QPushButton#IconButton {{
    background-color: transparent;
    border: none;
    padding: {S["xs"]}px;
    min-height: 0;
    color: {P["fg_muted"]};
}}
QPushButton#IconButton:hover {{
    color: {P["accent"]};
    background: {P["surface_alt"]};
    border-radius: {R["sm"]}px;
}}

/* -----------------------------------------------------------------
 *  Inputs (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox)
 * ----------------------------------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background-color: {P["surface_alt"]};
    color: {P["fg"]};
    border: 1px solid {P["border"]};
    border-radius: {R["sm"]}px;
    padding: {S["xs"]}px {S["sm"]}px;
    selection-background-color: {P["accent"]};
    selection-color: {P["bg"]};
}}
QPlainTextEdit#Console {{
    background-color: #0a0b0d;
    color: #d4d7dc;
    border: 1px solid {P["border_soft"]};
    font-family: "JetBrains Mono", "Menlo", "Consolas", monospace;
    font-size: {F["small"]}px;
    padding: {S["sm"]}px;
    selection-background-color: {P["accent_lo"]};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {P["accent"]};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {{
    color: {P["fg_dim"]};
    background-color: {P["surface"]};
    border-color: {P["border_soft"]};
}}
QLineEdit::placeholder {{
    color: {P["fg_dim"]};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: transparent;
    border: none;
    width: 16px;
}}
QSpinBox::up-arrow, QSpinBox::down-arrow,
QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {{
    width: 8px; height: 8px;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border: none;
}}
QComboBox::down-arrow {{
    image: none;
    border: 4px solid transparent;
    border-top-color: {P["fg_muted"]};
    margin-top: 4px;
    width: 0;
    height: 0;
}}
QComboBox QAbstractItemView {{
    background-color: {P["surface_alt"]};
    color: {P["fg"]};
    border: 1px solid {P["border"]};
    border-radius: {R["sm"]}px;
    padding: {S["xs"]}px;
    selection-background-color: {P["accent"]};
    selection-color: {P["bg"]};
}}

/* -----------------------------------------------------------------
 *  Checkboxes + toggles
 * ----------------------------------------------------------------- */
QCheckBox {{
    color: {P["fg"]};
    background: transparent;
    spacing: {S["sm"]}px;
    padding: 2px 0px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {P["border"]};
    border-radius: {R["sm"]}px;
    background: {P["surface_alt"]};
}}
QCheckBox::indicator:hover {{
    border-color: {P["accent"]};
}}
QCheckBox::indicator:checked {{
    background: {P["accent"]};
    border-color: {P["accent"]};
    image: none;
}}
QCheckBox::indicator:disabled {{
    background: {P["surface"]};
    border-color: {P["border_soft"]};
}}
QRadioButton {{
    color: {P["fg"]};
    background: transparent;
    spacing: {S["sm"]}px;
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {P["border"]};
    border-radius: 8px;
    background: {P["surface_alt"]};
}}
QRadioButton::indicator:checked {{
    background: {P["accent"]};
    border-color: {P["accent"]};
}}

/* -----------------------------------------------------------------
 *  Scrollbars
 * ----------------------------------------------------------------- */
QScrollBar:vertical {{
    background: {P["bg"]};
    width: 10px;
    margin: 0px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {P["border"]};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {P["accent"]};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background: transparent; height: 0px;
}}
QScrollBar:horizontal {{
    background: {P["bg"]};
    height: 10px;
    margin: 0px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {P["border"]};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {P["accent"]};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    background: transparent; width: 0px;
}}

/* -----------------------------------------------------------------
 *  Progress bar
 * ----------------------------------------------------------------- */
QProgressBar {{
    background: {P["surface"]};
    border: none;
    border-radius: 4px;
    text-align: center;
    color: {P["fg_muted"]};
    height: 8px;
    max-height: 8px;
}}
QProgressBar::chunk {{
    background-color: {P["accent"]};
    border-radius: 4px;
}}
QProgressBar#UsageBar {{
    background: {P["surface_alt"]};
    height: 6px;
    max-height: 6px;
}}
QProgressBar#UsageBar::chunk {{
    background: {P["accent"]};
    border-radius: 3px;
}}
QProgressBar#UsageBarWarn::chunk {{
    background: {P["warning"]};
}}
QProgressBar#UsageBarError::chunk {{
    background: {P["error"]};
}}

/* -----------------------------------------------------------------
 *  Splitter handle
 * ----------------------------------------------------------------- */
QSplitter::handle {{
    background: {P["border_soft"]};
}}
QSplitter::handle:horizontal {{
    width: 1px;
}}
QSplitter::handle:vertical {{
    height: 1px;
}}
QSplitter::handle:hover {{
    background: {P["accent"]};
}}

/* -----------------------------------------------------------------
 *  Tooltip
 * ----------------------------------------------------------------- */
QToolTip {{
    background-color: {P["surface_alt"]};
    color: {P["fg"]};
    border: 1px solid {P["border"]};
    border-radius: {R["sm"]}px;
    padding: {S["xs"]}px {S["sm"]}px;
    font-size: {F["small"]}px;
}}

/* -----------------------------------------------------------------
 *  AI Console chat bubbles (legacy standalone panel)
 * ----------------------------------------------------------------- */
QLabel#ChatBubbleUser {{
    background-color: {P["accent_soft"]};
    color: {P["fg"]};
    border: 1px solid {P["accent_lo"]};
    border-radius: {R["md"]}px;
    padding: {S["sm"]}px {S["md"]}px;
    font-size: {F["body"]}px;
}}
QLabel#ChatBubbleAssistant {{
    background-color: {P["surface_alt"]};
    color: {P["fg"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["md"]}px;
    padding: {S["sm"]}px {S["md"]}px;
    font-size: {F["body"]}px;
}}

/* -----------------------------------------------------------------
 *  Merged Console (pipeline stdout + AI chat)
 * ----------------------------------------------------------------- */
QWidget#ConsolePanel {{
    background-color: {P["surface_alt"]};
    border-radius: {R["md"]}px;
}}
QWidget#ConsoleHolder {{
    background-color: {P["surface_alt"]};
}}
QScrollArea#ConsoleScroll {{
    background-color: {P["surface_alt"]};
    border: none;
}}
QFrame#ConsoleTopicBar {{
    background-color: {P["surface_hi"]};
    border-top: 1px solid {P["border_soft"]};
    border-bottom: 1px solid {P["border_soft"]};
}}
QLabel#ConsoleTopicLabel {{
    color: {P["fg_muted"]};
    font-size: {F["small"]}px;
    font-weight: 600;
    letter-spacing: 0.4px;
    background: transparent;
}}
QLabel#ConsoleStdoutBlock {{
    color: {P["fg"]};
    background-color: {P["surface_alt"]};
    font-family: "JetBrains Mono", "Menlo", "Consolas", monospace;
    font-size: {F["small"]}px;
    padding: {S["sm"]}px {S["md"]}px;
}}
QLabel#ConsoleStdoutBlockError {{
    color: {P["error"]};
    background-color: {P["surface_alt"]};
    font-family: "JetBrains Mono", "Menlo", "Consolas", monospace;
    font-size: {F["small"]}px;
    padding: {S["sm"]}px {S["md"]}px;
}}
QFrame#ConsoleBubbleUser {{
    background-color: #163b28;                 /* dark green */
    border: 1px solid #2a6a48;
    border-radius: {R["md"]}px;
}}
QFrame#ConsoleBubbleAI {{
    background-color: {P["accent_soft"]};      /* dark blue */
    border: 1px solid {P["accent_lo"]};
    border-radius: {R["md"]}px;
}}
QFrame#ConsoleInputBar {{
    background-color: {P["surface"]};
    border-top: 1px solid {P["border_soft"]};
}}

/* -----------------------------------------------------------------
 *  Section — collapsible dropdown (custom widget)
 * ----------------------------------------------------------------- */
QFrame#SectionCard {{
    background-color: {P["surface"]};
    border: 1px solid {P["border_soft"]};
    border-radius: {R["md"]}px;
    margin-bottom: {S["sm"]}px;
}}
QToolButton#SectionHeader {{
    background: transparent;
    color: {P["fg_muted"]};
    border: none;
    border-radius: {R["md"]}px;
    padding: {S["sm"]}px {S["md"]}px;
    text-align: left;
    font-size: {F["small"]}px;
    font-weight: 600;
    letter-spacing: 0.6px;
}}
QToolButton#SectionHeader:hover {{
    color: {P["fg"]};
    background: {P["surface_alt"]};
}}
QToolButton#SectionHeader:checked {{
    color: {P["fg"]};
    background: {P["surface_alt"]};
    border-bottom: 1px solid {P["border_soft"]};
    border-bottom-left-radius: 0px;
    border-bottom-right-radius: 0px;
}}
QWidget#SectionBody {{
    background-color: transparent;
    border-bottom-left-radius: {R["md"]}px;
    border-bottom-right-radius: {R["md"]}px;
}}

/* -----------------------------------------------------------------
 *  Group box (used by settings sections)
 * ----------------------------------------------------------------- */
QGroupBox {{
    background: transparent;
    border: 1px solid {P["border_soft"]};
    border-radius: {R["md"]}px;
    margin-top: {S["md"]}px;
    padding: {S["md"]}px {S["sm"]}px {S["sm"]}px;
    color: {P["fg_muted"]};
    font-weight: 600;
    font-size: {F["small"]}px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {S["md"]}px;
    top: -{S["xs"]}px;
    padding: 0px {S["xs"]}px;
    background: {P["bg"]};
}}

/* -----------------------------------------------------------------
 *  Status bar
 * ----------------------------------------------------------------- */
QStatusBar {{
    background: {P["surface"]};
    color: {P["fg_muted"]};
    border-top: 1px solid {P["border_soft"]};
    font-size: {F["small"]}px;
    padding: 0px {S["sm"]}px;
}}
"""
