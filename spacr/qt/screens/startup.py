"""StartupPage — the home screen.

Layout, top to bottom:

    ┌──────────────────────────────────────────────────────────┐
    │             spaCR  🖼    (wordmark + logo right)          │
    │        End-to-end microscopy → single-cell …            │
    ├──────────────────────────────────────────────────────────┤
    │  CORE  ──────────────────────────────────────            │
    │    🖼  Mask     — Cellpose segmentation of cells…         │
    │    📏  Measure  — Per-object feature extraction…         │
    │    …                                                    │
    │  ANALYSIS  ──────────────────────────────────            │
    │    …                                                    │
    └──────────────────────────────────────────────────────────┤
    │  Hover a tile to see what it does.                       │  ← sticky
    └──────────────────────────────────────────────────────────┘

The hint bar at the bottom stays pinned as the user scrolls so the
description of whatever's under the cursor is always visible — no
guessing what the tiny descriptions in the tiles mean.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..theme import PALETTE, SPACING
from ..widgets import Divider, HTile


def _find_logo_pixmap() -> Optional[QPixmap]:
    """Locate the bundled spaCR logo, returning ``None`` if absent."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in ("logo_spacr.png", "logo_spacr_v1.png"):
        p = os.path.normpath(
            os.path.join(here, "..", "..", "resources", "icons", candidate)
        )
        if os.path.isfile(p):
            pix = QPixmap(p)
            if not pix.isNull():
                return pix
    return None


_DEFAULT_HINT = "Hover a tile to see what it does."


class StartupPage(QWidget):
    """Home screen. ``tile_clicked(str key)`` fires when a row is pressed.

    :param apps: ``(key, name, description, section)`` tuples describing
        each app to render.
    :param icon_provider: callable that returns a QIcon for a given app
        key (or ``None`` for a text-fallback tile).
    :param parent: optional parent widget.
    :ivar tile_clicked: emitted with the app key when a tile is pressed.
    """

    tile_clicked = Signal(str)

    def __init__(
        self,
        apps: List[Tuple[str, str, str, str]],
        icon_provider: Callable[[str], Optional[QIcon]],
        parent=None,
    ):
        super().__init__(parent)

        # Two-row outer layout: scrollable content + sticky hint bar.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        content = QWidget()
        col = QVBoxLayout(content)
        col.setContentsMargins(SPACING["xxl"], SPACING["xxl"],
                                SPACING["xxl"], SPACING["xxl"])
        col.setSpacing(SPACING["xl"])

        col.addWidget(self._build_hero())

        # Sections — keep a mapping from every HTile to its description
        # so the sticky footer knows what to display on hover.
        # Each section is now a HORIZONTAL row of tiles (was a 2-col
        # grid) which collapses the middle of the page and leaves
        # room for the Insights dashboard below.
        self._tile_hints: dict = {}
        sections: dict[str, list[tuple[str, str, str]]] = {}
        for key, name, desc, section in apps:
            sections.setdefault(section, []).append((key, name, desc))

        for section_name, entries in sections.items():
            col.addWidget(self._build_section_header(section_name))
            col.addWidget(self._build_section_grid(entries, icon_provider))

        # Insights dashboard — three compact cards along the lower
        # half of the home screen. Replaces the old two-column grid's
        # tendency to shove everything upward.
        col.addSpacing(SPACING["md"])
        col.addWidget(self._build_insights_dashboard())

        # Reserved surface under the insights row. Deliberately empty
        # for now — will host featured content (news, tutorial
        # thumbnails, sponsored slots) later. Rendered as a subtle
        # rounded gray box so it's laid out and visible but doesn't
        # compete with the rest of the UI. Consumers can swap in a
        # widget via :meth:`set_reserved_content`.
        col.addSpacing(SPACING["md"])
        col.addWidget(self._build_reserved_surface())

        col.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # Sticky hint bar — always visible along the bottom, updates
        # on tile hover, reverts to the default when the mouse leaves.
        # Styled via the app stylesheet (#HintBar) so it renders white and
        # tracks the font-size preference instead of a fixed 13px.
        self._hint_bar = QLabel(_DEFAULT_HINT)
        self._hint_bar.setObjectName("HintBar")
        self._hint_bar.setAlignment(Qt.AlignHCenter)
        self._hint_bar.setMinimumHeight(36)
        outer.addWidget(self._hint_bar)

    # -- pieces --------------------------------------------------------
    def _build_hero(self) -> QWidget:
        """Single row: LOGO | spaCR wordmark | subtitle.

        Layout::

            [🖼   large logo]   spaCR   End-to-end microscopy → …

        Everything is baseline-aligned so the descender of "spaCR"
        sits on the same line as the subtitle text.
        """
        hero = QWidget()
        row = QHBoxLayout(hero)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["lg"])
        row.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # LEFT: logo (larger than before — 108 px, was 72).
        logo_pix = _find_logo_pixmap()
        if logo_pix is not None:
            logo_lbl = QLabel()
            logo_lbl.setPixmap(
                logo_pix.scaled(108, 108, Qt.KeepAspectRatio,
                                Qt.SmoothTransformation)
            )
            logo_lbl.setFixedSize(108, 108)
            logo_lbl.setStyleSheet("background: transparent;")
            row.addWidget(logo_lbl, alignment=Qt.AlignVCenter)

        # CENTER: wordmark
        title = QLabel("spaCR")
        title.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 300;"
            "font-size: 64px;"
            f"color: {PALETTE['accent']};"
            "letter-spacing: -1.2px;"
            "background: transparent;"
        )
        row.addWidget(title, alignment=Qt.AlignVCenter)

        # RIGHT: subtitle, in the same line rather than stacked below.
        # Styled via the app stylesheet (#HeroSubtitle) so it renders white
        # and tracks the font-size preference instead of a fixed 16px.
        subtitle = QLabel(
            "End-to-end microscopy → single-cell measurements "
            "→ genotype-phenotype mapping."
        )
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setMinimumWidth(320)
        row.addWidget(subtitle, 1, alignment=Qt.AlignVCenter)

        return hero

    def _build_insights_dashboard(self) -> QWidget:
        """Three glanceable cards along the lower half of Home.

        Layout::

            ┌ System ─────┐ ┌ Recent (3) ─────┐ ┌ Totals ────┐
            │ GPU 42%     │ │ 14:22 mask ✓    │ │ 47 plates  │
            │ VRAM 3 GB   │ │ 12:05 meas ✓    │ │ 128k cells │
            │ Disk 82%    │ │ 09:47 mask ✗    │ │ 12 models  │
            └─────────────┘ └─────────────────┘ └────────────┘

        Every card degrades gracefully — GPU section reports "no CUDA"
        when torch/cuda are unavailable, Recent shows the top 3 runs
        or a hint when the journal is empty, Totals shows zeros.
        """
        dash = QWidget()
        row = QHBoxLayout(dash)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["md"])
        row.addWidget(self._build_system_card(), 1)
        row.addWidget(self._build_recent_runs_card(), 2)
        row.addWidget(self._build_totals_card(), 1)
        return dash

    def _card_wrap(self, title: str, body: QWidget) -> QWidget:
        """Style helper: title label above a bordered content box."""
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        hdr = QLabel(title.upper())
        hdr.setStyleSheet(
            "font-family: 'Open Sans', sans-serif; font-weight: 600;"
            "font-size: 10px; letter-spacing: 2px;"
            f"color: {PALETTE['fg_muted']};"
        )
        col.addWidget(hdr)
        body.setStyleSheet(
            f"background: {PALETTE['surface_alt']};"
            f"border: 1px solid {PALETTE['border_soft']};"
            "border-radius: 8px;"
            f"padding: {SPACING['md']}px;"
        )
        col.addWidget(body, 1)
        return wrap

    def _build_system_card(self) -> QWidget:
        """GPU / VRAM / disk snapshot. Refreshes on Home-revisit only —
        the numbers move slowly enough that a live poll isn't worth
        the CPU."""
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self._stat_row("GPU",  self._gpu_util_pct()))
        lay.addWidget(self._stat_row("VRAM", self._gpu_vram_used()))
        lay.addWidget(self._stat_row("Disk", self._disk_used_pct()))
        lay.addStretch(1)
        return self._card_wrap("System", body)

    def _stat_row(self, label: str, value: str) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACING["sm"])
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {PALETTE['fg_muted']};"
                            "font-size: 11px; font-weight: 500;")
        lbl.setMinimumWidth(40)
        val = QLabel(value)
        val.setStyleSheet(f"color: {PALETTE['fg']}; font-size: 12px;"
                            "font-weight: 500;")
        lay.addWidget(lbl); lay.addWidget(val); lay.addStretch(1)
        return row

    def _gpu_util_pct(self) -> str:
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            return f"{u.gpu}%"
        except Exception:
            try:
                import torch
                return "idle" if torch.cuda.is_available() else "no CUDA"
            except Exception:
                return "n/a"

    def _gpu_vram_used(self) -> str:
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            used_gb = info.used / 1e9
            total_gb = info.total / 1e9
            return f"{used_gb:.1f} / {total_gb:.0f} GB"
        except Exception:
            try:
                import torch
                if torch.cuda.is_available():
                    used = torch.cuda.memory_allocated() / 1e9
                    return f"{used:.1f} GB"
            except Exception:
                pass
            return "n/a"

    def _disk_used_pct(self) -> str:
        try:
            import shutil as _sh
            usage = _sh.disk_usage(os.path.expanduser("~"))
            pct = int(100 * usage.used / usage.total)
            return f"{pct}%"
        except Exception:
            return "n/a"

    def _build_recent_runs_card(self) -> QWidget:
        """Compact 3-row list of the most recent runs (or an empty
        hint). Each row is clickable and emits the run's app_key."""
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        try:
            from spacr.run_journal import recent_runs
            runs = recent_runs(limit=3)
        except Exception:
            runs = []
        if not runs:
            hint = QLabel("No runs yet — start one from the tiles above.")
            hint.setStyleSheet(f"color: {PALETTE['fg_muted']};"
                                "font-style: italic;")
            hint.setWordWrap(True)
            lay.addWidget(hint)
        else:
            for entry in runs:
                lay.addWidget(self._recent_run_row(entry))
        lay.addStretch(1)
        return self._card_wrap("Recent runs", body)

    def _recent_run_row(self, entry: dict) -> QWidget:
        """One clickable row inside the Recent-Runs card."""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACING["sm"])
        status_ok = entry.get("status") == "success"
        status_icon = "✓" if status_ok else "✗"
        colour = (PALETTE['success'] if status_ok
                    else PALETTE['error'])
        icon = QLabel(status_icon)
        icon.setStyleSheet(f"color: {colour}; font-weight: 700;"
                            "font-size: 13px;")
        icon.setFixedWidth(14)
        label_txt = f"{entry.get('app_key', '?')}"
        elapsed = entry.get('elapsed_s') or 0
        label = QLabel(f"{label_txt:<8s}  {int(elapsed):>4d}s")
        label.setStyleSheet(f"color: {PALETTE['fg']};"
                             "font-family: 'JetBrains Mono', monospace;"
                             "font-size: 11px;")
        lay.addWidget(icon)
        lay.addWidget(label, 1)
        # Wrap in a clickable button so the row navigates on click
        from PySide6.QtWidgets import QPushButton
        btn = QPushButton()
        btn.setLayout(lay)
        btn.setFlat(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; "
            "text-align: left; padding: 2px; } "
            f"QPushButton:hover {{ background: {PALETTE['surface_hi']}; }}"
        )
        app_key = entry.get("app_key", "")
        btn.clicked.connect(
            lambda checked=False, k=app_key: self.tile_clicked.emit(k))
        return btn

    def _build_totals_card(self) -> QWidget:
        """Aggregate journal counts: total runs, per-app, distinct models."""
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        try:
            from spacr.run_journal import journal_totals
            t = journal_totals()
        except Exception:
            t = {"total_runs": 0, "mask_runs": 0, "measure_runs": 0,
                    "classify_runs": 0, "models_recorded": 0}
        lay.addWidget(self._stat_row("Runs",   str(t["total_runs"])))
        lay.addWidget(self._stat_row("Mask",   str(t["mask_runs"])))
        lay.addWidget(self._stat_row("Meas.",  str(t["measure_runs"])))
        lay.addWidget(self._stat_row("Models", str(t["models_recorded"])))
        lay.addStretch(1)
        return self._card_wrap("Totals", body)

    def _build_reserved_surface(self) -> QWidget:
        """Empty gray placeholder under the insights dashboard.

        A future release will drop featured content here (news feed,
        tutorial thumbnails, sponsored panels, whatever). For now it
        renders as a rounded gray box with a subtle "Reserved for
        featured content" caption so the layout doesn't collapse.
        """
        self._reserved_content: Optional[QWidget] = None
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        hdr = QLabel("FEATURED".upper())
        hdr.setStyleSheet(
            "font-family: 'Open Sans', sans-serif; font-weight: 600;"
            "font-size: 10px; letter-spacing: 2px;"
            f"color: {PALETTE['fg_muted']};"
        )
        col.addWidget(hdr)

        surface = QWidget()
        surface.setObjectName("ReservedSurface")
        surface.setMinimumHeight(140)
        surface.setStyleSheet(
            f"background: {PALETTE['surface_alt']};"
            f"border: 1px solid {PALETTE['border_soft']};"
            "border-radius: 10px;"
        )
        inner = QVBoxLayout(surface)
        caption = QLabel("Reserved for featured content")
        caption.setAlignment(Qt.AlignCenter)
        caption.setStyleSheet(
            f"color: {PALETTE['fg_dim']}; font-style: italic;"
            "font-size: 12px; background: transparent;")
        inner.addWidget(caption)
        col.addWidget(surface)
        self._reserved_surface = surface
        return wrap

    def set_reserved_content(self, widget: QWidget) -> None:
        """Swap in a widget to fill the reserved surface.

        Later releases (news feed, tutorial links, etc.) can call
        this to promote the surface from placeholder to real content
        without touching layout code.
        """
        surface = getattr(self, "_reserved_surface", None)
        if surface is None:
            return
        # Clear existing children
        lay = surface.layout()
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        lay.addWidget(widget)
        self._reserved_content = widget

    def _build_recent_runs_section(self) -> Optional[QWidget]:
        """Return a "Recent runs" widget, or None if there's no history.

        Reads :func:`spacr.run_journal.recent_runs` (last 5 by default);
        each row is a clickable strip that emits
        :attr:`tile_clicked` for that run's app + fires a callback to
        load its settings into the target AppScreen.
        """
        # Import + fetch. Imports and disk I/O can fail cheaply on a
        # freshly-checked-out install — we log to help diagnose (the
        # older `except Exception: return None` swallowed silently).
        import logging as _lg
        _log = _lg.getLogger("spacr.qt.startup")
        try:
            # Absolute — spacr/qt/screens/startup.py needs three dots
            # to reach spacr.run_journal, but the absolute form is
            # unambiguous and refactor-safe.
            from spacr.run_journal import recent_runs, load_run_settings
        except Exception as e:
            _log.debug("recent runs import failed: %s", e)
            return None
        try:
            runs = recent_runs(limit=5)
        except Exception as e:
            _log.debug("recent_runs() raised: %s", e)
            return None
        if not runs:
            _log.debug("recent_runs() returned nothing — hiding section")
            return None

        from PySide6.QtWidgets import QGridLayout, QPushButton

        wrap = QWidget()
        outer = QVBoxLayout(wrap)
        outer.setContentsMargins(0, SPACING["sm"], 0, SPACING["md"])
        outer.setSpacing(6)

        hdr = QLabel("RECENT RUNS")
        hdr.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 600; font-size: 11px;"
            "letter-spacing: 2px;"
            f"color: {PALETTE['fg_muted']};"
        )
        outer.addWidget(hdr)
        outer.addWidget(Divider())

        # Store a per-row loader closure so clicking navigates AND
        # pushes the recorded settings back into the target screen.
        self._recent_loaders = []
        for r in runs:
            btn = QPushButton()
            btn.setCursor(Qt.PointingHandCursor)
            elapsed = r.get("elapsed_s") or 0
            status_dot = "●" if r["status"] == "success" else "○"
            label = (
                f"  {status_dot}  {r['app_key']:12s}  "
                f"{r['start_utc'][:19]}  "
                f"({elapsed:.0f}s)"
            )
            btn.setText(label)
            btn.setStyleSheet(
                "QPushButton {"
                "  text-align: left; padding: 6px 12px;"
                "  background: transparent;"
                "  border: 1px solid transparent;"
                "  border-radius: 4px;"
                "  font-family: 'JetBrains Mono', monospace;"
                "  font-size: 12px;"
                f"  color: {PALETTE['fg_muted']};"
                "}"
                "QPushButton:hover {"
                f"  background: {PALETTE['surface_alt']};"
                f"  border-color: {PALETTE['border_soft']};"
                f"  color: {PALETTE['fg']};"
                "}"
            )
            def _load_run(run=r):
                self.tile_clicked.emit(run["app_key"])
                # After navigation the target AppScreen exists —
                # tile_clicked.emit is synchronous inside MainWindow's
                # nav handler. Push the recorded settings if possible.
                try:
                    from PySide6.QtCore import QTimer
                    def _push():
                        try:
                            mw = self.window()
                            if mw is None or not hasattr(mw, "_screens"):
                                return
                            screen = mw._screens.get(run["app_key"])
                            if screen is None or not hasattr(
                                screen, "apply_settings_dict"
                            ):
                                return
                            settings = load_run_settings(run["dir"])
                            screen.apply_settings_dict(settings)
                        except Exception:
                            pass
                    QTimer.singleShot(50, _push)
                except Exception:
                    pass
            btn.clicked.connect(_load_run)
            outer.addWidget(btn)
            self._recent_loaders.append(_load_run)
        return wrap

    def _build_section_header(self, name: str) -> QWidget:
        wrap = QWidget()
        row = QVBoxLayout(wrap)
        row.setContentsMargins(0, SPACING["sm"], 0, 0)
        row.setSpacing(6)

        hdr = QLabel(name.upper())
        hdr.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 600;"
            "font-size: 11px;"
            "letter-spacing: 2px;"
            f"color: {PALETTE['fg_muted']};"
        )
        row.addWidget(hdr)
        row.addWidget(Divider())
        return wrap

    def _build_section_grid(
        self,
        entries: list[tuple[str, str, str]],
        icon_provider: Callable[[str], Optional[QIcon]],
    ) -> QWidget:
        """Return a horizontal row of tiles for this section.

        Wrapped in a :class:`QScrollArea` so a section wider than the
        window gets a horizontal scroll rather than pushing the whole
        page sideways. Tile min-width shrinks (compared to the old
        two-column grid) so more fit on screen at once.
        """
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, SPACING["xs"], 0, SPACING["md"])
        row.setSpacing(SPACING["sm"])

        # Tiles in horizontal rows: show ONLY the name (no wrapped
        # description). The description still surfaces in the sticky
        # hint bar at the bottom on hover, so users don't lose that
        # information — they just don't have to squint at a cut-off
        # two-line label. The icon jumps up to 44 px so the tile
        # reads as an app launcher, not a menu entry.
        for key, name, desc in entries:
            icon = icon_provider(key)
            tile = HTile(text=name, description="", icon=icon,
                          icon_size=44)
            tile.setMinimumWidth(180)
            tile.setMaximumWidth(240)
            self._tile_hints[tile] = desc
            tile.installEventFilter(self)
            tile.clicked.connect(lambda checked=False, k=key:
                                   self.tile_clicked.emit(k))
            row.addWidget(tile)
        row.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(row_widget)
        from ..preferences import scaled_px
        # Slightly taller now to accommodate the bigger icon.
        scroll.setFixedHeight(scaled_px(112))
        return scroll

    # -- sticky hint bar wiring ----------------------------------------
    def eventFilter(self, obj, event):
        """Update the pinned hint bar as the cursor enters/leaves tiles."""
        if event.type() == QEvent.Enter:
            hint = self._tile_hints.get(obj)
            if hint:
                self._hint_bar.setText(hint)
        elif event.type() == QEvent.Leave:
            self._hint_bar.setText(_DEFAULT_HINT)
        return super().eventFilter(obj, event)
