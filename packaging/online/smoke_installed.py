"""Clean-install smoke test: build one module screen and measure one field."""
from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from spacr.measure import measure_crop
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.synthetic import demo_settings, generate_measure_demo

    app = QApplication.instance() or QApplication([])
    screen = AppScreen("measure")
    screen.close()
    screen.deleteLater()
    app.processEvents()

    with tempfile.TemporaryDirectory(prefix="spacr-installed-smoke-") as tmp:
        layout = generate_measure_demo(
            Path(tmp) / "experiment",
            wells=("A01",),
            fields=1,
            channels=(0, 1, 2, 3),
        )
        settings = demo_settings("measure", str(layout.src))
        settings.update(
            save_png=False,
            representative_images=False,
            n_jobs=1,
            verbose=False,
        )
        measure_crop(settings)
        database = layout.src / "measurements" / "measurements.db"
        if not database.is_file():
            raise RuntimeError("measure smoke run produced no measurements.db")
        with sqlite3.connect(database) as connection:
            status = connection.execute(
                "SELECT status, n_succeeded, n_failed FROM run_status "
                "ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            cells = connection.execute("SELECT COUNT(*) FROM cell").fetchone()[0]
        if status != ("complete", 1, 0) or cells < 1:
            raise RuntimeError(
                f"measure smoke run incomplete: status={status!r}, cells={cells}"
            )
    print("installed spaCR module screen and one-field measure run passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
