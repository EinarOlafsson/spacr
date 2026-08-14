"""Clean-install smoke test: build one module screen and measure one field."""
from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


def _read_measure_result(database: Path) -> tuple[tuple[object, ...] | None, int]:
    """Return the terminal run status and cell count, then release the DB."""
    # sqlite3.Connection's context manager commits or rolls back but does not
    # close the handle.  Keeping it open is harmless when unlinking on POSIX,
    # but Windows refuses to remove measurements.db during temp-dir cleanup.
    with closing(sqlite3.connect(database)) as connection:
        status = connection.execute(
            "SELECT status, n_succeeded, n_failed FROM run_status "
            "ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        cells = connection.execute("SELECT COUNT(*) FROM cell").fetchone()[0]
    return status, cells


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QLabel
    from spacr.measure import measure_crop
    from spacr.qt.install_consent import InstallerConsentDialog
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.synthetic import demo_settings, generate_measure_demo

    app = QApplication.instance() or QApplication([])
    screen = AppScreen("measure")
    screen.close()
    screen.deleteLater()
    app.processEvents()

    consent = InstallerConsentDialog()
    if any(consent.choices().values()):
        raise RuntimeError("fresh-install consent choices did not start off")
    consent_text = " ".join(
        label.text() for label in consent.findChildren(QLabel)
    )
    consent.close()
    consent.deleteLater()
    app.processEvents()
    if "PUBLIC spaCR GitHub repository" not in consent_text or (
        "cannot be reliably unpublished" not in consent_text
    ):
        raise RuntimeError("fresh-install consent page omitted public-report warning")

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
        status, cells = _read_measure_result(database)
        if status != ("complete", 1, 0) or cells < 1:
            raise RuntimeError(
                f"measure smoke run incomplete: status={status!r}, cells={cells}"
            )
    print("installed spaCR module screen and one-field measure run passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
