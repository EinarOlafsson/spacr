"""The macOS .pkg installs spaCR.app into /Applications, never onto another copy.

Found on an Intel Mac 2026-10-18: pkgbuild's default component plist marks
the app bundle relocatable, so Installer updated a backup copy of spaCR.app
that Spotlight knew about and left /Applications/spaCR.app missing; the
installer still reported success.
"""
import pathlib

SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
          / "packaging" / "online" / "build_macos_online.sh").read_text(encoding="utf-8")


def test_the_component_plist_pins_every_bundle():
    assert 'pkgbuild --analyze --root "$ROOT" "$COMPONENTS"' in SCRIPT
    assert 'plutil -replace "$index.BundleIsRelocatable" -bool NO "$COMPONENTS"' in SCRIPT


def test_the_package_is_built_from_that_plist():
    build = SCRIPT[SCRIPT.index("pkgbuild \\"):]
    assert '--component-plist "$COMPONENTS"' in build.split('"$OUT"')[0]
