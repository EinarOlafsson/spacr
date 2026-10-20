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


def _postinstall():
    start = SCRIPT.index("cat > \"$SCRIPTS/postinstall\" <<'EOF'\n") + len("cat > \"$SCRIPTS/postinstall\" <<'EOF'\n")
    return SCRIPT[start:SCRIPT.index("\nEOF\n", start)] + "\n"


def test_the_1499_system_runtime_is_retired_and_its_cache_kept(tmp_path):
    """1.4.9.9 put a root runtime in the support folder; an upgrade left it there unused."""
    import subprocess

    legacy = tmp_path / "Library" / "Application Support" / "spaCR"
    for part in ("venv/bin", "python/cpython", "bootstrap", "cache/wheels-v6"):
        (legacy / part).mkdir(parents=True)
    (legacy / "install-online.sh").write_text("helper")
    elsewhere = tmp_path / "keep"
    elsewhere.mkdir()
    (legacy / "linked").symlink_to(elsewhere)
    script = (_postinstall().replace('"/Library/Application Support/spaCR"', f'"{legacy}"')
              .replace("mkdir -p /usr/local/bin\n", "").replace(
                  'ln -sfn "/Applications/spaCR.app/Contents/MacOS/spaCR" /usr/local/bin/spacr\n', ""))
    subprocess.run(["/bin/sh", "-c", script], check=True)
    assert sorted(p.name for p in legacy.iterdir()) == ["cache", "install-online.sh", "linked"]
    assert (legacy / "cache" / "wheels-v6").is_dir() and elsewhere.is_dir()
