"""The install windows ask, they do not explain: the privacy text lives in the app and the terms.

Maintainer decision 2026-10-19: the paragraph about anonymised error reports is removed from the macOS,
Linux and Windows install windows (the app and the user agreement already carry it), and every consent
box starts ticked. "Install without wgpu" is an opt-out of the 3D renderer and stays unticked.
"""
from pathlib import Path

ONLINE = Path(__file__).resolve().parents[1] / "packaging" / "online"


def test_no_install_window_shows_the_report_paragraph():
    assert "gui_public_reports" not in (ONLINE / "macos_installer.m").read_text(encoding="utf-8")
    assert "gui_public_reports" not in (ONLINE / "install_spacr_unix.sh").read_text(encoding="utf-8")
    assert "Anonymised error reports" not in (ONLINE / "spacr_online_installer.nsi").read_text(encoding="utf-8")


def test_every_consent_box_starts_ticked_and_wgpu_stays_off():
    mac = (ONLINE / "macos_installer.m").read_text(encoding="utf-8")
    assert 'NSArray *keys = @[@"nsis_gpu", @"gui_share_diagnostics", @"gui_report_issues", @"without_wgpu"];' in mac
    assert "choice.state = index < 3 ? NSControlStateValueOn : NSControlStateValueOff;" in mac
    unix = (ONLINE / "install_spacr_unix.sh").read_text(encoding="utf-8")
    assert 'TRUE diagnostics "$(spacr_say gui_share_diagnostics)"' in unix
    assert 'TRUE issues "$(spacr_say gui_report_issues)"' in unix
    assert '"$wgpu_default" without_wgpu' in unix
    nsis = (ONLINE / "spacr_online_installer.nsi").read_text(encoding="utf-8")
    assert "${NSD_Check} $ConsentShare" in nsis and "${NSD_Check} $ConsentIssues" in nsis
