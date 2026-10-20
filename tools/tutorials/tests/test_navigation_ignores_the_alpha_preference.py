"""The tutorial navigation lists the Home tiles a lesson is recorded on, whatever this machine shows.

Found 2026-10-20: the published navigation carried candida, giardia, leishmania, mammalian, plasmodium,
trypanosoma and virus in its main section. build_navigation read app.tiled_apps(), which follows the live
Show alpha species preference, so a build on a machine with it on listed the alpha organism tiles that no
lesson is recorded with.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_navigation import build

ROOT = Path(__file__).resolve().parents[1] / 'release_candidate'


def test_alpha_tiles_shown_on_this_machine_stay_out_of_the_navigation(monkeypatch):
    from spacr.qt import app
    from spacr.settings import _SPECIES_WITH_PUBLISHED_LESSONS, _alpha_names

    hidden = set(_alpha_names('apps')) - set(_SPECIES_WITH_PUBLISHED_LESSONS)
    original = app.tiled_apps
    seen = {}

    def with_alpha_shown():
        rows = list(original())
        keys = {row[0] for row in rows}
        rows += [row for row in app.APPS if row[0] in hidden and row[0] not in keys]
        seen["shown"] = rows
        return rows

    monkeypatch.setattr(app, "tiled_apps", with_alpha_shown)
    english = json.loads((ROOT / 'web/catalog/lessons_en.json').read_text(encoding="utf-8"))
    navigation = build(english)
    shown = seen["shown"]
    assert {row[0] for row in shown} & hidden, "no alpha tile was shown to the builder"
    keys = {key for section in navigation['sections'] for group in section['groups']
            for key in group.get('module_keys', [])}
    assert not keys & hidden
    assert {row[0] for row in shown if row[0] not in hidden} <= keys | {
        lesson.get('app_key') for lesson in english['lessons']}
