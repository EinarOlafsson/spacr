"""Build explicitly unavailable tutorial routes without inventing recordings.

The maintainer approved these five screens on 2026-09-11. Original staged
catalogs and held recordings remain unchanged; only release copies transform.
"""
from copy import deepcopy

HELD = ('12_map_barcodes', '21_model_compare', '22_model_zoo', '71_investigate_hit')
OPS = '76_ops'
# Embeddings shipped a Home tile on 2026-09-11 (instruction 386 step 6) after
# this candidate was verified, so the course had a registered module with no
# route at all -- which the route audit is there to catch and did. A
# placeholder is the honest entry: the module exists and the lesson does not.
#
# NO host_app_key, unlike OPS. `tiled_apps()` reports Embeddings AS a Home
# tile and OPS as a fold on Mask, so OPS is a submodule route and this is a
# main one. Giving it a host would nest it under a module it does not belong
# to and the navigation contract would place it wrongly.
EMBEDDINGS = '77_embeddings'
PLACEHOLDERS = (*HELD, OPS, EMBEDDINGS)

# Interface copy only: no narration is synthesized for an unavailable lesson.
COPY = {
    'en': ('Coming soon', 'This tutorial is not available yet. Please explore the other tutorials.'),
    'es': ('Próximamente', 'Este tutorial aún no está disponible. Mientras tanto, puedes explorar los otros tutoriales.'),
    'fr': ('Bientôt disponible', 'Ce tutoriel n’est pas encore disponible. En attendant, découvrez les autres tutoriels.'),
    'hi': ('जल्द आ रहा है', 'यह ट्यूटोरियल अभी उपलब्ध नहीं है। तब तक अन्य ट्यूटोरियल देखें।'),
    'it': ('Prossimamente', 'Questo tutorial non è ancora disponibile. Nel frattempo, esplora gli altri tutorial.'),
    'ja': ('近日公開', 'このチュートリアルはまだ公開されていません。ほかのチュートリアルをご覧ください。'),
    'pt-BR': ('Em breve', 'Este tutorial ainda não está disponível. Enquanto isso, explore os outros tutoriais.'),
    'zh-CN': ('即将推出', '本教程尚未上线。请先浏览其他教程。'),
    'da': ('Kommer snart', 'Denne vejledning er endnu ikke tilgængelig. Se de andre vejledninger imens.'),
    'de': ('Demnächst verfügbar', 'Dieses Tutorial ist noch nicht verfügbar. Entdecke in der Zwischenzeit die anderen Tutorials.'),
    'is': ('Væntanlegt', 'Þetta kennsluefni er ekki tiltækt enn. Skoðaðu annað kennsluefni á meðan.'),
    'ko': ('곧 공개됩니다', '이 튜토리얼은 아직 공개되지 않았습니다. 다른 튜토리얼을 먼저 살펴보세요.'),
    'nb': ('Kommer snart', 'Denne veiledningen er ikke tilgjengelig ennå. Se de andre veiledningene i mellomtiden.'),
    'sv': ('Kommer snart', 'Den här handledningen är inte tillgänglig ännu. Utforska de andra handledningarna under tiden.'),
}


def release_catalog(source, language):
    """Preserve every ready lesson verbatim and replace only approved holds."""
    result = deepcopy(source)
    ids = [lesson['id'] for lesson in result['lessons']]
    recorded_embedding = next((lesson for lesson in result['lessons']
                               if lesson['id'] == EMBEDDINGS), None)
    valid_promotion = (recorded_embedding is not None
                       and recorded_embedding.get('number') == 77
                       and recorded_embedding.get('app_key') == 'embeddings'
                       and 'host_app_key' not in recorded_embedding
                       and recorded_embedding.get('status') != 'coming_soon'
                       and bool(recorded_embedding.get('scenes')))
    if (len(ids) != len(set(ids)) or not set(HELD) <= set(ids)
            or OPS in ids or (EMBEDDINGS in ids and not valid_promotion)):
        raise ValueError('Expected distinct original lessons, four holds, '
                         'no OPS placeholder, and only a recorded Embeddings promotion')
    title, description = COPY[language]
    for lesson in result['lessons']:
        if lesson['id'] in HELD:
            # Retain identity, title and routing, not stale promises or media.
            for field in ('silent', 'poster', 'example_files'):
                lesson.pop(field, None)
            lesson.update(status='coming_soon', availability_title=title,
                          description=description, objectives=[], prerequisite='', scenes=[])
    ops = {
        'id': OPS, 'number': 76, 'slug': 'ops', 'title': 'OPS', 'series': 2,
        'app_key': 'ops', 'host_app_key': 'mask', 'section': 'Segmentation models',
        'status': 'coming_soon', 'availability_title': title,
        'description': description, 'objectives': [], 'prerequisite': '', 'scenes': [],
    }
    # Preserve legacy caption entries and their original order. Some have
    # no numeric metadata until write_catalogs copies the English routing.
    # Only the reserved OPS position needs insertion before recorded 77.
    position = ids.index(EMBEDDINGS) if recorded_embedding is not None else len(ids)
    result['lessons'].insert(position, ops)
    if recorded_embedding is None:
        result['lessons'].append({
            'id': EMBEDDINGS, 'number': 77, 'slug': 'embeddings',
            'title': 'Embeddings', 'series': 2, 'app_key': 'embeddings',
            'section': 'Data', 'status': 'coming_soon',
            'availability_title': title, 'description': description,
            'objectives': [], 'prerequisite': '', 'scenes': [],
        })
    return result
