"""Item 972: the high-content choices offer an optical pooled screen (OPS) that opens the OPS module."""
from tools import build_module_workflows as workflows


def _nodes(node):
    yield node
    for child in node.get('children', ()):
        yield from _nodes(child)


def test_wizard_offers_ops_under_many_images_high_content_and_routes_to_ops():
    from spacr.qt.widgets.sample_project import _protocol_options, _protocol_result

    data = workflows.load()
    workflows._validate_protocols(data)
    workflows._validate_protocol_wizard(data)
    family = data['protocol_wizard']['families']['plate_images']
    assert '>50' in family['label']
    option = family['formats']['ops_cycles']
    assert option['label'] == 'Optical pooled screen (OPS)'
    assert 'in situ sequencing' in option['explanation'] and 'per-gene' in option['explanation']
    assert option['starts'] == ['ops'] and 'ops' in data['modules']
    assert 'ops_cycles' not in data['protocol_wizard']['families']['small_images']['formats']
    assert 'ops_cycles' in {row['key'] for row in _protocol_options('plate_images')}
    result = _protocol_result('plate_images', 'ops_cycles', {'goal': 'recommended'})
    assert [row[0] for row in result['actions']] == ['ops']
    assert result['protocol']['active_start'] == 'ops'
    assert result['protocol']['recipe_title'] == data['protocols']['optical_screen']['title']


def test_tree_and_route_link_the_existing_optical_screen_guide_without_a_copy():
    data = workflows.load()
    workflows._validate_protocols(data)
    images = next(node for node in data['protocol_tree']['children'] if node['id'] == 'images')
    quantify = next(node for node in images['children'] if node['id'] == 'images_quantify')
    ids = [child['id'] for child in quantify['children']]
    assert ids.index('images_quantify_optical_screen') == ids.index('images_quantify_high_content') + 1
    node = next(child for child in quantify['children'] if child['id'] == 'images_quantify_optical_screen')
    assert (node['protocol'], node['start_module']) == ('optical_screen', 'ops')
    assert 'optical_screen' in data['protocols'] and len([k for k in data['protocols'] if 'optical' in k]) == 1
    routes = [r for r in data['protocol_routes']
              if r['protocol'] == 'optical_screen' and r['start_module'] == 'ops']
    assert routes and all(r['data'] == 'images' and r['details'] == 'cycle_images' for r in routes)
