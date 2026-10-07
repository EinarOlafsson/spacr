KEEP = {
    'tests/qt/test_641_module_first_open_budgets.py::test_a_module_first_open_stays_inside_its_budget[mask]',
    'tests/qt/test_471c_screens_p_to_z_sections_fold.py::test_every_figure_and_table_folds_to_its_heading_at_the_bottom[pipeline_graph]',
}


def pytest_collection_modifyitems(config, items):
    selected = [item for item in items if item.nodeid in KEEP]
    deselected = [item for item in items if item.nodeid not in KEEP]
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
