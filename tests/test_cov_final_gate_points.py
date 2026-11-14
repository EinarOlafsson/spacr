"""Which outline anchors are new, and the Cluster button's hand-off."""


from spacr.qt.widgets.gate_editor import GateEditorPanel


def test_the_cluster_button_asks_before_running_on_a_worker():
    asked = []

    class Stub:
        def run_cluster_in_background(self, ask=False):
            asked.append(ask)

    GateEditorPanel._on_cluster_clicked(Stub(), False)
    assert asked == [True]
