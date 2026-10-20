"""Tiny protocol regressions, not Qt/native/scientific acceptance."""
import unittest
from types import SimpleNamespace

from proposed_adapter import run_owned_preview


class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self, slot):
        self.slots.remove(slot)

    def emit(self, value):
        for slot in list(self.slots):
            slot(value)


class ProtocolTests(unittest.TestCase):
    def panel(self):
        button = lambda: SimpleNamespace(isVisible=lambda: True, isEnabled=lambda: True)
        p = SimpleNamespace(preview_ready=Signal(), _run_btn=button(),
                            _cancel_btn=button(), busy=False, _plaque_result=None)
        p.mode = lambda: "plaque"
        p.current_path = lambda: "/private/input.tif"
        p.current_settings = lambda: {"plaque_model": "selected", "diameter": 30}
        p.preview_running = lambda: p.busy
        self.addCleanup(lambda: self.assertEqual(p.preview_ready.slots, []))
        return p

    def test_fast_completion_before_click_returns_is_observed(self):
        p = self.panel()
        result = {"path": p.current_path(), "labels": object(), "count": 4}
        def click(_button):
            p._plaque_result = result
            p.preview_ready.emit(result)
        self.assertIs(run_owned_preview(p, click, lambda _: None, 1), result)

    def test_preexisting_result_does_not_certify_a_new_pass(self):
        p = self.panel()
        p._plaque_result = {"path": p.current_path(), "labels": object(), "count": 4}
        with self.assertRaisesRegex(ValueError, "without an accepted result"):
            run_owned_preview(p, lambda _: None, lambda _: None, 1)

    def test_other_image_completion_is_refused(self):
        p = self.panel()
        def click(_button):
            p._plaque_result = {"path": "/other.tif", "labels": object(), "count": 4}
            p.preview_ready.emit(p._plaque_result)
        with self.assertRaisesRegex(ValueError, "does not identify"):
            run_owned_preview(p, click, lambda _: None, 1)

    def test_timeout_uses_actual_cancel_control(self):
        p = self.panel()
        clicks = []
        def click(button):
            clicks.append(button)
            p.busy = button is p._run_btn
        ticks = iter([0, 2])
        with self.assertRaises(TimeoutError):
            run_owned_preview(p, click, lambda _: None, 1, clock=lambda: next(ticks))
        self.assertEqual(clicks, [p._run_btn, p._cancel_btn])


if __name__ == "__main__":
    unittest.main()
