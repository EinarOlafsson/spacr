"""Half precision: asked for by name, taken only where it works, and said.

Instruction 236 B5 wanted the training loop driven on a GPU, and the one
thing a CPU cannot exercise is mixed precision. Driving it turned up that
there was nothing to exercise:

    grep -rn "autocast\\|GradScaler" spacr/*.py   ->  nothing
    "mixed_precision" in spacr.settings.expected_types        ->  False

So a run with `mixed_precision=True` and a run with `mixed_precision=False` were the same run
twice. Measured on plate1's three classes, 16,875 crops, resnet18, two
epochs on a 3090: 476.4 s and 479.3 s. Identical, because the flag was
dropped by `check_settings` before it reached anything.

WHY IT IS WORTH HAVING RATHER THAN JUST RECORDING. The forward pass and
the loss run in float16 while the weights and the optimiser stay in
float32. On a card with tensor cores that is most of the speed and about
half the activation memory -- which on a screen is not a nicety, it is the
difference between a batch of 32 and a batch of 64 at 224 px.

TWO THINGS IT MUST NOT DO. It must not silently do nothing where there are
no tensor cores -- that is the defect it replaces -- and it must not
silently change the numbers, because a score from a half-precision run and
a score from a full-precision one are not comparable.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spacr.deep_spacr import (autocasting,                   # noqa: E402
                              resolve_mixed_precision)


class TestWhenItIsTaken:
    def test_not_asked_for_is_not_taken(self):
        on, note = resolve_mixed_precision(False, torch.device("cuda"))
        assert on is False
        assert note == ""

    def test_asked_for_on_a_card_is_taken(self):
        on, note = resolve_mixed_precision(True, torch.device("cuda"))
        assert on is True
        assert note

    def test_asked_for_on_a_cpu_is_answered_rather_than_obeyed(self):
        """`torch.autocast` accepts a CPU device type, and on the CPUs
        spaCR runs on it is slower rather than faster. Taking it there
        would be obeying the letter of the setting against its point."""
        on, note = resolve_mixed_precision(True, torch.device("cpu"))
        assert on is False
        assert "cpu" in note.lower()

    def test_the_cpu_refusal_says_what_it_did_instead(self):
        """Silently doing nothing is the defect this replaces."""
        _on, note = resolve_mixed_precision(True, torch.device("cpu"))
        assert "full precision" in note.lower()


class TestWhatItSays:
    def test_it_warns_that_the_numbers_move(self):
        """A score from a half-precision run and a score from a full one
        are not comparable, and a screen is a pile of scores."""
        _on, note = resolve_mixed_precision(True, torch.device("cuda"))
        assert "compare" in note.lower()

    def test_it_says_what_stays_in_full_precision(self):
        """"Mixed" is the whole point: a reader who thinks the weights are
        float16 will not trust the result."""
        _on, note = resolve_mixed_precision(True, torch.device("cuda"))
        lowered = note.lower()
        assert "float16" in lowered and "float32" in lowered


class TestTheContext:
    def test_it_is_a_context_manager_either_way(self):
        """So the training loop has ONE shape rather than a branch around
        every forward pass."""
        for device in ("cpu", "cuda"):
            with autocasting(False, torch.device(device)) as handed_back:
                # It yields nothing and enables nothing; what matters is
                # that it IS a context, so the loop needs no branch.
                assert handed_back is None
                assert not torch.is_autocast_enabled()
            assert not torch.is_autocast_enabled()

    @pytest.mark.gpu
    def test_it_turns_autocast_on_and_off_again(self):
        with autocasting(True, torch.device("cuda")):
            assert torch.is_autocast_enabled()
        assert not torch.is_autocast_enabled()

    def test_off_never_enables_autocast(self):
        with autocasting(False, torch.device("cuda")):
            assert not torch.is_autocast_enabled()


class TestItIsARealSetting:
    def test_it_is_declared(self):
        """A key `check_settings` does not know is a key it DROPS, which is
        exactly how the flag reached nothing."""
        from spacr.settings import expected_types

        assert expected_types["mixed_precision"] is bool

    def test_it_has_a_default(self):
        """Off. Turning it on changes the numbers, and no existing run's
        scores should move because spaCR gained a feature."""
        from spacr.settings import get_train_test_model_settings

        assert get_train_test_model_settings({}).get("mixed_precision") is False

    def test_the_panel_offers_it_beside_the_other_memory_knob(self):
        """`gradient_accumulation` is the other way to fit a bigger
        effective batch in the same VRAM; a user choosing between them
        should see both at once."""
        from spacr.settings import categories

        training = categories["Computer Vision Training"]
        assert "mixed_precision" in training
        assert abs(training.index("mixed_precision")
                   - training.index("gradient_accumulation")) <= 2

    def test_its_help_says_what_it_costs(self):
        from spacr.settings import tooltips

        said = tooltips["mixed_precision"].lower()
        # "a modern graphics card" rather than "CUDA": the hover is read by
        # a biologist, and the word that has to be there is the WARNING --
        # that the numbers move, so two runs are not comparable.
        assert "graphics card" in said
        assert "compare" in said
        assert "efault" in said

    def test_its_help_says_what_it_BUYS(self):
        """A setting whose help says only what it costs is one nobody
        turns on. Measured on an RTX 3090 at 224 px over 30 training steps
        after 5 warm-up: resnet50 1.77x on 0.58x the memory at batch 32 and
        1.78x on 0.55x at batch 64, maxvit_t 1.62x on 0.60x, vit_b_16 2.46x
        on 0.67x. `bench_amp.py` in the GPU queue folder reproduces it."""
        from spacr.settings import tooltips

        said = tooltips["mixed_precision"].lower()
        assert "twice as fast" in said
        assert "memory" in said
        # The numbers live in `resolve_mixed_precision`'s docstring rather
        # than the hover: four model names and six ratios is a table, and a
        # tooltip is a sentence. It is also what the translator choked on --
        # a hover that is mostly protected literals comes back untranslated.
        import inspect

        from spacr.deep_spacr import resolve_mixed_precision

        table = inspect.getdoc(resolve_mixed_precision)
        assert "3090" in table and "1.77x" in table


class TestTheLoopUsesIt:
    def test_the_gradients_are_scaled(self):
        """float16 gradients underflow to zero without a scaler, and the
        model simply does not learn -- silently, which is worse than not
        having the feature."""
        import inspect

        from spacr import deep_spacr

        source = inspect.getsource(deep_spacr.train_model)
        scaler_source = inspect.getsource(deep_spacr._gradient_scaler)
        assert "_gradient_scaler" in source
        assert "GradScaler" in scaler_source
        assert "scaler.scale(loss).backward()" in source
        assert "scaler.step(optimizer)" in source

    def test_the_forward_and_the_loss_are_inside_the_context(self):
        """Only those. The optimiser step must run in full precision, and
        a context that wrapped it would defeat the scaler."""
        import inspect
        import re

        from spacr import deep_spacr

        source = inspect.getsource(deep_spacr.train_model)
        block = source[source.index("with autocasting("):]
        block = block[:block.index("scaler.scale(loss).backward()")]
        assert "model(data)" in block
        assert "loss_fn(" in block
        assert "optimizer.step" not in block

    def test_the_accumulation_flush_scales_too(self):
        """The leftover-gradient flush at the end of an epoch is a second
        optimiser step, and an unscaled one would step on float16
        gradients."""
        import inspect

        from spacr import deep_spacr

        source = inspect.getsource(deep_spacr.train_model)
        assert source.count("scaler.step(optimizer)") == 2
        assert "optimizer.step()" not in source
