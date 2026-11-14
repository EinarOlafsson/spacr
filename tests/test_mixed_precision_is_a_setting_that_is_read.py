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

    @pytest.mark.parametrize("device, fragment", [
        # Silently doing nothing is the defect this replaces.
        ("cpu", "full precision"),
        # A score from a half-precision run and a score from a full one are
        # not comparable, and a screen is a pile of scores.
        ("cuda", "compare"),
    ], ids=["cpu-refusal-says-what-it-did-instead", "numbers-move-warning"])
    def test_the_note_says_what_it_did_or_what_moves(self, device, fragment):
        _on, note = resolve_mixed_precision(True, torch.device(device))
        assert fragment in note.lower()


class TestWhatItSays:
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
        """`gradient_accumulation_steps` is the other way to fit a bigger
        effective batch in the same VRAM; a user choosing between them
        should see both at once.

        THE NAME HERE WAS THE OLD ONE and this test failed on it rather
        than on the adjacency it is about. `gradient_accumulation` was a
        boolean beside a step count saying the same thing twice, and
        `_fold_gradient_accumulation` merged the pair -- but nothing
        brought this assertion with it, so the failure read as "the panel
        does not offer it" when the panel offers both, adjacent.
        """
        from spacr.settings import categories

        training = categories["Computer Vision Training"]
        assert "mixed_precision" in training
        assert abs(training.index("mixed_precision")
                   - training.index("gradient_accumulation_steps")) <= 2

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
        # tooltip is a sentence.


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="AMP is CUDA-only; run via tools/gpu_turn.sh")
def test_a_real_cuda_training_run_uses_half_precision(tmp_path, capsys,
                                                      monkeypatch):
    """Instruction 236: the AMP path, driven on a GPU rather than reasoned.

    A real resnet18 from spaCR's own ``choose_model``, trained two epochs on
    random crops with ``mixed_precision=True``. The training forward pass
    must come out in float16 (autocast took effect), the weights must stay
    float32 and finite (the scaler stepped them), and a checkpoint must be
    written. Every other torch test runs on CPU, where this path is refused.
    """
    import os

    from spacr import deep_spacr, utils

    dtypes = []
    original = utils.choose_model

    def recording(*args, **kwargs):
        model = original(*args, **kwargs)
        model.register_forward_hook(
            lambda _m, _i, out: dtypes.append(
                out.dtype if torch.is_tensor(out) else None))
        return model

    monkeypatch.setattr("spacr.utils.choose_model", recording)
    generator = torch.Generator().manual_seed(0)
    batches = [(torch.rand(8, 3, 64, 64, generator=generator),
                torch.tensor([0, 1] * 4),
                [f"b{b}_f{i}.png" for i in range(8)]) for b in range(3)]
    src = tmp_path / "data"
    src.mkdir()
    dst = tmp_path / "model"
    dst.mkdir()

    model, path = deep_spacr.train_model(
        str(src), str(dst), "resnet18", batches, epochs=2,
        init_weights=False, num_classes=2, image_size=64, schedule=None,
        settings={"mixed_precision": True}, tensorboard=False,
        write_card=False)

    out = capsys.readouterr().out
    assert "amp=True: the forward pass" in out
    assert torch.float16 in dtypes
    params = list(model.parameters())
    assert params and all(p.dtype == torch.float32 for p in params)
    assert all(bool(torch.isfinite(p).all()) for p in params)
    assert next(iter(params)).device.type == "cuda"
    assert os.path.exists(path)


class _RecordingScaler:
    """A gradient scaler that multiplies the loss and writes down each step."""

    def __init__(self, factor=1.0):
        self.factor, self.steps, self.updates = factor, 0, 0

    def scale(self, loss):
        return loss * self.factor

    def is_enabled(self):
        return self.factor != 1.0

    def get_scale(self):
        return self.factor

    def step(self, optimizer):
        self.steps += 1
        optimizer.step()

    def update(self):
        self.updates += 1


class TestTheLoopUsesIt:
    def test_the_gradients_are_scaled(self):
        """float16 gradients underflow to zero without a scaler, and the
        model simply does not learn -- silently, which is worse than not
        having the feature."""
        from spacr.deep_spacr import _train_batch

        data = torch.rand(4, 3, generator=torch.Generator().manual_seed(0))
        target = torch.tensor([0, 1, 0, 1])

        def gradient(factor):
            torch.manual_seed(1)
            model = torch.nn.Linear(3, 2)
            _train_batch(model, data, target, scaler=_RecordingScaler(factor),
                         loss_fn=torch.nn.functional.cross_entropy,
                         mixed_precision=False, device=torch.device("cpu"),
                         head_dim=2)
            return model.weight.grad.clone()

        assert torch.allclose(gradient(8.0), 8.0 * gradient(1.0))

    def test_the_optimiser_only_steps_through_the_scaler(self, monkeypatch):
        """Each optimiser step goes through the scaler, and the leftover
        gradient flush at the end of an epoch is a second one: an unscaled
        step would move the weights on float16 gradients."""
        from spacr import deep_spacr

        scaler = _RecordingScaler()
        monkeypatch.setattr(deep_spacr, "_gradient_scaler",
                            lambda device, enabled: scaler)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        model = torch.nn.Sequential(torch.nn.Linear(4, 2))
        monkeypatch.setattr("spacr.utils.choose_model", lambda *a, **k: model)
        monkeypatch.setattr(
            "spacr.deep_spacr.evaluate_model_performance",
            lambda model, loader, epoch, **kw: ({
                "accuracy": 0.5, "Accuracy": 0.5, "neg_accuracy": 0.5,
                "pos_accuracy": 0.5, "prauc": 0.5, "optimal_threshold": 0.5,
                "f1_macro": 0.5, "loss": 0.7, "epoch": int(epoch)},
                [None, None]))
        monkeypatch.setattr("spacr.io._save_progress", lambda *a: None)
        generator = torch.Generator().manual_seed(0)
        loaders = [(torch.rand(4, 4, generator=generator),
                    torch.tensor([0, 1, 0, 1]), [f"b{n}_{i}.png" for i in range(4)])
                   for n in range(3)]
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            deep_spacr.train_model(
                folder, folder, "resnet18", loaders, epochs=1,
                optimizer_type="adam", num_classes=2, schedule=None,
                gradient_accumulation=True, gradient_accumulation_steps=2,
                intermedeate_save=False, tensorboard=False)
        assert (scaler.steps, scaler.updates) == (2, 2)

