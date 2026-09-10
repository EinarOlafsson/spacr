"""Inference, driven end to end on a model trained in the same run.

Instruction 236 B5. Two defects, both found by pointing `apply_model` at a
real folder and reading what it said:

1. `print(f'Loading dataset in {src} with {len(src)} images')` counts the
   CHARACTERS IN THE PATH. So did the line after it. A run over a folder
   whose path happened to be 98 characters long announced "Loading dataset
   ... with 98 images", then "Loaded 98 images", and returned an empty
   frame. The number was plausible, it was printed twice, and it had
   nothing to do with the data.

2. A folder with no images to score produced a frame with the right columns
   and no rows. Downstream that reads as "the model scored nothing", which
   is a result, rather than "there was nothing to score", which is a
   mistake in the path.

`apply_model` reads the pictures lying DIRECTLY in the folder it is given
-- it does not walk class subfolders, because inference has no classes to
walk -- and pointing it at a dataset root is the easy way to hit both.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")

from PIL import Image                                        # noqa: E402


def _crops(where, how_many, seed=0):
    where.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(how_many):
        pixels = (rng.random((32, 32, 3)) * 255).astype("uint8")
        Image.fromarray(pixels).save(
            where / f"plate1_A{index + 1:02d}_f1_o1.png")
    return where


@pytest.fixture(scope="module")
def a_model(tmp_path_factory):
    """A three-class head, saved the way spaCR saves one."""
    from spacr.utils import choose_model

    model = choose_model("mobilenet_v3_small", torch.device("cpu"),
                         init_weights=False, num_classes=3, height=32)
    where = tmp_path_factory.mktemp("model") / "three_classes.pth"
    torch.save(model, where)
    return str(where)


class TestTheCount:
    def test_it_reports_the_images_it_found(self, tmp_path, a_model, capsys):
        """THE DEFECT: this was the length of the path string."""
        from spacr.deep_spacr import apply_model

        folder = _crops(tmp_path / "crops", 7)
        apply_model(str(folder), a_model, image_size=32, batch_size=4,
                    n_jobs=0)
        said = capsys.readouterr().out
        assert "with 7 images" in said, said[:400]

    def test_the_count_does_not_follow_the_path_length(self, tmp_path,
                                                       a_model, capsys):
        """The same seven crops under a much longer path. If the number
        moves, it is still counting the wrong thing."""
        from spacr.deep_spacr import apply_model

        deep = tmp_path
        for part in ("a_rather_long_directory_name", "and_another_one",
                     "and_a_third_for_good_measure"):
            deep = deep / part
        folder = _crops(deep / "crops", 7)
        apply_model(str(folder), a_model, image_size=32, batch_size=4,
                    n_jobs=0)
        assert "with 7 images" in capsys.readouterr().out


class TestNothingToScore:
    def test_an_empty_folder_is_refused_rather_than_returned(self, tmp_path,
                                                             a_model):
        from spacr.deep_spacr import apply_model

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No images to score"):
            apply_model(str(empty), a_model, image_size=32, n_jobs=0)

    def test_a_dataset_root_says_what_it_reads(self, tmp_path, a_model):
        """The easy mistake: pointing it at `test/` rather than at
        `test/<class>/`. The message has to name the difference, or the
        user reads it as "my crops are broken"."""
        from spacr.deep_spacr import apply_model

        root = tmp_path / "dataset" / "test"
        for name in ("c1", "c2", "c3"):
            _crops(root / name, 3)
        with pytest.raises(ValueError) as raised:
            apply_model(str(root), a_model, image_size=32, n_jobs=0)
        assert "subfolder" in str(raised.value)

    def test_the_class_folders_themselves_score(self, tmp_path, a_model):
        """The refusal must be about the ROOT, not about the crops."""
        from spacr.deep_spacr import apply_model

        root = tmp_path / "dataset" / "test"
        _crops(root / "c1", 5)
        scored = apply_model(str(root / "c1"), a_model, image_size=32,
                             batch_size=4, n_jobs=0)
        assert len(scored) == 5


class TestWhatComesBack:
    def test_three_classes_get_a_column_each(self, tmp_path, a_model):
        """A three-class model reporting one `pred` column would have lost
        two thirds of what it computed."""
        from spacr.deep_spacr import apply_model

        folder = _crops(tmp_path / "crops", 6)
        scored = apply_model(str(folder), a_model, image_size=32,
                             batch_size=4, n_jobs=0)
        assert len(scored) == 6
        for column in ("path", "pred", "predicted_label",
                       "prob_class_0", "prob_class_1", "prob_class_2"):
            assert column in scored.columns, column

    def test_every_row_is_scored(self, tmp_path, a_model):
        from spacr.deep_spacr import apply_model

        folder = _crops(tmp_path / "crops", 6)
        scored = apply_model(str(folder), a_model, image_size=32,
                             batch_size=4, n_jobs=0)
        assert scored["pred"].notna().all()
        assert scored["predicted_label"].isin([0, 1, 2]).all()

    def test_the_probabilities_sum_to_one(self, tmp_path, a_model):
        """A softmax that did not would mean the columns are logits wearing
        a probability's name."""
        from spacr.deep_spacr import apply_model

        folder = _crops(tmp_path / "crops", 6)
        scored = apply_model(str(folder), a_model, image_size=32,
                             batch_size=4, n_jobs=0)
        totals = scored[["prob_class_0", "prob_class_1",
                         "prob_class_2"]].sum(axis=1)
        assert np.allclose(totals.to_numpy(), 1.0, atol=1e-5)
