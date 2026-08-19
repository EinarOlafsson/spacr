"""Every crop the montage opens names itself in the console.

Asked for 2026-08-19 while working out whether the montage was looking in the
right place: "can you have the software print the path for each cell it loads
in the console".
"""
import pytest


@pytest.fixture()
def crop(tmp_path):
    import numpy as np
    from PIL import Image

    folder = tmp_path / "plate1" / "data" / "w" / "cell_png"
    folder.mkdir(parents=True)
    path = folder / "plate1_A10_4_143.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype="uint8")).save(path)
    return tmp_path / "plate1", str(path)


def test_it_prints_the_path_it_opened(crop, capsys):
    from spacr.crops import PngCropSource, forget_announced_crops, say_crop_paths

    root, path = crop
    say_crop_paths(True)
    forget_announced_crops()

    PngCropSource(root=str(root)).get(path)

    assert path in capsys.readouterr().out


def test_the_same_path_is_not_printed_twice(crop, capsys):
    """A montage redrawn five times must not print three hundred lines five
    times."""
    from spacr.crops import PngCropSource, forget_announced_crops, say_crop_paths

    root, path = crop
    say_crop_paths(True)
    forget_announced_crops()
    source = PngCropSource(root=str(root))

    source.get(path)
    capsys.readouterr()
    source.get(path)

    assert capsys.readouterr().out == ""


def test_a_new_montage_announces_them_again(crop, capsys):
    from spacr.crops import PngCropSource, forget_announced_crops, say_crop_paths

    root, path = crop
    say_crop_paths(True)
    forget_announced_crops()
    source = PngCropSource(root=str(root))
    source.get(path)
    capsys.readouterr()

    forget_announced_crops()
    source.get(path)

    assert path in capsys.readouterr().out


def test_it_can_be_turned_off(crop, capsys):
    from spacr.crops import PngCropSource, forget_announced_crops, say_crop_paths

    root, path = crop
    forget_announced_crops()
    say_crop_paths(False)
    try:
        PngCropSource(root=str(root)).get(path)
        assert capsys.readouterr().out == ""
    finally:
        say_crop_paths(True)


def test_a_path_that_is_not_there_says_so(capsys):
    from spacr.crops import _say_which_crop, forget_announced_crops, say_crop_paths

    say_crop_paths(True)
    forget_announced_crops()

    _say_which_crop("/gone/plate1/data/w/cell_png/x.png")

    assert "NOT ON DISK" in capsys.readouterr().out
