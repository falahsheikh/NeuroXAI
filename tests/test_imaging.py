import numpy as np
from PIL import Image

from neuroxai.imaging import crop_and_resize_slice, find_brain_bbox, load_slice, save_slice_png


def ellipse_slice(shape=(256, 200), center=(120, 90), radii=(90, 60), seed=0):
    rows, cols = np.indices(shape)
    inside = ((rows - center[0]) / radii[0]) ** 2 + ((cols - center[1]) / radii[1]) ** 2 < 1
    return inside * np.random.default_rng(seed).uniform(40, 200, size=shape)


def test_bbox_contains_the_brain_with_padding():
    row_min, row_max, col_min, col_max = find_brain_bbox(ellipse_slice())
    assert (row_min, col_min) == (30 - 10 + 1, 30 - 10 + 1)
    assert (row_max, col_max) == (210 + 10, 150 + 10)
    assert find_brain_bbox(np.zeros((10, 20))) == (0, 10, 0, 20)


def test_crop_and_resize_keeps_the_aspect_ratio():
    canvas = crop_and_resize_slice(ellipse_slice())
    assert canvas.shape == (224, 224)
    rows = np.flatnonzero(canvas.any(axis=1))
    cols = np.flatnonzero(canvas.any(axis=0))
    height, width = np.ptp(rows) + 1, np.ptp(cols) + 1
    assert height > 190  # the taller axis fills the canvas, except for the padding
    np.testing.assert_allclose(width / height, 119 / 179, rtol=0.05)  # the brain is 179 x 119 pixels


def test_extracted_slices_are_used_as_in_training(tmp_path):
    path = tmp_path / "slice.png"
    save_slice_png(crop_and_resize_slice(ellipse_slice()), path)
    # Keras flow_from_dataframe read the training PNG files as RGB without changes.
    as_trained = np.asarray(Image.open(path).convert("RGB"))
    loaded = load_slice(path)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, as_trained)


def test_other_images_are_cropped_and_scaled(tmp_path):
    path = tmp_path / "raw.png"
    Image.fromarray((ellipse_slice() / 2).astype(np.uint8)).save(path)
    loaded = load_slice(path)
    assert loaded.shape == (224, 224, 3)
    assert loaded.max() == 255 and loaded.min() == 0
    assert (loaded[..., 0] == loaded[..., 2]).all()
