"""Preprocessing of 2D slices, shared by slice extraction and the Analysis Tool.

The model was trained on the PNG files that neuroxai.preprocessing.extract_slices makes:
each coronal slice is cropped to the brain, scaled into a zero-padded 224x224 canvas,
rotated so that superior is at the top, scaled to 0-255 and saved as a grayscale PNG.
"""

import cv2
import numpy as np
from matplotlib import image as mpimg
from PIL import Image
from scipy.ndimage import zoom
from skimage import measure, morphology

from .config import IMG_SIZE


def find_brain_bbox(slice_data, threshold_percentile=5, padding=10):
    """Bounding box (row_min, row_max, col_min, col_max) of the largest foreground region, with padding."""
    rows, cols = slice_data.shape
    if np.max(slice_data) == 0:
        return 0, rows, 0, cols
    threshold = np.percentile(slice_data[slice_data > 0], threshold_percentile)
    mask = morphology.remove_small_objects(slice_data > threshold, max_size=255)
    mask = morphology.closing(mask, morphology.disk(5))
    labels = measure.label(mask)
    if labels.max() == 0:
        return 0, rows, 0, cols
    min_row, min_col, max_row, max_col = max(measure.regionprops(labels), key=lambda r: r.area).bbox
    return (
        max(0, min_row - padding),
        min(rows, max_row + padding),
        max(0, min_col - padding),
        min(cols, max_col + padding),
    )


def crop_and_resize_slice(slice_data, target_size=IMG_SIZE):
    """Crop to the brain and scale it into a zero-padded target_size canvas, keeping the aspect ratio."""
    min_row, max_row, min_col, max_col = find_brain_bbox(slice_data)
    cropped = slice_data[min_row:max_row, min_col:max_col]
    if cropped.size == 0:
        return np.zeros(target_size, dtype=slice_data.dtype)
    scale = min(target_size[0] / cropped.shape[0], target_size[1] / cropped.shape[1])
    new_shape = (int(cropped.shape[0] * scale), int(cropped.shape[1] * scale))
    resized = zoom(cropped, (new_shape[0] / cropped.shape[0], new_shape[1] / cropped.shape[1]), order=1)
    canvas = np.zeros(target_size, dtype=resized.dtype)
    top, left = (target_size[0] - new_shape[0]) // 2, (target_size[1] - new_shape[1]) // 2
    canvas[top : top + new_shape[0], left : left + new_shape[1]] = resized
    return canvas


def to_uint8(slice_data):
    """Scale the intensities linearly so that the minimum is 0 and the maximum is 255."""
    return cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def save_slice_png(slice_data, path):
    """Rotate a (left-right, inferior-superior) slice for display, scale it to 0-255 and save it as a PNG."""
    mpimg.imsave(path, to_uint8(np.rot90(slice_data, k=1)), cmap="gray")


def load_slice(path, size=IMG_SIZE):
    """Load a coronal slice as an RGB uint8 image of the model input size.

    An image that already has the input size is an extracted slice, so it is used without change,
    as in training. Another image is cropped to the brain, scaled into the canvas and scaled to 0-255.
    """
    gray = np.asarray(Image.open(path).convert("L"))
    if gray.shape != tuple(size):
        gray = crop_and_resize_slice(gray.astype(np.float32), tuple(size))
        gray = to_uint8(gray) if gray.max() > 0 else gray.astype(np.uint8)
    return np.repeat(gray[..., np.newaxis], 3, axis=-1)
