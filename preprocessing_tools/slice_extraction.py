"""Extract 224x224 coronal PNG slices from skull-stripped T1-weighted MRI volumes.

Input:  <input-dir>/<class>/**/*.nii.gz (or .nii), with <class> in cn, emci, lmci.
Output: <output-dir>/<class>/<class>_<volume>_s<index>.png.

Each volume is reoriented to RAS. The script takes 30 contiguous coronal slices centered on the
middle coronal index, crops each slice to the brain bounding box, and pads it to 224x224 without
distortion. A trailing "_stripped" in a volume name (added by skull_stripping.py) is removed.
"""

import argparse
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from skimage import measure, morphology

CLASSES = ("cn", "emci", "lmci")
IMG_SIZE = (224, 224)


def volume_name(path):
    name = re.sub(r"\.nii(\.gz)?$", "", Path(path).name)
    return re.sub(r"_stripped$", "", name)


def find_brain_bbox(slice_data, threshold_percentile=5):
    """Bounding box (row_min, row_max, col_min, col_max) of the largest foreground region, with 10 px padding."""
    if np.max(slice_data) == 0:
        return 0, slice_data.shape[0], 0, slice_data.shape[1]
    threshold = np.percentile(slice_data[slice_data > 0], threshold_percentile)
    mask = morphology.remove_small_objects(slice_data > threshold, min_size=256)
    mask = morphology.binary_closing(mask, morphology.disk(5))
    labels = measure.label(mask)
    if labels.max() == 0:
        return 0, slice_data.shape[0], 0, slice_data.shape[1]
    min_row, min_col, max_row, max_col = max(measure.regionprops(labels), key=lambda r: r.area).bbox
    pad = 10
    return (
        max(0, min_row - pad),
        min(slice_data.shape[0], max_row + pad),
        max(0, min_col - pad),
        min(slice_data.shape[1], max_col + pad),
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


def save_slice_as_png(slice_data, png_path):
    """Rotate for display, scale intensities to 0-255 and save as a grayscale PNG."""
    rotated = np.rot90(slice_data, k=1)
    normalized = cv2.normalize(rotated, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    plt.imsave(png_path, normalized, cmap="gray")


def extract_class(input_dir, output_dir, class_name, num_slices=30, per_class_cap=1000):
    """Write slices for one class and return (number of PNG files, number of volumes used)."""
    out = Path(output_dir) / class_name
    out.mkdir(parents=True, exist_ok=True)
    volumes = sorted([*(Path(input_dir) / class_name).rglob("*.nii.gz"), *(Path(input_dir) / class_name).rglob("*.nii")])
    saved, used = 0, set()
    for path in volumes:
        if saved >= per_class_cap:
            break
        try:
            data = nib.as_closest_canonical(nib.load(path)).get_fdata()
        except Exception as e:  # unreadable volumes are skipped, as in the original run
            print(f"  skipped {path}: {e}")
            continue
        mid = data.shape[1] // 2
        for idx in range(max(0, mid - num_slices // 2), min(data.shape[1], mid + num_slices // 2)):
            if saved >= per_class_cap:
                break
            coronal = crop_and_resize_slice(data[:, idx, :])
            if np.sum(coronal) > 0:
                save_slice_as_png(coronal, out / f"{class_name}_{volume_name(path)}_s{idx:03d}.png")
                saved += 1
                used.add(path)
    return saved, len(used)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, help="folder with cn/, emci/ and lmci/ subfolders of skull-stripped volumes")
    p.add_argument("--output-dir", required=True, help="folder for the PNG slices")
    p.add_argument("--num-slices", type=int, default=30, help="coronal slices per volume")
    p.add_argument("--per-class-cap", type=int, default=1000, help="maximum number of slices per class")
    args = p.parse_args(argv)
    for class_name in CLASSES:
        saved, n_volumes = extract_class(args.input_dir, args.output_dir, class_name, args.num_slices, args.per_class_cap)
        print(f"{class_name}: {saved} slices from {n_volumes} volumes")


if __name__ == "__main__":
    main()
