"""Extract 224x224 coronal PNG slices from skull-stripped T1-weighted MRI volumes.

Usage: python -m neuroxai.preprocessing.extract_slices --input-dir VOLUMES --output-dir SLICES

Input:  <input-dir>/<class>/**/*.nii.gz (or .nii), with <class> in cn, emci, lmci.
Output: <output-dir>/<class>/<class>_<volume>_s<index>.png

Each volume is reoriented to RAS. The script takes 30 contiguous coronal slices centered on the
middle coronal index, crops each slice to the brain bounding box, and pads it to 224x224 without
distortion. A trailing "_stripped" in a volume name (added by skull_strip) is removed.
The eAlz repository (github.com/falahsheikh/eAlz) uses the same steps.
"""

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np

from ..config import CLASS_NAMES
from ..imaging import crop_and_resize_slice, save_slice_png

CLASSES = tuple(name.lower() for name in CLASS_NAMES)


def volume_name(path):
    name = re.sub(r"\.nii(\.gz)?$", "", Path(path).name)
    return re.sub(r"_stripped$", "", name)


def extract_class(input_dir, output_dir, class_name, num_slices=30, per_class_cap=1000):
    """Write the slices of one class and return (number of PNG files, number of volumes used)."""
    out = Path(output_dir) / class_name
    out.mkdir(parents=True, exist_ok=True)
    class_dir = Path(input_dir) / class_name
    volumes = sorted([*class_dir.rglob("*.nii.gz"), *class_dir.rglob("*.nii")])
    saved, used = 0, set()
    for path in volumes:
        if saved >= per_class_cap:
            break
        try:
            data = nib.as_closest_canonical(nib.load(path)).get_fdata()
        except Exception as e:  # nibabel raises many types of errors; skip unreadable volumes, as in the original run
            print(f"  skipped {path}: {e}")
            continue
        mid = data.shape[1] // 2
        for index in range(max(0, mid - num_slices // 2), min(data.shape[1], mid + num_slices // 2)):
            if saved >= per_class_cap:
                break
            coronal = crop_and_resize_slice(data[:, index, :])
            if np.sum(coronal) > 0:
                save_slice_png(coronal, out / f"{class_name}_{volume_name(path)}_s{index:03d}.png")
                saved += 1
                used.add(path)
    return saved, len(used)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m neuroxai.preprocessing.extract_slices",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, help="folder with cn/, emci/ and lmci/ subfolders of volumes")
    parser.add_argument("--output-dir", required=True, help="folder for the PNG slices")
    parser.add_argument("--num-slices", type=int, default=30, help="coronal slices per volume")
    parser.add_argument("--per-class-cap", type=int, default=1000, help="maximum number of slices per class")
    args = parser.parse_args(argv)
    for class_name in CLASSES:
        saved, n_volumes = extract_class(
            args.input_dir, args.output_dir, class_name, args.num_slices, args.per_class_cap
        )
        print(f"{class_name}: {saved} slices from {n_volumes} volumes")


if __name__ == "__main__":
    main()
