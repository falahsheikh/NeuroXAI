import tkinter as tk

import numpy as np
import pytest
import SimpleITK as sitk

# Voxel size (x, y, z) in mm and size (x, y, z) in voxels of the test volume. Different values
# for each axis make errors in the order of the axes visible.
SPACING = (1.2, 1.0, 1.5)
SIZE = (20, 30, 40)


def itk_index_values(size=SIZE):
    """Array (z, y, x) whose value encodes the ITK index (i, j, k): i + 100 j + 10000 k."""
    k, j, i = np.indices(size[::-1])
    return (i + 100 * j + 10000 * k).astype(np.float32)


@pytest.fixture
def ras_volume_file(tmp_path):
    """NIfTI volume stored in RAS orientation, as ADNI files often are (ITK direction diag(-1, -1, 1))."""
    image = sitk.GetImageFromArray(itk_index_values())
    image.SetSpacing(SPACING)
    image.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    path = tmp_path / "ras.nii.gz"
    sitk.WriteImage(image, str(path))
    return path


@pytest.fixture
def tk_root():
    """A Tk root window that the user cannot see. Tests that need it are skipped without a display."""
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    root.attributes("-alpha", 0.0)
    yield root
    root.destroy()
