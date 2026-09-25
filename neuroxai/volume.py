"""MRI volumes and their slices in the three anatomical planes."""

import hashlib
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

AXIAL, CORONAL, SAGITTAL = 0, 1, 2
VIEWS = {"axial": AXIAL, "coronal": CORONAL, "sagittal": SAGITTAL}
PLANE_NAMES = {plane: name.capitalize() for name, plane in VIEWS.items()}

# For a view of each plane: the planes whose slice numbers are the (horizontal, vertical) pixel coordinates.
IN_PLANE_AXES = {AXIAL: (SAGITTAL, CORONAL), CORONAL: (SAGITTAL, AXIAL), SAGITTAL: (CORONAL, AXIAL)}

# Anatomical direction of each view axis, read from left to right and from bottom to top of the view.
# The axial view shows anterior at the top, so its vertical axis is inverted (see FLIPPED_VERTICAL).
AXIS_LABELS = {
    AXIAL: ("Right → Left", "Posterior → Anterior"),
    CORONAL: ("Right → Left", "Inferior → Superior"),
    SAGITTAL: ("Anterior → Posterior", "Inferior → Superior"),
}
FLIPPED_VERTICAL = {AXIAL}

FILE_TYPES = [("All supported", "*.nii *.nii.gz *.mhd"), ("NIfTI files", "*.nii *.nii.gz"), ("MetaImage", "*.mhd")]


@dataclass
class Volume:
    """A 3D image in LPS orientation.

    data has the shape (z, y, x): x increases toward the patient's left, y toward posterior and
    z toward superior. spacing is the voxel size in mm in the same (z, y, x) order. Thus
    data.shape[plane] is the number of slices in a plane, and spacing[plane] is the slice spacing.
    """

    data: np.ndarray
    spacing: tuple[float, float, float]
    path: str = ""

    @classmethod
    def load(cls, path):
        """Read a NIfTI or MetaImage file and reorient it to LPS.

        Without the reorientation, the planes and the anatomical labels would depend on how the file
        stores its axes.
        """
        image = sitk.DICOMOrient(sitk.ReadImage(str(path)), "LPS")
        data = sitk.GetArrayFromImage(image).astype(np.float32)
        return cls(data, tuple(float(s) for s in reversed(image.GetSpacing())), str(path))

    @property
    def shape(self):
        return self.data.shape

    def slice(self, plane, index):
        """2D image of one slice. Its columns are the horizontal axis and its rows the vertical axis of the view."""
        index = int(np.clip(index, 0, self.shape[plane] - 1))
        if plane == AXIAL:
            return self.data[index, :, :]
        if plane == CORONAL:
            return self.data[:, index, :]
        return self.data[:, :, index]

    def pixel_spacing(self, plane):
        """(horizontal, vertical) size in mm of one pixel of a slice in the plane."""
        z, y, x = self.spacing
        return {AXIAL: (x, y), CORONAL: (x, z), SAGITTAL: (y, z)}[plane]

    def aspect(self, plane):
        """Height-to-width ratio of one pixel of a slice in the plane, for Axes.set_aspect."""
        width, height = self.pixel_spacing(plane)
        return height / width

    def value_range(self):
        return float(self.data.min()), float(self.data.max())

    def checksum(self):
        """SHA-256 of the shape, the spacing and the voxel values. It does not change between runs."""
        digest = hashlib.sha256(repr((self.shape, self.spacing)).encode())
        digest.update(np.ascontiguousarray(self.data).tobytes())
        return digest.hexdigest()
