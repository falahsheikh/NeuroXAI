import numpy as np
import pytest
import SimpleITK as sitk
from conftest import SIZE, SPACING, itk_index_values

from neuroxai.volume import AXIAL, CORONAL, SAGITTAL, Volume


def test_a_ras_volume_is_reoriented_to_lps(ras_volume_file):
    volume = Volume.load(ras_volume_file)
    nx, ny, nz = SIZE
    assert volume.shape == (nz, ny, nx)
    assert volume.spacing == pytest.approx(SPACING[::-1])
    # In the file, x and y increase toward right and anterior. In LPS they increase toward left and posterior.
    expected = itk_index_values()[:, ::-1, ::-1]
    np.testing.assert_array_equal(volume.data, expected)


def test_the_axis_order_of_the_file_does_not_change_the_volume(ras_volume_file, tmp_path):
    permuted = sitk.PermuteAxes(sitk.ReadImage(str(ras_volume_file)), [2, 0, 1])
    sitk.WriteImage(permuted, str(tmp_path / "permuted.nii.gz"))
    a, b = Volume.load(ras_volume_file), Volume.load(tmp_path / "permuted.nii.gz")
    np.testing.assert_array_equal(a.data, b.data)
    assert a.spacing == pytest.approx(b.spacing)


def test_slices_spacing_and_aspect():
    sx, sy, sz = SPACING
    volume = Volume(np.zeros(SIZE[::-1], dtype=np.float32), (sz, sy, sx))
    nx, ny, nz = SIZE
    assert volume.slice(AXIAL, 5).shape == (ny, nx)
    assert volume.slice(CORONAL, 5).shape == (nz, nx)
    assert volume.slice(SAGITTAL, 5).shape == (nz, ny)
    assert volume.slice(AXIAL, 10_000).shape == (ny, nx)  # the index is clipped
    assert volume.pixel_spacing(AXIAL) == (sx, sy)
    assert volume.pixel_spacing(CORONAL) == (sx, sz)
    assert volume.pixel_spacing(SAGITTAL) == (sy, sz)
    assert volume.aspect(AXIAL) == pytest.approx(sy / sx)
    assert volume.aspect(SAGITTAL) == pytest.approx(sz / sy)


def test_checksum_is_stable_and_detects_changes():
    data = np.random.default_rng(0).random((4, 5, 6)).astype(np.float32)
    a = Volume(data, (1.0, 1.0, 1.0))
    assert a.checksum() == Volume(data.copy(), (1.0, 1.0, 1.0)).checksum()
    changed = data.copy()
    changed[0, 0, 0] += 1
    assert a.checksum() != Volume(changed, (1.0, 1.0, 1.0)).checksum()
    assert a.checksum() != Volume(data, (1.0, 1.0, 2.0)).checksum()
