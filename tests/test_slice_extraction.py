import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import slice_extraction


def write_volume(path, seed):
    rng = np.random.default_rng(seed)
    grid = np.indices((64, 80, 64)).astype(float)
    center = np.array([32, 40, 32])[:, None, None, None]
    brain = (((grid - center) / np.array([24, 30, 26])[:, None, None, None]) ** 2).sum(axis=0) < 1
    volume = brain * rng.uniform(50, 200, size=brain.shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype("float32"), np.eye(4)), path)


def test_names_count_and_size(tmp_path):
    for i, name in enumerate(["cn", "emci", "lmci"]):
        write_volume(tmp_path / "in" / name / f"ADNI_002_S_000{i}_stripped.nii.gz", seed=i)

    slice_extraction.main(["--input-dir", str(tmp_path / "in"), "--output-dir", str(tmp_path / "out"), "--per-class-cap", "25"])

    files = sorted(p.name for p in (tmp_path / "out" / "cn").iterdir())
    # The middle coronal index is 40, so the 30-slice window is 25-54; the cap keeps the first 25.
    assert files == [f"cn_ADNI_002_S_0000_s{i:03d}.png" for i in range(25, 50)]
    assert plt.imread(tmp_path / "out" / "cn" / files[0]).shape[:2] == (224, 224)
