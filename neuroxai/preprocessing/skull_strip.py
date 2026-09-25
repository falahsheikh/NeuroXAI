"""Remove the skull from T1-weighted MRI volumes with SynthStrip (Hoopes et al., 2022).

Usage: python -m neuroxai.preprocessing.skull_strip INPUT_DIR OUTPUT_DIR [--single-file FILE]

The script uses the freesurfer/synthstrip Docker image. If Docker is not available or fails,
it uses mri_synthstrip from a FreeSurfer installation. For each volume <name>.nii(.gz) it writes:

    <output>/skull_stripped_brains/<name>_stripped.nii.gz
    <output>/brain_masks/<name>_mask.nii.gz
    <output>/metadata/<subject>_metadata.json
    <output>/processing_log.json
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib

logger = logging.getLogger(__name__)
TIMEOUT_S = 600


def base_name(path):
    return re.sub(r"\.nii(\.gz)?$", "", Path(path).name)


def subject_id(path):
    """ADNI subject ID (for example 002_S_4225) from an ADNI file name, or else the base name."""
    parts = Path(path).name.split("_")
    if len(parts) >= 4 and parts[0] == "ADNI":
        return "_".join(parts[1:4])
    return base_name(path)


class SkullStripper:
    """Runs SynthStrip on NIfTI files and writes the brains, the masks and metadata."""

    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()  # Docker needs absolute paths for its volume mounts
        self.brains_dir = self.output_dir / "skull_stripped_brains"
        self.masks_dir = self.output_dir / "brain_masks"
        self.metadata_dir = self.output_dir / "metadata"
        for folder in (self.brains_dir, self.masks_dir, self.metadata_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.log = []

    def find_volumes(self):
        volumes = sorted([*self.input_dir.rglob("*.nii"), *self.input_dir.rglob("*.nii.gz")])
        logger.info("Found %d NIfTI files", len(volumes))
        return volumes

    def _commands(self, volume, brain, mask):
        """The Docker command, then the FreeSurfer command."""
        yield [
            "docker", "run", "--rm",
            "-v", f"{volume.parent}:/input",
            "-v", f"{self.brains_dir}:/output_brain",
            "-v", f"{self.masks_dir}:/output_mask",
            "freesurfer/synthstrip:latest",
            "-i", f"/input/{volume.name}", "-o", f"/output_brain/{brain.name}", "-m", f"/output_mask/{mask.name}",
        ]  # fmt: skip
        yield ["mri_synthstrip", "-i", str(volume), "-o", str(brain), "-m", str(mask)]

    def skull_strip(self, volume):
        """Return (brain file, mask file), or (None, None) if SynthStrip failed."""
        volume = Path(volume).resolve()
        brain = self.brains_dir / f"{base_name(volume)}_stripped.nii.gz"
        mask = self.masks_dir / f"{base_name(volume)}_mask.nii.gz"
        if brain.exists() and mask.exists():
            logger.info("Already processed: %s", volume.name)
            return brain, mask
        for command in self._commands(volume, brain, mask):
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_S)
            except FileNotFoundError:
                logger.warning("%s is not installed", command[0])
                continue
            except subprocess.TimeoutExpired:
                logger.warning("%s timed out on %s", command[0], volume.name)
                continue
            if result.returncode == 0 and brain.exists() and mask.exists():
                logger.info("Processed %s with %s", volume.name, command[0])
                return brain, mask
            logger.warning("%s failed on %s: %s", command[0], volume.name, result.stderr.strip())
        logger.error("Skull stripping failed for %s", volume.name)
        return None, None

    def process(self, volume):
        """Skull-strip one volume and write its metadata. Return True if it succeeded."""
        subject = subject_id(volume)
        brain, mask = self.skull_strip(volume)
        metadata = {
            "subject_id": subject,
            "original_file": str(volume),
            "processing_date": datetime.now().isoformat(timespec="seconds"),
            "skull_stripped_file": str(brain) if brain else None,
            "mask_file": str(mask) if mask else None,
            "processing_successful": brain is not None,
        }
        if brain is not None:
            image = nib.load(brain)
            metadata.update(
                dimensions=list(image.shape),
                voxel_size=[float(z) for z in image.header.get_zooms()[:3]],
                data_type=str(image.get_data_dtype()),
            )
        with open(self.metadata_dir / f"{subject}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        self.log.append(
            {key: metadata[key] for key in ("subject_id", "original_file", "processing_date", "processing_successful")}
        )
        return brain is not None

    def process_all(self):
        volumes = self.find_volumes()
        if not volumes:
            logger.error("No NIfTI files in %s", self.input_dir)
            return
        successful = sum(self.process(volume) for volume in volumes)
        summary = {"total_files": len(volumes), "successful": successful, "failed": len(volumes) - successful}
        with open(self.output_dir / "processing_log.json", "w") as f:
            json.dump({"summary": summary, "details": self.log}, f, indent=2)
        logger.info("Done: %d successful, %d failed", summary["successful"], summary["failed"])


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m neuroxai.preprocessing.skull_strip", description="Skull stripping with SynthStrip"
    )
    parser.add_argument("input_dir", help="folder with .nii or .nii.gz files (subfolders are searched)")
    parser.add_argument("output_dir", help="folder for the results")
    parser.add_argument("--single-file", help="process only this file")
    args = parser.parse_args(argv)
    stripper = SkullStripper(args.input_dir, args.output_dir)
    if args.single_file:
        if not Path(args.single_file).exists():
            parser.error(f"file not found: {args.single_file}")
        sys.exit(0 if stripper.process(Path(args.single_file)) else 1)
    stripper.process_all()


if __name__ == "__main__":
    main()
