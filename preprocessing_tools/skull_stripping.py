#!/usr/bin/env python3
"""
ADNI Skull Stripping Script
============================
This script processes ADNI brain MRI data by:
1. Performing skull stripping using SynthStrip
2. Organizing skull-stripped brains and masks into separate folders
3. Creating metadata for processed files

Author: Falah Sheikh
"""

import os
import sys
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
import logging
from typing import List, Tuple, Optional
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ADNISkullStripper:
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the ADNI Skull Stripper
        
        Args:
            input_dir: Directory containing ADNI .nii files
            output_dir: Directory to save processed files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.skull_stripped_dir = self.output_dir / "skull_stripped_brains"
        self.brain_masks_dir = self.output_dir / "brain_masks"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.skull_stripped_dir, self.brain_masks_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.processing_log = []
    
    def find_nii_files(self) -> List[Path]:
        """Find all .nii and .nii.gz files in the input directory"""
        nii_files = []
        for pattern in ["**/*.nii", "**/*.nii.gz"]:
            nii_files.extend(self.input_dir.glob(pattern))
        
        logger.info(f"Found {len(nii_files)} NIfTI files")
        return nii_files
    
    def skull_strip(self, input_file: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Perform skull stripping using SynthStrip
        
        Args:
            input_file: Path to input .nii file
            
        Returns:
            Tuple of (skull_stripped_file_path, mask_file_path) or (None, None) if failed
        """
        # Generate output filenames
        base_name = input_file.stem.replace('.nii', '')  # Handle .nii.gz
        stripped_file = self.skull_stripped_dir / f"{base_name}_stripped.nii.gz"
        mask_file = self.brain_masks_dir / f"{base_name}_mask.nii.gz"
        
        # Skip if already processed
        if stripped_file.exists() and mask_file.exists():
            logger.info(f"Files already exist for: {base_name}")
            return stripped_file, mask_file
        
        try:
            # Try containerized version first (more reliable)
            docker_cmd = [
                "docker", "run", "--rm",
                "-v", f"{input_file.parent}:/input",
                "-v", f"{self.skull_stripped_dir}:/output_brain",
                "-v", f"{self.brain_masks_dir}:/output_mask",
                "freesurfer/synthstrip:latest",
                "-i", f"/input/{input_file.name}",
                "-o", f"/output_brain/{stripped_file.name}",
                "-m", f"/output_mask/{mask_file.name}"
            ]
            
            logger.info(f"Running SynthStrip (Docker) on {input_file.name}")
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                # Fallback to FreeSurfer installation
                logger.warning("Docker version failed, trying FreeSurfer installation")
                fs_cmd = [
                    "mri_synthstrip",
                    "-i", str(input_file),
                    "-o", str(stripped_file),
                    "-m", str(mask_file)
                ]
                result = subprocess.run(fs_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and stripped_file.exists() and mask_file.exists():
                logger.info(f"Successfully processed: {input_file.name}")
                return stripped_file, mask_file
            else:
                logger.error(f"Skull stripping failed for {input_file.name}: {result.stderr}")
                return None, None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Skull stripping timed out for {input_file.name}")
            return None, None
        except Exception as e:
            logger.error(f"Error during skull stripping {input_file.name}: {str(e)}")
            return None, None
    
    def create_metadata(self, subject_id: str, original_file: Path, 
                       skull_stripped_file: Optional[Path], mask_file: Optional[Path]) -> None:
        """Create metadata file for the processed subject"""
        
        try:
            metadata = {
                "subject_id": subject_id,
                "original_file": str(original_file),
                "processing_date": datetime.now().isoformat(),
                "skull_stripped_file": str(skull_stripped_file) if skull_stripped_file else None,
                "mask_file": str(mask_file) if mask_file else None,
                "processing_successful": skull_stripped_file is not None and mask_file is not None
            }
            
            # Add image information if processing was successful
            if skull_stripped_file and skull_stripped_file.exists():
                img = nib.load(skull_stripped_file)
                header = img.header
                metadata.update({
                    "original_dimensions": img.shape,
                    "voxel_size": list(header.get_zooms()[:3]),
                    "data_type": str(img.get_data_dtype())
                })
            
            metadata_file = self.metadata_dir / f"{subject_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error creating metadata for {subject_id}: {str(e)}")
    
    def extract_subject_id(self, file_path: Path) -> str:
        """Extract subject ID from ADNI filename"""
        filename = file_path.name
        # ADNI format: ADNI_XXX_S_XXXX_...
        parts = filename.split('_')
        if len(parts) >= 4 and parts[0] == 'ADNI':
            return f"{parts[1]}_{parts[2]}_{parts[3]}"
        else:
            # Fallback to filename without extension
            return file_path.stem.replace('.nii', '')
    
    def process_single_file(self, nii_file: Path) -> bool:
        """Process a single NIfTI file"""
        subject_id = self.extract_subject_id(nii_file)
        logger.info(f"Processing {subject_id}: {nii_file.name}")
        
        try:
            # Perform skull stripping
            skull_stripped_file, mask_file = self.skull_strip(nii_file)
            
            success = skull_stripped_file is not None and mask_file is not None
            
            # Create metadata
            self.create_metadata(subject_id, nii_file, skull_stripped_file, mask_file)
            
            # Log processing info
            log_entry = {
                "subject_id": subject_id,
                "original_file": str(nii_file),
                "processing_date": datetime.now().isoformat()
            }
            
            if success:
                log_entry.update({
                    "status": "success",
                    "skull_stripped_file": str(skull_stripped_file),
                    "mask_file": str(mask_file)
                })
                logger.info(f"Successfully processed {subject_id}")
            else:
                log_entry.update({
                    "status": "failed",
                    "error": "Skull stripping failed"
                })
                logger.error(f"Failed to process {subject_id}")
            
            self.processing_log.append(log_entry)
            return success
            
        except Exception as e:
            logger.error(f"Error processing {subject_id}: {str(e)}")
            self.processing_log.append({
                "subject_id": subject_id,
                "status": "failed",
                "error": str(e),
                "original_file": str(nii_file),
                "processing_date": datetime.now().isoformat()
            })
            return False
    
    def process_all(self) -> None:
        """Process all NIfTI files in the input directory"""
        nii_files = self.find_nii_files()
        
        if not nii_files:
            logger.error("No NIfTI files found in input directory")
            return
        
        successful = 0
        failed = 0
        
        for nii_file in nii_files:
            if self.process_single_file(nii_file):
                successful += 1
            else:
                failed += 1
        
        # Save processing log
        log_file = self.output_dir / "processing_log.json"
        with open(log_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_files": len(nii_files),
                    "successful": successful,
                    "failed": failed,
                    "processing_date": datetime.now().isoformat()
                },
                "details": self.processing_log
            }, f, indent=2)
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        logger.info(f"Processing log saved to: {log_file}")
        
        # Create dataset info
        self.create_dataset_info()
    
    def create_dataset_info(self) -> None:
        """Create information file for the processed dataset"""
        
        # Count processed files
        brain_files = list(self.skull_stripped_dir.glob("*.nii.gz"))
        mask_files = list(self.brain_masks_dir.glob("*.nii.gz"))
        
        dataset_info = {
            "dataset_info": {
                "total_skull_stripped_brains": len(brain_files),
                "total_brain_masks": len(mask_files),
                "file_format": "NIfTI (.nii.gz)",
                "processing_tool": "SynthStrip"
            },
            "directory_structure": {
                "skull_stripped_brains": str(self.skull_stripped_dir),
                "brain_masks": str(self.brain_masks_dir),
                "metadata": str(self.metadata_dir)
            },
            "file_naming_convention": {
                "skull_stripped": "[original_name]_stripped.nii.gz",
                "brain_mask": "[original_name]_mask.nii.gz",
                "metadata": "[subject_id]_metadata.json"
            },
            "usage_examples": {
                "load_brain": """
import nibabel as nib
brain = nib.load('path/to/brain_stripped.nii.gz')
brain_data = brain.get_fdata()
""",
                "load_mask": """
import nibabel as nib
mask = nib.load('path/to/brain_mask.nii.gz')
mask_data = mask.get_fdata()
""",
                "apply_mask": """
# Apply mask to original image
masked_brain = brain_data * mask_data
"""
            },
            "quality_control": {
                "recommended_checks": [
                    "Visual inspection of skull stripping quality",
                    "Check mask coverage of brain tissue",
                    "Verify no excessive brain tissue removal",
                    "Compare with original images"
                ]
            }
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Dataset info saved to: {info_file}")

def main():
    parser = argparse.ArgumentParser(description="Perform skull stripping on ADNI brain MRI data")
    parser.add_argument("input_dir", help="Directory containing ADNI .nii files")
    parser.add_argument("output_dir", help="Directory to save processed files")
    parser.add_argument("--single-file", help="Process only a single file")
    
    args = parser.parse_args()
    
    # Create skull stripper
    skull_stripper = ADNISkullStripper(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    if args.single_file:
        # Process single file
        single_file = Path(args.single_file)
        if not single_file.exists():
            logger.error(f"File not found: {single_file}")
            sys.exit(1)
        skull_stripper.process_single_file(single_file)
    else:
        # Process all files
        skull_stripper.process_all()

if __name__ == "__main__":
    main()