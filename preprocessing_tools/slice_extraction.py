import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from skimage import measure, morphology
import cv2

def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorients a NIfTI image to the RAS (Right-Anterior-Superior) orientation."""
    return nib.as_closest_canonical(img)

def find_brain_bbox(slice_data, threshold_percentile=5):
    """
    Finds the bounding box of the brain in a 2D slice to crop out empty space.
    """
    if np.max(slice_data) == 0:
        return 0, slice_data.shape[0], 0, slice_data.shape[1]
        
    threshold = np.percentile(slice_data[slice_data > 0], threshold_percentile)
    binary_mask = slice_data > threshold
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=256)
    binary_mask = morphology.binary_closing(binary_mask, morphology.disk(5))
    
    labeled_mask = measure.label(binary_mask)
    if labeled_mask.max() == 0:
        return 0, slice_data.shape[0], 0, slice_data.shape[1]
        
    props = measure.regionprops(labeled_mask)
    largest_region = max(props, key=lambda x: x.area)
    min_row, min_col, max_row, max_col = largest_region.bbox
    
    padding = 10
    return (max(0, min_row - padding), min(slice_data.shape[0], max_row + padding),
            max(0, min_col - padding), min(slice_data.shape[1], max_col + padding))

def crop_and_resize_slice(slice_data, target_size=(224, 224)):
    """Crops and resizes a 2D slice to a target size."""
    min_row, max_row, min_col, max_col = find_brain_bbox(slice_data)
    cropped = slice_data[min_row:max_row, min_col:max_col]
    
    if cropped.size == 0:
        return np.zeros(target_size, dtype=slice_data.dtype)
        
    scale = min(target_size[0] / cropped.shape[0], target_size[1] / cropped.shape[1])
    new_shape = (int(cropped.shape[0] * scale), int(cropped.shape[1] * scale))
    resized = zoom(cropped, (new_shape[0] / cropped.shape[0], new_shape[1] / cropped.shape[1]), order=1)
    
    final_image = np.zeros(target_size, dtype=resized.dtype)
    start_h = (target_size[0] - new_shape[0]) // 2
    start_w = (target_size[1] - new_shape[1]) // 2
    final_image[start_h:start_h + new_shape[0], start_w:start_w + new_shape[1]] = resized
    
    return final_image

def save_slice_as_png(slice_data, png_path):
    """Saves a 2D slice as a normalized PNG image."""
    rotated = np.rot90(slice_data, k=1)
    normalized = cv2.normalize(rotated, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    plt.imsave(png_path, normalized, cmap='gray')

def extract_coronal_slices(base_dir, class_name, slice_step, target_count, target_size):
    """
    Extracts, saves coronal slices, and counts the number of unique subjects used.
    """
    print(f"-> Processing class: {class_name.upper()}")
    
    input_dir = base_dir / "stripped_data" / class_name
    output_dir = base_dir / class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_saved = 0
    # Use a set to automatically handle unique subject IDs
    subjects_used = set()
    nii_files = sorted(list(input_dir.rglob("*.nii.gz")))
    
    for nii_file in nii_files:
        if files_saved >= target_count:
            print(f"   Target of {target_count} files reached for class '{class_name}'.")
            break
        try:
            img = nib.load(nii_file)
            data = reorient_to_ras(img).get_fdata()
            # The base filename without extensions serves as the subject/scan ID
            base_name = nii_file.stem.replace(".nii", "")
            
            # Medial temporal region: 30 slices around the center
            mid = data.shape[1] // 2
            start_slice = max(0, mid - 15)
            end_slice = min(data.shape[1], mid + 15)
            
            for idx in range(start_slice, end_slice, slice_step):
                if files_saved >= target_count:
                    break
                    
                coronal_slice = crop_and_resize_slice(data[:, idx, :], target_size)
                
                if np.sum(coronal_slice) > 0:
                    filename = f"{class_name}_{base_name}_s{idx:03d}.png"
                    png_path = output_dir / filename
                    save_slice_as_png(coronal_slice, png_path)
                    files_saved += 1
                    # Add the subject ID to our set. Duplicates are ignored.
                    subjects_used.add(base_name)
                    
        except Exception as e:
            print(f"    Could not process {nii_file.name}: {e}")
            
    print(f" Finished '{class_name}'. Total PNGs: {files_saved}. Unique subjects used: {len(subjects_used)}")
    return len(subjects_used)

def main():
    """Main function to configure and run the dataset extraction process."""
    base_dir = Path.home() / "Desktop/research2025"
    class_names = ['cn', 'emci', 'lmci']
    
    slice_step = 1
    target_files_per_class = 1000
    target_image_size = (224, 224)
    subject_counts = {}
    
    print("Starting Coronal Slice Extraction...")
    
    for class_name in class_names:
        num_subjects = extract_coronal_slices(
            base_dir=base_dir,
            class_name=class_name,
            slice_step=slice_step,
            target_count=target_files_per_class,
            target_size=target_image_size
        )
        subject_counts[class_name] = num_subjects
        
    print("\n" + "="*40)
    print(" DATASET SUMMARY")
    print("="*40)
    for class_name, count in subject_counts.items():
        print(f"  - {class_name.upper()}: {count} unique subjects")
    print("="*40)
    print("\n All classes processed successfully!")

if __name__ == "__main__":
    main()