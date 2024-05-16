import numpy as np
import cv2
import tifffile
import os

def process_slice(slice_16bit, scale_factor):
    """Convert a 16-bit slice to 8-bit and downscale it."""
    slice_8bit = ((slice_16bit.astype(np.float32) / np.max(slice_16bit)) * 255).round().astype(np.uint8)
    height, width = slice_8bit.shape
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    slice_resized = cv2.resize(slice_8bit, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return slice_resized

def save_individual_slices(input_file, output_folder, scale_factor, num_timepoints=None):
    with tifffile.TiffFile(input_file) as tif:
        # Assuming the OME-TIFF has T, Y, X dimensions
        if 'ImageDescription' in tif.pages[0].tags:
            metadata = tif.pages[0].tags['ImageDescription'].value
        else:
            metadata = None
        total_timepoints = len(tif.pages)
        
        if num_timepoints is None or num_timepoints > total_timepoints:
            num_timepoints = total_timepoints
        
        print(f"Starting processing: {num_timepoints} timepoints to process.")
        
        for t in range(num_timepoints):
            # Process each slice one at a time to save memory
            slice_16bit = tif.pages[t].asarray()
            processed_slice = process_slice(slice_16bit, scale_factor)
            
            # Construct the filename
            filename = f"slice_T{t}_C0.tif"
            full_path = os.path.join(output_folder, filename)
            
            # Save the slice
            tifffile.imwrite(full_path, processed_slice)
            print(f"Saved {filename}")
    
    print("Processing complete.")

input_file = r'Z:\Garrett\Livecell\072723_50hr_20uM_TBHP\livecell_B02\down\WellB02_ChannelCFP_Seq0000 - Stitched.ome.tif'
output_folder = r'Z:\Garrett\Livecell\072723_50hr_20uM_TBHP\livecell_B02\down\slices'
scale_factor = 0.5  # Adjust as needed
num_timepoints = 188  # Specify the number of timepoints to process, for testing or full processing

save_individual_slices(input_file, output_folder, scale_factor, num_timepoints)