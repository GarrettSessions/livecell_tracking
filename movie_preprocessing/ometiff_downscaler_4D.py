import numpy as np
import cv2
import tifffile
import os

def process_slice(slice_16bit, scale_factor):
    """Convert a 16-bit slice to 8-bit and downscale it."""
    # Convert to 8-bit
    slice_8bit = ((slice_16bit.astype(np.float32) / np.max(slice_16bit)) * 255).round().astype(np.uint8)
    
    # Downscale
    height, width = slice_8bit.shape
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    slice_resized = cv2.resize(slice_8bit, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return slice_resized

def save_individual_slices(input_file, output_folder, scale_factor, num_timepoints=None):
    with tifffile.TiffFile(input_file) as tif:
        # Extract metadata
        metadata = tif.ome_metadata
        images = tif.series[0]  # Assume the first series is the one we want to process
        T, C, Y, X = images.shape
        
        # If num_timepoints is not specified or larger than the total, process all timepoints
        if num_timepoints is None or num_timepoints > T:
            num_timepoints = T
        
        print(f"Starting processing: {num_timepoints * C} total slices to process.")
        
        for t in range(num_timepoints):
            for c in range(C):
                # Correctly access each slice based on timepoint and channel
                slice_index = t * C + c
                slice_16bit = images.pages[slice_index].asarray()
                
                # Construct the filename
                filename = f"slice_T{t}_C{c}.tif"
                full_path = os.path.join(output_folder, filename)
                
                # Check if the slice has already been saved
                if os.path.exists(full_path):
                    print(f"{filename} already exists. Skipping.")
                    continue  # Skip processing and saving this slice
                
                processed_slice = process_slice(slice_16bit, scale_factor)
                
                # Save the slice
                tifffile.imwrite(full_path, processed_slice)
                    
                print(f"Saved {filename}")
    
    print("Processing complete.")

# Example usage
input_file = r'Z:\Garrett\Livecell\030124_Lorenzo_livecell_TBHP_Dose_Curve\Well_B05\full\WellB05_ChannelCFP,YFP,mCherry_Seq0003 - Shading Correction - Stitched.ome.tif'
output_folder = r'Z:\Garrett\Livecell\030124_Lorenzo_livecell_TBHP_Dose_Curve\Well_B05\down\slices'
scale_factor = 0.5  # Adjust as needed
num_timepoints = 240  # Specify the number of timepoints to process, for testing or full processing

save_individual_slices(input_file, output_folder, scale_factor, num_timepoints)
