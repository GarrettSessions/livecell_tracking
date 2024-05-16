import os
import numpy as np
import tifffile

def reassemble_movie_per_channel(input_folder, output_folder):
    filenames = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    timepoints_channels = [(int(f.split('_T')[1].split('_')[0]), int(f.split('_C')[1].split('.tif')[0])) for f in filenames]
    max_timepoint = max(tc[0] for tc in timepoints_channels) + 1
    max_channel = max(tc[1] for tc in timepoints_channels) + 1
    
    filenames_sorted = sorted(filenames, key=lambda x: (int(x.split('_T')[1].split('_')[0]), int(x.split('_C')[1].split('.tif')[0])))
    first_slice = tifffile.imread(os.path.join(input_folder, filenames_sorted[0]))
    Y, X = first_slice.shape
    
    for channel in range(max_channel):
        # Initialize movie array for the current channel
        movie = np.empty((max_timepoint, Y, X), dtype=first_slice.dtype)
        
        for filename in filenames_sorted:
            timepoint, file_channel = (int(filename.split('_T')[1].split('_')[0]), int(filename.split('_C')[1].split('.tif')[0]))
            # Only process files for the current channel
            if file_channel == channel:
                slice_path = os.path.join(input_folder, filename)
                movie_slice = tifffile.imread(slice_path)
                movie[timepoint] = movie_slice
        
        output_file = os.path.join(output_folder, f"channel_{channel}_output_movie.ome.tif")
        print(f"Saving assembled movie for channel {channel} to {output_file}")
        # Save as OME-TIFF without specifying ImageJ compatibility
        tifffile.imwrite(output_file, movie, metadata={'axes': 'TYX'})

        print(f"Reassembly for channel {channel} complete.")

# Adjust the paths as necessary
input_folder = r'Z:\Garrett\Livecell\030124_Lorenzo_livecell_TBHP_Dose_Curve\Well_B03\down\slices'
output_folder = r'Z:\Garrett\Livecell\030124_Lorenzo_livecell_TBHP_Dose_Curve\Well_B03\down'

reassemble_movie_per_channel(input_folder, output_folder)
