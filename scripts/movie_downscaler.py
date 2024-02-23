import numpy as np
import nd2reader
import os
import tifffile
import cv2

print("Import done")

# User-defined parameters
input_file = r'Z:\Garrett\Livecell\072723_50hr_20uM_TBHP\livecell_B02\data\WellB02_ChannelCFP_Seq0000 - Stitched.nd2'
output_folder = r'Z:\Garrett\Livecell\072723_50hr_20uM_TBHP\livecell_B02\data\downscaled'
channel_index = 0  # Specify the index of the channel you want to extract
scale_factor = 2  # Specify the factor by which you want to downscale the image

print("User Defined Parameters Accepted")

def read_nd2_frame(file_path, channel_index, time_index):
    with nd2reader.ND2Reader(file_path) as reader:
        # Select the desired channel and timepoint
        image = reader.get_frame_2D(c=channel_index, t=time_index)
    return image

def convert_to_8bit(image):
    # Convert to 8-bit depth
    max_val = np.max(image)
    scale_factor = 255.0 / max_val
    image_8bit = (image * scale_factor).astype(np.uint8)
    
    return image_8bit

def downscale_image(image, scale_factor):
    # Downscale the image using pooling
    downscaled_image = cv2.resize(image, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_AREA)
    
    return downscaled_image

def main():
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print("Output folder created:", output_folder)

    # Extract base name from input file
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    print("Base name extracted from input file:", base_name)

    with nd2reader.ND2Reader(input_file) as reader:
        # Get the number of timepoints in the ND2 file
        time_points = len(reader)
        print("Total timepoints found:", time_points)

        # Create an empty list to store images
        images = []

        for t in range(time_points):
            # Read ND2 file and extract the desired channel for the current timepoint
            image = read_nd2_frame(input_file, channel_index, t)
            print("Reading frame", t+1, "of", time_points)

            # Check if the image is empty
            if image.size == 0:
                print("Skipping empty frame: Consider this a failed run", t+1)
                continue

            # Convert to 8-bit depth
            image_8bit = convert_to_8bit(image)

            # Downscale the image
            downscaled_image = downscale_image(image_8bit, scale_factor)

            # Append the processed image to the list
            images.append(downscaled_image)
            print("Image processed and appended to list")

        # Save the list of images as a TIFF stack with the base name of the input file
        output_file = os.path.join(output_folder, f"{base_name}_downscaled.tif")
        tifffile.imwrite(output_file, np.array(images))
        print("Processed stack saved as TIFF:", output_file)

if __name__ == "__main__":
    main()
