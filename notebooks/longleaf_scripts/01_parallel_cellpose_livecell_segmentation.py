# # Cellpose segmentation
# Notebook showing how to segment cells using Cellpose.
import os
import sys
from skimage.io import imsave
import tensorflow as tf
import cellpose
from cellpose import models
from cellpose import utils

sys.path.append(r'/proj/purvislb/users/Garrett/libraries/')
import input_functions as inp_f

print("Packages loaded")

# Parse input arguments
FIRST_FRAME = int(sys.argv[1])
LAST_FRAME = int(sys.argv[2])

print(FIRST_FRAME)
print(LAST_FRAME)

info_file_path = r'/proj/purvislb/users/Garrett/072723_50hr_20uM_TBHP/info_B02.txt'

# load cellpose model - Longleaf can only use CPU at this time
model = models.Cellpose(gpu=False, model_type='cyto')
print("Model Loaded")

# read the file
info_file = open(info_file_path, 'r')
info_lines = info_file.readlines()
info_file.close()
print("Info file read")

# read info about the data frame
exp_dir,df_name = inp_f.read_df_info(info_lines)

# get info about the channels
channel_list = inp_f.read_channels(info_lines)
print("Dataframe and channel info read")

# read the movie to segment
file_name = [x['file_name'] for x in channel_list if x['tracking']][0]
c = [x['channel_in_file'] for x in channel_list if x['tracking']][0]
im_path = os.path.join(exp_dir,'data',file_name)
im_path

im = inp_f.open_movie(im_path,c)
print("Movie read")

# check how many timepoints are there in the file
frames_num = im.shape[0]
print(f'Total frame number: {frames_num}')

    # Loop for segmentation within the specified frame range
for i in range(FIRST_FRAME, min(LAST_FRAME + 1, frames_num)):
    print(f"Segmenting frame {i}/{frames_num}...")
    # get an image
    im_frame = im[i,:,:]

    # segment the right plane
    labels, _, _, _ = model.eval(im_frame, diameter=30, channels=[0,0])

    # save segmentation
    save_dir = os.path.join(exp_dir,'segmentation')
    save_file = file_name[:-4]+f'_{str(i).zfill(3)}_label.png'
    save_path = os.path.join(save_dir,save_file)
    print(f"Saving segmentation to {save_path}...")
    imsave(save_path,labels.astype('uint16'))