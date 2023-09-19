#!/usr/bin/env python
# coding: utf-8

# # Cellpose segmentation

import os
import sys
from skimage.io import imsave
from cellpose import models

sys.path.append('../libraries')
import input_functions as inp_f

# get the path to the info file as an argument
info_file_path = sys.argv[1]

# load cellpose model
model = models.Cellpose(gpu=False, model_type='cyto')

# read the file
info_file = open(info_file_path, 'r')
info_lines = info_file.readlines()
info_file.close()

# read info about the data frame
exp_dir,df_name = inp_f.read_df_info(info_lines)

# get info about the channels
channel_list = inp_f.read_channels(info_lines)

# read the movie to segment
file_name = [x['file_name'] for x in channel_list if x['tracking']][0]
c = [x['channel_in_file'] for x in channel_list if x['tracking']][0]
im_path = os.path.join(exp_dir,'data',file_name)
im_path

im = inp_f.open_movie(im_path,c)

# check how many timepoints are there in the file
frames_num = im.shape[0]

print(f'Total frame number: {frames_num}')

# loop for segmentation 
for i in range(0,frames_num): # it's a small example - just 4 first frames
    
    # get an image
    im_frame = im[i,:,:]

    # segment the right plane
    labels, _, _, _ = model.eval(im_frame, diameter=30, channels=[0,0])

    # save segmentation
    save_dir = os.path.join(exp_dir,'segmentation')
    save_file = file_name[:-4]+f'_{str(i).zfill(3)}_label.png'
    save_path = os.path.join(save_dir,save_file)
    imsave(save_path,labels.astype('uint16')) 