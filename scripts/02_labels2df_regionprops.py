#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd

from skimage import measure
from skimage.io import imread
from skimage.segmentation import clear_border

sys.path.append('../libraries')
from ring_functions import make_rings
import input_functions as inp_f

# get the path to the info file as an argument
info_file_path = sys.argv[1]

# read the info file
info_file = open(info_file_path, 'r')
info_lines = info_file.readlines()
info_file.close()

# read info about the data frame
exp_dir,df_name = inp_f.read_df_info(info_lines)

# get info about the channels
channel_list = inp_f.read_channels(info_lines)

# setting directories
labels_dir = os.path.join(exp_dir,'segmentation')
im_dir = os.path.join(exp_dir,'data')
df_dir = os.path.join(exp_dir,'df')

# reading labels 
file_list = [x for x in os.listdir(labels_dir) if 'label' in x]

# setting properties to calculate
properties = ['label', 'area','centroid','orientation','major_axis_length','minor_axis_length','bbox','image','mean_intensity']
properties_ring = ['label','centroid','mean_intensity']

cellDataList=[]
for label_file in file_list:

    frame = int(label_file.split('_')[-2])
    
    # update about progress
    print(f'processing {frame}/{len(file_list)-1}')
    
    # open labels
    label_path = os.path.join(labels_dir,label_file)
    labels_2D = imread(label_path)
    
    # clear border objects
    labels_2D = clear_border(labels_2D)
    
    # read images for intensity calculations
    intensity_list = []
    for i in np.arange(len(channel_list)):
        
        im_name = [ch['file_name'] for ch in channel_list if ch['channel_number']==i][0]
        ch_number = [ch['channel_in_file'] for ch in channel_list if ch['channel_number']==i][0]
        im_path = os.path.join(exp_dir,'data',im_name)
        im = inp_f.open_image(im_path,c=ch_number,t=frame)
        
        intensity_list.append(im)

    int_im = np.moveaxis(np.array(intensity_list),0,2)

    # calculate properties of regions
    cellData = pd.DataFrame(measure.regionprops_table(labels_2D, properties=properties,intensity_image=int_im))

    # add info of these measurements
    cellData['file'] = label_file
    cellData['t'] = frame
    
    # calculate signals in rings
    rings = make_rings(labels_2D,width=6,gap=1)
    rings_prop = measure.regionprops_table(rings, properties=properties_ring,intensity_image=int_im)
    rings_prop = pd.DataFrame(rings_prop)

    cellData = pd.merge(cellData,rings_prop,how='inner',on='label',suffixes=('_nuc', '_ring'))

    cellDataList.append(cellData)

# put all together
cellDataAll = pd.concat(cellDataList,ignore_index=True)

# rename columns
cellDataAll.columns = ['label', 'area', 'centroid-0', 'centroid-1', 'orientation',
                       'major_axis_length', 'minor_axis_length', 'bbox-0', 'bbox-1', 
                       'bbox-2','bbox-3', 'image']+[f'mean_intensity-{x}_nuc' for x in 
                       np.arange(len(channel_list))]+['file', 't', 'centroid-0_ring',
                       'centroid-1_ring'] +[f'mean_intensity-{x}_ring' for x in np.arange(len(channel_list))]


# add info
cellDataAll['size_x'] = labels_2D.shape[0]
cellDataAll['size_y'] = labels_2D.shape[1]

# save calculations
cellDataAll.to_pickle(os.path.join(df_dir,df_name))
cellDataAll.to_csv(os.path.join(df_dir,df_name.replace('pkl','csv')),index=False)