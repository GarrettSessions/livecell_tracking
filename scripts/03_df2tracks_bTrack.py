#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import numpy as np
import pandas as pd
import btrack

sys.path.append('../libraries')
import input_functions as inp_f

# get the path to the info file as an argument
info_file_path = sys.argv[1]

# read the file
info_file = open(info_file_path, 'r')
info_lines = info_file.readlines()
info_file.close()

# read info about the data frame
exp_dir,df_name = inp_f.read_df_info(info_lines)

df_dir = os.path.join(exp_dir,'df')
save_dir = df_dir

modelPath = os.path.join(exp_dir,'code','libraries','cell_config.json')

# ## Read in the data frame objects data
data_df = pd.read_pickle(os.path.join(df_dir,df_name))

objects_gen = data_df.loc[:,['label','area','centroid-1','centroid-0','major_axis_length','minor_axis_length','t']]

objects_gen.columns=['ID', 'area', 'x', 'y', 'major_axis_length','minor_axis_length','t']
objects_gen['z']=0
objects_gen['label']=5
objects_gen['prob']=0
objects_gen['dummy']=False
objects_gen['states']=0

objects_gen.head()

# ## Tracking proper

# initialise a tracker session using a context manager
with btrack.BayesianTracker() as tracker:

    # configure the tracker using a config file
    tracker.configure_from_file(modelPath)

    # append the objects to be tracked
    tracker.append(objects_gen)

    # set the volume (Z axis volume is set very large for 2D data)
    tracker.volume=((0, data_df.size_x[0]), (0, data_df.size_y[0]), (-1e5, 1e5))

    # track them (in interactive mode)
    tracker.track_interactive(step_size=100)

    # generate hypotheses and run the global optimizer
    tracker.optimize()

    # get the tracks as a python list
    tracks = tracker.tracks

    # optional: get the data in a format for napari
    data, properties, graph = tracker.to_napari(ndim=2)
    # pickle Napari data
    with open(os.path.join(df_dir,'track.pkl'),'wb') as f:
        pickle.dump([data,properties,graph],f)


# ## Merging objects and tracking information
trackDataAll = pd.DataFrame(data,columns=['track_id','t','x','y'])
trackDataAll['parent'] = properties['parent']
trackDataAll['generation'] = properties['generation']
trackDataAll['root'] = properties['root']

allData = pd.merge(left=data_df,right=trackDataAll,left_on=['centroid-0','centroid-1','t'],right_on=['x','y','t'],how='left')

print(f'Number of all objects: {len(allData)}')

allData['accepted'] = False
allData['rejected'] = False
allData['promise'] = False

# mark tracks longer than 100 as promising
tracks_set = set(allData.track_id)

for track in tracks_set:
    
    track_len = np.sum(allData.track_id==track)
    
    if (track_len>100):
        
        allData.loc[allData.track_id==track,'promise'] = True

# save df
allData.to_pickle(os.path.join(df_dir,df_name))
allData.to_csv(os.path.join(df_dir,df_name.replace('pkl','csv')),index=False)