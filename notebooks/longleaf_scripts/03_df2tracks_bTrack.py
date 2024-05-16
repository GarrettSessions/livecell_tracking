import os
import sys
import time
import pickle
import napari
import numpy as np
import pandas as pd

from skimage.io import imread

import btrack
from btrack.constants import BayesianUpdates

sys.path.append(r'/proj/purvislb/users/Garrett/libraries/')
import input_functions as inp_f

info_file_path = r'/proj/purvislb/users/Garrett/072723_50hr_20uM_TBHP/info_B02.txt'

# read the file
info_file = open(info_file_path, 'r')
info_lines = info_file.readlines()
info_file.close()

# read info about the data frame
exp_dir,df_name = inp_f.read_df_info(info_lines)

df_dir = os.path.join(exp_dir,'df')
save_dir = df_dir

frames_to_exclude = inp_f.read_frames_2_exclude(info_lines)
#frames_to_exclude = eval(frames_to_exclude)

modelPath = os.path.join(exp_dir,'code','libraries','cell_config.json')

# ## Read in the data frame objects data
data_df = pd.read_pickle(os.path.join(df_dir,df_name))

len(data_df)

# create a structure suitable for tracking
# choose objects 
sel_vector = [not(x in frames_to_exclude) for x in data_df.t]

objects_gen = data_df.loc[sel_vector,['label','area','centroid-1','centroid-0','major_axis_length','minor_axis_length','t']]

objects_gen.columns=['ID', 'area', 'x', 'y', 'major_axis_length','minor_axis_length','t']
objects_gen['z']=0
objects_gen['label']=5
objects_gen['prob']=0
objects_gen['dummy']=False
objects_gen['states']=0

objects_gen.head()

len(objects_gen)

# ## Tracking proper
# initialise a tracker session using a context manager
with btrack.BayesianTracker() as tracker:

    # configure the tracker using a config file
    tracker.configure_from_file(modelPath)
    
    # approximate
    tracker.update_method = BayesianUpdates.APPROXIMATE
    tracker.max_search_radius = 100

    # append the objects to be tracked
    tracker.append(objects_gen)

    # set the volume (Z axis volume is set very large for 2D data)
    tracker.volume=((0, data_df.size_x[0]), (0, data_df.size_y[0]), (-1e5, 1e5))

    # track them (in interactive mode)
    tracker.track_interactive(step_size=100)

    # generate hypotheses and run the global optimizer
    tracker.optimize()

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
len(trackDataAll)

trackDataAll

allData = pd.merge(left=data_df,right=trackDataAll,left_on=['centroid-0','centroid-1','t'],right_on=['x','y','t'],how='left')
print(f'Number of all objects: {len(allData)}')

allData.columns

# check how many objects doesn't have a track_id
test = np.sum(allData.track_id!=allData.track_id)
print(f'Number of objects without track_id: {test}')

# consider removing
allData = allData.loc[allData.track_id==allData.track_id,:]
print(f'Number of all objects: {len(allData)}')

# ## Define promising tracks
my_tracks = set(allData.track_id)
print(len(my_tracks))

allData['accepted'] = False
allData['rejected'] = False
allData['promise'] = False

# mark tracks longer than a defined value as promising
tracks_set = set(allData.track_id)

track_len=[]
promise_list = []
for track in tracks_set:
    
    # prepare signals for this track
    sel_signal = allData.loc[allData.track_id == track,['t','mean_intensity-0_nuc','mean_intensity-0_ring']]
    sel_signal.sort_values(by='t',inplace=True)
    sel_mean = sel_signal.rolling(9,min_periods = 9,center=True).mean()
    
    # test - length
    track_test = len(sel_signal)>50
    
    track_len.append(len(sel_signal))
    
    # test - DHB presence
    dhb_test = np.sum(sel_mean['mean_intensity-0_nuc'] > (sel_mean['mean_intensity-0_ring']+100)) > 10
    
    if (track_test and dhb_test):
        
        promise_list.append(track)
        
        allData.loc[allData.track_id==track,'promise'] = True

len(promise_list)

# ## Create columns for requested annotations
# get info about the tags (for annotating points on the tracks)
flag_list = inp_f.read_flags(info_lines,df=allData)

for flag in flag_list:
    
    allData[flag['flag_column']]=False

# save df
allData.to_pickle(os.path.join(df_dir,df_name))
allData.to_csv(os.path.join(df_dir,df_name.replace('pkl','csv')),index=False)