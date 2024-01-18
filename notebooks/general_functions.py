# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:34:01 2024

@author: Garrett Sessions
"""

import importlib
import sys
import re

import pandas as pd
from pandas.core.base import NoNewAttributesMixin
import numpy as np
from skimage import measure

#new packages for logging
import sys
import logging
from logging_config import configure_logging

configure_logging()

fov_f = importlib.import_module('ring_functions')

def update_dataFrame(channel_list,my_labels,df,current_frame,active_label,object_properties,flag_list):
    
    '''
    Function to use viewer data to modify data frame with all data (for a specific object in a specific frame)
    
    input:
        channel_list
        my_labels - sent as a layer from the viewer
        df
        current_frame
        active_label
    
    output:
       df 
    '''
    logging.debug("update dataframe loaded")
    # create intensity image
    signal_image = create_intensityImage(channel_list,current_frame)
    logging.debug("signal image")

    # create mask with only a selected object
    single_label_im = create_singleLabel(my_labels,current_frame,active_label)
    logging.debug("single label im")
    
    # characterize new nucleus
    cellData = characterize_newNucleus(single_label_im,signal_image,object_properties)
    logging.debug("cell data")
    
    # create ring image
    x = int(cellData['centroid-0'])
    y = int(cellData['centroid-1'])
    single_label_ring = make_ringImage(single_label_im,x,y,imSize=200)
    logging.debug("ring image created")
    

    # measure properties of the ring
    ringData = characterize_newRing(single_label_ring,signal_image)
    logging.debug("ring data created")
    
    # put data frames together
    labels_set = np.unique(my_labels[current_frame,:,:])
    logging.debug("labels set")
    df = mod_dataFrame(df,cellData,ringData,current_frame,labels_set,flag_list)
    logging.debug("dataframe created")
    
    return df

def create_singleLabel(my_labels,current_frame,active_label):
    
    '''
    Function to create a label image containing only a single cell
    
    input:
        my_labels
        current_frame
        active_label
    
    output:
       single_label_im 
    '''
    
    # create mask with only a selected object
    single_label_im = my_labels[current_frame,:,:].copy()
    single_label_im[single_label_im != active_label]=0
    
    return single_label_im
    
def create_intensityImage(channel_list,current_frame):

    '''
    Function to create intensity image for calculation for a single object.
    This has original size
    
    input:
        channel_list
        current_frame
        active_label
    
    output:
       signal_image 
    '''
    
    im_size_x = channel_list[0]['image'].shape[1]
    im_size_y = channel_list[0]['image'].shape[2]
    
    signal_image = np.zeros([im_size_x,im_size_y,len(channel_list)]).astype('uint16')
 
    for ch in channel_list:
        
        signal_image[:,:,ch['channel_number']] = ch['image'][current_frame,:,:] 
    
    return signal_image
    
def characterize_newNucleus(single_label_im,signal_image,object_properties):
    
    '''
    Function to get properties of a single cell
    
    input:
        single_label_im
        signal_image
        properties
    
    output:
        cellData - data frame with regionprops of a single object    
    '''
    
    # find features of the new object
    
    cellData = measure.regionprops_table(single_label_im, properties=object_properties,intensity_image=signal_image)
    
    cellData = pd.DataFrame(cellData)
    
    return cellData

def calculate_frame(im,imSize,x,y):

    '''
    Function that cuts out a small image
    It takes care of a possible edge problem
    '''
    
    frame = int(imSize/2)

    # calculate max possible frame size
    x1 = frame + np.min((x-frame),0)
    y1 = frame + np.min((y-frame),0)

    x2 = frame - (np.max([im.shape[0],x+frame]) - im.shape[0])
    y2 = frame - (np.max([im.shape[1],y+frame]) - im.shape[1])
    return x1,x2,y1,y2

def make_ringImage(single_label_im,x,y,imSize=200):
    
    '''
    Function to get properties of a single cell
    
    input:
        single_label_im
    
    output:
        single_label_ring  
    '''
    
    # calculate frame
    x1,x2,y1,y2 = calculate_frame(single_label_im,imSize,x,y)
    
    # cut small image
    small_im = single_label_im[x-x1:x+x2,y-y1:y+y2]
    
    # change small image into a ring
    rings = fov_f.make_rings(small_im,width=6,gap=1)
    
    # put small rings image back into the whole frame
    single_label_ring = single_label_im.copy()
    single_label_ring[x-x1:x+x2,y-y1:y+y2]=rings
    
    return single_label_ring
    
def characterize_newRing(single_label_ring,signal_image):
    
    '''
    Function to get properties of a single cell
    
    input:
        single_label_im
        signal_image
    
    output:
        cellData - data frame with regionprops of a single object    
    '''
    # define properties to calculate
    properties_ring = ['label','mean_intensity']
    
    # find features of the new object
    ringData = measure.regionprops_table(single_label_ring, properties=properties_ring,intensity_image=signal_image)
    ringData = pd.DataFrame(ringData)
    
    return ringData

def mod_dataFrame(df,cellData,ringData,current_frame,labels_set,flag_list):
    
    '''
    function to modify gneral data frame with updated modified single object data
    
    input:
        df - original general data frame
        cellData
        ringData
        current_frame
        labels_set - set of labels present in the current frame
        
    output:
        df - modified general data frame
    '''
    logging.debug("mod dataframe loaded")
    # check which cell it is
    active_label = list(cellData['label'])[0]
    logging.debug("mod dataframe active label")

    # put nucleus and ring data together
    cellData = pd.merge(cellData,ringData,how='inner',on='label',suffixes=('_nuc', '_ring'))
    logging.debug("mod dataframe cell data")
    
    # add aditional info
    cellData['t'] = current_frame
    cellData['track_id'] = active_label
    cellData['x'] = cellData['centroid-0']
    cellData['y'] = cellData['centroid-1']
    logging.debug("mod dataframe additional cell data info")

    # add necessary tags
    for flag in flag_list:
        col = flag['flag_column']
        cellData[col] = False
    logging.debug("mod dataframe needed tags")

    # collect information about this label and this time point to calculate 
    info_track = df.loc[:,['track_id','parent','root','generation','accepted','promise','rejected']].drop_duplicates()
    logging.debug("mod dataframe info track")
    
    # merge it to the data of this frame
    cellData = cellData.merge(info_track,on='track_id',how='left')  
    logging.debug("mod dataframe cell data merge") 
    
    # take care of the totally new tracks
    if (cellData.loc[0,'parent'] == cellData.loc[0,'parent']):
        pass
    else:
        cellData.parent = cellData.track_id
        cellData.generation = 0
        cellData.root = cellData.track_id
    logging.debug("mod dataframe celldata new tracks")
       
    # swap in the general data frame
    curr_df = df.loc[df.t==current_frame,:]
    logging.debug("mod dataframe current dataframe")
    
    drop_modified = (curr_df.track_id==active_label)
    logging.debug("mod dataframe drop modified")
    
    # close overlaping objects
    #drop_overlaping_neighbours = ((abs(df['centroid-0']-cellData['centroid-0'][0])<10) & (abs(df['centroid-1']-cellData['centroid-1'][0])<10))
    
    # objects that were removed
    drop_missing = [not(x in labels_set) for x in curr_df.track_id]
    logging.debug("mod dataframe drop missing")
    
    what_to_drop = (drop_modified | drop_missing)
    logging.debug("mod dataframe what to drop")
    
    try:
        curr_df.drop(curr_df.loc[what_to_drop].index, axis=0, inplace=True)
        logging.debug("mod dataframe this is the line that's showing up")

        # Convert curr_df to DataFrame if it's not already
        if not isinstance(curr_df, pd.DataFrame):
            curr_df = pd.DataFrame(curr_df)
            logging.debug("Converting into Dataframe")

        curr_df = pd.concat([curr_df, cellData], ignore_index=True)

        logging.debug("mod dataframe dropping more stuff using concat instead of append")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    # drop current frame 
    df.drop(df[df.t==current_frame].index, axis=0, inplace=True)
    logging.debug("df drop completed")

    # Convert curr_df to DataFrame if it's not already
    if not isinstance(curr_df, pd.DataFrame):
        curr_df = pd.DataFrame(curr_df)
        logging.debug("Modifying curr_df to make it a dataframe. Again.")

    # Concatenate DataFrames
    df = pd.concat([df, curr_df], ignore_index=True)
    logging.debug("mod dataframe completed")
    
    return df

def mod_trackLayer(data,properties,df,current_frame,active_label):
    
    '''
    function to modify tracking layer for the viewer
    
    input:
        data
        properties
        df
        current_frame
        active_label
        
    output:
        data
        properties
    '''
    # choose the data for the specific object
    selData = df.loc[((df.t == current_frame) & (df.track_id == active_label)),:]
    
    # prepare in the right format
    frameData = np.array(selData.loc[:,['label','t','centroid-0','centroid-1']])
    
    # find position of this cell in the tracking data structure
    changeIndex = ((data[:,1]==current_frame) & (data[:,0]==active_label))
    
    # change data
    data = np.delete(data,changeIndex,axis=0)
    data = np.vstack([data, frameData])
    
    # modify properties of the track layer

    selData.loc[:,'state'] = 5
    
    for tProp in properties.keys():
    
        properties[tProp] = np.delete(properties[tProp],changeIndex)
        properties[tProp] = np.append(properties[tProp], selData[tProp])
    
    return data, properties

def newTrack_number(vector):
    
    '''
    Function to find the smallest unused number for a track that can be used
    
    input:
        
        vector - array like with numbers used for tracks
        
    output:
        
        newTrack - number to be used for a new track
    
    '''
    # find number of independent tracks
    tracksSetLength = len(set(vector))
    
    # find maximum track number
    trackMax = np.max(vector)
    
    # check if all are used
    if (trackMax >= (tracksSetLength+1)):
        
        unusedTracks = set(vector).symmetric_difference(np.arange(trackMax+1))
        unusedTracks = np.array(list(unusedTracks))
        unusedTracks = unusedTracks[unusedTracks>0][0]
        newTrack = np.nanmin(unusedTracks)
        
    else:
        newTrack = trackMax + 1 
    
    return newTrack

def trackData_from_df(df,col_list=['promise'],create_graph = True):
    
    '''
    Function to extract tracking data from a data frame
    
    input:
        df - sorted
        create_graph - toggle if graph is needed
    
    output:
        data
        properties

    '''

    #############################################
    # prepare data
    #############################################
    
    # avoid objects without tracking data
    exist_vector = (df['track_id']==df['track_id'])
    
    # select only objects that have specific labels
    sel_vector = False*len(df)
    
    for i in range(len(col_list)):
 
        sel_vector = sel_vector | df[col_list[i]].astype('bool')

    selVector = exist_vector & sel_vector
    
    #gather data in a form of numpy array
    data = np.array(df.loc[selVector,['track_id','t','centroid-0','centroid-1']])
    
    # change format of tracks id
    data[:,0]=data[:,0].astype(int)

    if len(data)>0:

        #############################################
        # prepare properties
        #############################################
        # specify columns to extract properties
        properties = {}
        prop_prop = ['t', 'generation', 'root', 'parent']
        
        for tProp in prop_prop:
        
            properties[tProp] = df.loc[selVector,tProp]
        
        properties['state'] = [5]*len(properties['t'])
        
        #############################################
        # prepare graph
        #############################################
        if create_graph:
            graph = df.loc[(~(df.track_id == df.parent) & selVector),['track_id','parent']].drop_duplicates().to_numpy()
            
            graph = graph.astype(int)
            graph = dict(graph)


            # remove entries that are not available 
            valid_set=set(data[:,0])
            rem_list = [x for x in graph.keys() if not(graph[x] in valid_set)]
            for rem_key in rem_list:
                graph.pop(rem_key)

        else:
            graph = {}
    else:
        # create a dummy in case no data for this layer
        data = np.array([[0,0,0,0],[0,1,0,0]])
        properties = {'t':[0,1], 'generation':[0,0], 'root':[0,0], 'parent':[0,0], 'state':[5,5]}
        graph = {}

    return data,properties,graph

def labels_from_df(cell_data_all):
    
    max_frame = int(np.max(cell_data_all.t))
    row_total = int(cell_data_all.size_x[0])
    column_total = int(cell_data_all.size_y[0])
    
    labels = []

    for i in np.arange(max_frame + 1):
        # choose data from this frame
        sel_data = cell_data_all.loc[cell_data_all.t==i,:]
    
        # create an empty image
        label_image = np.zeros([row_total,column_total]).astype('uint16')
    
        # add objects
        for ind, my_cell in sel_data.iterrows():
            if (my_cell.label == my_cell.label):  # if it's a real object

                min_row = int(my_cell['bbox-0'])
                max_row = int(my_cell['bbox-2'])
                min_col = int(my_cell['bbox-1'])
                max_col = int(my_cell['bbox-3'])

                segment = label_image[min_row:max_row, min_col:max_col]
                image_segment = (my_cell.image * my_cell.track_id)

                # Adjust rows
                row_diff = segment.shape[0] - image_segment.shape[0]
                if row_diff > 0:  # need to pad
                    image_segment = np.pad(image_segment, ((0, row_diff), (0, 0)), mode='constant', constant_values=0)
                elif row_diff < 0:  # need to trim
                    image_segment = image_segment[:row_diff, :]

                # Adjust columns
                col_diff = segment.shape[1] - image_segment.shape[1]
                if col_diff > 0:  # need to pad
                    image_segment = np.pad(image_segment, ((0, 0), (0, col_diff)), mode='constant', constant_values=0)
                elif col_diff < 0:  # need to trim
                    image_segment = image_segment[:, :col_diff]

                # Assign to label_image
                label_image[min_row:max_row, min_col:max_col] = segment + image_segment
                                   
        labels.append(label_image)
    
    labels = np.array(labels)
    return labels


def tags_from_df(df,tag_list):
    
    '''
    Function to extract data for tags from df
    input:
        df
        tag_list
    output:
        tag_list
    '''
    
    tag_data = []
    
    for tag_column in [x['tag_column'] for x in tag_list]: 
        
        # select points for a given tag
        sel_data = df.loc[df[tag_column] == True,:]
        
        # create tag data
        tag_points = np.array([sel_data['t'],sel_data['centroid-0'],sel_data['centroid-1']]).T
        
        tag_data.append(tag_points)
        
    return tag_data
    
def find_all_paths(graph, node, path=[]):
    
    '''
    Function to find all the paths coming through a node in a graph 
    
    input:
        graph
        node
    output:
        list of paths
    '''
    
    path = path + [node]
    paths = [path]
    
    offspring_list = []
    for key, value in graph.items():   # iter on both keys and values
            if (value == [node]):
                offspring_list.append(key)
    
    for node in offspring_list:
        newpaths = find_all_paths(graph, node, path)
        for newpath in newpaths:
            paths.append(newpath)
            
    return paths

def forward_labels(my_labels,df,current_frame,active_label,newTrack):
    
    '''
    Function to modify labels layer.
    input:
        my_labels
        df
        current_frame
        active_label
        newTrack
    output:
        my_labels
    '''

    for myInd in df.index[(df.track_id==active_label) & (df.t>=current_frame)]:
        
        row_start = df.loc[myInd,'bbox-0']
        row_stop = df.loc[myInd,'bbox-2']
        column_start = df.loc[myInd,'bbox-1']
        column_stop = df.loc[myInd,'bbox-3']
        
        if np.isnan(row_start and row_stop and column_start and column_stop):
            
            pass
        
        else:
            
            myFrame = int(df.loc[myInd,'t'])
    
            # cut and replace
            temp = my_labels[myFrame,int(row_start):int(row_stop),int(column_start):int(column_stop)]
            temp[temp == active_label] = int(newTrack)
            my_labels[myFrame,int(row_start):int(row_stop),int(column_start):int(column_stop)] = temp
    
    return my_labels

def forward_df(df,current_frame,active_label,newTrack,connectTo=0):
    
    '''
    Function to modify forward data frame structure after linking changes
    input:
        df
        current_frame
        active_label
        newTrack
        graph
    output:
        df
    '''
    
    # find info about the cut track
    active_label_generation = list(df.loc[df.track_id==active_label,'generation'].drop_duplicates())[0]
    
    # find info about the new label
    genList = list(df.loc[df.track_id==newTrack,'generation'].drop_duplicates())
    if len(genList)>0:
        new_generation = genList[0]
        new_root = list(df.loc[df.track_id==newTrack,'root'].drop_duplicates())[0]
        new_parent = list(df.loc[df.track_id==newTrack,'parent'].drop_duplicates())[0]
        
    else: # so this is a completely new number for a track
        
        if connectTo == 0: # and nothing to connect to
            
            new_generation = 0
            new_root = newTrack
            new_parent = newTrack
        
        else: # check data of a track we connect to
        
            new_generation = list(df.loc[df.track_id==connectTo,'generation'].drop_duplicates())[0] + 1
            new_root = list(df.loc[df.track_id==connectTo,'root'].drop_duplicates())[0]
            new_parent = connectTo
            
    
    # get a graph
    data,properties,graph = trackData_from_df(df,col_list=['t'])
    
    # find kids
    kids_list = []
    for key, value in graph.items():   # iter on both keys and values
            if (value == [active_label]):
                kids_list.append(key)
    
    # find all family members
    all_paths = find_all_paths(graph, active_label)
    family_members = [item for sublist in all_paths for item in sublist]
    
    for myDescendant in family_members:
        
        # find which rows need to be changed
        changeIndex = (df.t>=current_frame) & (df.track_id==myDescendant)
        
        df.loc[changeIndex,'root'] = new_root
        df.loc[changeIndex,'generation'] = df.loc[changeIndex,'generation'] - active_label_generation + new_generation
        
        if(myDescendant == active_label):
        
            df.loc[changeIndex,'track_id'] = newTrack
            df.loc[changeIndex,'parent'] = new_parent
              
        elif (myDescendant in kids_list): #2nd generation
            
            df.loc[changeIndex,'parent'] = newTrack
            
    return df
  
def extract_graph_data(graph_list,df_sel):
    
    '''
    Function to translate input file info to signals for plotting.
    '''
 
    results_list = []   
 
    key_words = ['nuc','ring']
    
    for graph in graph_list:
        
        function = graph['function']

        if function=='family':

            function_value = np.zeros([len(df_sel),1])

        else:
            request_list = []
            replacement_list = []
        
            for key_word in key_words:
        
                signal_list = [x.end() for x in re.finditer(f'{key_word}_',function)]
        
                for signal in signal_list:
                    
                    # for which channel it's requested
                    ch_number = eval(function[signal])  
                    
                    # get a column name
                    col = f'mean_intensity-{ch_number}_{key_word}'
        
                    # get data
                    request_list.append(f'{key_word}_{function[signal]}')
                    replacement_list.append(f"df_sel['{col}']")


                    
            # translate the function
            for request_signal,replacement_name in zip(request_list,replacement_list): 
        
                function = function.replace(request_signal,replacement_name)
                
            # evaluate the function
            function_value = eval(function)
  
 
        # collect results
        results_list.append(function_value)
 
    return results_list

def calculate_graph_offset(df,current_track):

    
    sel_t = df.loc[df.track_id == current_track,'t']
    
    graph_offset = np.min(sel_t)
    
    return graph_offset

def find_empty_frames(t):

    '''
    Function to find empty frames in a time series.
    '''
     
    t_min = np.min(t)
    t_max = np.max(t)
    
    empty_frames_list = list(set(np.arange(t_min,t_max+1)) - set(t))
    empty_frames_list.sort()

    return empty_frames_list