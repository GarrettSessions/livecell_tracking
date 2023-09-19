# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:59:43 2021

@author: kmkedz
"""

import os
import re
from os import path
import pandas as pd
from tifffile import imread
from nd2reader import ND2Reader
import numpy as np

def clean_string(string):
    
    '''
    Function to clean string read from a txt file.
    Deals with trailing '\n'
    '''
    
    if '\n' in string:
        
        string = string.split('\n')[0]
        
    return string

def build_channel(info_lines,ind,end_ind):
    
    '''
    Function building a dictionary with all the channel info.
    '''
    
    # create dictionary
    channel = {}
    
    # add all the info
    keyword_list = [x.split(':')[0] for x in info_lines[ind:end_ind] if ((':' in x) and not(':\\' in x))]
    
    for key in keyword_list:
    
        info_ind = [f'{key}:' in x for x in info_lines].index(True,ind,end_ind) + 1
        
        key_value = info_lines[info_ind]
        
        try:
            key_value = eval(key_value)
        except:
            pass
        
        try:
            key_value = clean_string(key_value)
        except:
            pass
        
        try:
            key_value = eval(key_value)
        except:
            pass

            
        channel[key] = key_value

    
    return channel

def build_graph(info_lines,ind,end_ind):
    
    '''
    Function building a dictionary with all the graph info.
    '''
    
    # create dictionary
    graph = {}
    
    # add compulsory info
    info_ind = ['graph_name:' in x for x in info_lines].index(True,ind,end_ind) + 1
    graph['graph_name'] = clean_string(info_lines[info_ind])
    
    info_ind = ['graph_function:' in x for x in info_lines].index(True,ind,end_ind) + 1
    graph['function'] = clean_string(info_lines[info_ind])
    
    info_ind = ['graph_color:' in x for x in info_lines].index(True,ind,end_ind) + 1
    graph['color'] = clean_string(info_lines[info_ind])
    
    # add any additional info
    
    return graph

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

def check_channel(df,channel,exp_dir):
    
    '''
    Function to check if the info for a given channel corresponds with what is available in the data frame.
    '''
    
    # test if specified image exists
    im_path = os.path.join(exp_dir,'data',channel['file_name'])
    if path.exists(im_path):
        pass
    else:
        raise(InputError('Wrong pathway.',f"{channel['file_path']} does not exist."))
    
    # check if there are data in the table for this channel
    c_num = channel['channel_number']
    for col in [f'mean_intensity-{c_num}_nuc',f'mean_intensity-{c_num}_ring']:  
        
        # test if specified columns exist in the data frame
        if (col in df.columns):
            pass
        else:
            raise(InputError('Error in txt file.',f"Column for {col} of {channel['channel_name']} does not exist."))
            
def check_graph(df,graph,channel_list):
    
    '''
    Function to check if the info for a given graph corresponds with what is available in the data frame.
    '''
    
    key_words = ['nuc','ring']
    function = graph['function']

    request_list = []
    replacement_list = []

    for key_word in key_words:

        signal_list = [x.end() for x in re.finditer(f'{key_word}_',function)]

        for signal in signal_list:
            
            # check if requested channel is defined by number
            try:
                ch_number = eval(function[signal])

            except: # if it's not a number
                raise(InputError('Error in txt file.',f"{key_word}_{function[signal]} requested for graph '{graph['graph_name']}' is not a valid signal."))  
            
            # check if the name is in the data frame
            col = f'mean_intensity-{ch_number}_{key_word}'

            if (col in df.columns):

                request_list.append(f'{key_word}_{function[signal]}')
                replacement_list.append(f'{df.loc[0,col]}')

            else:
                raise(InputError('Error in txt file.',f"'{col}' requested for '{graph['graph_name']}' graph as c does not exist."))

            # check that requested channel will be loaded
            channel_num_list = [x['channel_number'] for x in channel_list]
            if (ch_number in channel_num_list):  
                pass
            else:
                raise(InputError('Error in txt file.',f"{key_word}_{function[signal]} requested for graph {graph['graph_name']} does not correspond to any loaded channel."))
                
    # check that the function can be evaluated
    for request_signal,replacement_name in zip(request_list,replacement_list): 

            function = function.replace(request_signal,replacement_name)

    try:
        eval(function)
    except:
        raise(InputError('Error in txt file.',f"Function requested for '{graph['graph_name']}' cannot be executed."))
        
def read_channels(info_lines,check=False,df=0,exp_dir=0):
    
    '''
    Function to read info about all the channels from the info file
    input:
        handle to file
    output:
        channel_list
    '''

    channel_list = []

    for ind,inf_l in enumerate(info_lines):

        if 'channel_name' in inf_l:

            try:
                end_ind = info_lines.index('\n',ind)
            except:
                end_ind = len(info_lines)

            # create a dictionary for a given channel
            temp_channel = build_channel(info_lines,ind,end_ind)

            # check if the dictionary matches data frame
            if check == True:
                check_channel(df,temp_channel,exp_dir)

            # put this channel dictionary in the list
            channel_list.append(temp_channel)

    # check that all channels are unique
    channel_num_list = [x['channel_number'] for x in channel_list]
    if len(channel_num_list) == len(set(channel_num_list)):
        pass
    else:
        raise(InputError('Error in txt file.',"Channels are not unique."))
        
    return(channel_list)

def read_tags(info_lines,df):
    
    '''
    Function to read info about all the requested tags from the info file
    input:
        handle to file
    output:
        tag_list
    '''

    tag_list = []

    for ind,inf_l in enumerate(info_lines):

        if 'tag_name' in inf_l:

            try:
                end_ind = info_lines.index('\n',ind)
            except:
                end_ind = len(info_lines)

            # create a dictionary for a given channel
            temp_tag = build_channel(info_lines,ind,end_ind)

            # check if the dictionary matches data frame
            #check_tag(df,temp_tag)

            # put this channel dictionary in the list
            tag_list.append(temp_tag)
        
    return(tag_list)

def read_flags(info_lines,df):
    
    '''
    Function to read info about all the requested flags. Flags are annotations of frames 
    within traces so a flag belongs to a point in time for a given cell.
    input:
        handle to file
    output:
        flag_list
    '''

    flag_list = []

    for ind,inf_l in enumerate(info_lines):

        if 'flag_name' in inf_l:

            try:
                end_ind = info_lines.index('\n',ind)
            except:
                end_ind = len(info_lines)

            # create a dictionary for a given channel
            temp_tag = build_channel(info_lines,ind,end_ind)

            # check if the dictionary matches data frame
            #check_tag(df,temp_tag)

            # put this channel dictionary in the list
            flag_list.append(temp_tag)
        
    return(flag_list)

def read_graphs(info_lines,df,channel_list):
    
    '''
    Function to read info about all the requested graphs from the info file
    input:
        handle to file
        channel_list - list of channels that will be loaded
    output:
        channel_list
    '''
    
    # build the graphs
    graph_list = []        

    for ind,inf_l in enumerate(info_lines):

        if 'graph_name' in inf_l:
    
            try:
                end_ind = info_lines.index('\n',ind)
            except:
                end_ind = len(info_lines)
    
            temp_graph = build_graph(info_lines,ind,end_ind)
    
            # check if the graph can be executed
            if temp_graph['function'] == 'family':
                pass
            else:
                check_graph(df,temp_graph,channel_list)
    
            graph_list.append(temp_graph)
            
    return(graph_list)

def read_df_info(info_lines):
    
    '''
    Function to read df based on info_file
    input:
        handle to file
    output:
        df
    '''        

    for ind,inf_l in enumerate(info_lines):
    
        if 'exp_dir' in inf_l:
    
            exp_dir = clean_string(info_lines[ind+1])
            
        if 'df_name' in inf_l:
    
            df_name = clean_string(info_lines[ind+1])
    
    return exp_dir,df_name

def read_frames_2_exclude(info_lines):
    
    '''
    Function to read df based on info_file
    input:
        info lines
    output:
        frames_to_exclude
    '''        

    frames_to_exclude = []

    for ind,inf_l in enumerate(info_lines):
    
        if 'frames_to_exclude' in inf_l:
    
            frames_to_exclude = clean_string(info_lines[ind+1])
    
    return frames_to_exclude

def read_settings(info_lines):
    
    '''
    Function to read settings from the txt file
    input:
        handle to file
    output:
        df
    '''        
    
    small_im_size = 120
    label_contour = 0
    time_threshold = 0
    gen_track_columns = ['label']

    for ind,inf_l in enumerate(info_lines):
    
        if 'small_im_size' in inf_l:
    
            small_im_size = eval(clean_string(info_lines[ind+1]))
            
        if 'time_threshold' in inf_l:
    
            time_threshold = eval(clean_string(info_lines[ind+1]))
            
        if 'label_contour' in inf_l:
    
            label_contour = eval(clean_string(info_lines[ind+1]))
            
        if 'gen_track_columns' in inf_l:
    
            gen_track_columns = eval(clean_string(info_lines[ind+1]))
    
    return time_threshold,small_im_size,label_contour,gen_track_columns

def read_properties(info_lines):

    
    '''
    Function to read what calculations have to be performed on objects
    input:
        handle to file
    output:
        df
    '''        
    
    object_properties = ['label', 'area','centroid','bbox','image','mean_intensity']

    for ind,inf_l in enumerate(info_lines):
    
        if 'properties:' in inf_l:
    
            object_properties = eval(clean_string(info_lines[ind+1]))
    
    return object_properties

def open_movie(im_path,c):

    if 'tif' in im_path:

        im = imread(im_path)
            
    if 'nd2' in im_path:
        
        temp_reader = ND2Reader(im_path) 
        
        frame_num = temp_reader.sizes['t']
        
        im = []
        
        for i in range(frame_num):
            
            try:
                temp_im = temp_reader.get_frame_2D(c=c, t=i)
                im.append(temp_im)
            except KeyError: # in case more frames are reported than available
                pass

        im = np.moveaxis(im,0,2)

    return im

def open_image(im_path,c,t):

    if 'tif' in im_path:

        im = imread(im_path,key=t)
            
    if 'nd2' in im_path:
        
        temp_reader = ND2Reader(im_path) 
            
        im = temp_reader.get_frame_2D(c=c, t=t)


    return im