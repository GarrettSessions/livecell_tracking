# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:31:52 2021

@author: Kasia Kedziora
"""

import numpy as np
from itertools import product

def calculate_cut(imSize,orgImShape,x,y):
    
    '''
    Function to calculate how to cut a rectange of a small object
    (mostly copes with edge effects):
        
    input:
    imSize - size of a small image (square)
    orgImShape - tuple with a size of an org image
    x - centroid (row)
    y - centroid (column)
    
    output:
    row_start,row_stop,column_start,column_stop - how to cut
    row_in_start,row_in_stop,column_in_start,column_in_stop - how to put in
    '''
    
    # calculate how to cut
    row_start = np.max([x-int(imSize/2),0])
    row_stop = np.min([x+int(imSize/2),orgImShape[1]])
    
    column_start = np.max([y-int(imSize/2),0])
    column_stop = np.min([y+int(imSize/2),orgImShape[2]])
    
    
    # calculate how to place
    row_in_start = int((imSize - (row_stop-row_start))/2)
    row_in_stop = int(row_in_start + (row_stop-row_start))
    
    column_in_start = int((imSize - (column_stop-column_start))/2)
    column_in_stop = int(column_in_start + (column_stop-column_start))
    
    return row_start,row_stop,column_start,column_stop,row_in_start,row_in_stop,column_in_start,column_in_stop


def smallStack_generate(myIm,cellDataAll,myTrack,imSize=100):
    
    '''
    function that generates small stacks
    input:
    
    myTrack - number of a track
    im - image data from which to cut a small stack
    data - tracking data (in a form from tracking layer)
    imSize - size of an image to cut 
    
    in the future:
        consider an option of clearing surroundings
    
    '''
    data = np.array(cellDataAll.loc[cellDataAll.track_id == myTrack,['track_id','t','centroid-0','centroid-1']])
    data[:,0]=data[:,0].astype(int)
    
    # look up start frame
    startFrame = int(np.min(data[data[:,0]==myTrack,1]))
    
    # look up stop frame
    stopFrame = int(np.max(data[data[:,0]==myTrack,1]))
    
    # calculate how many images there will be
    imNum = stopFrame-startFrame+1
    
    # prepare small stack container
    small_stack = np.zeros([imNum,imSize,imSize]).astype('uint16')
    
    # shape of the original image
    orgImShape = myIm.shape
    
    # put data into a small stack
    for myInd in np.where(data[:,0]==myTrack)[0]:
        
        x = data[myInd,2]
        y = data[myInd,3]
        
        if np.isnan(x and y):
            pass
        else:
            x=int(x)
            y=int(y)
            myFrame = int(data[myInd,1])
            
            t = myFrame - startFrame
            
            # calculate cut parameters
            row_start,row_stop,column_start,column_stop,row_in_start,row_in_stop,column_in_start,column_in_stop = calculate_cut(imSize,orgImShape,x,y) 
            
            
            small_stack[t,row_in_start:row_in_stop,column_in_start:
                        column_in_stop] = myIm[myFrame,row_start:row_stop,column_start:column_stop]

        
    return small_stack

def stack_create_all(channel_list,labels,cell_data_all,active_track,imSize=100):
    
    '''
    function that generates a collection of stacks for a given track
    input:
        channel_list
        labels
        cell_data_all
        active_track
    output:
        stack_im_list - this is a list itself for all the signal images
        stack_labels
    '''
    
    # generate a stack for labels
    stack_labels = smallStack_generate(labels,cell_data_all,active_track,imSize)
    
    # generate stacks for all tracking channels
    stack_im_list = []
    for ch in channel_list:
    
        temp_stack = smallStack_generate(ch['image'],cell_data_all,active_track,imSize)
        
        stack_im_list.append(temp_stack)
        
    return stack_im_list,stack_labels

def gallery_generate(smallIm):
    
    '''
    function to generate a gallery view from a small stack
    with an attempt to make it square
    
    input:
        small stack
    output:
        gallery
    '''
    
    imNum = smallIm.shape[0]
    imSize = smallIm.shape[1]
    
    if np.sqrt(imNum).is_integer():
    
        col = row = int(np.sqrt(imNum))

    else:
        
        col = int(np.sqrt(imNum))
        row = int(np.floor(imNum/col)+1)
            
    # create canvas
    myGallery = np.zeros([row*imSize,col*imSize]).astype('uint16')

    # put images into canvas
    for pair in product(range(row),range(col)):
            
            i = pair[1] # column iterator (fast)
            j = pair[0] # row iterator (slow)
            
            myFrame = j*col+i
            
            if myFrame<imNum:
            
                myGallery[j*imSize:(j+1)*imSize,i*imSize:(i+1)*imSize] = smallIm[myFrame,:,:] 
            
            else:
                break
            
    return myGallery

def gallery_create_all(myIm,myLabels,myIm_signal_list,data,myTrack,imSize=100):
    
    '''
    function that generates a collection of galleries for a given track
    input:
        myIm - tracking channel image
        myLabels - labels stack
        myIm_signal_list - list of signal channels
        data - tracking data (matching tracking layer format)
        myTrack - number of track to process
        imSize - size for an image to cut
    output:
        gallery_track
        gallery_labels
        gallery_signal_list - this is a list itself for all the signal images
    '''
    
    # generate a gallery for tracking channel
    small_stack_track = smallStack_generate(myIm,data,myTrack,imSize=100)
    gallery_track = gallery_generate(small_stack_track)
    
    # generate a gallery for labels
    small_stack_labels = smallStack_generate(myLabels,data,myTrack,imSize=100)
    gallery_labels = gallery_generate(small_stack_labels)
    
    # generate galleries for all tracking channels
    gallery_signal_list = []
    for myIm_signal in myIm_signal_list:
    
        temp_stack = smallStack_generate(myIm_signal,data,myTrack,imSize=100)
        temp_gallery = gallery_generate(temp_stack)
        
        gallery_signal_list.append(temp_gallery)
        
    return gallery_track,gallery_labels,gallery_signal_list
    
