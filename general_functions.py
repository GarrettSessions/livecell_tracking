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
    Function to use viewer data to modify data frame with all data
    (for a specific object in a specific frame).

    The napari selected label is treated as the authoritative track ID.
    Regionprops is used only to recalculate object measurements.
    
    input:
        channel_list
        my_labels - sent as a layer from the viewer
        df
        current_frame
        active_label - authoritative napari label / track ID
        object_properties
        flag_list
    
    output:
       df 
    '''
    logging.debug("update dataframe loaded")

    active_label = int(active_label)
    current_frame = int(current_frame)

    if active_label <= 0:
        raise ValueError("A non-zero label must be selected before modifying an object.")

    # create intensity image
    signal_image = create_intensityImage(channel_list,current_frame)
    logging.debug("signal image")

    # create mask with only the selected object
    single_label_im = create_singleLabel(my_labels,current_frame,active_label)
    logging.debug("single label im")

    if not np.any(single_label_im == active_label):
        raise ValueError(
            f"Selected label {active_label} is not present in frame {current_frame}. "
            "Select the label that was painted/filled before clicking Modify Label."
        )

    # Preserve the exact user-authored mask, even if the same label currently
    # contains multiple disconnected components.  In a manual-correction
    # workflow the napari Labels pixels are authoritative; regionprops can
    # measure a single integer label spanning disconnected components and the
    # stored binary ``image`` can reproduce that mask exactly on reload.

    # characterize new nucleus
    cellData = characterize_newNucleus(single_label_im,signal_image,object_properties)
    logging.debug("cell data")

    if len(cellData) != 1:
        raise ValueError(
            f"Expected one region for label {active_label} in frame {current_frame}, "
            f"but regionprops returned {len(cellData)} regions."
        )
    
    # create ring image
    x = int(cellData.loc[0,'centroid-0'])
    y = int(cellData.loc[0,'centroid-1'])
    single_label_ring = make_ringImage(single_label_im,x,y,imSize=200)
    logging.debug("ring image created")
    
    # measure properties of the ring
    ringData = characterize_newRing(single_label_ring,signal_image)
    logging.debug("ring data created")

    if len(ringData) != 1:
        raise ValueError(
            f"Expected one ring for label {active_label} in frame {current_frame}, "
            f"but regionprops returned {len(ringData)} regions."
        )
    
    # put data frames together
    labels_set = np.unique(my_labels[current_frame,:,:])
    logging.debug("labels set")
    df = mod_dataFrame(
        df, cellData, ringData, current_frame, active_label, labels_set, flag_list
    )
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

def mod_dataFrame(df,cellData,ringData,current_frame,active_label,labels_set,flag_list):
    
    '''
    Modify the general dataframe with recalculated measurements for one object.

    The supplied active_label is authoritative for both ``label`` and
    ``track_id``. Regionprops-derived label values are not used to infer track
    identity.
    
    input:
        df - original general data frame
        cellData
        ringData
        current_frame
        active_label - authoritative napari label / track ID
        labels_set - set of labels present in the current frame
        flag_list
        
    output:
        df - modified general data frame
    '''
    logging.debug("mod dataframe loaded")

    active_label = int(active_label)
    current_frame = int(current_frame)
    labels_set = set(int(x) for x in np.asarray(labels_set).tolist())

    # Put nucleus and ring data together using the regionprops label only as
    # an internal join key. Tracking identity is imposed explicitly below.
    cellData = pd.merge(cellData,ringData,how='inner',on='label',suffixes=('_nuc', '_ring'))
    logging.debug("mod dataframe cell data")

    if len(cellData) != 1:
        raise ValueError(
            f"Expected exactly one measured object for label {active_label} in "
            f"frame {current_frame}; found {len(cellData)}."
        )

    # Explicitly preserve the napari-selected identity. This is the critical
    # distinction between segmentation geometry and tracking identity.
    cellData.loc[:, 'label'] = active_label
    cellData.loc[:, 'track_id'] = active_label
    cellData.loc[:, 't'] = current_frame
    cellData.loc[:, 'x'] = cellData['centroid-0']
    cellData.loc[:, 'y'] = cellData['centroid-1']
    logging.debug("mod dataframe additional cell data info")

    # add necessary tags
    for flag in flag_list:
        col = flag['flag_column']
        cellData[col] = False
    logging.debug("mod dataframe needed tags")

    # Preserve lineage / track-level information for an existing track ID.
    # Do this by direct assignment from ONE representative row rather than a
    # dataframe merge. A merge can silently duplicate the measured object when
    # track-level status columns are not perfectly constant across every row.
    track_meta_cols = ['parent','root','generation','accepted','promise','rejected']
    existing_track = df.loc[df.track_id == active_label, track_meta_cols]
    logging.debug("mod dataframe existing track metadata rows: %s", len(existing_track))

    if len(existing_track) > 0:
        representative = existing_track.iloc[0]
        for col in track_meta_cols:
            cellData.loc[:, col] = representative[col]
    else:
        # This is a completely new track ID.
        cellData.loc[:, 'parent'] = active_label
        cellData.loc[:, 'generation'] = 0
        cellData.loc[:, 'root'] = active_label
        for col in ['accepted','promise','rejected']:
            cellData.loc[:, col] = False
    logging.debug("mod dataframe track metadata assigned")

    # Work on a copy to avoid chained-assignment/view behavior.
    curr_df = df.loc[df.t==current_frame,:].copy()
    logging.debug("mod dataframe current dataframe")

    # Remove the old row for the object being modified.
    drop_modified = (curr_df.track_id==active_label)
    logging.debug("mod dataframe drop modified")

    # Remove rows for labels that were erased/reassigned in napari. This is
    # what lets Fill-based source-ID -> destination-ID corrections remove the
    # stale source row from the dataframe.
    numeric_track_ids = pd.to_numeric(curr_df.track_id, errors='coerce')
    drop_missing = ~numeric_track_ids.isin(labels_set)
    logging.debug("mod dataframe drop missing")

    what_to_drop = (drop_modified | drop_missing)
    logging.debug("mod dataframe what to drop")

    curr_df = curr_df.loc[~what_to_drop].copy()
    curr_df = pd.concat([curr_df, cellData], ignore_index=True)
    logging.debug("mod dataframe replacement row added")

    # Replace the entire current frame in the master dataframe atomically.
    df_without_frame = df.loc[df.t!=current_frame,:].copy()
    df = pd.concat([df_without_frame, curr_df], ignore_index=True)
    logging.debug("mod dataframe completed")
    
    return df

def label_frame_from_df(df, current_frame):
    """Reconstruct one labels frame from the authoritative dataframe.

    This is used to roll back an uncommitted napari paint/fill edit when
    Modify Label fails. It mirrors ``labels_from_df`` but only for one frame.
    """
    current_frame = int(current_frame)
    row_total = int(df.size_x.iloc[0])
    column_total = int(df.size_y.iloc[0])
    label_image = np.zeros([row_total, column_total], dtype='uint16')

    sel_data = df.loc[df.t == current_frame, :]
    for _, my_cell in sel_data.iterrows():
        if pd.isna(my_cell.get('label', np.nan)) or pd.isna(my_cell.get('track_id', np.nan)):
            continue

        min_row = int(my_cell['bbox-0'])
        max_row = int(my_cell['bbox-2'])
        min_col = int(my_cell['bbox-1'])
        max_col = int(my_cell['bbox-3'])

        segment = label_image[min_row:max_row, min_col:max_col]
        image_segment = np.asarray(my_cell.image) * int(my_cell.track_id)

        row_diff = segment.shape[0] - image_segment.shape[0]
        if row_diff > 0:
            image_segment = np.pad(image_segment, ((0, row_diff), (0, 0)), mode='constant')
        elif row_diff < 0:
            image_segment = image_segment[:segment.shape[0], :]

        col_diff = segment.shape[1] - image_segment.shape[1]
        if col_diff > 0:
            image_segment = np.pad(image_segment, ((0, 0), (0, col_diff)), mode='constant')
        elif col_diff < 0:
            image_segment = image_segment[:, :segment.shape[1]]

        # Reconstruct stored masks by assigning their track ID only where the
        # saved binary mask is true. Addition can manufacture invalid numeric
        # labels when bounding boxes overlap.
        target = label_image[min_row:max_row, min_col:max_col]
        mask = image_segment != 0
        target[mask] = int(my_cell.track_id)
        label_image[min_row:max_row, min_col:max_col] = target

    return label_image


def validate_frame_label_sync(my_labels, df, current_frame):
    """Validate that a napari Labels frame and dataframe encode the same track IDs.

    Returns
    -------
    is_valid : bool
    message : str
    details : dict
        Contains labels_only, dataframe_only, and duplicate_dataframe_tracks.
    """
    current_frame = int(current_frame)

    frame_labels = set(
        int(x) for x in np.unique(my_labels[current_frame])
        if not pd.isna(x) and int(x) > 0
    )

    frame_df = df.loc[df.t == current_frame, :].copy()
    if 'label' in frame_df.columns:
        frame_df = frame_df.loc[frame_df['label'].notna(), :]

    frame_track_series = pd.to_numeric(frame_df['track_id'], errors='coerce').dropna().astype(int)
    frame_tracks = set(int(x) for x in frame_track_series if int(x) > 0)

    labels_only = sorted(frame_labels - frame_tracks)
    dataframe_only = sorted(frame_tracks - frame_labels)

    duplicate_dataframe_tracks = sorted(
        int(x) for x in frame_track_series[frame_track_series.duplicated(keep=False)].unique()
        if int(x) > 0
    )

    details = {
        'labels_only': labels_only,
        'dataframe_only': dataframe_only,
        'duplicate_dataframe_tracks': duplicate_dataframe_tracks,
    }

    is_valid = not labels_only and not dataframe_only and not duplicate_dataframe_tracks

    if is_valid:
        message = f"Frame {current_frame}: Labels layer and dataframe track IDs are synchronized."
    else:
        parts = [f"Frame {current_frame}: Labels/dataframe synchronization failed."]
        if labels_only:
            parts.append(f"IDs present only in Labels: {labels_only}")
        if dataframe_only:
            parts.append(f"IDs present only in dataframe: {dataframe_only}")
        if duplicate_dataframe_tracks:
            parts.append(f"Duplicate dataframe track IDs in this frame: {duplicate_dataframe_tracks}")
        message = ' '.join(parts)

    return is_valid, message, details


def _measure_label_row_from_frame(channel_list, my_labels, df, current_frame, label_id,
                                  object_properties, flag_list, signal_image=None):
    """Measure one label exactly as it exists in the current napari frame.

    This helper performs no frame-wide reconciliation.  It returns one dataframe
    row whose stored ``image``/bbox/centroid describe the actual pixels currently
    present in napari for ``label_id``.
    """
    current_frame = int(current_frame)
    label_id = int(label_id)

    if signal_image is None:
        signal_image = create_intensityImage(channel_list, current_frame)

    single_label_im = create_singleLabel(my_labels, current_frame, label_id)
    if not np.any(single_label_im == label_id):
        raise ValueError(f'Label {label_id} is not present in frame {current_frame}.')

    cellData = characterize_newNucleus(single_label_im, signal_image, object_properties)
    if len(cellData) != 1:
        raise ValueError(
            f'Expected one regionprops row for label {label_id}; found {len(cellData)}.'
        )

    x = int(cellData.loc[0, 'centroid-0'])
    y = int(cellData.loc[0, 'centroid-1'])
    single_label_ring = make_ringImage(single_label_im, x, y, imSize=200)
    ringData = characterize_newRing(single_label_ring, signal_image)
    if len(ringData) != 1:
        raise ValueError(
            f'Expected one ring row for label {label_id}; found {len(ringData)}.'
        )

    row = pd.merge(cellData, ringData, how='inner', on='label', suffixes=('_nuc', '_ring'))
    if len(row) != 1:
        raise ValueError(
            f'Expected one merged measurement row for label {label_id}; found {len(row)}.'
        )

    row.loc[:, 'label'] = label_id
    row.loc[:, 'track_id'] = label_id
    row.loc[:, 't'] = current_frame
    row.loc[:, 'x'] = row['centroid-0']
    row.loc[:, 'y'] = row['centroid-1']

    same_frame_track = df.loc[
        (df.t == current_frame) &
        (pd.to_numeric(df.track_id, errors='coerce') == label_id), :
    ]
    existing_track_rows = df.loc[
        pd.to_numeric(df.track_id, errors='coerce') == label_id, :
    ]
    same_frame_rep = same_frame_track.iloc[0] if len(same_frame_track) else None
    track_rep = existing_track_rows.iloc[0] if len(existing_track_rows) else None

    def _set_single_row_value(frame, col, value):
        """Assign *value* to the sole row without expanding array-like objects.

        Some dataframe fields (most importantly regionprops' ``image`` mask) are
        NumPy arrays.  ``frame.loc[:, col] = value`` makes pandas interpret such
        arrays as a vector to broadcast down the dataframe index.  For a one-row
        measurement dataframe this raises errors such as "length of values (187)
        does not match length of index (1)".  Store the value as one object cell
        instead.
        """
        if col not in frame.columns:
            frame[col] = pd.Series([None], dtype='object')
        frame.at[frame.index[0], col] = value

    # Preserve frame annotations when replacing an existing object.
    for flag in flag_list:
        col = flag['flag_column']
        if same_frame_rep is not None and col in same_frame_rep.index:
            _set_single_row_value(row, col, same_frame_rep[col])
        else:
            _set_single_row_value(row, col, False)

    track_meta_cols = ['parent','root','generation','accepted','promise','rejected']
    if track_rep is not None:
        for col in track_meta_cols:
            if col in track_rep.index:
                _set_single_row_value(row, col, track_rep[col])
    else:
        _set_single_row_value(row, 'parent', label_id)
        _set_single_row_value(row, 'root', label_id)
        _set_single_row_value(row, 'generation', 0)
        _set_single_row_value(row, 'accepted', False)
        _set_single_row_value(row, 'promise', False)
        _set_single_row_value(row, 'rejected', False)

    # Preserve columns not recalculated by regionprops from the old row/track.
    # Use scalar-cell assignment so object-valued fields such as ``image`` remain
    # a single dataframe value rather than being interpreted as an iterable.
    for col in df.columns:
        if col in row.columns:
            continue
        if same_frame_rep is not None and col in same_frame_rep.index:
            _set_single_row_value(row, col, same_frame_rep[col])
        elif track_rep is not None and col in track_rep.index:
            _set_single_row_value(row, col, track_rep[col])
        elif col in ('size_x', 'size_y'):
            vals = df[col].dropna() if col in df.columns else []
            _set_single_row_value(row, col, vals.iloc[0] if len(vals) else np.nan)
        else:
            _set_single_row_value(row, col, np.nan)

    return row


def reconcile_frame_to_labels(channel_list, my_labels, df, current_frame, object_properties,
                              flag_list, active_label=None):
    """Fast authoritative commit for a napari Modify Label operation.

    The active napari label is *always* remeasured and replaces its dataframe
    row.  Only additional labels whose pixels were actually touched by that edit
    are remeasured.  Labels removed completely by Fill are deleted.

    This keeps the common Modify Label case to one regionprops/intensity pass,
    while still serializing Fill operations that absorb or expose neighboring
    labels.  Unlike the previous incremental implementation, measurement failure
    raises an exception instead of silently returning the old dataframe.
    """
    if active_label is None:
        raise ValueError('active_label is required for an authoritative Modify Label commit.')

    current_frame = int(current_frame)
    active_label = int(active_label)
    frame_image = np.asarray(my_labels[current_frame])
    old_frame_image = label_frame_from_df(df, current_frame)

    if active_label <= 0 or not np.any(frame_image == active_label):
        raise ValueError(
            f'Selected label {active_label} is not present in frame {current_frame}.'
        )

    current_ids = set(int(x) for x in np.unique(frame_image) if int(x) > 0)
    old_ids = set(int(x) for x in np.unique(old_frame_image) if int(x) > 0)

    # Pixels in either the previous or current active-label footprint are the
    # only pixels this Modify Label operation can authoritatively claim changed.
    touched_pixels = ((old_frame_image == active_label) | (frame_image == active_label))
    touched_ids = {active_label}
    if np.any(touched_pixels):
        touched_ids.update(int(x) for x in np.unique(old_frame_image[touched_pixels]) if int(x) > 0)
        touched_ids.update(int(x) for x in np.unique(frame_image[touched_pixels]) if int(x) > 0)

    # Any label that vanished anywhere in the frame is stale and must not be
    # allowed to reappear on reload.
    deleted_ids = sorted(old_ids - current_ids)

    # Remeasure the active label every time.  Remeasure neighboring labels only
    # when they still exist after being touched by the active-label edit.
    measure_ids = [active_label]
    measure_ids.extend(sorted(
        label_id for label_id in touched_ids
        if label_id != active_label and label_id in current_ids
    ))

    signal_image = create_intensityImage(channel_list, current_frame)
    rebuilt_rows = []
    for label_id in measure_ids:
        rebuilt_rows.append(
            _measure_label_row_from_frame(
                channel_list, my_labels, df, current_frame, label_id,
                object_properties, flag_list, signal_image=signal_image
            )
        )

    # Remove exactly the rows that are being replaced plus labels the user
    # removed completely.  Untouched rows remain unchanged and therefore cheap.
    replace_ids = set(measure_ids) | set(deleted_ids)
    numeric_track_ids = pd.to_numeric(df.track_id, errors='coerce')
    remove_mask = (df.t == current_frame) & numeric_track_ids.isin(replace_ids)
    working_df = df.loc[~remove_mask, :].copy()

    rebuilt = pd.concat(rebuilt_rows, ignore_index=True)
    all_cols = list(dict.fromkeys(list(working_df.columns) + list(rebuilt.columns)))
    working_df = working_df.reindex(columns=all_cols)
    rebuilt = rebuilt.reindex(columns=all_cols)
    working_df = pd.concat([working_df, rebuilt], ignore_index=True)

    actions = {
        'affected_labels': sorted(touched_ids),
        'rebuilt_labels': measure_ids,
        'deleted_labels': deleted_ids,
        'errors': [],
    }
    return working_df.reset_index(drop=True), actions

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
                    image_segment = image_segment[:segment.shape[0], :]

                # Adjust columns
                col_diff = segment.shape[1] - image_segment.shape[1]
                if col_diff > 0:  # need to pad
                    image_segment = np.pad(image_segment, ((0, 0), (0, col_diff)), mode='constant', constant_values=0)
                elif col_diff < 0:  # need to trim
                    image_segment = image_segment[:, :segment.shape[1]]

                # Assign only the pixels belonging to this stored mask.  Do
                # not add integer label IDs together when bounding boxes overlap.
                target = label_image[min_row:max_row, min_col:max_col]
                mask = image_segment != 0
                target[mask] = int(my_cell.track_id)
                label_image[min_row:max_row, min_col:max_col] = target
                                   
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