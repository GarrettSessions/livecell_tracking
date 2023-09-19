# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:51:41 2021

@author: kmkedz
"""

import os
import time
import numpy as np
import napari
import matplotlib
from napari import Viewer
from napari.qt.threading import thread_worker
from magicgui import magicgui
from magicgui.widgets import Container, CheckBox

import pyqtgraph as pg
from qtpy.QtWidgets import QVBoxLayout
from PyQt5.QtCore import Qt

import napari_display_functions as my_napari
import general_functions as gen
import gallery_functions as g_f

# this variables will be injected from the main
global exp_dir
global df_name
global channel_list
global graph_list
global object_properties
global time_threshold

global viewer

##########################################################################
# specify for optional saving

def change_status(message):

    global viewer
    
    viewer.status = message

def should_i_save():
    
    global time_threshold
    
    mod_time = os.stat(os.path.join(exp_dir,'df',df_name)).st_mtime
    current_time = time.time()
    
    time_passed = current_time - mod_time
    
    if time_passed > time_threshold:
        
        return True
    else:
        return False

@thread_worker(connect={"returned": change_status})
def save_update():
    
    if should_i_save():
        

        df.to_pickle(os.path.join(exp_dir,'df',df_name))
    
        message = 'Data has been saved.'
    
    else:
        
        message = ' '
        
    return message

##########################################################################
# functions for buttons

def save_data(viewer: Viewer):
    
    global df
    global exp_dir
    global df_name
    
    df.to_pickle(os.path.join(exp_dir,'df',df_name), protocol=4)
    viewer.status = 'Data has been saved.'
       
def cut_track(viewer: Viewer):
    
    global df
    global gen_track_columns
    
    viewer,df = my_napari.cut_track(viewer,df,gen_track_columns)
    
    # optionally save dataframe
    if should_i_save():
        save_update()

def merge_track(viewer: Viewer):
    
    global df
    global gen_track_columns
    
    viewer,df = my_napari.merge_track(viewer,df,gen_track_columns)
    
    # optionally save dataframe
    if should_i_save():
        save_update()
    
def connect_track(viewer: Viewer):
    
    global df
    global gen_track_columns
    
    viewer,df = my_napari.connect_track(viewer,df,gen_track_columns)
    
    # optionally save dataframe
    if should_i_save():
        save_update()

def mod_label(viewer: Viewer):
    
    global df
    global channel_list
    global object_properties
    global gen_track_columns
    global flag_list

    viewer,df = my_napari.update_single_object(viewer,df,channel_list,object_properties,gen_track_columns,flag_list)
    
    active_label = viewer.layers['Labels'].selected_label
    viewer.status = f'Label {active_label} has been modified.'

    # trigger modification of the family tree
    update_lineage_display(None)
    
    # optionally save dataframe
    if should_i_save():
        save_update()
    
def select_label(layer, event):
    
    global viewer

    if(event.button == 2):
        
        # get data
        inProgressTracks = viewer.layers['Labels'].data
        
        # look up cursor position
        x = int(viewer.cursor.position[1])
        y = int(viewer.cursor.position[2])
        
        # check which cell was clicked
        myTrackNum = inProgressTracks[viewer.dims.current_step[0],x,y]
        
        # set track as active
        viewer.layers['Labels'].selected_label = int(myTrackNum)
        
def toggle_track(layer, event):

    global df
    global viewer
    global tag_list
    global gen_track_columns
    global widget_label_promise
    global widget_list_promise
    global promise_list
    
    if(event.button == 2):
    
        # look up cursor position
        t = viewer.dims.current_step[0]
        x = int(viewer.cursor.position[1])
        y = int(viewer.cursor.position[2])

        # get data
        labels_layer = viewer.layers['Labels']
        inProgressTracks = labels_layer.data
    
        # check which cell was clicked
        myTrackNum = inProgressTracks[t,x,y]
    
        if myTrackNum > 0:

            track_ind = [x['tag_name'] for x in tag_list].index(layer.name)
            tag_column = tag_list[track_ind]['tag_column']
    
            # check status
            track_status_old = list(df.loc[df.track_id==myTrackNum,tag_column])[0]
            track_status_new = not(track_status_old)
    
            # change status of this track
            df.loc[df.track_id==myTrackNum,tag_column] = track_status_new
    
            # in case a status is activated, others need to be deactivated
            if track_status_new == True:
    
                for tag in tag_list:
    
                    if (tag['tag_name'] != layer.name):
                        df.loc[df.track_id==myTrackNum,tag['tag_column']] = False
    
            ########################################################
            # modify points layer
            ########################################################
    
            # regenerate tags data
            tag_data = gen.tags_from_df(df,tag_list)
    
            # update viewer   
            for tag,tag_data in zip(tag_list,tag_data):
    
                viewer.layers[tag['tag_name']].data = tag_data
    
            ########################################################
            # modify tracking layer
            ########################################################
    
            # modify the data for the layer
            data,properties,graph = gen.trackData_from_df(df,col_list = gen_track_columns)
    
            # change tracks layer
            viewer.layers['Tracking'].data = data
            viewer.layers['Tracking'].color_by = 'track_id'
            viewer.layers['Tracking'].properties = properties
            viewer.layers['Tracking'].graph = graph 
                 
            
            ########################################################
            # modify promising tracks
            ########################################################
            promise_list = [int(x) for x in set(df.loc[df.promise==True,'track_id'])]
            promise_list.append(0)
            promise_list.sort()
             
            widget_label_promise.value = f'Number of promising tracks: {(len(promise_list)-1)}'
            widget_list_promise.choices = promise_list
 

            # optionally save dataframe
            if should_i_save():
                save_update()

def build_lineage_widget(t_max):

    '''
    Builds pyqt widget
    '''
    
    global plot_widget

    plot_widget = pg.GraphicsLayoutWidget()
    plot_view = plot_widget.addPlot(title="Lineage tree", labels={"bottom": "Time"})
    plot_view.hideAxis("left")
    plot_view.setXRange(0, t_max)

    return plot_widget

def build_small_stack_graph_widget(graph_list,df,current_track):

    global empty_frames_list
    
    '''
    Builds pyqt widget for small stack viewer.
    '''
    # select appropriate data
    df_sel = df.loc[df.track_id == current_track,:]
    df_sel = df_sel.sort_values(by='t')
    results_list = gen.extract_graph_data(graph_list,df_sel)

    # find empty frames
    empty_frames_list = gen.find_empty_frames(df_sel.t)

    # create a widget
    stack_plot_widget = pg.GraphicsLayoutWidget()

    symbol_list=['o','t','s','d','+']

    # populate
    for i,graph in enumerate(graph_list):

        signal = results_list[i]

        plot_view = stack_plot_widget.addPlot(title=graph['graph_name'], labels={"bottom": "Time"},col=0,row=i)
        
        hex_color = matplotlib.colors.cnames[graph['color']]
        pen = pg.mkPen(color=hex_color,width=3)

        # mark empty frames
        pen_empty = pg.mkPen(color = (255,0,0),xwidth=1,style=Qt.DotLine)

        for empty_frame in empty_frames_list:
            time_line = plot_view.addLine(x=empty_frame,pen=pen_empty)

            
        # plot from list or a single series
        if type(signal) == list:

            plot_view.addLegend()
            
            graph_ind = 0
            for sub_signal in signal:

                # extract name from function
                plot_name = graph['function'][1:-1].split(',')[graph_ind]

                # proper plotting
                plot_view.plot(np.array(df_sel.t),np.array(sub_signal),pen=pen,symbol=symbol_list[graph_ind],name=plot_name,
                symbolSize=6, symbolBrush = hex_color,symbolPen=None)
                
                graph_ind = graph_ind + 1
        
        else:
                
            plot_view.plot(np.array(df_sel.t),np.array(signal),pen=pen)

    

    return stack_plot_widget

def render_tree_view(plot_view,t,viewer):

    labels_layer = viewer.layers['Labels']

    y_max = 1

    for n in t.traverse():

        if n.is_root():
            pass
        else:

            node_name = n.name

            # get position in time
            x1 = n.start
            x2 = n.stop
            x_signal = [x1,x2]

            # get rendered position (y axis) 
            y_max = np.max([n.y,y_max])
            y_signal = [n.y,n.y]

            label_color = labels_layer.get_color(node_name)
            pen = pg.mkPen(color=pg.mkColor((label_color*255).astype(int)),width=5)

            plot_view.plot(x_signal, y_signal,pen=pen)

            text_item = pg.TextItem(str(node_name),anchor=(1,1))
            text_item.setPos(x2,n.y)
            plot_view.addItem(text_item)

            # check if children are present
            if len(n.children)>0:

                for child in n.children:

                    x_signal = [x2,x2]
                    y_signal = [n.y,child.y]
                    plot_view.plot(x_signal, y_signal,pen=pen)


    # set limits on axis
    t_max = viewer.dims.range[0][1]
    plot_view.setXRange(0, t_max)
    plot_view.setYRange(0, 1.1*y_max)

    return plot_view

def update_lineage_display(event):

    
    global plot_widget # it may be possible to extract from napari, at the moment a global thing
    global viewer
    global time_line
    
    # clear the widget
    plot_view = plot_widget.getItem(0,0)
    plot_view.clear()
    
    # get for whom the update will be
    active_label = viewer.layers['Labels'].selected_label

    if ((active_label > 0) and (active_label in set(df.track_id))):
    
        # init family line
        position = viewer.dims.current_step[0]
        init_family_line(position)

        # find graph for everyone
        _,_,graph = gen.trackData_from_df(df,col_list = ['track_id'])

        # find the root
        my_root = int(list(df.loc[df.track_id==active_label,'root'])[0])
        paths=gen.find_all_paths(graph,my_root)

        # generate the family tree
        t = my_napari.generate_tree_min(paths,df)
        
        # calculate rendering
        t_rendering = t.render('family_tree.png')
        
        # add positions to the tree
        t = my_napari.add_y_rendering(t,t_rendering)

        # create view
        plot_view = render_tree_view(plot_view,t,viewer)

def update_family_line(step_event):
    
    global time_line
    
    slider_pos = step_event.value
    
    time_line.setValue(slider_pos)

def init_family_line(position):
    
    global plot_widget
    global time_line
    global viewer

    plot_view = plot_widget.getItem(0,0)
    
    pen = pg.mkPen(color = (255,255,255),xwidth=2)
    time_line = plot_view.addLine(x=position,pen=pen)
    
    viewer.dims.events.current_step.connect(update_family_line)

def go_to_track_beginning():
    
    global df
    global viewer
    
    # find the beginning of a track
    current_track = viewer.layers['Labels'].selected_label
    t_begin = np.min(df.loc[df.track_id==current_track,'t'])

    # move to the beginning of a track
    viewer.dims.set_point(0, t_begin)

def go_to_track_end():
    
    global df
    global viewer
    
    # find the beginning of a track
    current_track = viewer.layers['Labels'].selected_label
    t_stop = np.max(df.loc[df.track_id==current_track,'t'])

    # move to the beginning of a track
    viewer.dims.set_point(0, t_stop)
    
def center_the_cell():
    
    global df
    global viewer
    
    # find position of a cell
    current_track = viewer.layers['Labels'].selected_label
    current_frame = viewer.dims.current_step[0]
    
    x = df.loc[((df.track_id==current_track) & (df.t==current_frame)),'x']
    
    if len(x)>0:
        
        x = list(x)[0]
        y = list(df.loc[((df.track_id==current_track) & (df.t==current_frame)),'y'])[0]
        
    else:
        
        t_begin = np.min(df.loc[df.track_id==current_track,'t'])
        x = list(df.loc[((df.track_id==current_track) & (df.t==t_begin)),'x'])[0]
        y = list(df.loc[((df.track_id==current_track) & (df.t==t_begin)),'y'])[0]

        # move the frame (to the beginning of the track)
        viewer.dims.set_point(0, t_begin)
        
    # move the camera   
    viewer.camera.center = (0,x,y)
    viewer.camera.zoom = 2

def select_promising_track(val_coming: int):

    global viewer
    
    # make selected track active
    viewer.layers['Labels'].selected_label = val_coming

    # center the selected track
    center_the_cell()

def next_label():
    
    global viewer
    
    # find new track number
    newTrack = gen.newTrack_number(df.track_id)
    
    # set new track id
    viewer.layers['Labels'].selected_label = newTrack

#############################################

def update_stack(viewer_stack,active_track):
    
    global viewer
    global channel_list
    global small_im_size
    global label_contour
    global df
    
    labels = viewer.layers['Labels'].data
        
    # ask for an update
    stack_im_list,stack_labels = g_f.stack_create_all(channel_list,labels,df,active_track,imSize=small_im_size)
    
    # display new layers
    my_napari.display_set(viewer_stack,stack_labels,stack_im_list,channel_list,label_contour = label_contour)

def update_graph_qt(viewer_stack,active_track):
    
    global df
    global viewer
    global graph_list
    global stack_plot_widget
    
    # remove previous graph
    viewer_stack.window.remove_dock_widget(stack_plot_widget)
        
    # add new graph
    stack_plot_widget = build_small_stack_graph_widget(graph_list,df,active_track)
    viewer_stack.window.add_dock_widget(stack_plot_widget,name='Single Track Data')

def update_flag_buttons(viewer_stack,active_track):

    global flag_buttons_widget
    global flag_buttons
    global flag_list

    viewer_stack.window.remove_dock_widget(flag_buttons_widget)

    # create single frame annotation buttons
    flag_buttons = create_annotation_options(flag_list)
    flag_buttons_widget = viewer_stack.window.add_dock_widget(flag_buttons,area='left',name='Annotate Events')

def update_stack_button_f(viewer_stack: Viewer):
    
    global viewer
    global channel_list
    global graph_list
    global small_im_size
    global label_contour
    global df
    global small_stack_widget
    global position_line_list
    global graph_offset
    global flag_buttons
    global flag_line_list

    active_label = viewer_stack.layers['Labels'].selected_label

    # update stack
    update_stack(viewer_stack,active_label)

    # update graph
    update_graph_qt(viewer_stack,active_label) 
    
    # calculate offset on x axis
    graph_offset = gen.calculate_graph_offset(df,active_label)
    
    # initiate the position line and connect to the function
    init_lines_qt(viewer_stack)

    # initiate visualization of single events on the graph
    flag_line_list = update_lines_flags(active_label)

    # connect flag buttons to slider
    update_flag_buttons(viewer_stack,active_label)

    # connect flag buttons to slider
    viewer_stack.dims.events.current_step.connect(update_annotation_buttons)

def init_lines_qt(viewer_stack):

    global stack_plot_widget
    global position_line_list
    global graph_offset
    
    x = viewer_stack.dims.current_step[0] + graph_offset  
    
    pen = pg.mkPen(color = (255,255,255),xwidth=2)

    i=0
    position_line_list = []
    plot_view = stack_plot_widget.getItem(i,0)
    while plot_view != None:

        i=i+1
        position_line = plot_view.addLine(x=x,pen=pen)
        position_line_list.append(position_line)
        
        plot_view = stack_plot_widget.getItem(i,0)
    
    viewer_stack.dims.events.current_step.connect(update_lines_qt)

def update_lines_qt(step_event):

    global stack_plot_widget
    global position_line_list
    global graph_offset

    steps = step_event.value
    slice_num = steps[0] + graph_offset
    
    for position_line in position_line_list:

        position_line.setValue(slice_num)

def clean_lines_flags(flag_line_list):

    global stack_plot_widget

    for widget_plot,flag_line in flag_line_list:

        widget_plot.removeItem(flag_line)

    flag_line_list = []

    return flag_line_list

def get_plots_from_widget(widget):

    i = 0
    plot_list = []
    plot_view = widget.getItem(i,0)
    while plot_view != None:

        i=i+1
        plot_list.append(plot_view)
        
        plot_view = stack_plot_widget.getItem(i,0)

    return plot_list

def update_lines_flags(active_label):

    global stack_plot_widget
    global graph_offset
    global flag_list

    # get part of the dataframe
    df_sel = df.loc[df.track_id==active_label,:]

    # get all the graphs
    plot_list = get_plots_from_widget(stack_plot_widget)

    ind = 0 
    flag_line_list = []
    for flag in flag_list:

        sel_color = flag['flag_color']
        hex_color = matplotlib.colors.cnames[sel_color]

        pen = pg.mkPen(color = hex_color,xwidth=2)
        
        b_column = flag['flag_column']
        positions = df_sel.loc[df_sel[b_column] == True,'t']

        for pos in positions:

            for widget_plot in plot_list:

                flag_line=widget_plot.addLine(x=pos,pen=pen)
                flag_line_list.append([widget_plot,flag_line])
        
        ind = ind + 1

    return flag_line_list
    
def update_annotation_buttons(step_event):

    global graph_offset
    global df
    global flag_list
    global viewer_stack
    global flag_buttons

    # position in time
    slice_num = step_event.value[0] + graph_offset

    # current label
    active_label = viewer_stack.layers['Labels'].selected_label
    
    # selected entry
    sel_vector = ((df.t == slice_num) & (df.track_id == active_label))
    
    # if the object exists
    if(len(list(df.loc[sel_vector,'t'])) > 0):

        ind = 0 
        for flag in flag_list:
            
            b_column = flag['flag_column']
            flag_buttons[ind].value = list(df.loc[sel_vector,b_column])[0]
            
            ind = ind + 1
    else:
        ind = 0 
        for flag in flag_list:
            
            b_column = flag['flag_column']
            flag_buttons[ind].value = False
            
            ind = ind + 1
        
def show_stack(viewer: Viewer):
    
    global stack_plot_widget
    global position_line_list
    global graph_offset
    global empty_frames_list
    global df
    global viewer_stack
    global flag_buttons
    global flag_buttons_widget
    global flag_line_list
    
    # find current track
    active_label = viewer.layers['Labels'].selected_label
    
    # init stack viewer
    try:
        viewer_stack.close()
        # save data
        save_update()
    except:
        pass
    viewer_stack = napari.Viewer()
    update_stack(viewer_stack, active_label)
    
    # set the right label
    viewer_stack.layers['Labels'].selected_label = active_label
    
    # init graph
    stack_plot_widget = build_small_stack_graph_widget(graph_list,df,active_label)
    viewer_stack.window.add_dock_widget(stack_plot_widget,name='Signals')
    
    # calculate offset on x axis
    graph_offset = gen.calculate_graph_offset(df,active_label)
    
    # initiate the position line and connect to the function
    init_lines_qt(viewer_stack)

    # create empty frames navigation
    backward_empty_button = magicgui(backward_button_f, call_button='<')
    sync_frames_button = magicgui(sync_frames_f, call_button='<->')
    forward_empty_button = magicgui(forward_button_f, call_button='>')

    container = Container(widgets=[backward_empty_button,sync_frames_button, forward_empty_button],layout='horizontal',labels=False)
    viewer_stack.window.add_dock_widget(container,area='left',name='Navigate Frames')

    # initiate visualization of single events on the graph
    flag_line_list = update_lines_flags(active_label)

    # create single frame annotation buttons
    flag_buttons = create_annotation_options(flag_list)
    flag_buttons_widget = viewer_stack.window.add_dock_widget(flag_buttons,area='left',name='Annotate Events')

    # connect flag buttons to slider
    viewer_stack.dims.events.current_step.connect(update_annotation_buttons)

    # create an update button and connect to the function
    update_stack_button = magicgui(update_stack_button_f, call_button='Update Stack')
    viewer_stack.window.add_dock_widget(update_stack_button,area='bottom')

def create_annotation_options(flag_list):

    w_list = []
    for flag in flag_list:
    
        button = create_annotation_button(flag)
        w_list.append(button)
    
    c = Container(widgets=w_list,labels=False)

    return c

def create_annotation_button(flag):
    
    global viewer_stack
    global graph_offset
    global df
    global flag_line_list

    # find current track
    active_label = viewer_stack.layers['Labels'].selected_label
       
    b_name = f"{flag['flag_name']} {flag['flag_color']}".replace(' ','_')
    b_column = flag['flag_column']
    b_unique = flag['flag_unique']
    temp = CheckBox(value=False,name=b_name)
    
    @temp.changed.connect
    def button_changed(event):

        global viewer_stack
        global flag_line_list
        
        t_position = viewer_stack.dims.current_step[0] + graph_offset

        if (b_unique == True and temp.value == True):

            df.loc[df.track_id == active_label,b_column] = False
        
        df.loc[((df.t == t_position ) & (df.track_id == active_label)),b_column] = temp.value

        # update the graphs
        # clean lines
        flag_line_list = clean_lines_flags(flag_line_list)
        flag_line_list = update_lines_flags(active_label)

        # optionally save dataframe
        if should_i_save():
            save_update()
        
    return temp  

def backward_button_f(viewer_stack: Viewer):

    global empty_frames_list
    global graph_offset
    global viewer

    current_frame = viewer_stack.dims.current_step[0]

    # find first backward empty frame
    empty_frames_array = np.array(empty_frames_list) - graph_offset

    if np.sum([empty_frames_array < current_frame])>0:
        
        go_empty = empty_frames_array[empty_frames_array < current_frame][-1]

        # set viewer position
        viewer_stack.dims.set_point(0, go_empty)
        viewer.dims.set_point(0, go_empty+graph_offset)
    else:
        viewer_stack.status = 'No empty frames in this direction.'

    return viewer_stack

def forward_button_f(viewer_stack: Viewer):

    global empty_frames_list
    global graph_offset
    global viewer

    current_frame = viewer_stack.dims.current_step[0]

    # find first backward empty frame
    empty_frames_array = np.array(empty_frames_list) - graph_offset
    if np.sum([empty_frames_array > current_frame])>0:
        
        go_empty = empty_frames_array[empty_frames_array > current_frame][0]

        # set viewer position
        viewer_stack.dims.set_point(0, go_empty)
        viewer.dims.set_point(0, go_empty+graph_offset)
    else:
        viewer_stack.status = 'No empty frames in this direction.'

    return viewer_stack

def sync_frames_f(viewer_stack:Viewer):
    
    global graph_offset
    global viewer

    current_frame = viewer_stack.dims.current_step[0]

    viewer.dims.set_point(0, current_frame+graph_offset)

##########################
# DEPRECATED

'''
def update_stack(viewer_stack,active_track):
    
    global viewer
    global channel_list
    global small_im_size
    global label_contour
    global df
    
    labels = viewer.layers['Labels'].data
        
    # ask for an update
    stack_im_list,stack_labels = g_f.stack_create_all(channel_list,labels,df,active_track,imSize=small_im_size)
    
    # display new layers
    my_napari.display_set(viewer_stack,stack_labels,stack_im_list,channel_list,label_contour = label_contour)
      
def update_graph(viewer_stack,active_track):
    
    global df
    global viewer
    global graph_list
    global mpl_widget
    
    # remove previous graph
    h = viewer_stack.window._dock_widgets['']
    viewer_stack.window.remove_dock_widget(h)
        
    # add new graph
    mpl_widget = my_napari.create_graph_widget(graph_list,df,active_track,viewer) 
    h = viewer_stack.window.add_dock_widget(mpl_widget)
   
def update_stack_button_f(viewer_stack: Viewer):
    
    global viewer
    global channel_list
    global graph_list
    global small_im_size
    global label_contour
    global df
    global mpl_widget
    global position_line_list
    global graph_offset

    active_label = viewer_stack.layers['Labels'].selected_label

    # update stack
    update_stack(viewer_stack,active_label)

    # update graph
    update_graph(viewer_stack,active_label) 
    
    # calculate offset on x axis
    graph_offset = gen.calculate_graph_offset(df,active_label)
    
    # initiate the position line and connect to the function
    init_lines(viewer_stack)

def init_lines(viewer_stack):

    global mpl_widget
    global position_line_list
    global graph_offset
    
    x = viewer_stack.dims.current_step[0] + graph_offset  
 
    position_line_list = []
    for lin_num in range(len(mpl_widget.figure.axes)):
        
        position_line = mpl_widget.figure.axes[lin_num].axvline(x=x,color='black')
        position_line_list.append(position_line)
    
        # connect the function to the dims axis
        viewer_stack.dims.events.current_step.connect(update_lines)
    
def update_lines(step_event):

    global mpl_widget
    global position_line_list
    global graph_offset

    steps = step_event.value
    slice_num = steps[0] + graph_offset
    
    for position_line in position_line_list:

        current_pos = position_line.get_data()[0][0]

        if slice_num == current_pos:
            return
        position_line.set_data([slice_num, slice_num], [0, 1])

        mpl_widget.draw_idle()    

def show_stack(viewer: Viewer):
    
    global mpl_widget
    global position_line_list
    global graph_offset
    
    # find current track
    active_label = viewer.layers['Labels'].selected_label
    
    # init stack viewer
    viewer_stack = napari.Viewer()
    update_stack(viewer_stack, active_label)
    
    # set the right label
    viewer_stack.layers['Labels'].selected_label = active_label
    
    # init graph
    mpl_widget = my_napari.create_graph_widget(graph_list,df,active_label,viewer) 
    viewer_stack.window.add_dock_widget(mpl_widget)
    
    # calculate offset on x axis
    graph_offset = gen.calculate_graph_offset(df,active_label)
    
    # initiate the position line and connect to the function
    init_lines(viewer_stack)
    
    # create an update button and connect to the function
    update_stack_button = magicgui(update_stack_button_f, call_button='Update Stack')
    viewer_stack.window.add_dock_widget(update_stack_button,area='bottom')

'''