import os
import sys
import shutil

sys.path.append(r'/proj/purvislb/users/Garrett/libraries/')
import input_functions as inp_f

# ## Define experiment
info_file_path = r'/proj/purvislb/users/Garrett/072723_50hr_20uM_TBHP/data/info_B02.txt'

# read the file
info_file = open(info_file_path, 'r')
info_lines = info_file.readlines()
info_file.close()

# read info about the data frame
exp_dir,df_name = inp_f.read_df_info(info_lines)

# get info about the channels
channel_list = inp_f.read_channels(info_lines)

# list files present
file_list = os.listdir(exp_dir)

# create all directories
try:
    os.mkdir(os.path.join(exp_dir,'data'))
except FileExistsError:
    pass

try:
    os.mkdir(os.path.join(exp_dir,'df'))
except FileExistsError:
    pass

try:
    os.mkdir(os.path.join(exp_dir,'analysis'))
except FileExistsError:
    pass

try:
    os.mkdir(os.path.join(exp_dir,'segmentation'))
except FileExistsError:
    pass

# move files to the data folder
for im_file in file_list:
    
    file_path = os.path.join(exp_dir,im_file)
    
    if os.path.isfile(file_path):
        
        os.rename(file_path,os.path.join(exp_dir,'data',im_file))
        
# move info file
shutil.move(info_file_path,os.path.join(exp_dir,os.path.basename(info_file_path)))

# check that all requested files are present
for ch in channel_list:
    
    im_path = os.path.join(exp_dir,'data',ch['file_name'])
    
    if os.path.isfile(im_path):
        pass
    else:
        print('Missing Imaging Files')
        break
        
print('All imaging files present.')