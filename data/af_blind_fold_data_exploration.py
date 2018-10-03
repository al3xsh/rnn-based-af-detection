"""
af_blind_fold_data_exporation.py

exploring the atrial fibrillation data set

author:     alex shenfield
date:       01/04/2018
"""

# file handling functionality
import os
import glob

# let's do datascience ...
import pandas as pd
import numpy as np

# fix random seed for reproduciblity
seed = 1337
np.random.seed(seed)

#
# read and save the data to use in testing
#

# set the directory where the data lives
root_dir = ('/home/alex/data/medical/atrial_fibrillation/' +
            'patient_data_100_beat_window_99_beat_overlap/test/')

# get the patient IDs
filelist = glob.glob(root_dir + '*.csv', recursive=True)
patients  = [(os.path.split(i)[1]).split('_')[0] for i in filelist]

# read all the data into a single data frame
frames = [pd.read_csv(p, header=None) for p in filelist]
data   = pd.concat(frames, ignore_index=True)
    
# show what our data looks like
print('we have {0} data points from {1} patients!'.format(data.shape, 
      len(patients)))

# split the data into variables and targets (0 = no af, 1 = af)
x_data = data.iloc[:,1:].values
y_data = data.iloc[:,0].values
y_data[y_data > 0] = 1

# save the data with numpy so we can use it later
datafile = './data/test_data.npz'
np.savez(datafile, 
         x_data=x_data, y_data=y_data)

# count the number of samples exhibitning af
print('there are {0} out of {1} samples that have at least one beat that '
      'is classified as atrial fibrillation'.format(sum(y_data), len(y_data)))