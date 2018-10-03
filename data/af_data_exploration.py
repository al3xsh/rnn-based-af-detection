"""
af_data_exploration.py

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
# read and save the data to use in training
#

# set the directory where the data lives
root_dir = ('/home/alex/data/medical/atrial_fibrillation/' +
            'patient_data_100_beat_window_99_beat_overlap/train/')

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
datafile = './data/training_data.npz'
np.savez(datafile, 
         x_data=x_data, y_data=y_data)

# count the number of samples exhibitning af
print('there are {0} out of {1} samples that have at least one beat that '
      'is classified as atrial fibrillation'.format(sum(y_data), len(y_data)))

#
# lets visualise some of the data 
#

# import matplotlib plotting magic
import matplotlib.pyplot as plt

# lets plot every patient in our training and validation sets
i = 0
for patient in frames:

    # reshape the data sequences of a patient until we have a single 
    # consecutive heart rate trace (like the original data)
    patient.drop(patient.columns[0], axis=1, inplace=True)
    hr_trace = patient.iloc[::100, :]
    hr_trace_seq = hr_trace.values.reshape(1,-1)
    plt.figure()
    plt.title('patient id = {0}'.format(patients[i]))
    plt.plot(hr_trace_seq.transpose(), '.')
    plt.savefig('./log/figures/4_fixed_data/{0}.png'.format(patients[i]))
    i = i + 1

#
# split the data into training (75%) and validation (25%) to use in our 
# initial model development
#
# note: when we start to properly fine tune the models we should use 10-fold 
# cross validation to evaluate the effects of the model structure and
# parameters 
#

# create train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    stratify=y_data, 
                                                    test_size=0.25,
                                                    random_state=seed)

# reformat the training and test inputs to be in the format that the lstm 
# wants
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# save the data with numpy so we can use it later
datafile = './data/training_and_validation.npz'
np.savez(datafile, 
         x_train=x_train, x_test=x_test, 
         y_train=y_train, y_test=y_test)