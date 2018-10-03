"""
af_results_analysis_crossvalidation.py

load up the results we have got from the stratified 10 fold crossvalidation
process, analyse the data, and show some pretty graphs 

author:     alex shenfield
date:       27/04/2018
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# my visualisation utilities for plotting some pretty graphs of classifier 
# performance
import visualisation_utils as my_vis

#
# main code
#

# we used 10-folds for the stratified cross validation process
nfolds = 10

# set the root directory for the data
results_path = './model/cross_validation_20180417_1801/'

# get the predictions and ground truths from each cross validation fold
predictions = list()
true_labels = list()
for f in range(0, nfolds):
    predictions.append(np.load(results_path + 
                               'fold_{0}/test_predictions.npy'.format(f)))
    true_labels.append(np.load(results_path + 
                          'fold_{0}/test_labels.npy'.format(f)))
   
# concatenate into a single vector
y_pred = np.concatenate(predictions)
y_true = np.concatenate(true_labels)

#
# confusion matrix and plot
#

# set the names of the classes
classes = ['normal', 'af']

# get the confusion matrix and plot both the un-normalised and normalised
# confusion plots 
cm = confusion_matrix(y_true, np.round(y_pred))

plt.figure(figsize=[5,5])
my_vis.plot_confusion_matrix(cm, 
                      classes=classes,
                      title=None)
plt.savefig('./results/cv_confusion_plot.png',
            dpi=600, bbox_inches='tight', pad_inches=0.5)

plt.figure(figsize=[5,5])
my_vis.plot_confusion_matrix(cm, 
                      classes=classes,
                      normalize=True,
                      title=None)
plt.savefig('./results/cv_confusion_plot_normalised.png',
            dpi=600, bbox_inches='tight', pad_inches=0.5)
plt.show()

#
# calculate and plot the roc curve
# 
plt.figure(figsize=[5,5])
title = 'Receiver operating characteristic curve showing ' \
        'AF diagnostic performance'
my_vis.plot_roc_curve(y_pred, y_true, title=None)        
plt.savefig('./results/cv_roc_curve.png',
            dpi=600, bbox_inches='tight', pad_inches=0.5)
plt.show()