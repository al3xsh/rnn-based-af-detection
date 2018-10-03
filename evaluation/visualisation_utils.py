"""
visualisation_utils.py

make pretty graphs to show classifier performance

(most of these are based on the really useful examples from the 
scikit learn user guides!)

author:     alex shenfield
date:       27/04/2018
"""

# numpy is needed for everything :)
import numpy as np
import matplotlib.pyplot as plt

# utilities for managing the data
import itertools
from scipy import interp

# data analysis functions from scikit learn
from sklearn.metrics import roc_curve, auc

# define a function for plotting a confusion matrix
def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    # should we normalise the confusion matrix?
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Confusion matrix, with normalization')
    else:
        print('Confusion matrix, without normalization')

    # display in command windows
    print(cm)

    # create a plot for the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)    
    
    # if we want a title displayed
    if title:        
        plt.title(title)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# define a simple function for plotting a roc curve
def plot_roc_curve(y_pred, y_true,
                   title='ROC curve'):

    # get fpr, tpr, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    tpr[0] = 0.0

    # get auc
    roc_auc = auc(fpr, tpr)
    print('AUC was:')
    print(roc_auc)

    # plot data
    plt.plot(fpr, tpr, lw=2,
             label='ROC curve (AUC = %0.4f)' % (roc_auc))
    
    # plot the line showing luck
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    
    # configure the plot
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    # if we want a title displayed
    if title:        
        plt.title(title)
    
    
# plot a set of roc curves from the crossvalidation process
def plot_roc_curve_folds(predictions, true_labels,
                         title='ROC curve'):

    # we'll store the tprs and aucs from each crossvalidation fold so we can
    # calculate averages and standard deviations after
    tprs = list()
    aucs = list()
    mean_fpr = np.linspace(0, 1, 100)    

    # for every fold in our data ...
    nfolds = len(predictions)
    for f in range(0, nfolds):
    
        # get fpr, tpr, and thresholds
        fpr, tpr, thresholds = roc_curve(true_labels[f], predictions[f])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        # get auc
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # plot this fold
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (f, roc_auc))
    
    # calculate averages and standard deviations
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    # plot the average ROC curve
    plt.plot(mean_fpr, mean_tpr, 
             label=r'Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_auc, std_auc),
             color='b', lw=2, alpha=.8)
    
    # plot the line showing luck
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    # calculate confidence bounds
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, 
                     color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    # format the plot neatly
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")