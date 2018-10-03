"""
af_initial_model_fitting.py

fitting a simple bidirectional LSTM model to the af data - now including extra
dropout, an extra fully connected layer, and using the keras functional model

the ideas behind the bidirectional lstm model come from: 
    
https://machinelearningmastery.com/
develop-bidirectional-lstm-sequence-classification-python-keras/

author:     alex shenfield
date:       01/04/2018
"""

# file handling functionality
import os

# useful utilities
import time
import pickle

# let's do datascience ...
import numpy as np

# import keras deep learning functionality
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import GlobalMaxPool1D
from keras.layers import Dense
from keras.layers import Dropout

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

# fix random seed for reproduciblity
seed = 1337
np.random.seed(seed)

# tell the application whether we are running on a server or not (so as to
# influence which backend matplotlib uses for saving plots)
headless = False

#
# get the data
#

# load the npz file
data_path = './data/training_and_validation.npz'
af_data   = np.load(data_path)

# extract the training and validation data sets from this data
x_train = af_data['x_train']
y_train = af_data['y_train']
x_test  = af_data['x_test']
y_test  = af_data['y_test']

#
# create and train the model
#

# set the model parameters
n_timesteps = x_train.shape[1]
mode = 'concat'
n_epochs = 200
batch_size = 1024

# create a bidirectional lstm model (based around the model in:
# https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
# )
inp = Input(shape=(n_timesteps,1,))
x = Bidirectional(LSTM(200, 
                       return_sequences=True, 
                       dropout=0.1, recurrent_dropout=0.1))(inp)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inp, outputs=x)

# set the optimiser
opt = Adam()

# compile the model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

# set up a model checkpoint callback (including making the directory where to 
# save our weights)
directory = './model/initial_runs_{0}/'.format(time.strftime("%Y%m%d_%H%M"))
os.makedirs(directory)
filename  = 'af_lstm_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpointer = ModelCheckpoint(filepath=directory+filename, 
                               verbose=1, 
                               save_best_only=True)

# fit the model
history = model.fit(x_train, y_train,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=[checkpointer])

# get the best validation accuracy
best_accuracy = max(history.history['val_acc'])
print('best validation accuracy = {0:f}'.format(best_accuracy))

# pickle the history so we can use it later
with open(directory + 'training_history', 'wb') as file:
    pickle.dump(history.history, file)

# set matplotlib to use a backend that doesn't need a display if we are 
# running remotely
if headless:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# plot the results

# accuracy
f1 = plt.figure()
ax1 = f1.add_subplot(111)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('training and validation accuracy of af diagnosis')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.text(0.4, 0.05, 
         ('validation accuracy = {0:.3f}'.format(best_accuracy)), 
         ha='left', va='center', 
         transform=ax1.transAxes)
plt.savefig('af_lstm_training_accuracy_{0}.png'
            .format(time.strftime("%Y%m%d_%H%M")))

# loss
f2 = plt.figure()
ax2 = f2.add_subplot(111)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('training and validation loss of af diagnosis')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.text(0.4, 0.05, 
         ('validation loss = {0:.3f}'
          .format(min(history.history['val_loss']))), 
         ha='right', va='top', 
         transform=ax2.transAxes)
plt.savefig('af_lstm_training_loss_{0}.png'
            .format(time.strftime("%Y%m%d_%H%M")))

# we're all done!
print('all done!')