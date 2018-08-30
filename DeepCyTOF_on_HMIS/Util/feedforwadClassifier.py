#!/usr/bin/env python3
'''
Created on Oct 10, 2016

@author: huaminli
'''
from keras.layers import Input, Dense
from keras.models import Model
import keras.optimizers
from keras.regularizers import l2
from keras import callbacks as cb
from keras.callbacks import LearningRateScheduler
import numpy as np
from Util import Monitoring as mn
import sklearn.metrics
import matplotlib
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt1
import math
import random
import tensorflow as tf
from keras.metrics import sparse_categorical_crossentropy
from keras.metrics import sparse_categorical_accuracy
import keras.backend as K
import os.path
from Util import FileIO as io


def step_decay(epoch):
    '''
    Learning rate schedule.
    '''
    initial_lrate = 1e-3
    drop = 0.5
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

def f1score(confusionMatrix):
    '''
    Calculate the F1 score of a given confusion matrix.
    '''
#    col1 = confusionMatrix[1:,:1]
#    confusionMatrix = confusionMatrix[1:,1:]
#    temp = np.zeros(confusionMatrix.shape)
#  
#    for i in range(0, confusionMatrix.shape[0]):
#        for j in range(0, confusionMatrix.shape[1]):
#            if col1[i,0] > 0:
#                temp[i,j] = random.randint(0, col1[i,0])
#                col1[i,0] = col1[i,0] - temp[i,j]
    
#    confusionMatrix = confusionMatrix + temp
#    confusionMatrix = confusionMatrix.astype(int)
    
    sum_C = np.sum(confusionMatrix, axis = 1) # sum of each row
    sum_K = np.sum(confusionMatrix, axis = 0) # sum of each column
    
    Pr = np.divide(confusionMatrix, np.matlib.repmat(np.array([sum_C]).T, 1, 
                                                     confusionMatrix.shape[0]))
    Re = np.divide(confusionMatrix, 
                   np.matlib.repmat(sum_K, confusionMatrix.shape[1], 1))

    F = np.divide(2 * np.multiply(Pr, Re), Pr + Re)

    for i in range(0, F.shape[0]):
        for j in range(0, F.shape[1]):
            if np.isnan(F[i,j]):
                F[i,j] = 0
            
    F = np.max(F, axis = 1)
    return np.dot(sum_C, F)/np.sum(sum_C)

def trainClassifier(trainSample, mode = 'None', i = 0,
                    hiddenLayersSizes = [12, 6, 3],
                    activation = 'softplus', l2_penalty = 1e-4,
                    path = 'None'):
    # Remove unlabeled cells for training.
    x_train = trainSample.X[trainSample.y != 0]
    y_train = trainSample.y[trainSample.y != 0]
    
    # Labels start from 0.
    y_train = np.int_(y_train) - 1

    
    # Special case in GvHD: label in those files are 0,1,3,4 with no 2.
    if mode == 'GvHD' and (i == 5 or i == 9 or 
                           i == 10 or i == 11):
        y_train[y_train != 0] = y_train[y_train != 0] - 1

    # Expand labels, to work with sparse categorical cross entropy.
    y_train = np.expand_dims(y_train, -1)
    
    # Construct a feed-forward neural network.
    inputLayer = Input(shape = (x_train.shape[1],))
    hidden1 = Dense(hiddenLayersSizes[0], activation = activation,
                    kernel_regularizer = l2(l2_penalty))(inputLayer)
    hidden2 = Dense(hiddenLayersSizes[1], activation = activation,
                    kernel_regularizer = l2(l2_penalty))(hidden1)
    hidden3 = Dense(hiddenLayersSizes[2], activation = activation,
                    kernel_regularizer = l2(l2_penalty))(hidden2)
#    numClasses = len(np.unique(trainSample.y)) - 1   # with 0 class
    numClasses = len(np.unique(trainSample.y))       # without 0 class
#    numClasses = 57                                   # for HMIS-2
    outputLayer = Dense(numClasses, activation = 'softmax')(hidden3)
    
    encoder = Model(inputs = inputLayer, outputs = outputLayer)
    net = Model(inputs = inputLayer, outputs = outputLayer)
    lrate = LearningRateScheduler(step_decay)
    optimizer = keras.optimizers.rmsprop(lr = 0.0)

    net.compile(optimizer = optimizer, 
                loss = 'sparse_categorical_crossentropy')
    net.fit(x_train, y_train, epochs = 80, batch_size = 128, shuffle = True,
            validation_split = 0.1, verbose = 0, 
            callbacks=[lrate, mn.monitor(),
            cb.EarlyStopping(monitor = 'val_loss',
                             patience = 25, mode = 'auto')])
    try:
        net.save(os.path.join(io.DeepLearningRoot(),
                              'savemodels/' + path + '/cellClassifier.h5'))
    except OSError:
        pass
    #plt.close('all')
    
    return net

def prediction(testSample, mode, i, net):
    # Labels start from 0.
    y_test = np.int_(testSample.y)
    
    # Special case in GvHD: label in those files are 0,1,3,4 with no 2.
    if mode == 'GvHD' and (i == 5 or i == 9 or 
                           i == 10 or i == 11):
        y_test[y_test > 1] = y_test[y_test > 1] - 1
        
    # Expand labels, to work with sparse categorical cross entropy.
    y_test = np.expand_dims(y_test, -1)
    
    y_test_pred_prob = net.predict(testSample.X, verbose = 0)
    y_test_pred = np.argmax(y_test_pred_prob, axis = 1) + 1
    y_test_pred[np.max(y_test_pred_prob, axis = 1) < .4] = 0
    y_test = np.squeeze(y_test)
    
    # Calculate accuracy.
    acc = np.mean(y_test[y_test!=0] == y_test_pred[y_test!=0])
#    acc = np.mean(y_test == y_test_pred)
    confusionMatrix = sklearn.metrics.confusion_matrix(y_test, y_test_pred,
                                                       labels=None)
#    confusionMatrix = sklearn.metrics.confusion_matrix(y_test[y_test!=0], 
#                                                       y_test_pred[y_test!=0],
#                                                       labels=None)  # for HMIS-2
    F1 = f1score(confusionMatrix)
    
    y_true = y_test[y_test!=0]
    y_true = np.int_(y_true) - 1
    
    print('sample ', i+1)
    print('accuracy: ',np.round(acc*100, 2), '%')

    print('F-measure: ',np.round(F1*100, 2))
    print('confusion matrix:\n', confusionMatrix)
    
    return acc, F1, y_test_pred

def plotHidden(trainSample, testSample, mode = 'None', i = 0,
                    hiddenLayersSizes = [12, 6, 3],
                    activation = 'softplus', l2_penalty = 1e-4,
                    path = 'None'):
    # Remove unlabeled cells for training.
    x_train = trainSample.X[trainSample.y != 0]
    y_train = trainSample.y[trainSample.y != 0]
    x_test = testSample.X[testSample.y != 0]
    y_test = testSample.y[testSample.y != 0]
    
    # Labels start from 0.
    y_train = np.int_(y_train) - 1
    y_test = np.int_(y_test) - 1

    
    # Special case in GvHD: label in those files are 0,1,3,4 with no 2.
    if mode == 'GvHD' and (i == 5 or i == 9 or 
                           i == 10 or i == 11):
        y_train[y_train != 0] = y_train[y_train != 0] - 1

    # Expand labels, to work with sparse categorical cross entropy.
    y_train = np.expand_dims(y_train, -1)
    y_test = np.expand_dims(y_test, -1)
    
    # Construct a feed-forward neural network.
    inputLayer = Input(shape = (x_train.shape[1],))
    hidden1 = Dense(hiddenLayersSizes[0], activation = activation,
                    W_regularizer = l2(l2_penalty))(inputLayer)
    hidden2 = Dense(hiddenLayersSizes[1], activation = activation,
                    W_regularizer = l2(l2_penalty))(hidden1)
    hidden3 = Dense(hiddenLayersSizes[2], activation = activation,
                    W_regularizer = l2(l2_penalty))(hidden2)
    numClasses = len(np.unique(trainSample.y)) - 1
    outputLayer = Dense(numClasses, activation = 'softmax')(hidden3)
    
    encoder = Model(input = inputLayer, output = hidden3)
    # plot data in the 3rd hidden layer
    h3_data = encoder.predict(x_test, verbose = 0)
    #fig, (ax1) = plt1.subplots(1,1, subplot_kw={'projection':'3d'})
    #ax1.scatter(h3_data[:,0], h3_data[:,1], h3_data[:,2], s = 20, c = np.squeeze(y_test))
    
    fig = plt1.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(h3_data[:,0], h3_data[:,1], h3_data[:,2], s = 20, c = np.squeeze(y_test))
    #ax1.set_title('data in 3rd hidden layer')
    plt1.show()
    
    net = Model(input = inputLayer, output = outputLayer)
    lrate = LearningRateScheduler(step_decay)
    optimizer = keras.optimizers.rmsprop(lr = 0.0)

    net.compile(optimizer = optimizer, 
                loss = 'sparse_categorical_crossentropy')
    net.fit(x_train, y_train, nb_epoch = 80, batch_size = 128, shuffle = True,
            validation_split = 0.1, verbose = 0, 
            callbacks=[lrate, mn.monitor(),
            cb.EarlyStopping(monitor = 'val_loss',
                             patience = 25, mode = 'auto')])
    try:
        net.save(os.path.join(io.DeepLearningRoot(),
                              'savemodels/' + path + '/cellClassifier.h5'))
    except OSError:
        pass
    #plt.close('all')
