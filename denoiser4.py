"""
Title: Convolutional Autoencoder For Image Denoising
Author: [Santiago L. Valdarrama](https://twitter.com/svpino)
Date created: 2021/03/01
Last modified: 2021/03/01
Description: How to train a deep convolutional autoencoder for image denoising.
"""

"""
## Introduction
This example demonstrates how to implement a deep convolutional autoencoder
for image denoising, mapping noisy digits images from the MNIST dataset to
clean digits images. This implementation is based on an original blog post
titled [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
by [François Chollet](https://twitter.com/fchollet).
"""

"""
## Setup
"""

import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from keras.datasets import mnist
from keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler

import numpy as np
import matplotlib.pyplot as plt
import os

import random
import keras.optimizers#.Adam
from  keras.optimizers import Adam

import tensorflow as tf

import librosa
import librosa.display
import pandas as pd
import warnings
import math

dataSize=128

timesteps = 128 # Length of your sequences
input_dim = 128 
latent_dim = 8
epochs=150
# Your data source for wav files
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'


def preprocess(array, labels):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    lookback = latent_dim
    #array=np.array(array)
    maxi=0
    #for i in range(array.shape[0]):
    #   if (maxi<np.max(array[i]):
    #       maxi= np.max(array[i])
    #print("arrshape1:", array.shape)
    #print("labshape:", labels)
    #array, labels =  temporalize(array, labels, lookback)
    #print("arrshape2:", array.shape)
    #array = np.expand_dims(array, -1)
    array = np.expand_dims(array, -1).astype("float32") / np.max(array)
    array = np.reshape(array, (array.shape[0], dataSize*dataSize,1))
     
    return array, labels

def importData():
    dataSet = []
    lblmap ={}
    lblid=0
    totalCount = 0
    progressThreashold = 100
    dirlist = os.listdir(dataSourceBase)
    for dr in dirlist:
      dataSource = os.path.join(dataSourceBase,dr)
      for root, _, files in os.walk(dataSource):
        for file in files:
            fileName, fileExtension = os.path.splitext(file)
            if fileExtension != '.wav': continue
            if totalCount % progressThreashold == 0:
                print('Importing data count:{}'.format(totalCount))
            wavFilePath = os.path.join(root, file)
            y, sr = librosa.load(wavFilePath, duration=2.97)
            ps = librosa.feature.melspectrogram(y=y, sr=sr)
            if ps.shape != (128, 128): continue
            
            # extract the class label from the FileName
            label0 = dr.split('-')[1]
            if label0 not in lblmap:
               lblmap[label0] =lblid
               lblid+=1
            label=lblmap[label0]
            #label = dr#fileName.split('-')[1]
            print(fileName, label0, label)
            dataSet.append( (ps, label) )
            totalCount += 1
    f = open('dict50.csv','w')
    f.write("classID,class")
    for lb in lblmap:
       f.write(str(lblmap[lb])+','+lb)
    f.close()

    global totalRecordCount
    totalRecordCount = totalCount
    
    print('TotalCount: {}'.format(totalRecordCount))
    trainDataEndIndex = int(totalRecordCount*0.8)
    random.shuffle(dataSet)

    train = dataSet[:trainDataEndIndex]
    test = dataSet[trainDataEndIndex:]

    print('Total training data:{}'.format(len(train)))
    print('Total test data:{}'.format(len(test)))

    # Get the data (128, 128) and label from tuple
    print("train 0 shape is ",train[0][0].shape)
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    

    return (X_train, y_train), (X_test, y_test)#dataSet


def temporalize(X, y, lookback):
    '''
    Inputs
    X         A 3D numpy array ordered by time of shape: 
              (n_observations x steps_per_ob x n_features)
    y         A 1D numpy array with indexes aligned with 
              X, i.e. y[i] should correspond to X[i]. 
              Shape: n_observations.
    lookback  The window size to look back in the past 
              records. Shape: a scalar.

    Output
    output_X  A 4D numpy array of shape: 
              ((n_observations-lookback-1) x steps_per_ob x lookback x 
              n_features)
    output_y  A 1D array of shape: 
              (n_observations-lookback-1), aligned with X.
    '''
    output_X = []
    output_y = []
    for i in range(len(X) - lookback - 1):
        print('look', i, len(output_X), len(output_y))
        t=[]
        for j in range(1, lookback + 1):
            # Gather the past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
        output_y.append(y[i + lookback + 1])
    #return np.array(output_X), np.array(output_y)
    return np.squeeze(np.array(output_X)), np.array(output_y)

# Display the train data and a version of it with added noise
#display(train_data, noisy_train_data)

"""
## Build the autoencoder
We are going to use the Functional API to build our convolutional autoencoder.
"""
"""
## Prepare the data
"""

# Since we only need images from the dataset to encode and decode, we
# won't use the labels.
(train_data,train_labels), (test_data, test_labels) = importData()#.load_data()

# Normalize and reshape the data
train_data, train_labels = preprocess(train_data,train_labels)
print(train_data.shape)
test_data, test_labels = preprocess(test_data,test_labels)
print(test_data.shape)

def build():
    inputs = layers.Input(shape=(128*128, 1))
    #conv = layers.Reshape((16384, 1))(inputs)
    # Encoder
    conv = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)  # 16384x16
    conv = layers.MaxPooling1D(pool_size=2, padding='same')(conv)  # 8192x64

    conv = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv)  # 8192x128
    conv = layers.MaxPooling1D(pool_size=2, padding='same')(conv)  # 4096x128

    # conv = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv)  # 4096x256
    # conv = MaxPooling1D(pool_size=2, padding='same')(conv)  # 2048x256
    #
    # # Decoder
    # conv = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv)  # 2048x64
    # conv = UpSampling1D(size=2)(conv)  # 4096x256
    print('abt to decode', conv.shape)
    deconv = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv)  # 4096x32
    deconv = layers.UpSampling1D(size=2)(deconv)  # 8192x128
    print('abt to decode2', deconv.shape)

    deconv = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(deconv)  # 8192x16
    deconv = layers.UpSampling1D(size=2)(deconv)  # 16384x64

    deconv = layers.Conv1D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(deconv)  # 16384x1

    autoencoder = keras.Model(inputs, deconv, name="auto")
    encoder = keras.Model(inputs, conv, name="encoder")
    autoencoder.summary()
    return autoencoder, encoder#print(' inputs shape is ', inputs.shape)
#mergedModel = Model(inputs=[firstModel.input, secondModel.input], outputs=secondModel.layers[-1].output)


autoencoder, encoder = build()
adamOpt = Adam(lr= 0.001)
autoencoder.compile(optimizer=adamOpt, loss="mean_absolute_error")
#autoencoder.compile(optimizer="adam", loss="mse", learning_rate=0.0001)
#autoencoder.summary()
"""
Now we can train our autoencoder using `train_data` as both our input data
and target. Notice we are setting up the validation data using the same
format.
"""
initial_learning_rate = 0.005
#epochs = 100
decay = initial_learning_rate / epochs
drop = 0.5
epochs_drop = 10.0
def lr_time_based_decay(epoch, lr):
    if True:#epoch < 5:
        #return decay *epochs
        lrate = initial_learning_rate * math.pow(drop,  
             math.floor((1+epoch)/epochs_drop))
        return lrate

    else:
        #return decay*lr#epochs
        #return lr * epoch / (epoch + decay * epoch)
        #return initial_learning_rate / (1 + decay * epoch)
        lrate = initial_learning_rate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
        return lrate


autoencoder.fit(
    x=train_data,
    y=train_data,
    epochs=epochs,
    batch_size=128,
    shuffle=True,
    #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    validation_data=(test_data, test_data)
)

"""
Let's predict on our test dataset and display the original image together with
the prediction from our autoencoder.
Notice how the predictions are pretty close to the original images, although
not quite the same.
"""
autoencoder.save("denoiser_"+str(latent_dim)+".hdf5")
encoder.save("encoder"+str(latent_dim)+".hdf5")

#predictions = autoencoder.predict(test_data)
#display(test_data, predictions)

"""
Now that we know that our autoencoder works, let's retrain it using the noisy
data as our input and the clean data as our target. We want our autoencoder to
learn how to denoise the images.
"""

