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
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import layers
from keras.datasets import mnist
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
import os

import random
import keras.optimizers
import librosa
import librosa.display
import pandas as pd
import warnings
import tensorflow as tf

dataSize=128

timesteps = 128 # Length of your sequences
input_dim = 128 
latent_dim = 16

# Your data source for wav files
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'


def preprocess(array, labels):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    lookback = latent_dim
    array=np.array(array)
    maxi=0
    #for i in range(array.shape[0]):
    #   if (maxi<np.max(array[i]):
    #       maxi= np.max(array[i])
    print("arrshape:", array.shape)
    array, labels =  temporalize(array, labels, lookback)
    print("arrshape:", array.shape)
    array = np.array(array).astype("float32") / np.max(array)
    array = np.reshape(array, (len(array), dataSize, dataSize,1))
     
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
    trainDataEndIndex = int(totalRecordCount*0.7)
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
        for j in range(1, lookback + 1):
            # Gather the past records upto the lookback period
            #t.append(X[[(i + j + 1)], :])
            #output_X.append(t)
            output_X.append(X[[(i + j + 1)], :])
        output_y.append(y[i + lookback + 1])
    return np.array(output_X), np.array(output_y)
    #np.squeeze(np.array(output_X)), np.array(output_y)

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


inp = keras.Input(shape=[timesteps, input_dim,1])
#inputs = layers.Reshape(target_shape=(timesteps,input_dim,1),input_shape =(None,timesteps,input_dim,1))(inputs)

#inp2 = layers.Reshape(target_shape=(input_dim,input_dim,1))(inp)
#print(inp.shape)
filter1 = layers.MaxPool2D(pool_size=(1, input_dim//latent_dim))(inp2) 
print(filter1.shape, ' vs (',input_dim, ',' ,input_dim//latent_dim*2, ',',1,')' )
filter1  = layers.Reshape((input_dim, input_dim//latent_dim *2))(filter1) 
print('after reshape', filter1.shape)
filter1  = layers.Bidirectional(layers.LSTM(input_dim//latent_dim, return_sequences=True))(filter1)
filter1  = layers.LSTM(input_dim//latent_dim)(filter1)
#encoded  = layers.Reshape((input_dim, input_dim//latent_dim *2))(encoded) 

filter2 = layers.MaxPool2D(pool_size=(1, latent_dim))(inp2) 
print(filter2.shape, ' vs (',input_dim, ',' ,latent_dim, ',',1,')' )
filter2  = layers.Reshape((input_dim, latent_dim//2))(filter2) 
print('after reshape', filter2.shape)
filter2  = layers.Bidirectional(layers.LSTM(input_dim//latent_dim, return_sequences=True))(filter2)
filter2  = layers.LSTM(latent_dim)(filter2)
#encoded  = layers.Reshape((input_dim, input_dim//latent_dim *2))(encoded) 

encoded = layers.Concatenate(axis=1)([filter1,filter2])
 

print('after ls', encoded.shape)
decoded = layers.RepeatVector(timesteps)(encoded)
decoded = layers.LSTM(input_dim, return_sequences=True)(decoded)    
decoded = layers.Bidirectional(layers.LSTM(input_dim//2, return_sequences=True))(decoded)
print(decoded.shape, ' vs (',input_dim, ',' ,input_dim//latent_dim * 2,',',1,')' )
decoded  = layers.Reshape((input_dim, input_dim,1))(decoded) 
print(decoded.shape, ' b4 con2d ', )
#decoded = layers.Conv2DTranspose( kernel_size=(2,2), activation='relu', padding='valid',input_shape=decoded.shape)(decoded) 
#print(decoded.shape, ' after con2d ', )

#decoded = layers.Reshape((timesteps, input_dim, 1))(decoded)
   
'''
  inp = layers.Input((latent_dim))
    x   = layers.Reshape((1, latent_dim))(inp)
    x   = layers.ZeroPadding1D((0, length - 1))(x)
    x   = layers.LSTM(latent_dim, return_sequences=True)(x)    
    x   = layers.Bidirectional(layers.LSTM(output_dim // 2, return_sequences=True))(x)
    x   = layers.Reshape((length, output_dim, 1))(x)
    
    x   = layers.Conv2DTranspose(n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x) 
    x   = layers.Conv2DTranspose(1, kernel_size=(1, 1), activation='linear', padding='same')(x) 

'''
#print(' inputs shape is ', inputs.shape)
#mergedModel = Model(inputs=[firstModel.input, secondModel.input], outputs=secondModel.layers[-1].output)
autoencoder = keras.Model(inputs=inp,outputs= decoded)
encoder = keras.Model(inp, encoded)


'''
input = layers.Input(shape=(dataSize, dataSize, 1))

# Encoder
x = layers.Conv2D(latent_dim, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(latent_dim, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
#x = layers.Conv2D(latent_dim, (3, 3), activation="relu", padding="same")(x)
#x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
#x = layers.Conv2DTranspose(latent_dim, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(latent_dim, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(latent_dim, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
'''

# Autoencoder
#autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="mse")
#autoencoder.summary()
"""
Now we can train our autoencoder using `train_data` as both our input data
and target. Notice we are setting up the validation data using the same
format.
"""

autoencoder.fit(
    x=train_data,
    y=train_data,
    epochs=20,
    batch_size=128,
    shuffle=True,
    validation_data=(test_data, test_data)
)

"""
Let's predict on our test dataset and display the original image together with
the prediction from our autoencoder.
Notice how the predictions are pretty close to the original images, although
not quite the same.
"""
autoencoder.save("denoiser_"+str(latent_dim)+".hdf5")
autoencoder.save("encoder"+str(latent_dim)+".hdf5")

#predictions = autoencoder.predict(test_data)
#display(test_data, predictions)

"""
Now that we know that our autoencoder works, let's retrain it using the noisy
data as our input and the clean data as our target. We want our autoencoder to
learn how to denoise the images.
"""

