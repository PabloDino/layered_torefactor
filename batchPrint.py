"""
Title: Variational AutoEncoder
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/03
Last modified: 2020/05/03
Description: Convolutional Variational AutoEncoder (VAE) trained on MNIST digits.
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from keras_self_attention import SeqSelfAttention
#from keras.layers import  Input, Dense,Activation, Conv2D,\
#	 MaxPooling2D, Reshape
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras import layers

import random
import keras.optimizers as ko
import librosa
import librosa.display
import pandas as pd
import warnings
import os
import time
import matplotlib.pyplot as plt
"""
## Create a sampling layer
"""

# Your data source for wav files
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'

# Total wav records for training the model, will22 be updated by the program
totalRecordCount = 0

# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 50

# model parameters for training
viewBatch=2
batchSize = 128
epochs =10
dataSize = 128
dataSize2 = 256
latent_dim = 64#256
embed_dim=4096#16384
digitSize = dataSize-2#124
#digitSize = 124

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
            ps = ps[0:dataSize,0:dataSize]
            dataSet.append( [ps, label] )
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
    #print(dataSet)
    #print (len(dataSet))
    #print(type(dataSet[0][0]))
    #print(dataSet[0][0].shape)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaler = scaler.fit(dataSet[0][0])
    for i in range(len(dataSet)):
       dataSet[i][0] = scaler.transform(dataSet[i][0])

    train = dataSet[:trainDataEndIndex]
    test = dataSet[trainDataEndIndex:]

    print('Total training data:{}'.format(len(train)))
    print('Total test data:{}'.format(len(test)))

    # Get the data (128, 128) and label from tuple
    #print("train 0 shape is ",train[0][0].shape)
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    #print(X_train)
    return (X_train, y_train), (X_test, y_test)#dataSet



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon






def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    numrows = x_train.shape[0]
    for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample = x_train[i*viewBatch:i*viewBatch+viewBatch,]
      z_mean8, _, _ = encoder.predict([[sample, sample]])
      #z_mean8, _, _ = vae.encoder.predict([[sample, sample]])
      if (i==0):
        z_mean=z_mean8
      else:
        z_mean = np.concatenate((z_mean,z_mean8), axis=0)
      print(z_mean.shape)
    #z_mean8, _, _ = vae.encoder.predict([[data, data]])
    ###################################################
    #pca = PCA(n_components=2)
    #z_mean = pca.fit_transform(z_mean8)
    ####################################################
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
    z_mean = tsne.fit_transform(z_mean)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    #plt.show()
    plt.savefig("encoder.att256.png")




"""
## Train the VAE
"""
#testdata = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = importData()#keras.datasets.mnist.load_data()
x_train = np.array(x_train)
x_test = np.array(x_test)
# One-Hot encoding for classes
#y_train = np.array(keras.utils.to_categorical(y_train, totalLabel))#.reshape(1,-1)
#y_test = np.array(keras.utils.to_categorical(y_test, totalLabel))#.reshape(1,-1)
print('x_train, y_train shape is ',x_train.shape, y_train.shape)
x_train *= int(1.0*embed_dim/x_train.max())
x_test *= int(1.0*embed_dim/x_test.max())

mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
print("about to init VAE")

#vae = VAE(encoder, decoder)
encoder = load_model("encoder.att256.h5", custom_objects={'Sampling': Sampling}, compile =False)

numItems = x_train.shape[0]
x_train = x_train.reshape(1,-1)
numvars =x_train.shape[1]
x_train = x_train.reshape(numItems,int(numvars/numItems))
#y_train = y_train.reshape(1,-1)#.reshape(1,-1)
#np.pad(y_train, 128)
#x_train = x_train.reshape( 256, 256, 1)





q1 = np.array([[1, 2, 0]])
q = x_train#np.array([[1, 2, 0]])
print("Q SHAPE IS",q.shape)

x_train = np.expand_dims(x_train, -1).astype("float32") / 255
print(y_train.shape)#, y_train)

'''
vae.built=True
lst = os.listdir()
coarseList=[]
fineList=[]
for f in lst:
  if (f.startswith("coarse2") and f.endswith("hdf5")):
     coarseList.append(f)
  if (f.startswith("fine3") and f.endswith("hdf5")):
     fineList.append(f)
     
print(coarseList)
print(fineList)
'''
plot_label_clusters(encoder, x_train, y_train)

