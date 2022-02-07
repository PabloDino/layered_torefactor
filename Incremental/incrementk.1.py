from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
#from tensorflow.python.keras.callbacks import TensorBoard
#from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Conv1D,Conv2D, GlobalAveragePooling2D, InputLayer, \
                         Flatten, MaxPooling2D,MaxPooling1D, LSTM, ConvLSTM2D, Reshape, Concatenate,concatenate, Input, AveragePooling2D
from keras.layers.normalization import BatchNormalization                      
import tensorflow as tf
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import denseBase
from denseBase import DenseBase

from time import time
import numpy as np
#import matplotlib.pyplot as plt
import os
from config import *
import datetime

import random
import keras.optimizers
import librosa
import librosa.display
import pandas as pd
import warnings




#import tensorflow as tf
#global totalRecordCount
#global totalLabel
totalRecordCount=0
totalLabel=0
lblmap={} 
lblid=0
    
# Your data source for wav files
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-base50/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-Base50p/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-base50p/'
#baseFolder = '/home/paul/Downloads/ESC-50-tst2b/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-Next30p/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-next30p/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-next30p/'
#nextFolder = '/home/paul/Downloads/ESC-50-tst2b/'
lastFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-last20p/'
#lastFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-last20p/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0
dataSize = 128
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)

# model parameters for training
batchSize = 64
epochs = 1000#0


filepath = "ESCvae-model-{epoch:02d}-{loss:.2f}.hdf5"
#checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

#tf.compat.v1.disable_eager_execution()

def init_layer(layer):
    session = K.get_session()
    weights_initializer = tf.variables_initializer(layer.weights)
    session.run(weights_initializer)

def encPredict(enc, x_train):
   viewBatch=1#batchSize#8
   numrows = x_train.shape[0]
   z_mean=[]
   #enc.summary()
   for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      #sample = x_train[i*viewBatch:min(i*viewBatch+viewBatch,len(x_train)),]
      sample =  x_train[i*viewBatch:min(i*viewBatch+viewBatch,len(x_train)),]
      #z_mean8, _, _ = enc.predict([[sample, sample]])
      z_mean8 = enc.predict(sample)
      #print(z_mean8)
      try:
        #print(z_mean8.shape)
        #if len(z_mean8[0].shape)==4:
        #   z_mean8[0]=np.reshape(z_mean8[0],(z_mean8[0].shape[1],z_mean8[0].shape[2],z_mean8[0].shape[3]))
        z_mean.append(np.array(z_mean8[0]))
      except:
        print('error adding wheni=',i, sample) 
      #print('Sample ', i, ' shape ', sample.shape , ' converted to ', z_mean8.shape)
      #if (i==0):
      #  z_mean=z_mean8[0]
      #else:
      #  z_mean = np.concatenate((z_mean,z_mean8[0]))#, axis=0)
      #if True:#(i%200==0) and i>1:  
      #  print("enc stat",z_mean.shape)
   print(len(z_mean) ,' things ::')#, z_mean[0].shape)
   z_mean=np.array(z_mean) 
   return z_mean


def decPredict(dec, x_train):
   viewBatch=1
   numrows = x_train.shape[0]
   for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample = x_train[i*viewBatch:i*viewBatch+viewBatch,]
      #sample = np.reshape(sample, (sample.shape[1], sample.shape[2]))
      #print(sample.shape)
      z_mean8 = dec.predict(sample)
      #z_mean8, _, _ = dec.predict([[sample, sample]])
      if (i==0):
        z_mean=z_mean8
      else:
        z_mean = np.concatenate((z_mean,z_mean8), axis=0)
      if (i%200==0) and i>1:  
        print("dec stat",z_mean.shape)
   return z_mean




def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    print('z_mean shape is ',z_mean.shape, z_log_var.shape)
    
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1] # Returns the shape of tensor or variable as a tuple of int or None entries.
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    #return z_mean*z_mean+ K.exp(0.5 * z_log_var) * epsilon
    #return K.exp(0.5 * z_log_var) * epsilon
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# VAE model = encoder + decoder
# build encoder model
def encoder_model(inputs):
    print('starting encoder model -inputs shape is ', inputs.shape)
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder, z_mean, z_log_var


# build decoder model
def decoder_model():
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder


# This function will import wav files by given data source path.
# And will extract wav file features using librosa.feature.melspectrogram.
# Class label will be extracted from the file name
# File name pattern: {WavFileName}-{ClassLabel}
# e.g. 0001-0 (0001 is the name for the wav and 0 is the class label)
# The program only interested in the class label and doesn't care the wav file name
def importData(setname,train=False):
    global totalRecordCount
    global totalLabel
    global lblmap
    global lblid
    dataSet = []
    totalCount = 0
    progressThreashold = 100
    lblid=totalLabel
    if (setname) == 'base':
        dataSourceBase=baseFolder
        totalLabel+=25
    if (setname) == 'next':
        dataSourceBase=nextFolder
        totalLabel+=15
    if (setname) == 'last':
        dataSourceBase=lastFolder
        totalLabel+=10
    
    dirlist = os.listdir(dataSourceBase)
    for dr in dirlist:
     if not train:
        label0 = dr.split('-')[1]
        if label0 not in lblmap:
           lblmap[label0] =lblid
           lblid+=1
        label=lblmap[label0]
        #label = dr#fileName.split('-')[1]
        print('Not re-training for ', label0, label)
        #dataSet.append( (ps, label) )  
     else: 
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
            dataSet.append( (ps, label) )
            print(fileName, label0, label)
           
            totalCount += 1
    f = open('dict50.csv','w')
    f.write("classID,class")
    for lb in lblmap:
       f.write(str(lblmap[lb])+','+lb)
    f.close()

    totalRecordCount += totalCount
    

    '''
    print('Total training data:{}'.format(len(train)))
    print('Total test data:{}'.format(len(test)))

    # Get the data (128, 128) and label from tuple
    print("train 0 shape is ",train[0][0].shape)
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    '''
    

    #return (X_train, y_train), (X_test, y_test)#dataSet
    #return (train,test)#dataSet
    return dataSet
    
 
def mergeSets2(dset,  nextdset):
    combSet =[]
    for i in range(max(len(dset),len(nextdset))):
        if (i<len(dset)):
           combSet.append(dset[i])
        if (i<len(nextdset)):
           combSet.append(nextdset[i])
    return combSet
        


filepath = 'Model.1.{epoch:02d}-{loss:.2f}.hdf5'
#filepath = 'Model.1.{epoch:02d}-{loss:.2f}.{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=100)


def fitCombined(X_train50p1_encoded, X_train30p1_encoded,X_train20p1_encoded, X_train50p5_encoded, X_train30p5_encoded,X_train20p5_encoded, X_test50p1_encoded, X_test30p1_encoded, X_test20p1_encoded, X_test50p5_encoded, X_test30p5_encoded, X_test20p5_encoded, y_train, y_test,combModel, layernum1, layernum2,perc):#,fixedLayers):
    #print('xtrain shape1:',X_train50p1_encoded.shape)
    #print('xTest shape1:',X_test50p1_encoded.shape)
    '''
    X_train50p1_encoded = np.array([x.reshape( (1,64) ) for x in X_train50p1_encoded])
    X_test50p1_encoded = np.array([x.reshape( (1,64) ) for x in X_test50p1_encoded])
    X_train30p1_encoded = np.array([x.reshape( (1,64) ) for x in X_train30p1_encoded])
    X_test30p1_encoded = np.array([x.reshape( (1,64) ) for x in X_test30p1_encoded])
    X_train50p5_encoded = np.array([x.reshape( (1,84) ) for x in X_train50p5_encoded])
    X_test50p5_encoded = np.array([x.reshape( (1,84) ) for x in X_test50p5_encoded])
    X_train30p5_encoded = np.array([x.reshape( (1,264) ) for x in X_train30p5_encoded])
    X_test30p5_encoded = np.array([x.reshape( (1,264) ) for x in X_test30p5_encoded])
    '''
    '''
    if len(xTrain.shape)==5:
        #inLyr = Input(shape=(int(xTrain[0]),int(xTrain[1]),int(xTrain[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        #branchIn = Input(shape=(int(xTrain[0]),int(xTrain[1]),int(xTrain[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        
        xTrain = np.array([x.reshape( int(xTrain.shape[2]), int(xTrain.shape[3]), int(xTrain.shape[4]) ) for x in xTrain])
        xTest = np.array([x.reshape( int(xTest.shape[2]), int(xTest.shape[3]), int(xTest.shape[4]))  for x in xTest])
        
    else:
      if len(xTrain.shape)==4:
        #inLyr = Input(shape=(int(xTrain.shape[1]),int(xTrain.shape[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        #branchIn = Input(shape=(int(xTrain.shape[1]),int(xTrain.shape[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        xTrain = np.array([x.reshape( int(xTrain.shape[1]),int(xTrain.shape[2]) ) for x in xTrain])
        xTest = np.array([x.reshape( (int(xTest.shape[1]), int(xTest.shape[2])))  for x in xTest])
      #else:
      #  if fixedLayers==14:# and fixedLayers <16:
      #     xTrain = np.array([x.reshape( int(xTrain.shape[2])) for x in xTrain])
      #     xTest = np.array([x.reshape( int(xTest.shape[2]))  for x in xTest])
    '''
    #combModel.summary() 
    #print('xtrain shape2 is ',xTrain.shape)
    #xTrain = np.array([x.reshape( (128, 128, 1) ) for x in xTrain])
    #xTest = np.array([x.reshape( (128, 128, 1 ) ) for x in xTest])
    #xTrain = np.array([x.reshape( (8,8,48) ) for x in xTrain])
    #xTest = np.array([x.reshape( (8,8,48) ) for x in xTest])

    #xTrain = np.expand_dims(xTrain,-1)
    #xTest = np.expand_dims(xTest,-1)

    combModel.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=['accuracy'])
    initial_learning_rate = 0.01
    #epochs = 100
    drop = 0.75
    epochs_drop = 10.0
    decay = initial_learning_rate / epochs
    def lr_time_based_decay(epoch, lr):
       if epoch < 50:
            return initial_learning_rate
       else:
            lrate = initial_learning_rate * math.pow(drop,  
             math.floor((1+epoch)/epochs_drop))
       return lrate
    #print ('xtrain3:', xTrain.shape)
    #print ('xTest:', xTest.shape)
    #print ('xtraincoded:', X_train50p1_encoded.shape)
    
    print('X_train30p1_encoded',X_train30p1_encoded.shape, layernum1, layernum2)
    print('X_train50p5_encoded',X_train50p5_encoded.shape)
    #print('X_train30p5_encoded',X_train30p5_encoded.shape)


    if len(X_train50p1_encoded.shape)==3:
       if  (layernum1>20):
         X_train50p1_encoded=np.reshape(X_train50p1_encoded,(X_train50p1_encoded.shape[0],X_train50p1_encoded.shape[2]))
         X_train30p1_encoded=np.reshape(X_train30p1_encoded,(X_train30p1_encoded.shape[0],X_train30p1_encoded.shape[2])) 
         X_train20p1_encoded=np.reshape(X_train20p1_encoded,(X_train20p1_encoded.shape[0],X_train20p1_encoded.shape[2]))
 
         X_test50p1_encoded=np.reshape(X_test50p1_encoded,(X_test50p1_encoded.shape[0],X_test50p1_encoded.shape[2]))
         X_test30p1_encoded=np.reshape(X_test30p1_encoded,(X_test30p1_encoded.shape[0],X_test30p1_encoded.shape[2]))
         X_test20p1_encoded=np.reshape(X_test20p1_encoded,(X_test20p1_encoded.shape[0],X_test20p1_encoded.shape[2]))
       else:
         X_train50p1_encoded=np.reshape(X_train50p1_encoded,(X_train50p1_encoded.shape[0],1,X_train50p1_encoded.shape[2]))
         X_train30p1_encoded=np.reshape(X_train30p1_encoded,(X_train30p1_encoded.shape[0],1,X_train30p1_encoded.shape[2])) 
         X_train20p1_encoded=np.reshape(X_train20p1_encoded,(X_train20p1_encoded.shape[0],1,X_train20p1_encoded.shape[2])) 
          
         X_test50p1_encoded=np.reshape(X_test50p1_encoded,(X_test50p1_encoded.shape[0],1,X_test50p1_encoded.shape[2]))
         X_test30p1_encoded=np.reshape(X_test30p1_encoded,(X_test30p1_encoded.shape[0],1,X_test30p1_encoded.shape[2]))
         X_test20p1_encoded=np.reshape(X_test20p1_encoded,(X_test20p1_encoded.shape[0],1,X_test20p1_encoded.shape[2]))
                
    if len(X_train50p5_encoded.shape)==3:
       if (layernum2>127):
         X_train50p5_encoded=np.reshape(X_train50p5_encoded,(X_train50p5_encoded.shape[0],X_train50p5_encoded.shape[2]))
         X_train30p5_encoded=np.reshape(X_train30p5_encoded,(X_train30p5_encoded.shape[0],X_train30p5_encoded.shape[2]))
         X_train20p5_encoded=np.reshape(X_train20p5_encoded,(X_train20p5_encoded.shape[0],X_train20p5_encoded.shape[2]))
  
         X_test50p5_encoded=np.reshape(X_test50p5_encoded,(X_test50p5_encoded.shape[0],X_test50p5_encoded.shape[2]))
         X_test30p5_encoded=np.reshape(X_test30p5_encoded,(X_test30p5_encoded.shape[0],X_test30p5_encoded.shape[2]))
         X_test20p5_encoded=np.reshape(X_test20p5_encoded,(X_test20p5_encoded.shape[0],X_test20p5_encoded.shape[2]))
       else:
         X_train50p5_encoded=np.reshape(X_train50p5_encoded,(X_train50p5_encoded.shape[0],1,X_train50p5_encoded.shape[2]))
         X_train30p5_encoded=np.reshape(X_train30p5_encoded,(X_train30p5_encoded.shape[0],1,X_train30p5_encoded.shape[2]))
         X_train20p5_encoded=np.reshape(X_train20p5_encoded,(X_train20p5_encoded.shape[0],1,X_train20p5_encoded.shape[2]))
         
         X_test50p5_encoded=np.reshape(X_test50p5_encoded,(X_test50p5_encoded.shape[0],1,X_test50p5_encoded.shape[2]))
         X_test30p5_encoded=np.reshape(X_test30p5_encoded,(X_test30p5_encoded.shape[0],1,X_test30p5_encoded.shape[2]))
         X_test20p5_encoded=np.reshape(X_test30p5_encoded,(X_test20p5_encoded.shape[0],1,X_test20p5_encoded.shape[2]))



    #print('X_train30p1_encoded',X_train30p1_encoded.shape)
    #print('X_test30p1_encoded',X_test30p1_encoded.shape)
    #print('X_train50p5_encoded',X_train50p5_encoded.shape)
    #print('X_test50p5_encoded',X_test50p5_encoded.shape)
    #print('X_train30p5_encoded',X_train30p5_encoded.shape)
    #print('X_test30p5_encoded',X_test30p5_encoded.shape)


    '''
    if len(X_train50p1_encoded.shape)==4:
         X_train50p1_encoded=np.reshape(X_train50p1_encoded,(X_train50p1_encoded.shape[0],X_train50p1_encoded.shape[2],X_train50p1_encoded.shape[3]))
         X_train30p1_encoded=np.reshape(X_train30p1_encoded,(X_train30p1_encoded.shape[0],X_train30p1_encoded.shape[2],X_train30p1_encoded.shape[3]))
         X_test50p1_encoded=np.reshape(X_test50p1_encoded,(X_test50p1_encoded.shape[0],X_test50p1_encoded.shape[2],X_test50p1_encoded.shape[3]))
         X_test30p1_encoded=np.reshape(X_test30p1_encoded,(X_test30p1_encoded.shape[0],X_test30p1_encoded.shape[2],X_test30p1_encoded.shape[3]))
         
    if len(X_train50p5_encoded.shape)==4:
       X_train50p5_encoded=np.reshape(X_train50p5_encoded,(X_train50p5_encoded.shape[0],X_train50p5_encoded.shape[2],X_train50p5_encoded.shape[3]))
       X_train30p5_encoded=np.reshape(X_train30p5_encoded,(X_train30p5_encoded.shape[0],X_train30p5_encoded.shape[2],X_train30p5_encoded.shape[3]))
       X_test50p5_encoded=np.reshape(X_test50p5_encoded,(X_test50p5_encoded.shape[0],X_test50p5_encoded.shape[2],X_test50p5_encoded.shape[3]))
       X_test30p5_encoded=np.reshape(X_test30p5_encoded,(X_test30p5_encoded.shape[0],X_test30p5_encoded.shape[2],X_test30p5_encoded.shape[3]))
    
    
    if len(X_train50p1_encoded.shape)==5:
         X_train50p1_encoded=np.reshape(X_train50p1_encoded,(X_train50p1_encoded.shape[0],X_train50p1_encoded.shape[2],X_train50p1_encoded.shape[3],X_train50p1_encoded.shape[4]))
         X_train30p1_encoded=np.reshape(X_train30p1_encoded,(X_train30p1_encoded.shape[0],X_train30p1_encoded.shape[2],X_train30p1_encoded.shape[3],X_train30p1_encoded.shape[4]))
         X_test50p1_encoded=np.reshape(X_test50p1_encoded,(X_test50p1_encoded.shape[0],X_test50p1_encoded.shape[2],X_test50p1_encoded.shape[3],X_test50p1_encoded.shape[4]))
         X_test30p1_encoded=np.reshape(X_test30p1_encoded,(X_test30p1_encoded.shape[0],X_test30p1_encoded.shape[2],X_test30p1_encoded.shape[3],X_test30p1_encoded.shape[4]))
         
    if len(X_train50p5_encoded.shape)==5:
       X_train50p5_encoded=np.reshape(X_train50p5_encoded,(X_train50p5_encoded.shape[0],X_train50p5_encoded.shape[2],X_train50p5_encoded.shape[3],X_train50p5_encoded.shape[4]))
       X_train30p5_encoded=np.reshape(X_train30p5_encoded,(X_train30p5_encoded.shape[0],X_train30p5_encoded.shape[2],X_train30p5_encoded.shape[3],X_train30p5_encoded.shape[4]))
       X_test50p5_encoded=np.reshape(X_test50p5_encoded,(X_test50p5_encoded.shape[0],X_test50p5_encoded.shape[2],X_test50p5_encoded.shape[3],X_test50p5_encoded.shape[4]))
       X_test30p5_encoded=np.reshape(X_test30p5_encoded,(X_test30p5_encoded.shape[0],X_test30p5_encoded.shape[2],X_test30p5_encoded.shape[3],X_test30p5_encoded.shape[4]))
    '''    
    #print('X_train50p1_encoded',X_train50p1_encoded.shape)
    #print('X_train30p1_encoded',X_train30p1_encoded.shape)
    #print('X_train50p5_encoded',X_train50p5_encoded.shape)
    #print('X_train30p5_encoded',X_train30p5_encoded.shape)
   

    #print('X_train50p1_encoded s',X_train50p1_encoded.shape)
    #print('X_train30p1_encoded s',X_train30p1_encoded.shape)
    #print('X_train50p5_encoded s',X_train50p5_encoded.shape)
    #print('X_train30p5_encoded s',X_train30p5_encoded.shape)
    indata = [X_train50p1_encoded, X_train30p1_encoded,X_train20p1_encoded, X_train50p5_encoded, X_train30p5_encoded, X_train20p5_encoded]
    #  indata=[X_train50p1_encoded, X_train30p1_encoded, X_train50p5_encoded, X_train30p5_encoded,]
    #print('started fit at ', datetime.datetime.now())
    #batchSize=1
    combModel.fit(indata,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= ([X_test50p1_encoded, X_test30p1_encoded,X_test20p1_encoded, X_test50p5_encoded, X_test30p5_encoded,X_test20p5_encoded], y_test),#,
        #validation_data= (xTest, y_test),#,
        callbacks=[checkpoint]

        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    )
    #print('finished fit at ', datetime.datetime.now())

    score = combModel.evaluate([X_test50p1_encoded, X_test30p1_encoded,X_test20p1_encoded, X_test50p5_encoded, X_test30p5_encoded,X_test20p5_encoded],
        y=y_test)
    #score = combModel.evaluate(xTest,
    #0    y=y_test)

    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])

   
    combModel.save('comb.branch.'+str(perc)+'.perc.'+str(round(score[1],3))+'.'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.hdf5')
    #print('Model exported and finished')
    

def findLayer(lst, layer):
    pos=-1
    namecheck = layer.name.split('/')[0]
    #print('looking for ',namecheck)
    for i in range(len(lst)):
        #print('@', i, ', comparing ', lst[i].name, 'with ', namecheck)
        if (lst[i].name.split('/')[0]==namecheck):
           pos=i
    #print('found at pos ', pos)
    return pos
    
def mergeModels(mod50p10, mod30p10,mod20p10,mod50p50, mod30p50, mod20p50, layernum1, layernum2):

    mod50pstart10 = mod50p10.layers[0].get_output_at(0)
    mod50pend10 = mod50p10.layers[-1-(18-layernum1)].get_output_at(0)
    mod30pstart10 = mod30p10.layers[0].get_output_at(0)
    mod30pend10 = mod30p10.layers[-1-(18-layernum1)].get_output_at(0)
    mod20pstart10 = mod20p10.layers[0].get_output_at(0)
    mod20pend10 = mod20p10.layers[-1-(18-layernum1)].get_output_at(0)
    
    mod50pstart50 = mod50p50.layers[0].get_output_at(0)
    mod50pend50 = mod50p50.layers[-2-(127-layernum2)].get_output_at(0)
    mod30pstart50 = mod30p50.layers[0].get_output_at(0)
    mod30pend50 = mod30p50.layers[-2-(127-layernum2)].get_output_at(0)
    mod20pstart50 = mod20p50.layers[0].get_output_at(0)
    mod20pend50 = mod20p50.layers[-2-(127-layernum2)].get_output_at(0)
    
    
    mod50p1 = Model(inputs=mod50pstart10,outputs=mod50pend10)
    mod30p1 = Model(inputs=mod30pstart10,outputs=mod30pend10)
    mod20p1 = Model(inputs=mod20pstart10,outputs=mod20pend10)
    mod50p5 = Model(inputs=mod50pstart50,outputs=mod50pend50)
    mod30p5 = Model(inputs=mod30pstart50,outputs=mod30pend50)
    mod20p5 = Model(inputs=mod20pstart50,outputs=mod20pend50)
    
    #mod50p.summary()
    mod50pstart1 = mod50p1.layers[0].get_output_at(0)
    mod50pend1 = mod50p1.layers[-1].get_output_at(0)
    mod30pstart1 = mod30p1.layers[0].get_output_at(0)
    mod30pend1 = mod30p1.layers[-1].get_output_at(0)
    mod20pstart1 = mod20p1.layers[0].get_output_at(0)
    mod20pend1 = mod20p1.layers[-1].get_output_at(0)


    mod50pstart5 = mod50p5.layers[0].get_output_at(0)
    mod50pend5 = mod50p5.layers[-1].get_output_at(0)
    mod30pstart5 = mod30p5.layers[0].get_output_at(0)
    mod30pend5 = mod30p5.layers[-1].get_output_at(0)
    mod20pstart5 = mod20p5.layers[0].get_output_at(0)
    mod20pend5 = mod20p5.layers[-1].get_output_at(0)

    modelComplete=True
    print('l1:',layernum1, '/',len(mod50p10.layers),';l2:',layernum2, '/',len(mod50p50.layers))
    if (layernum1==19) and (layernum2==128):
       in50p1 = Input((64,))
       in30p1 = Input(( 64,))
       in20p1 = Input(( 64,))
       in50p5 = Input(( 264,))
       in30p5 = Input(( 264,))
       in20p5 = Input(( 264,))

 
       bn50p1= BatchNormalization()(in50p1)
       bn30p1= BatchNormalization()(in30p1)
       bn20p1= BatchNormalization()(in20p1)
       bn50p5= BatchNormalization()(in50p5)
       bn30p5= BatchNormalization()(in30p5)
       bn20p5= BatchNormalization()(in20p5)

       drop50p1= Dropout(0.5)(bn50p1)
       drop30p1= Dropout(0.5)(bn30p1)
       drop20p1= Dropout(0.5)(bn20p1)
       drop50p5= Dropout(0.5)(bn50p5)
       drop30p5= Dropout(0.5)(bn30p5)
       drop20p5= Dropout(0.5)(bn20p5)
     
     
       act50p1= Activation('relu')(drop50p1)
       act30p1= Activation('relu')(drop30p1)
       act20p1= Activation('relu')(drop20p1)
       act50p5= Activation('relu')(drop50p5)
       act30p5= Activation('relu')(drop30p5)
       act20p5= Activation('relu')(drop20p5)
           
       
       
       concatC = Concatenate(name='outconCat',axis=1) ([act50p1,act30p1,act20p1,act50p5,act30p5,act20p5])

    else:
     #in50p1 = Input(K.int_shape(mod50pend1)[1:])
     #in30p1 = Input(K.int_shape(mod30pend1)[1:])
     #in20p1 = Input(K.int_shape(mod20pend1)[1:])
     in50p5 = Input(K.int_shape(mod50pend5)[1:])
     in30p5 = Input(K.int_shape(mod30pend5)[1:])
     in20p5 = Input(K.int_shape(mod20pend5)[1:])


     newshape=[]
     oldshape=K.int_shape(mod50pend1)
     for k in range(len(oldshape)):
        if oldshape[k]!=None:
             newshape.append(oldshape[k])

     in50p1 = Input(newshape)
     in30p1 = Input(newshape)
     in20p1 =Input(newshape)
     #in50p5 = Input(K.int_shape(mod50pend5))
     #in30p5 = Input(K.int_shape(mod30pend5))
     #in20p5 = Input(K.int_shape(mod20pend5))
     
     print("layernum1, layernum2, in50p1 shape is ", in50p1.shape, layernum1, layernum2, mod50pend1.shape)
     print("layernum1, layernum2, in50p5 shape is ", in50p5.shape, layernum1, layernum2, mod50pend5.shape)
        #in50p5 = Input(mod50pend5.shape)
        #in30p5 = Input(mod30pend5.shape)
     mod50p10.summary()
     inLyrs=[]
     lyrs=[] 
     nextLyr = in50p1
     inLyr = in50p1# mod50p10.layers[0].get_output_at(0)
     preBatchLyr=None
     if (layernum1<=20) and (layernum2<=127):
      for i in range (layernum1+1,len(mod50p10.layers)-1):
       layer = mod50p10.layers[i]
      #if (currlyr >= startLayer):
       if True:#currlyr <(numlyrs-2):
        if isinstance(layer, keras.layers.Conv2D):
            print('passing ', nextLyr, ' to Conv2D as ', nextLyr.shape, K.int_shape(nextLyr))
            nextLyr = Conv2D(filters=layer.filters,kernel_size=layer.kernel_size, padding=layer.padding, strides=layer.strides,  kernel_regularizer=layer.kernel_regularizer)(nextLyr)#, layer.get_output_at(0).shape
        if isinstance(layer, keras.layers.MaxPooling2D):
           nextLyr = MaxPooling2D(pool_size=layer.pool_size)(nextLyr)
        if isinstance(layer, keras.layers.Concatenate):
           pos0 =findLayer(lyrs, layer.get_input_at(0)[0])
           pos1 =findLayer(lyrs, layer.get_input_at(0)[1])
           if ((pos1>0) and (pos0>=0)):
             nextLyr = Concatenate()([inLyrs[pos0],inLyrs[pos1]])
           else:
              modelComplete=False 
        if isinstance(layer, keras.layers.Activation):
           nextLyr = Activation('relu')(nextLyr)
        if isinstance(layer, keras.layers.Dropout):
           nextLyr = Dropout(rate=0.5)(nextLyr)
        if isinstance(layer, keras.layers.Flatten):
           print('passing ', nextLyr, ' to flatten as ', nextLyr.shape, K.int_shape(nextLyr))
           newshape=[]
           oldshape=K.int_shape(nextLyr)
           for k in range(len(oldshape)):
               if oldshape[k]==None:
                   newshape.append(1)
               else:
                   newshape.append(oldshape[k])
           #if (len(K.int_shape(nextLyr)) ==3):
           nextLyr = Reshape(newshape)(nextLyr)
           nextLyr = Flatten()(nextLyr)
        if isinstance(layer, keras.layers.Dense):
           nextLyr = Dense(units=layer.units,name ='dense_orig50_'+str(i))(nextLyr)
        if isinstance(layer, keras.layers.LSTM):
           newshape=[]
           print(layernum1,':b4 reshape retrying to apply lstm to ', K.int_shape(nextLyr))
           oldshape=K.int_shape(nextLyr)
           for k in range(len(oldshape)):
               if oldshape[k]!=None:
                   newshape.append(oldshape[k])
           #if (len(K.int_shape(nextLyr)) ==3):
           nextLyr = Reshape(newshape)(nextLyr)
           print('after reshape retrying to apply lstm to ', K.int_shape(nextLyr),newshape)

           nextLyr = LSTM(units= layer.units,return_sequences=layer.return_sequences,unit_forget_bias=layer.unit_forget_bias,dropout=layer.dropout)(nextLyr)
        if isinstance(layer, keras.layers.Reshape):
           print('trying to reshape', K.int_shape(nextLyr),' to ', layer.target_shape)
           nextLyr = Reshape(layer.target_shape)(nextLyr)
        if isinstance(layer, keras.layers.BatchNormalization):           
           if (preBatchLyr==None):
              nextLyr = BatchNormalization()(nextLyr)
           else:
              nextLyr = BatchNormalization()(preBatchLyr)
        if isinstance(layer, keras.layers.AveragePooling2D):
           nextLyr = AveragePooling2D(pool_size=layer.pool_size, strides=layer.strides)(nextLyr)
        lyrModel = Model(inLyr,nextLyr)
        preBatchLyr = lyrModel.layers[-1].get_output_at(0)

       #print('ncelyr shape is ', nextLyr.shape)
        lyrs.append(layer.get_output_at(0))
        inLyrs.append(nextLyr)
     numdims=1
     #print('drop50p1 shape is ', drop50p1.shape)

     for i in range(len(K.int_shape(nextLyr))):
       #print('i=',i,'numdims=',numdims)
       if K.int_shape(nextLyr)[i]!=None:
           numdims=numdims*K.int_shape(nextLyr)[i]
           
     re50p1 =Reshape((numdims,))(nextLyr)       
     dense50p1 = Dense(units=64, name ='denseout50p1')(re50p1)       
       #mod50p1 = Model(in50p1,nextLyr)
    
     inLyrs=[]
     lyrs=[] 
     nextLyr = in30p1 
     inLyr = in30p1#mod30p10.layers[0].get_output_at(0)
     preBatchLyr=None
     print('will insert from layers ' , layernum1+1 , ' to ', len(mod30p10.layers)-1)
     for i in range (layernum1+1,len(mod30p10.layers)-1):
        layer = mod30p10.layers[i]
        #if (currlyr >= startLayer):
        if isinstance(layer, keras.layers.Conv2D):
            print('createing conv2dlayer from ', nextLyr.shape)
            nextLyr = Conv2D(filters=layer.filters,kernel_size=layer.kernel_size, padding=layer.padding, strides=layer.strides,  kernel_regularizer=layer.kernel_regularizer)(nextLyr)#, layer.get_output_at(0).shape
        if isinstance(layer, keras.layers.MaxPooling2D):
           print('createing MaxPooling2D from ', nextLyr.shape)
           nextLyr = MaxPooling2D(pool_size=layer.pool_size)(nextLyr)
        if isinstance(layer, keras.layers.Concatenate):
           print('createing Concatenate from ', nextLyr.shape)
           pos0 =findLayer(lyrs, layer.get_input_at(0)[0])
           pos1 =findLayer(lyrs, layer.get_input_at(0)[1])
           if ((pos1>0) and (pos0>=0)):
             nextLyr = Concatenate()([inLyrs[pos0],inLyrs[pos1]])
           else:
              modelComplete=False 
        if isinstance(layer, keras.layers.Activation):
           print('createing Activation from ', nextLyr.shape)
           nextLyr = Activation('relu')(nextLyr)
        if isinstance(layer, keras.layers.Dropout):
           print('createing Dropout from ', nextLyr.shape)
           nextLyr = Dropout(rate=0.5)(nextLyr)
        if isinstance(layer, keras.layers.Flatten):
           print('createing Flatten from ', nextLyr.shape)
           newshape=[]
           oldshape=K.int_shape(nextLyr)
           for k in range(len(oldshape)):
               if oldshape[k]==None:
                   newshape.append(1)
               else:
                   newshape.append(oldshape[k])
               
           #if (len(K.int_shape(nextLyr)) ==3):
           nextLyr = Reshape(newshape)(nextLyr)     
           nextLyr = Flatten()(nextLyr)
        if isinstance(layer, keras.layers.Dense):
           print('createing Dense from ', nextLyr.shape)
           nextLyr = Dense(units=layer.units,name ='dense_orig30_'+str(i))(nextLyr)
        if isinstance(layer, keras.layers.LSTM):
           print('createing LSTM from ', nextLyr.shape)
           newshape=[]
           oldshape=K.int_shape(nextLyr)
           for k in range(len(oldshape)):
               if oldshape[k]!=None:
                   newshape.append(oldshape[k])
           #if (len(K.int_shape(nextLyr)) ==3):
           nextLyr = Reshape(newshape)(nextLyr)           
           nextLyr = LSTM(units= layer.units,return_sequences=layer.return_sequences,unit_forget_bias=layer.unit_forget_bias,dropout=layer.dropout)(nextLyr)
        if isinstance(layer, keras.layers.Reshape):
           print('createing Reshape from ', nextLyr.shape)
           nextLyr = Reshape(layer.target_shape)(nextLyr)
        if isinstance(layer, keras.layers.BatchNormalization):           
           print('createing BatchNormalization from ', nextLyr.shape)
           if (preBatchLyr==None):
              nextLyr = BatchNormalization()(nextLyr)
           else:
              nextLyr = BatchNormalization()(preBatchLyr)
        if isinstance(layer, keras.layers.AveragePooling2D):
           print('createing AveragePooling2D from ', nextLyr.shape)
           nextLyr = AveragePooling2D(pool_size=layer.pool_size, strides=layer.strides)(nextLyr)
        lyrModel = Model(inLyr,nextLyr)
        preBatchLyr = lyrModel.layers[-1].get_output_at(0)
        lyrs.append(layer.get_output_at(0))
        inLyrs.append(nextLyr)
     numdims=1
     for i in range(len(K.int_shape(nextLyr))):
        #print('i=',i,'numdims=',numdims)
        if K.int_shape(nextLyr)[i]!=None:
           numdims=numdims*K.int_shape(nextLyr)[i]
     print('numdims=',numdims)
     
     re30p1 = Dense(units=numdims, name ='denseout30p1')(nextLyr)       
     print(re30p1.shape)
     dense30p1 = Reshape((64,))(re30p1)       
     #mod30p1 = Model(in30p1,nextLyr)
     #print('30p1: numdims=',numdims, drop30p1.shape,re30p1.shape,dense30p1.shape  )
     #mod30p1.summary()
    

     inLyrs=[]
     lyrs=[] 
     nextLyr = in20p1
     inLyr = in20p1#mod20p10.layers[0].get_output_at(0)
     preBatchLyr=None
     for i in range (layernum1+1,len(mod20p10.layers)-1):
        layer = mod20p10.layers[i]
        #if (currlyr >= startLayer):
        if isinstance(layer, keras.layers.Conv2D):
            nextLyr = Conv2D(filters=layer.filters,kernel_size=layer.kernel_size, padding=layer.padding, strides=layer.strides,  kernel_regularizer=layer.kernel_regularizer)(nextLyr)#, layer.get_output_at(0).shape
        if isinstance(layer, keras.layers.MaxPooling2D):
           nextLyr = MaxPooling2D(pool_size=layer.pool_size)(nextLyr)
        if isinstance(layer, keras.layers.Concatenate):
           pos0 =findLayer(lyrs, layer.get_input_at(0)[0])
           pos1 =findLayer(lyrs, layer.get_input_at(0)[1])
           if ((pos1>0) and (pos0>=0)):
             nextLyr = Concatenate()([inLyrs[pos0],inLyrs[pos1]])
           else:
              modelComplete=False 
        if isinstance(layer, keras.layers.Activation):
           nextLyr = Activation('relu')(nextLyr)
        if isinstance(layer, keras.layers.Dropout):
           nextLyr = Dropout(rate=0.5)(nextLyr)
        if isinstance(layer, keras.layers.Flatten):
           newshape=[]
           oldshape=K.int_shape(nextLyr)
           for k in range(len(oldshape)):
               if oldshape[k]==None:
                   newshape.append(1)
               else:
                   newshape.append(oldshape[k])
           #if (len(K.int_shape(nextLyr)) ==3):
           nextLyr = Reshape(newshape)(nextLyr)
           nextLyr = Flatten()(nextLyr)
        if isinstance(layer, keras.layers.Dense):
           nextLyr = Dense(units=layer.units,name ='dense_orig20_'+str(i))(nextLyr)
        if isinstance(layer, keras.layers.LSTM):
           newshape=[]
           oldshape=K.int_shape(nextLyr)
           for k in range(len(oldshape)):
               if oldshape[k]!=None:
                   newshape.append(oldshape[k])
           #if (len(K.int_shape(nextLyr)) ==3):
           nextLyr = Reshape(newshape)(nextLyr)        
           nextLyr = LSTM(units= layer.units,return_sequences=layer.return_sequences,unit_forget_bias=layer.unit_forget_bias,dropout=layer.dropout)(nextLyr)
        if isinstance(layer, keras.layers.Reshape):
           nextLyr = Reshape(layer.target_shape)(nextLyr)
        if isinstance(layer, keras.layers.BatchNormalization):           
           if (preBatchLyr==None):
              nextLyr = BatchNormalization()(nextLyr)
           else:
              nextLyr = BatchNormalization()(preBatchLyr)
        if isinstance(layer, keras.layers.AveragePooling2D):
           nextLyr = AveragePooling2D(pool_size=layer.pool_size, strides=layer.strides)(nextLyr)
        lyrModel = Model(inLyr,nextLyr)
        preBatchLyr = lyrModel.layers[-1].get_output_at(0)
        lyrs.append(layer.get_output_at(0))
        inLyrs.append(nextLyr)
     numdims=1
     for i in range(len(K.int_shape(nextLyr))):
        #print('i=',i,'numdims=',numdims)
        if K.int_shape(nextLyr)[i]!=None:
           numdims=numdims*K.int_shape(nextLyr)[i]
     #print('numdims=',numdims)
     re20p1 = Dense(units=numdims, name ='denseout20p1')(nextLyr)       
     dense20p1 = Reshape((64,))(re20p1)       
     #mod30p1 = Model(in30p1,nextLyr)
     #print('30p1: numdims=',numdims, drop30p1.shape,re30p1.shape,dense30p1.shape  )
     #mod30p1.summary()
    
    
     inLyrs=[]
     lyrs=[] 
     nextLyr = in50p5
     inLyr =in50p5# mod50p50.layers[0].get_output_at(0)
     preBatchLyr=None
     for i in range (layernum2+1,len(mod50p50.layers)-1):
      layer = mod50p50.layers[i]
      #if (currlyr >= startLayer):
      if True:#currlyr <(numlyrs-2):
        if isinstance(layer, keras.layers.Conv2D):
            nextLyr = Conv2D(filters=layer.filters,kernel_size=layer.kernel_size, padding=layer.padding, strides=layer.strides,  kernel_regularizer=layer.kernel_regularizer)(nextLyr)#, layer.get_output_at(0).shape
        if isinstance(layer, keras.layers.MaxPooling2D):
           nextLyr = MaxPooling2D(pool_size=layer.pool_size)(nextLyr)
        if isinstance(layer, keras.layers.Concatenate):
           pos0 =findLayer(lyrs, layer.get_input_at(0)[0])
           pos1 =findLayer(lyrs, layer.get_input_at(0)[1])
           if ((pos1>0) and (pos0>=0)):
             nextLyr = Concatenate()([inLyrs[pos0],inLyrs[pos1]])
           else:
              modelComplete=False 
        if isinstance(layer, keras.layers.Activation):
           nextLyr = Activation('relu')(nextLyr)
        if isinstance(layer, keras.layers.Dropout):
           nextLyr = Dropout(rate=0.5)(nextLyr)
        if isinstance(layer, keras.layers.Flatten):
           nextLyr = Flatten()(nextLyr)
        if isinstance(layer, keras.layers.Dense):
           nextLyr = Dense(units=layer.units,name ='dense_orig_'+str(i))(nextLyr)
        if isinstance(layer, keras.layers.LSTM):
           nextLyr = LSTM(units= layer.units,return_sequences=layer.return_sequences,unit_forget_bias=layer.unit_forget_bias,dropout=layer.dropout)(nextLyr)
        if isinstance(layer, keras.layers.Reshape):
           nextLyr = Reshape(layer.target_shape)(nextLyr)
        if isinstance(layer, keras.layers.BatchNormalization):           
           if (preBatchLyr==None):
              nextLyr = BatchNormalization()(nextLyr)
           else:
              nextLyr = BatchNormalization()(preBatchLyr)
        if isinstance(layer, keras.layers.AveragePooling2D):
           nextLyr = AveragePooling2D(pool_size=layer.pool_size, strides=layer.strides)(nextLyr)
        #print('nextLyr shape:', nextLyr.shape)          
        lyrModel = Model(inLyr,nextLyr)
        preBatchLyr = lyrModel.layers[-1].get_output_at(0)
        lyrs.append(layer.get_output_at(0))
        inLyrs.append(nextLyr)

     numdims=1
     for i in range(len(K.int_shape(nextLyr))):
        #print('i=',i,'numdims=',numdims)
        if K.int_shape(nextLyr)[i]!=None:
           numdims=numdims*K.int_shape(nextLyr)[i]
     #print('numdims=',numdims)
           
     re50p5 = Reshape((numdims,))(nextLyr)       
     dense50p5 = Dense(units=264, name ='denseout50p5')(re50p5)       
     #mod50p5 = Model(in50p5,nextLyr)
     #print('50p5: numdims=',numdims, drop50p5.shape,re50p5.shape,dense50p5.shape  )
     #mod50p5.summary()
    
    
     inLyrs=[]
     lyrs=[] 
     nextLyr = in30p5
     inLyr = in30p5#88888mod30p50.layers[0].get_output_at(0)
     preBatchLyr=None
     for i in range (layernum2+1,len(mod30p50.layers)-1):
      layer = mod30p50.layers[i]
      #if (currlyr >= startLayer):
      if True:#currlyr <(numlyrs-2):
        if isinstance(layer, keras.layers.Conv2D):
            nextLyr = Conv2D(filters=layer.filters,kernel_size=layer.kernel_size, padding=layer.padding, strides=layer.strides,  kernel_regularizer=layer.kernel_regularizer)(nextLyr)#, layer.get_output_at(0).shape
        if isinstance(layer, keras.layers.MaxPooling2D):
           nextLyr = MaxPooling2D(pool_size=layer.pool_size)(nextLyr)
        if isinstance(layer, keras.layers.Concatenate):
           pos0 =findLayer(lyrs, layer.get_input_at(0)[0])
           pos1 =findLayer(lyrs, layer.get_input_at(0)[1])
           if ((pos1>0) and (pos0>=0)):
             nextLyr = Concatenate()([inLyrs[pos0],inLyrs[pos1]])
           else:
              modelComplete=False 
        if isinstance(layer, keras.layers.Activation):
           nextLyr = Activation('relu')(nextLyr)
        if isinstance(layer, keras.layers.Dropout):
           nextLyr = Dropout(rate=0.5)(nextLyr)
        if isinstance(layer, keras.layers.Flatten):
           nextLyr = Flatten()(nextLyr)
        if isinstance(layer, keras.layers.Dense):
           nextLyr = Dense(units=layer.units,name ='dense_orig_'+str(i))(nextLyr)
        if isinstance(layer, keras.layers.LSTM):
           nextLyr = LSTM(units= layer.units,return_sequences=layer.return_sequences,unit_forget_bias=layer.unit_forget_bias,dropout=layer.dropout)(nextLyr)
        if isinstance(layer, keras.layers.Reshape):
           nextLyr = Reshape(layer.target_shape)(nextLyr)
        if isinstance(layer, keras.layers.BatchNormalization):           
           if (preBatchLyr==None):
              nextLyr = BatchNormalization()(nextLyr)
           else:
              nextLyr = BatchNormalization()(preBatchLyr)
        if isinstance(layer, keras.layers.AveragePooling2D):
           nextLyr = AveragePooling2D(pool_size=layer.pool_size, strides=layer.strides)(nextLyr)
        lyrModel = Model(inLyr,nextLyr)
        preBatchLyr = lyrModel.layers[-1].get_output_at(0)
        lyrs.append(layer.get_output_at(0))
        inLyrs.append(nextLyr)

     numdims=1
     for i in range(len(K.int_shape(nextLyr))):
        #print('i=',i,'numdims=',numdims)
        if K.int_shape(nextLyr)[i]!=None:
           numdims=numdims*K.int_shape(nextLyr)[i]
     #print('numdims=',numdims)
           
     inLyrs=[]
     lyrs=[] 
     re30p5=Reshape((numdims,))(nextLyr)
     dense30p5 = Dense(units=264, name ='denseout30p5')(re30p5)       
     #mod30p5 = Model(in30p5,nextLyr)
     #print('30p5: numdims=',numdims, drop30p5.shape,re30p5.shape,dense30p5.shape  )
     #mod30p5.summary()
     
     nextLyr = in20p5
     inLyr = in20p5#mod20p50.layers[0].get_output_at(0)
     preBatchLyr=None
     for i in range (layernum2+1,len(mod20p50.layers)-1):
      layer = mod20p50.layers[i]
      #if (currlyr >= startLayer):
      if True:#currlyr <(numlyrs-2):
        if isinstance(layer, keras.layers.Conv2D):
            nextLyr = Conv2D(filters=layer.filters,kernel_size=layer.kernel_size, padding=layer.padding, strides=layer.strides,  kernel_regularizer=layer.kernel_regularizer)(nextLyr)#, layer.get_output_at(0).shape
        if isinstance(layer, keras.layers.MaxPooling2D):
           nextLyr = MaxPooling2D(pool_size=layer.pool_size)(nextLyr)
        if isinstance(layer, keras.layers.Concatenate):
           pos0 =findLayer(lyrs, layer.get_input_at(0)[0])
           pos1 =findLayer(lyrs, layer.get_input_at(0)[1])
           if ((pos1>0) and (pos0>=0)):
             nextLyr = Concatenate()([inLyrs[pos0],inLyrs[pos1]])
           else:
              modelComplete=False 
        if isinstance(layer, keras.layers.Activation):
           nextLyr = Activation('relu')(nextLyr)
        if isinstance(layer, keras.layers.Dropout):
           nextLyr = Dropout(rate=0.5)(nextLyr)
        if isinstance(layer, keras.layers.Flatten):
           nextLyr = Flatten()(nextLyr)
        if isinstance(layer, keras.layers.Dense):
           nextLyr = Dense(units=layer.units,name ='dense_orig_'+str(i))(nextLyr)
        if isinstance(layer, keras.layers.LSTM):
           nextLyr = LSTM(units= layer.units,return_sequences=layer.return_sequences,unit_forget_bias=layer.unit_forget_bias,dropout=layer.dropout)(nextLyr)
        if isinstance(layer, keras.layers.Reshape):
           nextLyr = Reshape(layer.target_shape)(nextLyr)
        if isinstance(layer, keras.layers.BatchNormalization):           
           if (preBatchLyr==None):
              nextLyr = BatchNormalization()(nextLyr)
           else:
              nextLyr = BatchNormalization()(preBatchLyr)
        if isinstance(layer, keras.layers.AveragePooling2D):
           nextLyr = AveragePooling2D(pool_size=layer.pool_size, strides=layer.strides)(nextLyr)
        lyrModel = Model(inLyr,nextLyr)
        preBatchLyr = lyrModel.layers[-1].get_output_at(0)
        lyrs.append(layer.get_output_at(0))
        inLyrs.append(nextLyr)
        print('NXTLYR8 is ', nextLyr)
     numdims=1
     for i in range(len(K.int_shape(nextLyr))):
        #print('i=',i,'numdims=',numdims)
        if K.int_shape(nextLyr)[i]!=None:
           numdims=numdims*K.int_shape(nextLyr)[i]
     #print('numdims=',numdims)
           
     inLyrs=[]
     lyrs=[] 
     re20p5=Reshape((numdims,))(nextLyr)
     dense20p5 = Dense(units=264, name ='denseout20p5')(re20p5)       
     #mod30p5 = Model(in30p5,nextLyr)
     #print('30p5: numdims=',numdims, drop30p5.shape,re30p5.shape,dense30p5.shape  )
     #mod30p5.summary()
    
     bn50p1= BatchNormalization()(dense50p1)
     bn30p1= BatchNormalization()(dense30p1)
     bn20p1= BatchNormalization()(dense20p1)
     bn50p5= BatchNormalization()(dense50p5)
     bn30p5= BatchNormalization()(dense30p5)
     bn20p5= BatchNormalization()(dense20p5)

     drop50p1= Dropout(0.5)(bn50p1)
     drop30p1= Dropout(0.5)(bn30p1)
     drop20p1= Dropout(0.5)(bn20p1)
     drop50p5= Dropout(0.5)(bn50p5)
     drop30p5= Dropout(0.5)(bn30p5)
     drop20p5= Dropout(0.5)(bn20p5)
     
     
     act50p1= Activation('relu')(drop50p1)
     act30p1= Activation('relu')(drop30p1)
     act20p1= Activation('relu')(drop20p1)
     act50p5= Activation('relu')(drop50p5)
     act30p5= Activation('relu')(drop30p5)
     act20p5= Activation('relu')(drop20p5)
    
     concatC = Concatenate(name='outconCat',axis=1) ([act50p1,act30p1,act20p1,act50p5,act30p5,act20p5])

    

 
    x = Activation('relu', name = 'actOut0')(concatC)

    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    print("BatchNormalization shape is ", x.shape)
    
    #x = BatchNormalization()(nextLyr)
    x = Activation('relu', name = 'actOut')(x)

    print("Activation shape is ", x.shape)
    #x = GlobalAveragePooling2D()(x)
    #print("GlobalAveragePooling2D shape is ", x.shape)
    prediction = Dense(totalLabel, activation='softmax',name ='denseOut')(x)
    if modelComplete:
       newModel = Model([in50p1,in30p1,in20p1,in50p5,in30p5,in20p5], prediction)
    else:
        newModel=None
    #origBranch.summary()
    #print("NEW")
    #newModel.summary()  
    return newModel
    #print('this was newModel ', startLayer)      
    

     
    
def replicateListToMatch(inList, reqSize):
    outList =[]
    dx=0
    while (len(outList)<reqSize):
         print('dx=',dx)
         if (dx >=len(inList)):
            outList.append(inList[dx %len(inList)])
         else:
            outList.append(inList[dx])
         dx+=1
    return np.array(outList)

if __name__ == '__main__':
 #tensorboard = TensorBoard(log_dir = "logs/{}".format(time()))
 dataset =  importData('base',train=True)#(train, test) =  
 print('total recs =', totalRecordCount, '; Total Labels=', totalLabel, lblmap)
 nextdataset =   importData('next',train=True)#(nextTrain,nextTest) = 
 print('total recs =', totalRecordCount, '; Total Labels=', totalLabel, lblmap)
 lastdataset =   importData('last',train=True)#(nextTrain,nextTest) = 
 print('total recs =', totalRecordCount, '; Total Labels=', totalLabel, lblmap)
 #random.shuffle(dataset)
 availCount= totalRecordCount#*0.2)
 #availset = dataset#[:availCount]
 #availset = replicateListToMatch(availset,len(nextdataset)/2)
 #availset = mergeSets2(availset, nextdataset)#train, test, nextTrain, nextTest)
 availset = mergeSets2(dataset, nextdataset)#train, test, nextTrain, nextTest)
 availset = mergeSets2(availset, lastdataset)#train, test, nextTrain, nextTest)
 #availCount+=len(nextdataset)

 print('AvailCount: {}'.format(availCount))
 trainDataEndIndex = int(availCount*0.8)
 random.shuffle(availset)
 print('trainDataEndIndex: {}'.format(trainDataEndIndex))
   
 #train = availset[:trainDataEndIndex]
 #test = availset[trainDataEndIndex:]
 x, y = zip(*availset)
 x_train = x[:trainDataEndIndex]
 x_test = x[trainDataEndIndex:]   
 ycat = to_categorical(y)
 y_traincat = ycat[:trainDataEndIndex]
 y_testcat = ycat[trainDataEndIndex:]   

 x_train = np.array(x_train)
 x_test = np.array(x_test)
 x_train = np.expand_dims(x_train,-1)
 x_test = np.expand_dims(x_test,-1)
 x_train = x_train.astype('float32') / 255
 x_test = x_test.astype('float32') / 255
 #x_train = np.expand_dims(x_train,-1)
 #x_test = np.expand_dims(x_test,-1)




 #encfine.built=True
 #enccoarse.built=True
 #print(os.getcwd())
   
 mod50p1 =keras.models.load_model('50p/1/Model.1.final.hdf5')#,  custom_objects={'tf': tf})
 mod30p1 =keras.models.load_model('30p/1/Model.1.final.hdf5')#,  custom_objects={'tf': tf})
 mod20p1 =keras.models.load_model('20p/1/Model.1.final.hdf5')#,  custom_objects={'tf': tf})
 #################################################################################
 mod50p5 =keras.models.load_model('50p/5c/Model.5.final.hdf5')#,  custom_objects={'tf': tf})
 mod30p5 =keras.models.load_model('30p/5c/Model.5.final.hdf5')#,  custom_objects={'tf': tf})
 mod20p5 =keras.models.load_model('20p/5c/Model.5.final.hdf5')#,  custom_objects={'tf': tf})
 
 #avoid errors with duplicate layer names
 if True:
    for layer in mod50p1.layers:
        layer.name = layer.name + str("_1")
        layer.fixed=True
        #print("updated "+layer.name + " to " + layer._name)
        
    for layer in mod30p1.layers:
        layer.name = layer.name + str("_2")
        layer.fixed=True
        
    for layer in mod20p1.layers:
        layer.name = layer.name + str("_3")
        layer.fixed=True        

    for layer in mod50p5.layers:
        layer.name = layer.name + str("_4")
        layer.fixed=True
        #print("updated "+layer.name + " to " + layer._name)
        
    for layer in mod30p5.layers:
        layer.name = layer.name + str("_5")
        layer.fixed=True
        
    for layer in mod20p5.layers:
        layer.name = layer.name + str("_6")
        layer.fixed=True



 orig_in50p1 = mod50p1.layers[0].get_input_at(0)
 orig_in30p1 = mod30p1.layers[0].get_input_at(0)
 orig_in20p1 = mod20p1.layers[0].get_input_at(0)
 orig_in50p5 = mod50p5.layers[0].get_input_at(0)
 orig_in30p5 = mod30p5.layers[0].get_input_at(0)
 orig_in20p5 = mod20p5.layers[0].get_input_at(0)
 trimmed50p1out = mod50p1.layers[-3].get_output_at(0)
 trimmed30p1out = mod30p1.layers[-3].get_output_at(0)
 trimmed20p1out = mod20p1.layers[-3].get_output_at(0)
 trimmed50p5out = mod50p5.layers[-2].get_output_at(0)
 trimmed30p5out = mod30p5.layers[-2].get_output_at(0)
 trimmed20p5out = mod20p5.layers[-2].get_output_at(0)
  
 trimmed50p1=Model(inputs=orig_in50p1,outputs=trimmed50p1out)
 trimmed30p1=Model(inputs=orig_in30p1,outputs=trimmed30p1out)
 trimmed20p1=Model(inputs=orig_in20p1,outputs=trimmed20p1out)
 trimmed50p5=Model(inputs=orig_in50p5,outputs=trimmed50p5out)
 trimmed30p5=Model(inputs=orig_in30p5,outputs=trimmed30p5out)
 trimmed20p5=Model(inputs=orig_in20p5,outputs=trimmed20p5out)
  
 print('ytraincat', y_traincat.shape)
 print('ytestcat', y_testcat.shape)
 print('x_train', x_train.shape)
 print('x_test', x_test.shape)
 numtrain = x_train.shape[0]
 numtest = x_test.shape[0]
 c50p1Count = len(trimmed50p1.layers)
 c30p1Count = len(trimmed30p1.layers)
 c20p1Count = len(trimmed20p1.layers)
   
 c50p5Count = len(trimmed50p5.layers)
 c30p5Count = len(trimmed30p5.layers)
 c20p5Count = len(trimmed20p5.layers)

 c50p1Lev =  c50p1Count
 c30p1Lev =  c30p1Count
 c20p1Lev =  c20p1Count
    
 c50p5Lev =  c50p5Count
 c30p5Lev =  c30p5Count 
 c20p5Lev =  c20p5Count 
    
 prevc50p1Lev =  c50p1Count
 prevc30p1Lev =  c30p1Count
 prevc20p1Lev =  c20p1Count
    
 prevc50p5Lev =  c50p5Count
 prevc30p5Lev =  c30p5Count   
 prevc20p5Lev =  c20p5Count  
 #trimmed50p5.summary()
 #mod50p5.summary() 
 init=False
 for perc0 in range(69,100):
   print(perc0, '% of test')
   perc = 100-perc0
   outfile=open("iter"+str(perc)+".txt","w")
   outfile.write("started")
   outfile.close()
   #try:
   if True:
    c50p1Lev =  (int)(c50p1Count*perc/100)
    c30p1Lev =  (int)(c30p1Count*perc/100)
    c20p1Lev =  (int)(c20p1Count*perc/100)
     
    c50p5Lev =  (int)(c50p5Count*perc/100)
    c30p5Lev =  (int)(c30p5Count*perc/100)
    c20p5Lev =  (int)(c20p5Count*perc/100)
      
    combModel = mergeModels(trimmed50p1, trimmed30p1,trimmed20p1,trimmed50p5, trimmed30p5,trimmed20p5, c50p1Lev, c50p5Lev)
    if not (combModel==None):
     combModel.summary()
     if ((not(prevc50p1Lev==c50p1Lev)) or (not init)):#codifying data for mod50p1
      encPreModel = keras.models.clone_model(mod50p1)
      modcopy  = keras.models.clone_model(mod50p1)
      orig_in = mod50p1.layers[0].get_output_at(0)
   
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      print('popping ', c50p1Count - c50p1Lev, ' layers')
      for j in range(c50p1Count - c50p1Lev):
         encPreModel._layers.pop()
          
      encin = encPreModel._layers[0].get_output_at(0)

      if (c50p1Count==c50p1Lev):
         encout = encPreModel._layers[-3].get_output_at(0)
      else:
         encout = encPreModel._layers[-2].get_output_at(0)
      
      #encout = encPreModel.layers[-2].get_input_at(0)
      print('for 50p1 encout.shape is ', encout.shape, ', name is ',encout.name )
      #''' 
      encModel=Model(inputs=encin,outputs=encout)
      #encModel.summary()
      enc50p1 = keras.models.clone_model(encModel)
      
      #print('ENC MODELmod50p1')#, i)
      X_train50p1_encoded = encPredict(encModel,x_train)
      X_test50p1_encoded = encPredict(encModel,x_test)
      prevc50p1Lev=c50p1Lev
      
      
     if ((not(prevc30p1Lev==c30p1Lev)) or (not init)):#codifying data for mod50p1
      encPreModel = keras.models.clone_model(mod30p1)
      modcopy  = keras.models.clone_model(mod30p1)
      orig_in = mod30p1.layers[0].get_output_at(0)
   
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      #print('PRESUMMARY')
      for j in range(c30p1Count - c30p1Lev):
        encPreModel._layers.pop()
          
      encin = encPreModel._layers[0].get_output_at(0)
      if (c30p1Count==c30p1Lev):
         encout = encPreModel._layers[-3].get_output_at(0)
      else:
         encout = encPreModel._layers[-2].get_output_at(0)
      #''' 
      encModel=Model(inputs=encin,outputs=encout)
      #88888888 	encModel.summary()
      enc30p1 = keras.models.clone_model(encModel)
      
      #print('ENC MODELmod30p1')#, i)
      X_train30p1_encoded = encPredict(encModel,x_train)
      X_test30p1_encoded = encPredict(encModel,x_test)
      prevc30p1Lev=c30p1Lev
      
      
     if ((not(prevc20p1Lev==c20p1Lev)) or (not init)):#codifying data for mod50p1
      encPreModel = keras.models.clone_model(mod20p1)
      modcopy  = keras.models.clone_model(mod20p1)
      orig_in = mod20p1.layers[0].get_output_at(0)
   
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      #print('PRESUMMARY')
      for j in range(c20p1Count - c20p1Lev):
        encPreModel._layers.pop()
          
      encin = encPreModel._layers[0].get_output_at(0)
      if (c20p1Count==c20p1Lev):
         encout = encPreModel._layers[-3].get_output_at(0)
      else:
         encout = encPreModel._layers[-2].get_output_at(0)
      #''' 
      encModel=Model(inputs=encin,outputs=encout)
      #88888888 	encModel.summary()
      enc20p1 = keras.models.clone_model(encModel)
      
      #print('ENC MODELmod30p1')#, i)
      X_train20p1_encoded = encPredict(encModel,x_train)
      X_test20p1_encoded = encPredict(encModel,x_test)
      prevc20p1Lev=c20p1Lev
      
     if ((not(prevc50p5Lev==c50p5Lev)) or (not init)):#codifying data for mod50p1
      encPreModel = keras.models.clone_model(trimmed50p5)
      modcopy  = keras.models.clone_model(trimmed50p5)
      orig_in = trimmed50p5.layers[0].get_output_at(0)
   
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      #print('PRESUMMARY')
      for j in range(c50p5Count - c50p5Lev):
         print("popping a layer from c50p5")
         encPreModel._layers.pop()
          
      encin = encPreModel._layers[0].get_output_at(0)
      if (c50p5Count==c50p5Lev):
         encout = encPreModel._layers[-2].get_output_at(0)
      else:
         encout = encPreModel._layers[-1].get_output_at(0)

      encout = encPreModel._layers[-1].get_output_at(0)
      #''' 
      encModel=Model(inputs=encin,outputs=encout)
      encModel.summary()
      
      print('for 50p5 encout.shape is ', encout.shape, ', name is ',encout.name )
      encModel.summary()
      enc50p5 = keras.models.clone_model(encModel)
      print('ENC MODELmod50p5')#, i)
      X_train50p5_encoded = encPredict(encModel,x_train)
      X_test50p5_encoded = encPredict(encModel,x_test)
      prevc50p5Lev=c50p5Lev

     if ((not(prevc30p5Lev==c30p5Lev)) or (not init)):#codifying data for mod50p1
      encPreModel = keras.models.clone_model(trimmed30p5)
      modcopy  = keras.models.clone_model(trimmed30p5)
      orig_in = trimmed30p5.layers[0].get_output_at(0)
   
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      #print('PRESUMMARY')
      for j in range(c30p5Count - c30p5Lev):
           encPreModel._layers.pop()
          
      encin = encPreModel._layers[0].get_output_at(0)
      if (c30p5Count==c30p5Lev):
         encout = encPreModel._layers[-1].get_output_at(0)
      else:
         encout = encPreModel._layers[-1].get_output_at(0)
      #''' 
      encModel=Model(inputs=encin,outputs=encout)
      #88888888 	encModel.summary()

      enc30p5 = keras.models.clone_model(encModel)
      
      #print('ENC MODELmod30p1')#, i)
      X_train30p5_encoded = encPredict(encModel,x_train)
      X_test30p5_encoded = encPredict(encModel,x_test)
      prevc30p5Lev=c30p5Lev


     if ((not(prevc20p5Lev==c20p5Lev)) or (not init)):#codifying data for mod50p1
      encPreModel = keras.models.clone_model(trimmed20p5)
      modcopy  = keras.models.clone_model(trimmed20p5)
      orig_in = trimmed20p5.layers[0].get_output_at(0)
   
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      #print('PRESUMMARY')
      for j in range(c20p5Count - c20p5Lev):
           encPreModel._layers.pop()
          
      encin = encPreModel._layers[0].get_output_at(0)
      if (c20p5Count==c20p5Lev):
         encout = encPreModel._layers[-1].get_output_at(0)
      else:
         encout = encPreModel._layers[-1].get_output_at(0)
      #''' 
      encModel=Model(inputs=encin,outputs=encout)
      #88888888 	encModel.summary()

      enc20p5 = keras.models.clone_model(encModel)
      
      #print('ENC MODELmod30p1')#, i)
      X_train20p5_encoded = encPredict(encModel,x_train)
      X_test20p5_encoded = encPredict(encModel,x_test)
      prevc20p5Lev=c20p5Lev
      
     print(perc,'% 1[', c50p1Lev, ':', X_train50p1_encoded.shape, '], 5[', c50p5Lev, ':', X_train50p5_encoded.shape, ']')
    
     init=True
     enc50p1.summary()
     enc50p5.summary()
     #enc30p1.summary()
     #enc20p1.summary()
     #combModel = mergeModels(enc50p1, enc30p1,enc20p1,enc50p5, enc30p5,enc20p5, c50p1Lev, c50p5Lev)




     #try:
     if True:#888888888888888888
        fitCombined(X_train50p1_encoded, X_train30p1_encoded,X_train20p1_encoded, X_train50p5_encoded, X_train30p5_encoded,X_train20p5_encoded, X_test50p1_encoded, X_test30p1_encoded,X_test20p1_encoded, X_test50p5_encoded, X_test30p5_encoded,X_test20p5_encoded, y_traincat, y_testcat, combModel, c50p1Lev, c50p5Lev,perc)  

     #except:
     #   print('next time @', perc)  
    
   #except:
   #    print('couldnt create model @', perc)
    #fitCombined(x_train, x_test,  y_traincat, y_testcat, combModel)  
     

    #modelFixed.summary()
    

    
