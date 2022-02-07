from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
#from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
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
#
# Total wav records for training the model, will be updated by the program
totalRecordCount = 0
dataSize = 128
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)

# model parameters for training
batchSize = 64
epochs = 5000#0


filepath = "ESCvae-model-{epoch:02d}-{loss:.2f}.hdf5"
#checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

#tf.compat.v1.disable_eager_execution()

def init_layer(layer):
    session = K.get_session()
    weights_initializer = tf.variables_initializer(layer.weights)
    session.run(weights_initializer)

def encPredict(enc, x_train):
   viewBatch=1
   numrows = x_train.shape[0]
   z_mean=[]
   for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample = x_train[i*viewBatch:i*viewBatch+viewBatch,]
      #z_mean8, _, _ = enc.predict([[sample, sample]])
      z_mean8 = enc.predict(sample)
      z_mean.append(z_mean8)
      if (i==0):
         print('INPUT:::')
         print(sample.shape)
         print('OUTPUT:::')
         print(z_mean8.shape)
      #print('Sample ', i, ' shape ', sample.shape , ' converted to ', z_mean8.shape)
      #if (i==0):
      #  z_mean=z_mean8[0]
      #else:
      #  z_mean = np.concatenate((z_mean,z_mean8[0]))#, axis=0)
      #if True:#(i%200==0) and i>1:  
      #  print("enc stat",z_mean.shape)
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
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


def fitCombined(X_train50p1_encoded, X_train30p1_encoded,X_train20p1_encoded, X_train50p5_encoded, X_train30p5_encoded,X_train20p5_encoded, X_test50p1_encoded, X_test30p1_encoded, X_test20p1_encoded, X_test50p5_encoded, X_test30p5_encoded, X_test20p5_encoded, y_train, y_test,combModel):#,fixedLayers):
    print('xtrain shape1:',X_train50p1_encoded.shape)
    print('xTest shape1:',X_test50p1_encoded.shape)
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
    epochs_drop = 100
    decay = initial_learning_rate / epochs
    def lr_time_based_decay2(epoch, lr):
       if epoch < 50:
            return initial_learning_rate
       else:
            lrate = initial_learning_rate *drop/((1+epoch)/epochs_drop)
       return lrate
    def lr_time_based_decay(epoch, lr):
       if epoch < 500:
            return initial_learning_rate
       else:
            #if epoch%epochs_drop==0:
            #    lr=lr*drop
            return lr * epoch / (epoch + decay * epoch)
    #print ('xtrain3:', xTrain.shape)
    #print ('xTest:', xTest.shape)
    print ('xtraincoded:', X_train50p1_encoded.shape)
    
    X_train50p1_encoded=np.reshape(X_train50p1_encoded,(X_train50p1_encoded.shape[0],64))
    X_train30p1_encoded=np.reshape(X_train30p1_encoded,(X_train30p1_encoded.shape[0],64))
    X_train20p1_encoded=np.reshape(X_train20p1_encoded,(X_train20p1_encoded.shape[0],64))

    X_train50p5_encoded=np.reshape(X_train50p5_encoded,(X_train50p5_encoded.shape[0],264))
    X_train30p5_encoded=np.reshape(X_train30p5_encoded,(X_train30p5_encoded.shape[0],264))
    X_train20p5_encoded=np.reshape(X_train20p5_encoded,(X_train20p5_encoded.shape[0],264))
   

    
    X_test50p1_encoded=np.reshape(X_test50p1_encoded,(X_test50p1_encoded.shape[0],64))
    X_test30p1_encoded=np.reshape(X_test30p1_encoded,(X_test30p1_encoded.shape[0],64))
    X_test20p1_encoded=np.reshape(X_test20p1_encoded,(X_test20p1_encoded.shape[0],64))
    X_test50p5_encoded=np.reshape(X_test50p5_encoded,(X_test50p5_encoded.shape[0],264))
    X_test30p5_encoded=np.reshape(X_test30p5_encoded,(X_test30p5_encoded.shape[0],264))
    X_test20p5_encoded=np.reshape(X_test20p5_encoded,(X_test20p5_encoded.shape[0],264))
  
    print('X_train50p1_encoded s',X_train50p1_encoded.shape)
    print('X_train30p1_encoded s',X_train30p1_encoded.shape)
    print('X_train50p5_encoded s',X_train50p5_encoded.shape)
    print('X_train30p5_encoded s',X_train30p5_encoded.shape)
    indata = [X_train50p1_encoded, X_train30p1_encoded,X_train20p1_encoded, X_train50p5_encoded, X_train30p5_encoded, X_train20p5_encoded]
    #  indata=[X_train50p1_encoded, X_train30p1_encoded, X_train50p5_encoded, X_train30p5_encoded,]
    print('started fit at ', datetime.datetime.now())
    combModel.fit(indata,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= ([X_test50p1_encoded, X_test30p1_encoded,X_test20p1_encoded, X_test50p5_encoded, X_test30p5_encoded,X_test20p5_encoded], y_test),#,
        #validation_data= (xTest, y_test),#,
        ##callbacks=[checkpoint]
        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    )
    print('finished fit at ', datetime.datetime.now())

    score = combModel.evaluate([X_test50p1_encoded, X_test30p1_encoded,X_test20p1_encoded, X_test50p5_encoded, X_test30p5_encoded,X_test20p5_encoded],
        y=y_test)
    #score = combModel.evaluate(xTest,
    #0    y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    fixedLayers=50
    combModel.save('comb.branch.1.'+str(fixedLayers)+'_lyrs.'+str(round(score[1],3))+'.'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.hdf5')
    print('Model exported and finished')
    

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
    
def mergeModels(mod50p10, mod30p10,mod20p10,mod50p50, mod30p50, mod20p50):

    for layer in mod50p10.layers:
        layer.name = layer.name + str("_1")
        layer.fixed=True
        #print("updated "+layer.name + " to " + layer._name)
        
    for layer in mod30p10.layers:
        layer.name = layer.name + str("_2")
        layer.fixed=True
        
    for layer in mod20p10.layers:
        layer.name = layer.name + str("_3")
        layer.fixed=True        

    for layer in mod50p50.layers:
        layer.name = layer.name + str("_4")
        layer.fixed=True
        #print("updated "+layer.name + " to " + layer._name)
        
    for layer in mod30p50.layers:
        layer.name = layer.name + str("_5")
        layer.fixed=True
        
    for layer in mod20p50.layers:
        layer.name = layer.name + str("_6")
        layer.fixed=True

    mod50pstart10 = mod50p10.layers[0].get_output_at(0)
    mod50pend10 = mod50p10.layers[-2].get_input_at(0)
    mod30pstart10 = mod30p10.layers[0].get_output_at(0)
    mod30pend10 = mod30p10.layers[-2].get_input_at(0)
    mod20pstart10 = mod20p10.layers[0].get_output_at(0)
    mod20pend10 = mod20p10.layers[-2].get_input_at(0)
    
    mod50pstart50 = mod50p50.layers[0].get_output_at(0)
    mod50pend50 = mod50p50.layers[-2].get_input_at(0)
    mod30pstart50 = mod30p50.layers[0].get_output_at(0)
    mod30pend50 = mod30p50.layers[-2].get_input_at(0)
    mod20pstart50 = mod20p50.layers[0].get_output_at(0)
    mod20pend50 = mod20p50.layers[-2].get_input_at(0)
    
    
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


    print("mod50pend1 shape is ", mod50pend1.shape)
    print("mod30pend1 shape is ", mod30pend1.shape)
    print("mod50pend5 shape is ", mod50pend5.shape)
    print("mod30pend5 shape is ", mod30pend5.shape)

    in50p1 = Input((64,))
    in30p1 = Input(( 64,))
    in20p1 = Input(( 64,))
    in50p5 = Input(( 264,))
    in30p5 = Input(( 264,))
    in20p5 = Input(( 264,))
    '''
    conv50p1= Conv1D(1,(1,))(in50p1)
    conv30p1= Conv1D(1,(1,))(in30p1)
    conv50p5= Conv1D(1,(1,))(in50p5)
    conv30p5= Conv1D(1,(1,))(in30p5)
    '''

    bn50p1= BatchNormalization()(in50p1)
    bn30p1= BatchNormalization()(in30p1)
    bn20p1= BatchNormalization()(in20p1)
    bn50p5= BatchNormalization()(in50p5)
    bn30p5= BatchNormalization()(in30p5)
    bn20p5= BatchNormalization()(in20p5)

    act50p1= Activation('relu')(bn50p1)
    act30p1= Activation('relu')(bn30p1)
    act20p1= Activation('relu')(bn20p1)
    act50p5= Activation('relu')(bn50p5)
    act30p5= Activation('relu')(bn30p5)
    act20p5= Activation('relu')(bn20p5)
    
    drop50p1= Dropout(0.5)(act50p1)
    drop30p1= Dropout(0.5)(act30p1)
    drop20p1= Dropout(0.5)(act20p1)
    drop50p5= Dropout(0.5)(act50p5)
    drop30p5= Dropout(0.5)(act30p5)
    drop20p5= Dropout(0.5)(act20p5)

    print("in50p1 shape is ", in50p1.shape)
    print("in30p1 shape is ", in30p1.shape)         
    #concatC = Concatenate(name='outconCat',axis=1) ([drop50p1,drop30p1,drop50p5,drop30p5])
    concatC = Concatenate(name='outconCat',axis=1) ([act50p1,act30p1,act20p1,act50p5,act30p5,act20p5])
    #concatC = Concatenate(name='outconCat',axis=1) ([in50p1,in30p1,in50p5,in30p5])

    print("concatC shape is ", concatC.shape)
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
        
    #modelout = modelbase.layers[-1].get_output_at(0)
    
    #denseOut=Dense(totalLabel,name='denseout')(modelout)
    #out=Activation('softmax',name='actout')(denseOut)
    
    #dropCat=Dropout(rate=0.5)(concatC)
    #print('dropCat shape is ', dropCat.shape)
    #lastDense = Dense(totalLabel)(concatC)
    #newModel = Model(inLyr, prediction)
    #newModel = Model([mod50pstart1,mod30pstart1,mod50pstart5,mod30pstart5], prediction)
    newModel = Model([in50p1,in30p1,in20p1,in50p5,in30p5,in20p5], prediction)

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
    
    print('ytraincat', y_traincat.shape)
    print('ytestcat', y_testcat.shape)
    print('x_train', x_train.shape)
    print('x_test', x_test.shape)

    if True:#codifying data for mod50p1
      encPreModel = keras.models.clone_model(mod50p1)
      modcopy  = keras.models.clone_model(mod50p1)
      orig_in = mod50p1.layers[0].get_output_at(0)
   
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      print('PRESUMMARY')
      #for j in range( len(modelbase.layers)-i-1):
      encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-2].get_output_at(0)
      #''' 
      encModel=Model(inputs=encin,outputs=encout)
      #88888888 	encModel.summary()
      
      print('ENC MODELmod50p1')#, i)
      X_train50p1_encoded = encPredict(encModel,x_train)
      X_test50p1_encoded = encPredict(encModel,x_test)

    if True:#codifying data for mod30p1
      encPreModel = keras.models.clone_model(mod30p1)
      modcopy  = keras.models.clone_model(mod30p1)
      orig_in = mod30p1.layers[0].get_output_at(0)
   
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      print('PRESUMMARY')
      #for j in range( len(modelbase.layers)-i-1):
      encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-2].get_output_at(0)
      #''' 
      encModel=Model(inputs=encin,outputs=encout)
      #88888888 	encModel.summary()
      
      print('ENC MODELmod30p1')#, i)
      X_train30p1_encoded = encPredict(encModel,x_train)
      X_test30p1_encoded = encPredict(encModel,x_test)
      
    if True:#codifying data for mod20p1
      encPreModel = keras.models.clone_model(mod20p1)
      modcopy  = keras.models.clone_model(mod20p1)
      orig_in = mod20p1.layers[0].get_output_at(0)
   
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      print('PRESUMMARY')
      #for j in range( len(modelbase.layers)-i-1):
      encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-2].get_output_at(0)
      #''' 
      encModel=Model(inputs=encin,outputs=encout)
      #88888888 	encModel.summary()
      
      print('ENC MODELmod30p1')#, i)
      X_train20p1_encoded = encPredict(encModel,x_train)
      X_test20p1_encoded = encPredict(encModel,x_test)
      
    if True:#codifying data for mod50p5
      encPreModel = keras.models.clone_model(mod50p5)
      modcopy  = keras.models.clone_model(mod50p5)
      orig_in = mod50p5.layers[0].get_output_at(0)
   
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      print('PRESUMMARY')
      #for j in range( len(modelbase.layers)-i-1):
      encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-1].get_output_at(0)
      #''' 
      encModel=Model(inputs=encin,outputs=encout)
      #88888888 	encModel.summary()
      
      print('ENC MODELmod50p1')#, i)
      X_train50p5_encoded = encPredict(encModel,x_train)
      X_test50p5_encoded = encPredict(encModel,x_test)

    if True:#codifying data for mod30p5
      encPreModel = keras.models.clone_model(mod30p5)
      modcopy  = keras.models.clone_model(mod30p5)
      orig_in = mod30p5.layers[0].get_output_at(0)
   
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      print('PRESUMMARY')
      #for j in range( len(modelbase.layers)-i-1):
      encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-1].get_output_at(0)
      #''' 
      encModel=Model(inputs=encin,outputs=encout)
      #88888888 	encModel.summary()
      
      print('ENC MODELmod30p1')#, i)
      X_train30p5_encoded = encPredict(encModel,x_train)
      X_test30p5_encoded = encPredict(encModel,x_test)

    if True:#codifying data for mod30p58888888888888
      encPreModel = keras.models.clone_model(mod20p5)
      modcopy  = keras.models.clone_model(mod20p5)
      orig_in = mod20p5.layers[0].get_output_at(0)
   
      modcopy.build(orig_in) 
      encPreModel.build(orig_in)
      #encPreModel.summary()
      print('PRESUMMARY')
      #for j in range( len(modelbase.layers)-i-1):
      encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-1].get_output_at(0)
      #''' 
      encModel=Model(inputs=encin,outputs=encout)
      #88888888 	encModel.summary()
      
      print('ENC MODELmod30p1')#, i)
      X_train20p5_encoded = encPredict(encModel,x_train)
      X_test20p5_encoded = encPredict(encModel,x_test)


    
    combModel = mergeModels(mod50p1, mod30p1, mod20p1,mod50p5, mod30p5, mod20p5)
    combModel.summary()
    fitCombined(X_train50p1_encoded, X_train30p1_encoded,X_train20p1_encoded, X_train50p5_encoded, X_train30p5_encoded,X_train20p5_encoded, X_test50p1_encoded, X_test30p1_encoded,X_test20p1_encoded, X_test50p5_encoded, X_test30p5_encoded,X_test20p5_encoded, y_traincat, y_testcat, combModel)  
    #fitCombined(x_train, x_test,  y_traincat, y_testcat, combModel)  
     

    #modelFixed.sum    ary()
    

    
