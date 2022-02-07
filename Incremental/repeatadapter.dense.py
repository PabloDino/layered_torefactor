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
from keras import backend as K




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
nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-last20p/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-last20p/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-last20p/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-next30p/'
#nextFolder = '/home/paul/Downloads/ESC-50-tst2b/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0
dataSize = 128
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)

# model parameters for training
batchSize = 128
epochs = 100#0


from timeit import default_timer as timer

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

#tf.compat.v1.disable_eager_execution()
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


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
        


filepath = 'DenseRep-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-{val_f1_m:.2f}-{val_precision_m:.2f}-{val_recall_m:.2f}-{acc:.2f}-.hdf5'
#filepath = 'Model.1.{epoch:02d}-{loss:.2f}.{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
cb = TimingCallback()


def fitCombined(xTrain50,xTrain20,xTest50,xTest20, y_train, y_test,combModel,fixedLayers):

    if len(xTrain50.shape)==5:
        xTrain50 = np.array([x.reshape( int(xTrain50.shape[2]), int(xTrain50.shape[3]), int(xTrain50.shape[4]) ) for x in xTrain50])
        xTest50 = np.array([x.reshape( int(xTest50.shape[2]), int(xTest50.shape[3]), int(xTest50.shape[4]))  for x in xTest50])
        xTrain20 = np.array([x.reshape( int(xTrain20.shape[2]), int(xTrain20.shape[3]), int(xTrain20.shape[4]) ) for x in xTrain20])
        xTest20 = np.array([x.reshape( int(xTest20.shape[2]), int(xTest20.shape[3]), int(xTest20.shape[4]))  for x in xTest20])
    else:
      if len(xTrain50.shape)==4:
        xTrain50 = np.array([x.reshape( int(xTrain50.shape[2]),int(xTrain50.shape[3]) ) for x in xTrain50])
        xTest50 = np.array([x.reshape( (int(xTest50.shape[2]), int(xTest50.shape[3])))  for x in xTest50])
        xTrain20 = np.array([x.reshape( int(xTrain20.shape[2]),int(xTrain20.shape[3]) ) for x in xTrain20])
        xTest20 = np.array([x.reshape( (int(xTest20.shape[2]), int(xTest20.shape[3])))  for x in xTest20])
      else:
        xTrain50 = np.array([x.reshape( int(xTrain50.shape[2]),1) for x in xTrain50])
        xTest50 = np.array([x.reshape( int(xTest50.shape[2]),1)  for x in xTest50])
        xTrain20 = np.array([x.reshape( int(xTrain20.shape[2]),1) for x in xTrain20])
        xTest20 = np.array([x.reshape( int(xTest20.shape[2]),1)  for x in xTest20])

    combModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

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
 
    indata = [xTrain50,xTrain20]
    #''' 
    print('started fit at ', datetime.datetime.now())
    combModel.fit(indata,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= ([xTest50,xTest20], y_test),#,
        callbacks=[checkpoint, cb]

        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    )
    print('finished fit at ', datetime.datetime.now())


    loss , acc, valacc,f1,precision, recall  = evaluateCheckPoints("DenseRep")
    
    print('Loss for best accuracy:', loss)
    print('Best validation accuracy:', valacc)
    print('Best training accuracy:', acc)
    sumtime, avgtime, max_itertime,min_itertime = getAggregates(cb.logs)
    print('sumtime, avgtime, max_itertime,min_itertime :', sumtime, avgtime, max_itertime,min_itertime )
  



    outfile=open("denserep.perf.txt","a")
    outfile.write(str(fixedLayers)+","+ str(acc)+","+ str(valacc) +","+str(f1)+","+str(precision)+","+str(recall)+","+str(sumtime)+","+str(avgtime)+","+str(max_itertime)+","+str(min_itertime)+","+ str(loss)+"\n" )
    outfile.close()
    print('Log completed for layer '+str(fixedLayers)) 
    

def getAggregates(logs):
   mx=0
   mn=10000
   for i in range(len(logs)):
       if (logs[i]>mx):
          mx=logs[i]
       if (logs[i]<mn):
          mn=logs[i]
   sumtime = sum(logs)
   avgtime = 1.0*sumtime/len(logs)

   return sumtime, avgtime, mx,mn
   
def evaluateCheckPoints(prefix):
    files=[]
    for fle in os.listdir():
        if fle.startswith(prefix):
            files.append(fle)
    maxvacc=0
    maxacc=0
    maxdx=0
    for dx in range(len(files)):
        arr = files[dx].split("-")
        if  float(arr[3])>maxvacc:
            maxvacc = float(arr[3])
            maxdx = dx
        if  float(arr[7])>maxacc:
            maxacc = float(arr[7])
    arr = files[maxdx].split("-")
            
    retloss = float(arr[2])
    retf1 = float(arr[4])
    retprecision = float(arr[5])
    retrecall = float(arr[6])
    for fle in files:
        os.remove(fle)
    return retloss,maxacc, maxvacc, retf1, retprecision, retrecall
                        


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
    
def mergeModels(mod50p, mod20p,startLayer):
    #mod50p.summary()
    print("cloning from ", mod50p.layers[startLayer-1].name )
    input_shape_a=mod50p.layers[startLayer-1].get_output_at(0).shape#(128, 128,1)
    nextLyr=mod50p.layers[startLayer-1].get_output_at(0)
    print('startlyr = ',startLayer, 'input-shape -a', input_shape_a)
    #return   
    #if ((startLayer >=10) and (startLayer<=13)):
    #   startLayer=10
    inLyr50p=None
    inLyr20p=None
    if len(input_shape_a)==4:
        inLyr50p = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2]),int(input_shape_a[3])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        inLyr20p = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2]),int(input_shape_a[3])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
    else:#     
      if len(input_shape_a)==3:
        #print('inLyr.shp=>', input_shape_a)
        inLyr50p = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        inLyr20p = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
      else:
         oldshape=K.int_shape(nextLyr)
         #if startLayer==8:# and startLayer <16:
         inLyr50p = Input(shape=(oldshape[1],1))
         inLyr20p = Input(shape=(oldshape[1],1))
    nextLyr50p=inLyr50p
    nextLyr20p=inLyr20p
    print("b4 concat",nextLyr50p.shape)
    nextLyr = Concatenate(name='inconCat',axis=1) ([nextLyr50p,nextLyr20p])
    print("ftr concat",nextLyr.shape)
    if len(input_shape_a)==4:
       nextLyr = MaxPooling2D(pool_size=(2,1))(nextLyr)
    else:
       nextLyr = MaxPooling1D(pool_size=2)(nextLyr)
    print("ftr pool",nextLyr.shape)
    
    numlyrs =len(mod50p.layers)
    currlyr=0
    for layer in mod50p.layers:
     if (currlyr >= startLayer):
      print(currlyr,':layer name is ', layer.name, ' ; ', nextLyr.name)
      if not isinstance(nextLyr, keras.layers.Reshape):
          print(nextLyr.shape, '==>',str(layer.name), layer.get_output_at(0).shape)
      if currlyr <(numlyrs-2):
        if isinstance(layer, keras.layers.InputLayer):
           nextLyr = layer.get_output_at(0)
        if isinstance(layer, keras.layers.Conv2D):
           nextLyr = Conv2D(layer.filters,layer.kernel_size)(nextLyr)#, layer.get_output_at(0).shape
           #nextLyr = Conv2D(layer.filters,(1,1))(nextLyr)
        if isinstance(layer, keras.layers.MaxPooling2D):
           nextLyr = MaxPooling2D(pool_size=layer.pool_size)(nextLyr)
           #nextLyr = MaxPooling2D(layer.channels,(1,1),layer.strides,layer.shape)(nextLyr)
        if isinstance(layer, keras.layers.Activation):
           nextLyr = Activation('relu')(nextLyr)
        if isinstance(layer, keras.layers.Dropout):
           nextLyr = Dropout(rate=0.5)(nextLyr)
        if isinstance(layer, keras.layers.Flatten):
           nextLyr = Flatten()(nextLyr)
        if isinstance(layer, keras.layers.Dense):
           nextLyr = Dense(units=layer.units)(nextLyr)
        if isinstance(layer, keras.layers.LSTM):
           nextLyr = LSTM(units= layer.units,return_sequences=layer.return_sequences,unit_forget_bias=layer.unit_forget_bias,dropout=layer.dropout)(nextLyr)
        if isinstance(layer, keras.layers.Reshape):
           nextLyr = Reshape(layer.target_shape)(nextLyr)
     currlyr+=1 
    #'''
    numdims=1
    
    for i in range(len(K.int_shape(nextLyr))):
        if K.int_shape(nextLyr)[i]!=None:
           numdims=numdims*K.int_shape(nextLyr)[i]
    #'''       
    #nextLyr =Reshape((numdims,))(nextLyr) 
    x = BatchNormalization()(nextLyr)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(totalLabel, activation='softmax')(x)
    newModel = Model(inputs= [inLyr50p, inLyr20p], outputs=prediction)

    return newModel
    



if __name__ == '__main__':
    #tensorboard = TensorBoard(log_dir = "logs/{}".format(time()))
    dataset =  importData('base',train=True)#(train, test) =  
    print('total recs =', totalRecordCount, '; Total Labels=', totalLabel, lblmap)
    nextdataset =   importData('next',train=True)#(nextTrain,nextTest) = 
    print('total recs =', totalRecordCount, '; Total Labels=', totalLabel, lblmap)
    random.shuffle(dataset)
    availCount= int(totalRecordCount*0.5)
    availset = dataset[:availCount]
    #availset = replicateListToMatch(availset,len(nextdataset)/2)
    availset = mergeSets2(availset, nextdataset)#train, test, nextTrain, nextTest)
    availCount+=len(nextdataset)

    print('AvailCount: {}'.format(availCount))
    trainDataEndIndex = int(availCount*0.8)
    random.shuffle(availset)
    train = availset[:trainDataEndIndex]
    test = availset[trainDataEndIndex:]
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





    random.shuffle(nextdataset)
    nextRecordCount= len(nextdataset)

    print('nextRecordCount: {}'.format(nextRecordCount))
    nxttrainDataEndIndex = int(nextRecordCount*0.8)
    nx, ny = zip(*nextdataset)
    nx_train = nx[:nxttrainDataEndIndex]
    nx_test = nx[nxttrainDataEndIndex:]   
    nycat = to_categorical(ny)
    ny_traincat = nycat[:nxttrainDataEndIndex]
    ny_testcat = nycat[nxttrainDataEndIndex:]   

    nx_train = np.array(nx_train)
    nx_test = np.array(nx_test)
    nx_train = np.expand_dims(nx_train,-1)
    nx_test = np.expand_dims(nx_test,-1)
    nx_train = nx_train.astype('float32') / 255
    nx_test = nx_test.astype('float32') / 255



    mod50p =keras.models.load_model('50p/Model.3.hdf5',custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})#, custom_objects={'sampling': sampling}, compile =False)
    mod20p =keras.models.load_model('20p/Model.3.hdf5',custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})#, custom_objects={'sampling': sampling}, compile =False)
    #modelbase.summary()
    #################################################################################
    for layer in mod50p.layers:
        layer.name = layer.name + str("_1")
           
    for layer in mod20p.layers:
        layer.name = layer.name + str("_2")
       
    orig50_in = mod50p.layers[0].get_output_at(0)
    orig20_in = mod20p.layers[0].get_output_at(0)
    
    
    #modelout = modelbase.layers[-1].get_output_at(0)
    
    #denseOut=Dense(totalLabel,name='denseout')(modelout)
    #out=Activation('softmax',name='actout')(denseOut)
    for i in range( len(mod50p.layers[:-1])-2,1,-5):
    
      encPreModel = keras.models.clone_model(mod50p)
      
      encPreModel.build(orig50_in)
      #encPreModel.summary()
      print('PRESUMMARY')
      for j in range( len(mod50p.layers)-i):
          encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-1].get_output_at(0)
      encModel=Model(inputs=encin,outputs=encout)
      #encModel.summary()
      enc50p = keras.models.clone_model(encModel)
      X_train50p_encoded = encPredict(encModel,x_train)
      X_test50p_encoded = encPredict(encModel,x_test)
                  
      ###########20p#############################
      encPreModel = keras.models.clone_model(mod20p)
      
      encPreModel.build(orig20_in)
      #encPreModel.summary()
      print('PRESUMMARY')
      for j in range( len(mod20p.layers)-i):
          encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-1].get_output_at(0)
      encModel=Model(inputs=encin,outputs=encout)
      #encModel.summary()
      enc20p = keras.models.clone_model(encModel)
      X_train20p_encoded = encPredict(encModel,x_train)
      X_test20p_encoded = encPredict(encModel,x_test)

      startLyr =i
      
    
      combModel = mergeModels(enc50p,enc20p, startLyr)
      #combModel.summary()
      try:
        #if True:#
        fitCombined(X_train50p_encoded, X_train20p_encoded, X_test50p_encoded, X_test20p_encoded, y_traincat, y_testcat, combModel, startLyr)
      except: 
        print('next time @', startLyr)  
    

    
