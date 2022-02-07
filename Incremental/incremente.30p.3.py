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
                         Flatten, MaxPooling2D,MaxPooling1D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input
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
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-base50/'
baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-Base50p/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-base50p/'
baseFolder = '/home/paul/Downloads/ESC-50-tst2b/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-Next30p/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-next30p/'
nextFolder = '/home/paul/Downloads/ESC-50-tst2b/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0
dataSize = 128
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)

# model parameters for training
batchSize = 128
epochs = 100#0


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


def fitCombined(xTrain,xTest, y_train, y_test,combModel,fixedLayers):
    print('xtrain shape1:',xTrain.shape)
    print('xTest shape1:',xTest.shape)
    if len(xTrain.shape)==5:
        #inLyr = Input(shape=(int(xTrain[0]),int(xTrain[1]),int(xTrain[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        #branchIn = Input(shape=(int(xTrain[0]),int(xTrain[1]),int(xTrain[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        
        xTrain = np.array([x.reshape( int(xTrain.shape[2]), int(xTrain.shape[3]), int(xTrain.shape[4]) ) for x in xTrain])
        xTest = np.array([x.reshape( int(xTest.shape[2]), int(xTest.shape[3]), int(xTest.shape[4]))  for x in xTest])
        
    else:
      if len(xTrain.shape)==4:
        #inLyr = Input(shape=(int(xTrain.shape[1]),int(xTrain.shape[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        #branchIn = Input(shape=(int(xTrain.shape[1]),int(xTrain.shape[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        xTrain = np.array([x.reshape( int(xTrain.shape[2]),int(xTrain.shape[3]) ) for x in xTrain])
        xTest = np.array([x.reshape( (int(xTest.shape[2]), int(xTest.shape[3])))  for x in xTest])
      else:
        if fixedLayers==14:# and fixedLayers <16:
           xTrain = np.array([x.reshape( int(xTrain.shape[2])) for x in xTrain])
           xTest = np.array([x.reshape( int(xTest.shape[2]))  for x in xTest])

    #combModel.summary() 
    print('xtrain shape2 is ',xTrain.shape)
    #xTrain = np.array([x.reshape( (128, 128, 1) ) for x in xTrain])
    #xTest = np.array([x.reshape( (128, 128, 1 ) ) for x in xTest])
    #xTrain = np.array([x.reshape( (8,8,48) ) for x in xTrain])
    #xTest = np.array([x.reshape( (8,8,48) ) for x in xTest])

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
    print ('xtrain3:', xTrain.shape)
    #print ('xtraincoded:', xTrainEncoded.shape)

    indata = [xTrain,xTrain]
    #''' 
    print('started fit at ', datetime.datetime.now())
    combModel.fit(indata,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= ([xTest,xTest], y_test),#,
        ##callbacks=[checkpoint]

        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    )
    print('finished fit at ', datetime.datetime.now())

    score = combModel.evaluate([xTest,xTest],
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    #combModel.save('Inc.branch.3.'+str(fixedLayers)+'_lyrs.'+str(round(score[1],3))+'.'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.hdf5')
    #'''
    print('Model exported and finished')
    

    


def cloneBranchedModel(modelbase, startLayer,totalLabel):
    print("cloning from ", modelbase.layers[startLayer-1].name )
    #origBranch.layers[j].name 
    #encModel.summary()
    input_shape_a=modelbase.layers[startLayer-1].get_output_at(0).shape#(128, 128,1)
    print('startlyr = ',startLayer, 'input-shape -a', input_shape_a)
    #return   
    #if ((startLayer >=10) and (startLayer<=13)):
    #   startLayer=10
    inLyr=None
    branchIn=None
    if len(input_shape_a)==4:
        inLyr = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2]),int(input_shape_a[3])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        branchIn = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2]),int(input_shape_a[3])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
    else:#     
      if len(input_shape_a)==3:
        #print('inLyr.shp=>', input_shape_a)
        inLyr = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
        branchIn = Input(shape=(int(input_shape_a[1]),int(input_shape_a[2])))#(128,128,1))#modelbase.layers[startLayer].get_output_at(0))#Input(shape=input_shape_a)
      else:
        if startLayer==8:# and startLayer <16:
           
           inLyr = Input(shape=(None,4096))
           branchIn = Input(shape=(None,4096))
        else:
           if startLayer>8:
             inLyr=Input((None,4096))
             branchIn=Input((None,4096))
             #inLyr=Reshape((1,4096))(inLyr)
             #branchIn=Reshape((1,4096))(branchIn)
    nextLyr=inLyr
    nextBr=branchIn
    numlyrs =len(modelbase.layers)
    currlyr=0
    #print(modelbase.layers[startLayer].get_output_at(0).shape,'::inLyr.shape==>', inLyr.shape)
    #nextLyr = Reshape(modelbase.layers[startLayer].get_output_at(0).shape)(inLyr)
    for layer in modelbase.layers:
     if (currlyr >= startLayer):
      print(currlyr,':layer name is ', layer.name, ' ; ', nextLyr.name)
      if not isinstance(nextLyr, keras.layers.Reshape):
          print(nextLyr.shape, '==>',str(layer.name), layer.get_output_at(0).shape)
      if currlyr <(numlyrs-2):
        if isinstance(layer, keras.layers.InputLayer):
           nextLyr = layer.get_output_at(0)
           nextBr = layer.get_output_at(0)
        if isinstance(layer, keras.layers.Conv2D):
           nextLyr = Conv2D(layer.filters,layer.kernel_size)(nextLyr)#, layer.get_output_at(0).shape
           nextBr = Conv2D(layer.filters,layer.kernel_size)(nextBr)#, layer.get_output_at(0).shape
           #nextLyr = Conv2D(layer.filters,(1,1))(nextLyr)
        if isinstance(layer, keras.layers.MaxPooling2D):
           nextLyr = MaxPooling2D(pool_size=layer.pool_size)(nextLyr)
           nextBr = MaxPooling2D(pool_size=layer.pool_size)(nextBr)
           #nextLyr = MaxPooling2D(layer.channels,(1,1),layer.strides,layer.shape)(nextLyr)
        if isinstance(layer, keras.layers.Activation):
           nextLyr = Activation('relu')(nextLyr)
           nextBr = Activation('relu')(nextBr)
        if isinstance(layer, keras.layers.Dropout):
           nextLyr = Dropout(rate=0.5)(nextLyr)
           nextBr = Dropout(rate=0.5)(nextBr)
        if isinstance(layer, keras.layers.Flatten):
           nextLyr = Flatten()(nextLyr)
           nextBr = Flatten()(nextBr)
        if isinstance(layer, keras.layers.Dense):
           nextLyr = Dense(units=layer.units)(nextLyr)
           nextBr = Dense(units=layer.units)(nextBr)
           #nextLyr = Dense(units=layer.units)(nextLyr)
        if isinstance(layer, keras.layers.LSTM):
           nextLyr = LSTM(units= layer.units,return_sequences=layer.return_sequences,unit_forget_bias=layer.unit_forget_bias,dropout=layer.dropout)(nextLyr)
           nextBr = LSTM(units= layer.units,return_sequences=layer.return_sequences,unit_forget_bias=layer.unit_forget_bias,dropout=layer.dropout)(nextBr)
        if isinstance(layer, keras.layers.Reshape):
           nextLyr = Reshape(layer.target_shape)(nextLyr)
           nextBr = Reshape(layer.target_shape)(nextBr)
     currlyr+=1 
    #V2179842T3
    print('nextBr c shape is ', nextBr.shape)
  
    concatC = Concatenate(name='outconCat',axis=1) ([nextLyr,nextBr])
    print('concat c shape is ', concatC.shape)
    #recat = Reshape(target_shape=(4096,64))(concatC)
    #recat = Reshape(target_shape=(1,64))(concatC)
    #print('recat c shape is ', recat.shape)

    dropCat=Dropout(rate=0.5)(concatC)
    print('dropCat shape is ', dropCat.shape)
    lastDense = Dense(totalLabel)(dropCat)
    print('lastDense shape is ', lastDense.shape)
    out = Activation('softmax')(lastDense)
    newModel = Model(inputs= [inLyr,branchIn], outputs=out)
    origBranch = Model(inputs= inLyr, outputs=nextLyr)
    for j in range(1,len(origBranch.layers)-2):
        #print('@[',startLyr,'][',j,'] copying weights for ', origBranch.layers[j].name , ' from ', modelbase.layers[j+startLyr-1].name,'; src shape',modelbase.layers[j+startLyr-1].get_output_at(0).shape, ';dest shape is ' ,origBranch.layers[j].get_output_at(0).shape)
        origBranch.layers[j].set_weights(modelbase.layers[j+startLyr-1].get_weights())
        origBranch.layers[j].trainable=False
    otherBranch = Model(inputs= branchIn, outputs=nextBr)
    print("OTHER")
    #origBranch.summary()
    print("NEW")
    newModel.summary()  
    return newModel,origBranch,otherBranch
    print('this was newModel ', startLayer)      
    
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
    random.shuffle(dataset)
    availCount= int(totalRecordCount*0.2)
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



    image_size = nx_train[0].shape

    #encfine.built=True
    #enccoarse.built=True
    #print(os.getcwd())
    modelbase =keras.models.load_model('50p/3/Model.3.final.hdf5',  custom_objects={'tf': tf})
    modelbase.summary()
    #################################################################################
    
    orig_in = modelbase.layers[0].get_output_at(0)
    
    
    
    '''
   
    #print('inshape a', act_10a.shape) 
    lsinput = Input(shape=(8,8,48))
    latent_dim=8
    print('lsinput a shape is ', lsinput.shape) 
    
    re_10a = Reshape(target_shape=(latent_dim*latent_dim, 48),input_shape=(latent_dim,latent_dim ,48))(lsinput)
    ls11a= LSTM(latent_dim*latent_dim,return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(re_10a)
    print('ls11 a shape is ', ls11a.shape) 
    
    re11a = Reshape(target_shape=(latent_dim*latent_dim ,latent_dim*latent_dim ))(ls11a)
    #merge

      
    #at15 = Attention(latent_dim)(ls_5b)
    #print('at15 shape is ', at15.shape) 
    flat12 = Flatten()(re11a)
    drop13 = Dropout(rate=0.5)(flat12)
    dense14 =  Dense(64)(drop13)
    act15 = Activation('relu')(dense14)
    lsEndBranch = Model(inputs=[lsinput], outputs=act15)
    ############################################################
    re_10a_1 = Reshape(target_shape=(latent_dim*latent_dim, 48),input_shape=(latent_dim,latent_dim ,48))(lsinput)
    ls11a_1= LSTM(latent_dim*latent_dim,return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(re_10a_1)
    re11a_1 = Reshape(target_shape=(latent_dim*latent_dim ,latent_dim*latent_dim ))(ls11a_1)
    flat12_1 = Flatten()(re11a_1)
    drop13_1 = Dropout(rate=0.5)(flat12_1)
    dense14_1 =  Dense(64)(drop13_1)
    act15_1 = Activation('relu')(dense14_1)
    branch1 = Model(inputs=[lsinput], outputs=act15_1)

    srclayer =-11
    for layer in branch1.layers:
        if srclayer>=-10: 
           print(modelbase.layers[srclayer].name, '=>', layer.name)
           layer.set_weights(modelbase.layers[srclayer].get_weights())
        srclayer+=1
        layer.trainable=False
    
    
    ################################################################################
    ###
    weight_decay=1e-4
    denseIn1 = Input(shape=(8,8,48))
    
    b1_bnorm1a = BatchNormalization()(lsinput)
    b1_act1a = Activation('relu')(b1_bnorm1a)
    b1_conv1a = Conv2D(48, (1, 1), padding='same',
                                 kernel_regularizer=keras.regularizers.l2(weight_decay))(b1_act1a)
    b1_bnorm1b = BatchNormalization()(b1_conv1a)
    b1_act1b = Activation('relu')(b1_bnorm1b)
    b1_conv1b = Conv2D(12, (3, 3), padding='same')(b1_act1b)
    b1_flat=Flatten()(b1_conv1b)
    b1_denseout =  Dense(64)(b1_flat)
    b1_actout = Activation('relu')(b1_denseout)
    denseEndBranch=Model(inputs=lsinput,outputs=b1_actout)

        
        
    
    modConcatLayer=None
    origlen = len(modelbase.layers)
    for i in range(origlen):
      if (i<(origlen - 10)):
         modelbase.layers[i].trainable=False
      else:  #(i<=10):
        modelbase._layers.pop()
        if (i==(origlen -7)):
           modConcatLayer = modelbase.layers[i]
        #print(densebase.layers[-1].name, len(densebase.layers))
        
        
        
    modelout = modelbase.layers[-1].get_output_at(0)
    modelFixed=Model(inputs=orig_in,outputs=modelout)
    
    #concatC = Concatenate(axis=1, name = 'outconcat')([act15_1,act15])#Expt 1
    #concatC = Concatenate(axis=1, name = 'outconcat')([act15_1, b1_actout])#Expt 2
    concatC = Concatenate(axis=1, name = 'outconcat')([act15_1,act15, b1_actout])#Expt3
    
    #concatC = Concatenate( name = 'outconcat')([act15c,branch2_act15c])
    #drop16c=Dropout(rate=0.5)(concatC)
    #dense17c=Dense(totalLabel)(drop16c)
    #out2 = Activation('softmax', name='actout')(dense17c)
    

  

    drop16=Dropout(rate=0.5)(concatC)
    dense17=Dense(totalLabel)(drop16)
    print('dense shape',dense17.shape)  
    out = Activation('softmax')(dense17)

    comb_lsModel = Model(inputs=lsinput, outputs=out)


    print ("about to encode fine train",x_train.shape)
    #modelbase.summary()
    modelFixed.built=True


    ''' 


    #m = encPredict(densebase,x_train)
    #print ("about to encode coarse test")
    '''
    X_train_encoded = encPredict(modelFixed,x_train)
    X_test_encoded = encPredict(modelFixed,x_test)# enc32(x_test)



    nX_train_encoded = encPredict(modelFixed,nx_train)
    nX_test_encoded = encPredict(modelFixed,nx_test)# enc32(x_test)

    #print ("about to encode coarse test")
    #X_test_coarse_encoded = encPredict(enccoarse, x_test)
    #y_train = np.array(y_train)
    #y_test = np.array(y_test)
    #y_traincat = to_categorical(y_train)
    #y_testcat = to_categorical(y_test)
    x_train = np.array(x_train)
    trunc_testEncoded= X_test_encoded[:len(nX_test_encoded)]
    #print('ytrain', y_train.shape)
    print('ytraincat', y_traincat.shape)
    print('ytestcat', y_testcat.shape)
    print('xtrain', X_train_encoded.shape)
    '''
 
    #fitCombined(nX_train_encoded, nX_test_encoded,  ny_traincat, ny_testcat, comb_lsModel)
    #modelbase._layers.pop()
    #modelbase._layers.pop()
    
    modelout = modelbase.layers[-1].get_output_at(0)
    
    denseOut=Dense(totalLabel,name='denseout')(modelout)
    out=Activation('softmax',name='actout')(denseOut)
    for i in range(2, len(modelbase.layers[:-2])):
      encPreModel = keras.models.clone_model(modelbase)
      
      encPreModel.build(orig_in)
      #encPreModel.summary()
      print('PRESUMMARY')
      for j in range( len(modelbase.layers)-i):
          encPreModel._layers.pop()
          
      encin = encPreModel.layers[0].get_output_at(0)
      encout = encPreModel.layers[-1].get_output_at(0)
      encModel=Model(inputs=encin,outputs=encout)
      encModel.summary()
      
      print('ENC MODEL', i)
      X_train_encoded = encPredict(encModel,x_train)
      X_test_encoded = encPredict(encModel,x_test)
      startLyr =i
          
      modelNew,origBranch,otherBranch= cloneBranchedModel(modelbase, startLyr, totalLabel)
      #modelNew.summary()
      #for j in range(len(otherBranch.layers)):
      #  print('setting layer ', len(otherBranch.layers)-j-1) 
      #  init_layer(otherBranch.layers[j])
   

      #plot_model(encModel, 'modelplots/enc.1_'+str(i)+'.png')
      #plot_model(modelNew, 'modelplots/brch.1_'+str(i)+'.png')
      
      #try:
      try:
         fitCombined(X_train_encoded, X_test_encoded,  y_traincat, y_testcat, modelNew, startLyr)  
      except Exception as e:
          print("error fitting model with ",i, " frozen layers: ", e)

    #modelFixed.summary()
    

    
