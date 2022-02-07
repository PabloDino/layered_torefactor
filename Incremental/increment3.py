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
from keras.layers import Activation, Dense, Dropout, Conv1D,Conv2D, GlobalAveragePooling2D, \
                         Flatten, MaxPooling2D,MaxPooling1D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input
from keras.layers.normalization import BatchNormalization                      
import tensorflow as tf
import denseBase
from denseBase import DenseBase

from time import time
import numpy as np
#import matplotlib.pyplot as plt
import os
from config import *

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
baseFolder = '/home/paul/Downloads/ESC-50-tst2b/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-Next30p/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-next30p/'
#nextFolder = '/home/paul/Downloads/ESC-50-tst2b/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0
dataSize = 128
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)

# model parameters for training
batchSize = 128
epochs = 500#0


filepath = "ESCvae-model-{epoch:02d}-{loss:.2f}.hdf5"
#checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

#tf.compat.v1.disable_eager_execution()



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
def importData(setname):
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
    
 
def mergeSets2(train,  nextTrain):
    combTrain =[]
    for i in range(len(train)):
        if (i<len(train)):
           combTrain.append(train[i])
        if (i<len(nextTrain)):
           combTrain.append(nextTrain[i])
    return combTrain
        
def mergeSets(train, test, nextTrain, nextTest):
    combTrain =[]
    combTest =[]
    for i in range(len(train)):
        if (i<len(train)):
           combTrain.append(train[i])
        if (i<len(nextTrain)):
           combTrain.append(nextTrain[i])
        if (i<len(test)):
           combTest.append(test[i])
        if (i<len(nextTest)):
           combTest.append(nextTest[i])
    return combTrain,combTest

def fitCombined(xTrain,xTest, y_train, y_test,combModel):
    
    #combModel.summary() 
    #xTrain = np.array([x.reshape( (128, 128, 1) ) for x in xTrain])
    #xTest = np.array([x.reshape( (128, 128, 1 ) ) for x in xTest])
    xTrain = np.array([x.reshape( (7,7,252) ) for x in xTrain])
    xTest = np.array([x.reshape( (7,7,252) ) for x in xTest])

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
    print ('xtrain:', xTrain.shape)
    #print ('xtraincoded:', xTrainEncoded.shape)

    indata = [xTrain,xTrain]
  

    combModel.fit(indata,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= ([xTest,xTest], y_test),#,
        #callbacks=[early_stopping_monitor]

        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    )

    score = combModel.evaluate([xTest,xTest],
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    #partsModel.save('lsparts.'+format(latent_dim)+'.hdf5')
    #'''
    print('Model exported and finished')

    


if __name__ == '__main__':
    #tensorboard = TensorBoard(log_dir = "logs/{}".format(time()))
    dataset =  importData('base')#(train, test) =  
    print('total recs =', totalRecordCount, '; Total Labels=', totalLabel, lblmap)
    nextdataset =   importData('next')#(nextTrain,nextTest) = 
    print('total recs =', totalRecordCount, '; Total Labels=', totalLabel, lblmap)
    dataset = mergeSets2(dataset, nextdataset)#train, test, nextTrain, nextTest)

    print('TotalCount: {}'.format(totalRecordCount))
    trainDataEndIndex = int(totalRecordCount*0.8)
    random.shuffle(dataset)

    train = dataset[:trainDataEndIndex]
    test = dataset[trainDataEndIndex:]
    x, y = zip(*dataset)
    x_train = x[:trainDataEndIndex]
    x_test = x[trainDataEndIndex:]   
    ycat = to_categorical(y)
    y_traincat = ycat[:trainDataEndIndex]
    y_testcat = ycat[trainDataEndIndex:]   

    image_size = x_train[0].shape
    '''
    original_dim = image_size[0] * image_size[1]
    
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    '''
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = np.expand_dims(x_train,-1)
    x_test = np.expand_dims(x_test,-1)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    #encfine.built=True
    #enccoarse.built=True
    densebase =keras.models.load_model('model.aug.6.50p.h5')#, custom_objects={'sampling': sampling}, compile =False)
    densebase.summary()
    ###
    weight_decay=1e-4
    denseIn1 = Input(shape=(7,7, 252))
    
    b1_bnorm1a = BatchNormalization()(denseIn1)
    b1_act1a = Activation('relu')(b1_bnorm1a)
    b1_conv1a = Conv2D(48, (1, 1), padding='same',
                                 kernel_regularizer=keras.regularizers.l2(weight_decay))(b1_act1a)
    b1_bnorm1b = BatchNormalization()(b1_conv1a)
    b1_act1b = Activation('relu')(b1_bnorm1b)
    b1_conv1b = Conv2D(12, (3, 3), padding='same')(b1_act1b)

    branch1=Model(inputs=denseIn1,outputs=b1_conv1b)

    srclayer =-12
    for layer in branch1.layers:
        if srclayer>=-11: 
           print(densebase.layers[srclayer].name)
           layer.set_weights(densebase.layers[srclayer].get_weights())
        srclayer+=1
        
        
    b1_bnorm1a.trainable=False
    b1_act1a.trainable=False
    b1_conv1a.trainable=False
    b1_bnorm1b.trainable=False
    b1_act1b.trainable=False
    b1_conv1b.trainable=False
 
    
        
    #b1_gpool = GlobalAveragePooling2D()(b1_conv)
    #expanded_gpool = GlobalAveragePooling2D()(b1_act)
    #for i in range(4):
    #  expanded_gpool = Concatenate(axis=1, name = 'outconcat')([expanded_gpool,b1_gpool])
    #print('exp shape',expanded_gpool.shape)
    '''
    flat12c = Flatten()(denseIn1)
    drop13c = Dropout(rate=0.5)(flat12c)
    dense14c =  Dense(64)(drop13c)
    act15c = Activation('relu', name='actdense')(dense14c)
    branch1 = Model(inputs=denseIn1,outputs=act15c)
    branch1.fixed=True
    
    flat12c.weights = densebase.layers[].weights
    drop13c.weights = densebase.layers[].weights
    dense14c.weights = densebase.layers[].weights
    act15c.weights = densebase.layers[].weights
    
    origBranch = Model(inputs=orig_out, outputs=act15c)
    origBranch.built=True
    '''
    ###
    #densebase.summary()  
    for i in range(10):
        densebase.layers.pop()
        #print(densebase.layers[-1].name, len(densebase.layers))
    #x_train = np.array(x_train)
    #x_test = np.array(x_test)
    
    #densebase.summary()  
    orig_in = densebase.layers[0].get_output_at(0)
    denseBaseNew = DenseBase((128,128,1),  depth=25)
    print('building model')    
    densebaseout = densebase.layers[-1].get_output_at(0)
    denseFixed=Model(inputs=orig_in,outputs=densebaseout)
    
    newPart, new_in,newbranch  = denseBaseNew.build_model()
    print('newin shape is ', new_in.shape)
    print('densebase shape is ', densebaseout.shape)
    #parts_input1 = Input((64,64))
    #parts_input2 = Input((64,64))
    #parts_input1 = Input(model.layers[18].input_shape[1:])
    #parts_input2 = Input(model.layers[19].input_shape[1:])
    #print('partsmodel1 shape is ', parts_input1.shape) 
    #print('partsmodel2 shape is ', parts_input2.shape) 
    
    #combModel = Sequential()
    
    #denseIn = Input(shape=densebaseout.shape)
    #re_DenseIn = Reshape(target_shape=((7,7, 48)),input_shape=densebaseout.shape)(densebaseout)
    
    #denseIn = Reshape((1,7,7,48))(denseIn)
    #denseAdapt=Conv2D(48, (1,1), strides=(1, 1), name='conv_adapt')(re_DenseIn)
    #print(denseIn.shape,densebaseout.shape,re_DenseIn.shape,denseAdapt.shape)
    #denseAdapt=Conv2D(48, (1,1), strides=(1, 1), name='conv_adapt')(re_DenseIn)
    denseIn2 = Input(shape=(7,7, 252))
    
    b2_bnorm1a = BatchNormalization()(denseIn2)
    b2_act1a = Activation('relu')(b2_bnorm1a)   
    b2_conv1a = Conv2D(48, (1, 1), padding='same',
                                 kernel_regularizer=keras.regularizers.l2(weight_decay))(b2_act1a)
  
    b2_bnorm1b = BatchNormalization()(b2_conv1a)
    b2_act1b = Activation('relu')(b2_bnorm1b)
    b2_conv1b = Conv2D(12, (3, 3), padding='same')(b2_act1b)
    ########################################################################
    #''' 
    denseIn3 = Input(shape=(7,7, 252))
    b30_conv = Conv2D(12, (3, 3), padding='same')(denseIn3)
    print('b30_conv', b30_conv.shape)
    b3_flatten = Flatten()(b30_conv)
    print('b3_flatten', b3_flatten.shape)

    b3_drop = Dropout(rate=0.5)(b3_flatten)
    print('b3_drop', b3_drop.shape)
  
    b3_dense = Dense(48)(b3_drop)
    print('b3_dense', b3_dense.shape)

    b3_act = Activation('relu')(b3_dense)
    print('b3_act', b3_act.shape)

    b3_conv = Conv2D(12, (3, 3), padding='same')(b3_act)
    print('b3_conv', b3_conv.shape)

    #''' 
    ######################################################################
    
    '''
    latent_dim = 8
    re0a = Reshape(target_shape=(7*7*252,1),input_shape=(7,7,252))(denseIn2)
    #ft0a = Lambda(tf.signal.rfft)(re0a)
    ft0a = Lambda(lambda v: tf.to_float(tf.spectral.rfft(v)))(re0a)

	
    conv_1a = Conv1D(24, kernel_size=latent_dim, activation='relu')(ft0a)
    # Using CNN to build model
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24
    
    #conv_2a = Conv2D(24, (4,4), strides=(1, 1), input_shape=input_shape_a)(conv_1a)
    # 31 x 62 x 24
  
    pool_2a = MaxPooling1D(pool_size=latent_dim)(conv_1a)
    act_4a =Activation('relu')(pool_2a)

    print('act4 a', act_4a.shape) 
    #re_10a = Reshape(target_shape=(latent_dim*latent_dim, 48),input_shape=(latent_dim,latent_dim ,48))(act_10a)
    ls5a= LSTM(latent_dim*latent_dim,return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(act_4a)
    print('ls5 a shape is ', ls5a.shape) 
    
    #re5a = Reshape(target_shape=(latent_dim*latent_dim ,latent_dim*latent_dim ))(ls5a)
    #merge

    # 27 x 58 x 48
    conv_5a = Conv1D(48, kernel_size=latent_dim,  activation='relu')(ls5a)

    # 6 x 29 x 48
    #pool_6a=MaxPooling1D(pool_size=latent_dim)(conv_5a)
    act_7a = Activation('relu')(conv_5a)
    print('7a',act_7a.shape)
    # 27 x 58 x 48
    #conv_5aa = Conv1D(48, kernel_size=latent_dim,  activation='relu')(act_7a)

    # 6 x 29 x 48
    #pool_6aa=MaxPooling1D(pool_size=latent_dim//2)(act_7a)
    #act_7aa = Activation('relu')(pool_6aa)
    #print('7aa',act_7aa.shape)
    ift7a = Lambda(lambda v: tf.to_float(tf.spectral.irfft(tf.cast(v, dtype=tf.complex64))))(act_7a)

    #ift7a = Lambda(tf.signal.irfft)(act_7a)
    print('ifta',ift7a.shape)
    re_7a = Reshape(target_shape=(307,470,1))(ift7a)
    # 2 x 25 x 48
    conv_8a = Conv2D(12, (latent_dim,latent_dim), padding="valid")(re_7a)
    pool_8a = MaxPooling2D((latent_dim//2,latent_dim//2))(conv_8a)
    print('pool_8a', pool_8a.shape)
    conv_9a = Conv2D(12, (latent_dim, latent_dim), padding="valid")(pool_8a)
    print('conv_9a', conv_9a.shape)
    pool_9a = MaxPooling2D((7,7))(conv_9a)
    print('pool_9a', pool_9a.shape)
    conv_10a = Conv2D(12, (3, latent_dim+1), padding="valid")(pool_9a)
    print('conv_10a', conv_10a.shape)
    '''
    ########################################################################
    '''
    flat12c = Flatten()(denseIn1)
    drop13c = Dropout(rate=0.5)(flat12c)
    dense14c =  Dense(64)(drop13c)
    act15c = Activation('relu', name='actdense')(dense14c)
    branch1 = Model(inputs=denseIn1,outputs=act15c)
    branch1.fixed=True
    
    flat12c.weights = densebase.layers[].weights
    drop13c.weights = densebase.layers[].weights
    dense14c.weights = densebase.layers[].weights
    act15c.weights = densebase.layers[].weights
    
    origBranch = Model(inputs=orig_out, outputs=act15c)
    origBranch.built=True
    '''
    
    concatC = Concatenate(axis=3, name = 'outconcat')([denseIn1,b1_conv1b,b2_conv1b,conv_10a])
    
    #concatC = Concatenate( name = 'outconcat')([act15c,branch2_act15c])
    #drop16c=Dropout(rate=0.5)(concatC)
    #dense17c=Dense(totalLabel)(drop16c)
    #out2 = Activation('softmax', name='actout')(dense17c)
    
    bnorm_out = BatchNormalization()(concatC)
    act_out= Activation('relu')(bnorm_out)
    gpool_out= GlobalAveragePooling2D()(act_out)
    prediction = Dense(totalLabel, activation='softmax')(gpool_out)

    combModel = Model(inputs=[denseIn1,denseIn2], outputs=prediction)



    print ("about to encode fine train",x_train.shape)
    #densebase.summary()
    denseFixed.built=True
    #m = encPredict(densebase,x_train)
    #print ("about to encode coarse test")
    #'''
    X_train_encoded = encPredict(denseFixed,x_train)
    
    print ("about to encode fine test")
    X_test_encoded = encPredict(denseFixed,x_test)# enc32(x_test)
    #print ("about to encode coarse test")
    #X_test_coarse_encoded = encPredict(enccoarse, x_test)
    #y_train = np.array(y_train)
    #y_test = np.array(y_test)
    #y_traincat = to_categorical(y_train)
    #y_testcat = to_categorical(y_test)
    x_train = np.array(x_train)
    #print('ytrain', y_train.shape)
    print('ytraincat', y_traincat.shape)
    print('ytestcat', y_testcat.shape)
    print('xtrain', X_train_encoded.shape)
    #'''
 
    fitCombined(X_train_encoded, X_test_encoded,  y_traincat, y_testcat, combModel)

    
