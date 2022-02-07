from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input

from time import time
import numpy as np
import matplotlib.pyplot as plt
import os
from config import *

import random
import keras.optimizers
import librosa
import librosa.display
import pandas as pd
import warnings
import tensorflow as tf


# Your data source for wav files
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0
dataSize = 128
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 50

# model parameters for training
batchSize = 128
epochs = 1000#0


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
    trainDataEndIndex = int(totalRecordCount*0.5)
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
    
    


def buildModel(fineTrain,coarseTrain,fineTest,coarseTest, y_train, y_test):
    '''
    fineTrain = np.array(fineTrain)
    coarseTrain = np.array(coarseTrain)
    fineTest  = np.array(fineTest)
    coarseTest  = np.array(coarseTest)
    fineTrain = np.expand_dims(fineTrain,-1)
    coarseTrain = np.expand_dims(coarseTrain,-1)
    fineTest  = np.expand_dims(fineTest,-1)
    coarseTest  = np.expand_dims(coarseTest,-1)

    fineTrain = np.reshape(fineTrain, (fineTrain.shape[0], dataSize,dataSize,1))
    coarseTrain = np.reshape(coarseTrain, (coarseTrain.shape[0], dataSize,dataSize,1))
    fineTest  = np.reshape(fineTest, (fineTest.shape[0], dataSize,dataSize,1))
    coarseTest  = np.reshape(coarseTest, (coarseTest.shape[0], dataSize,dataSize,1))
    '''
    
    #'''
    fineTrain = np.array([x.reshape( (64, 64) ) for x in fineTrain])
    fineTest = np.array([x.reshape( (64, 64) ) for x in fineTest])
    
    coarseTrain = np.array([x.reshape( (64, 64) ) for x in coarseTrain])
    coarseTest = np.array([x.reshape( (64, 64) ) for x in coarseTest])
    #'''


    #fineEncoder.begin()
    model_a = Sequential()
    
    # Model Input

    

 
    ###################
    parts_input1 = Input((64,64))
    parts_input2 = Input((64,64))
    #parts_input1 = Input(model.layers[18].input_shape[1:])
    #parts_input2 = Input(model.layers[19].input_shape[1:])
    print('partsmodel1 shape is ', parts_input1.shape) 
    print('partsmodel2 shape is ', parts_input2.shape) 
    partsModel = Sequential()
    concatC = Concatenate(axis=1)([parts_input1,parts_input2])
    flat12c = Flatten()(concatC)
    drop13c = Dropout(rate=0.5)(flat12c)
    dense14c =  Dense(64)(drop13c)
    act15c = Activation('relu')(dense14c)
    drop16c=Dropout(rate=0.5)(act15c)
    dense17c=Dense(totalLabel)(drop16c)
    out2 = Activation('softmax')(dense17c)
    partsModel = Model(inputs=[parts_input1,parts_input2], outputs=out2)
    
    #partsModel.summary() 


    partsModel.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=['accuracy'])
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
       

    indata = [fineTrain, coarseTrain]
  

    partsModel.fit(indata,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= ([fineTest, coarseTest], y_test),#,
        #callbacks=[early_stopping_monitor]

        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    )

    score = partsModel.evaluate([fineTest, coarseTest],
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    #partsModel.save('lsparts.'+format(latent_dim)+'.hdf5')
    #'''
    print('Model exported and finished')

    
    ###################
    '''
    merged = Concatenate(axis=1)([ls11a,ls_5b])

    flat12 = Flatten()(merged)
    drop13 = Dropout(rate=0.5)(flat12)
    dense14 =  Dense(64)(drop13)
    act15 = Activation('relu')(dense14)
    drop16=Dropout(rate=0.5)(act15)
    dense17=Dense(totalLabel)(drop16)
    out = Activation('softmax')(dense17)
    model = Model(inputs=[model_a_in, model_b_in], outputs=out)
    #model.built=True
    #model.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=['accuracy'])
    initial_learning_rate = 0.01
    #epochs = 100
    decay = initial_learning_rate / epochs
    def lr_time_based_decay(epoch, lr):
       if epoch < 50:
            return decay *epochs
       else:
            return lr * epoch / (epoch + decay * epoch)


    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt,loss="categorical_crossentropy", metrics=['accuracy'])
    #print(model.summary())
    indata = [fineTrain,coarseTrain]
    print ('xtrain shape is ',fineTrain.shape)
    print ('xtest shape is ',fineTest.shape)
    print ('xbtrain shape is ',coarseTrain.shape)
    print ('indata[0] shape is ',indata[0].shape, '1', indata[1].shape,)
    print ('ytrain shape is ',y_train.shape)
    

    early_stopping_monitor = EarlyStopping(
       monitor='val_loss',
       min_delta=0,
       patience=50,
       verbose=0,
       mode='auto',
       baseline=None,
       restore_best_weights=True
)
    model.fit(indata,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= ([fineTest,coarseTest], y_test)#,
        #callbacks=[early_stopping_monitor]
    )#,
    #   callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    #)

    score = model.evaluate([fineTest,coarseTest],
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'esc50-sound-classification-{}.h5'.format(timestr)
    model.save('models/{}'.format(modelName))

    print('Model exported and finished')
    #***********************************************
    '''

if __name__ == '__main__':
    tensorboard = TensorBoard(log_dir = "logs/{}".format(time()))

    (x_train, y_train), (x_test, y_test) =    importData()
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
    encfine =tf.keras.models.load_model('lsfine.8.hdf5')#, custom_objects={'sampling': sampling}, compile =False)
    enccoarse =tf.keras.models.load_model('lscoarse.8.hdf5')#, custom_objects={'sampling': sampling}, compile =False)
    #vaecoarse.load_weights('ESCvae-finemodel-60-607.50.hdf5')
    

    #x_train = tf.expand_dims(x_train,-1)
    #x_test = tf.expand_dims(x_test,-1)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
      
    print ("about to encode fine train",x_train.shape)
    X_train_fine_encoded = encPredict(encfine,x_train)
    print ("about to encode coarse test")
    X_train_coarse_encoded = encPredict(enccoarse,x_train)
    
    print ("about to encode fine test")
    X_test_fine_encoded = encPredict(encfine,x_test)# enc32(x_test)
    print ("about to encode coarse test")
    X_test_coarse_encoded = encPredict(enccoarse, x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_traincat = to_categorical(y_train)
    y_testcat = to_categorical(y_test)
    x_train = np.array(x_train)
    print('ytrain', y_train.shape)
    print('ytraincat', y_traincat.shape)
    print('ytestcat', y_testcat.shape)
    print('xtrain', X_train_fine_encoded.shape)

 
    buildModel(X_train_fine_encoded,X_train_coarse_encoded, X_test_fine_encoded,X_test_coarse_encoded, y_traincat, y_testcat)

    
