'''
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
from tensorflow.keras import layers


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input
from keras.layers import Lambda, Input, Dense
from keras import backend as K

from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
'''
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
from time import time

from config import *

#import keras.optimizers
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random
import time
import warnings
import os
import time
warnings.filterwarnings('ignore')


# Your data source for wav files
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'

dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0

# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 50

# model parameters for training
batchSize = 128
epochs = 100
viewBatch=2
dataSize = 128
#latent_dim = 256
input_shape_b=(dataSize,dataSize,1)
conv_param   = (8, 8, 128)



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
    print('starting encoder model -inputs shape is ', inputs.shape, ';inter = ', intermediate_dim)
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

def importData0():
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

    
    #print(X_train)
    return (X_train, y_train), (X_test, y_test)#dataSet


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


def buildModel(fineTrain,coarseTrain,fineTest,coarseTest, y_train, y_test):

    fineTrain = np.array(fineTrain)
    coarseTrain = np.array(coarseTrain)
    fineTest  = np.array(fineTest)
    coarseTest  = np.array(coarseTest)
    #fineEncoder.begin()
    model_a = Sequential()
    
    # Model Input

    l_input_shape_a=(128, 128,1,1)
    input_shape_a=(128, 128,1)
    model_a_in = Input(shape=input_shape_a)
    
    conv_1a = Conv2D(24, (5,5), strides=(1, 1), input_shape=input_shape_a)(model_a_in)
    # Using CNN to build model
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24
    
    conv_2a = Conv2D(24, (5,5), strides=(1, 1), input_shape=input_shape_a)(conv_1a)
    # 31 x 62 x 24
  
    pool_3a = MaxPooling2D((4, 2), strides=(4, 2))(conv_2a)
    act_4a =Activation('relu')(pool_3a)

    # 27 x 58 x 48
    conv_5a = Conv2D(48, (5, 5), padding="valid")(act_4a)

    # 6 x 29 x 48
    pool_6a=MaxPooling2D((4, 2), strides=(4, 2))(conv_5a)
    act_7a = Activation('relu')(pool_6a)
  
    # 2 x 25 x 48
    conv_8a = Conv2D(48, (5, 5), padding="valid")(act_7a)

    act_9a = Activation('relu')(conv_8a)
    re_10a = Reshape(target_shape=(48,48),input_shape=(2,24,48))(act_9a)
    ls11a= LSTM(64,return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(re_10a)
    #merge

  
    model_b = Sequential()

    # Model Input

    l_input_shape_b=(128, 128,1,1)
    input_shape_b=(128, 128,1)

    model_b_in = Input(shape=input_shape_b)
    print(model_b_in.shape)

    conv_1b = Conv2D(24, (11,11), strides=(1, 1), input_shape=input_shape_a)(model_b_in)
    print(conv_1b.shape)
    # Using CNN to build model
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24   
    # 98x98x24    

    pool_2b = MaxPooling2D((8,8), strides=(8,8))(conv_1b)
    print(pool_2b.shape)
    conv_3b = Conv2D(48, (5,5), strides=(1, 1), input_shape=input_shape_a)(pool_2b)
    print(conv_3b.shape)
  
    act_3b =Activation('relu')(conv_3b)
    print(act_3b.shape)


    re_4b = Reshape(target_shape=(100,48),input_shape=(10,10,48))(act_3b)
    ls_5b= LSTM(64,return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(re_4b)
    #merge
 
 
    merged = Concatenate(axis=1)([ls11a,ls_5b])

    flat12 = Flatten()(merged)
    drop13 = Dropout(rate=0.5)(flat12)
    dense14 =  Dense(64)(drop13)
    act15 = Activation('relu')(dense14)
    drop16=Dropout(rate=0.5)(act15)
    dense17=Dense(totalLabel)(drop16)
    out = Activation('softmax')(dense17)
    model = Model(inputs=[model_a_in, model_b_in], outputs=out)
    model.built=True
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
    print(model.summary())
    indata = [fineTrain,coarseTrain]
    print ('xtrain shape is ',fineTrain.shape)
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


def encPredict(enc, x_train):
   viewBatch=1
   numrows = x_train.shape[0]
   for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample = x_train[i*viewBatch:i*viewBatch+viewBatch,]
      z_mean8, _, _ = enc.predict([[sample, sample]])
      #z_mean8 = enc.predict(sample)
      if (i==0):
        z_mean=z_mean8
      else:
        z_mean = np.concatenate((z_mean,z_mean8), axis=0)
      if (i%200==0):  
        print("enc stat",z_mean.shape)
   return z_mean


def decPredict(dec, x_train):
   viewBatch=1
   numrows = x_train.shape[0]
   for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample = x_train[i*viewBatch:i*viewBatch+viewBatch,]
      #z_mean8 = dec.predict(sample)
      z_mean8, _, _ = dec.predict([[sample, sample]])
      if (i==0):
        z_mean=z_mean8
      else:
        z_mean = np.concatenate((z_mean,z_mean8), axis=0)
      if (i%200==0):  
        print("dec stat",z_mean.shape)
   return z_mean



if __name__ == '__main__':
    #image_size = x_train[0].shape
    #original_dim = 128*128
    input_shape = (original_dim, )
    inputsfine = Input(shape=input_shape, name='encoder_input')
    encoderfine, z_mean, z_log_var = encoder_model(inputsfine)
    decoderfine = decoder_model()
    # instantiate VAE model
    outputsfine = decoderfine(encoderfine(inputsfine)[2])
    vaefine = Model(inputsfine, outputsfine, name='vaefine')
    vaefine.built=True
    vaefine.load_weights('vae_mlp_mnist_latent_dim_256.h5')
    
    inputscoarse = Input(shape=input_shape, name='encoder_input')
    encodercoarse, z_mean, z_log_var = encoder_model(inputscoarse)
    decodercoarse = decoder_model()
    # instantiate VAE model
    outputs = decodercoarse(encodercoarse(inputscoarse)[2])
    vaecoarse = Model(inputscoarse, outputscoarse, name='vaecoarse')

    vaecoarse.load_weights('vae_mlp_mnist_latent_dim_16,h5')
    

    x_train = tf.expand_dims(x_train,-1)
    x_test = tf.expand_dims(x_test,-1)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
      
    print ("about to encode fine train")
    X_train_fine_encoded = encPredict(encoderfine,x_train)
    print ("about to encode coarse test")
    X_train_coarse_encoded = encPredict(encodercoarse,x_train)
    
    print ("about to encode fine test")
    X_test_fine_encoded = encPredict(encoderfine,x_test)# enc32(x_test)
    print ("about to encode coarse test")
    X_test_coarse_encoded = encPredict(encodercoarse, x_test)
    
    
    print ("about to decode fine train")
    X_train_fine_decoded = decPredict(decoderfine,X_train_fine_encoded)#.astype("float32") / 255
    print ("about to decode coarse train")
    X_train_coarse_decoded = decPredict(decodercoarse, X_train_coarse_encoded)#.astype("float32") / 255
    
    print ("about to decode coarse train")
    X_test_fine_decoded = decPredict(decoderfine, X_test_fine_encoded)#.astype("float32") / 255
    print ("about to decode coarse test")
    X_test_coarse_decoded = decPredict(decodercoarse, X_test_coarse_encoded)#.astype("float32") / 255    
    '''
    #print('ytrain', y_train.shape)
    print('X_train_fine_encoded', X_train_fine_encoded.shape)
    print('X_test_fine_encoded', X_test_fine_encoded.shape)
    print('X_train_coarse_encoded', X_train_coarse_encoded.shape)
    print('X_test_coarse_encoded', X_test_coarse_encoded.shape)
    
    X_train_fine_encoded = np.expand_dims(X_train_fine_encoded, -1).astype("float32") / 255
    X_train_coarse_encoded = np.expand_dims(X_train_coarse_encoded, -1).astype("float32") / 255
    X_test_fine_encoded = np.expand_dims(X_test_fine_encoded, -1).astype("float32") / 255
    X_test_coarse_encoded = np.expand_dims(X_test_coarse_encoded, -1).astype("float32") / 255
    print('X_train_fine_encoded', X_train_fine_encoded.shape)
    print('X_test_fine_encoded', X_test_fine_encoded.shape)
    print('X_train_coarse_encoded', X_train_coarse_encoded.shape)
    print('X_test_coarse_encoded', X_test_coarse_encoded.shape)
    '''
    X_train_fine_decoded = np.expand_dims(X_train_fine_decoded, -1)
    X_train_coarse_decoded = np.expand_dims(X_train_coarse_decoded, -1)
    X_test_fine_decoded = np.expand_dims(X_test_fine_decoded, -1)
    X_test_coarse_decoded = np.expand_dims(X_test_coarse_decoded, -1)
     
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_traincat = to_categorical(y_train)
    y_testcat = to_categorical(y_test)
    x_train = np.array(x_train)
    print('ytrain', y_train)#.shape)
    print('ytraincat', y_traincat)#.shape)
    print('ytestcat', y_testcat.shape)
    print('xtrain', X_train_fine_decoded.shape)
    #X_train_fine_encoded = np.reshape(X_train_fine_encoded,(X_train_fine_encoded.shape[1],X_train_fine_encoded.shape[2],X_train_fine_encoded.shape[0],X_train_fine_encoded.shape[3]))
    #X_test_fine_encoded = np.reshape(X_test_fine_encoded,(X_test_fine_encoded.shape[1],X_test_fine_encoded.shape[2],X_test_fine_encoded.shape[0],X_test_fine_encoded.shape[3]))
 
    buildModel(X_train_fine_decoded,X_train_coarse_decoded, X_test_fine_decoded,X_test_coarse_decoded, y_traincat, y_testcat)
