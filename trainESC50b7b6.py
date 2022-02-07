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
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0
dataSize = 128
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 50

# model parameters for training
batchSize = 128
epochs = 100#0


filepath = "ESCvae-model-{epoch:02d}-{loss:.2f}.hdf5"
#checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

#tf.compat.v1.disable_eager_execution()



def encPredict(enc, x_train):
   viewBatch=1
   numrows = x_train.shape[0]
   for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample = x_train[i*viewBatch:i*viewBatch+viewBatch,]
      #z_mean8, _, _ = enc.predict([[sample, sample]])
      z_mean8 = enc.predict(sample)
      #print('Sample ', i, ' shape ', sample.shape , ' converted to ', z_mean8.shape)
      if (i==0):
        z_mean=z_mean8[0]
      else:
        z_mean = np.concatenate((z_mean,z_mean8[0]), axis=0)
      if (i%200==0) and i>1:  
        print("enc stat",z_mean.shape)
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
    
    


def buildModel(fineTrain,coarseTrain,fineTest,coarseTest, y_train, y_test):

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



if __name__ == '__main__':
    tensorboard = TensorBoard(log_dir = "logs/{}".format(time()))

    (x_train, y_train), (x_test, y_test) =    importData()
    image_size = x_train[0].shape
    original_dim = image_size[0] * image_size[1]
    
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    early_stopping_monitor = EarlyStopping(
       monitor='val_loss',
       min_delta=0,
       patience=10,
       verbose=0,
       mode='auto',
       baseline=None,
       restore_best_weights=True)
    input_shape = (original_dim, )

    inputsfine = Input(shape=input_shape, name='encoder_input')
    encoderfine, z_meanfine, z_log_varfine = encoder_model(inputsfine)
    decoderfine = decoder_model()
    # instantiate VAE model
    outputsfine = decoderfine(encoderfine(inputsfine)[2])
    vaefine = Model(inputsfine, outputsfine, name='vaefine')
   
    
    inputscoarse = Input(shape=input_shape, name='encoder_input')
    encodercoarse, z_meancoarse, z_log_varcoarse = encoder_model(inputscoarse)
    decodercoarse = decoder_model()
    # instantiate VAE model
    outputscoarse = decodercoarse(encodercoarse(inputscoarse)[2])
    vaecoarse = Model(inputscoarse, outputscoarse, name='vaecoarse')


    reconstruction_lossfine = mse(inputsfine, outputsfine)
    # reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_lossfine *= original_dim
    kl_lossfine = 1 + z_log_varfine - K.square(z_meanfine) - K.exp(z_log_varfine)
    kl_lossfine = K.sum(kl_lossfine, axis=-1)
    kl_lossfine *= -0.5
    vae_lossfine = K.mean(reconstruction_lossfine + kl_lossfine)
    
    reconstruction_losscoarse = mse(inputscoarse, outputscoarse)
    # reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_losscoarse *= original_dim
    kl_losscoarse = 1 + z_log_varcoarse - K.square(z_meancoarse) - K.exp(z_log_varcoarse)
    kl_losscoarse = K.sum(kl_losscoarse, axis=-1)
    kl_losscoarse *= -0.5
    vae_losscoarse = K.mean(reconstruction_losscoarse + kl_losscoarse)
    
    #vaefine.add_loss(vae_lossfine)
    vaecoarse.add_loss(vae_losscoarse)
    
    #vaefine.built=True
    #vaecoarse.built=True

    #vaefine.compile(optimizer='adam')
    #vaecoarse.compile(optimizer='adam')
    print(vaefine.summary())
    vaefine.built=True
    vaecoarse.built=True
    vaefine =tf.keras.models.load_model('ESCvae-finemodel-60-607.50.hdf5', custom_objects={'sampling': sampling}, compile =False)
    vaecoarse =tf.keras.models.load_model('ESCvae-model-62-504.26.hdf5', custom_objects={'sampling': sampling}, compile =False)
    #vaecoarse.load_weights('ESCvae-finemodel-60-607.50.hdf5')
    

    #x_train = tf.expand_dims(x_train,-1)
    #x_test = tf.expand_dims(x_test,-1)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
      
    print ("about to encode fine train",x_train.shape)
    X_train_fine_encoded = encPredict(encoderfine,x_train)
    print ("about to encode coarse test")
    X_train_coarse_encoded = encPredict(encodercoarse,x_train)
    
    print ("about to encode fine test")
    X_test_fine_encoded = encPredict(encoderfine,x_test)# enc32(x_test)
    print ("about to encode coarse test")
    X_test_coarse_encoded = encPredict(encodercoarse, x_test)
    
    
    print ("about to decode fine train", X_train_fine_encoded.shape)
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
    print('ytrain', y_train.shape)
    print('ytraincat', y_traincat.shape)
    print('ytestcat', y_testcat.shape)
    print('xtrain', X_train_fine_decoded.shape)
    #X_train_fine_encoded = np.reshape(X_train_fine_encoded,(X_train_fine_encoded.shape[1],X_train_fine_encoded.shape[2],X_train_fine_encoded.shape[0],X_train_fine_encoded.shape[3]))
    #X_test_fine_encoded = np.reshape(X_test_fine_encoded,(X_test_fine_encoded.shape[1],X_test_fine_encoded.shape[2],X_test_fine_encoded.shape[0],X_test_fine_encoded.shape[3]))
 
    buildModel(X_train_fine_decoded,X_train_coarse_decoded, X_test_fine_decoded,X_test_coarse_decoded, y_traincat, y_testcat)
    
    '''
    reconstruction_loss = mse(inputs, outputs)
    # reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    print(vae.summary())
    vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None),   callbacks=[checkpoint])
    vae.save_weights('vae_mlp_mnist_latent_dim_%s.h5' %latent_dim)
    '''
    
    
