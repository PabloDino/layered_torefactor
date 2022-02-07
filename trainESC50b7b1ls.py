
import tensorflow as tf
#import keras
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input
#from keras_self_attention import SeqSelfAttention
#from keras.models import Sequential, Model
from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
#from keras import layers
from sklearn.preprocessing import MinMaxScaler
import math


import keras.optimizers
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
tsplit = 0.8
# Total wav records for training the model, will be updated by the program
totalRecordCount = 0
dataSize=128
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 50
input_dim = 128
latent_dim = 8

# model parameters for training
batchSize = 128
epochs = 100
viewBatch=2
latent_dim = 8



def preProcess(array, labels):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    lookback = latent_dim
    array=np.array(array)
    maxi=0
    #for i in range(array.shape[0]):
    #   if (maxi<np.max(array[i]):
    #       maxi= np.max(array[i])
    print("arrshape1:", array.shape)
    #print("labshape:", labels)
    array, labels =  temporalize(array, labels, lookback)
    print("arrshape2:", array.shape)
    array = np.array(array).astype("float32") / np.max(array)
    array = np.reshape(array, (lookback*len(array), dataSize, dataSize,1))
     
    return array, labels


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
        #print('look', i, len(output_X), len(output_y))
        t=[]
        for j in range(1, lookback + 1):
            # Gather the past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
            output_y.append(y[i + lookback + 1])
        output_X.append(t)
        #output_y.append(y[i + lookback + 1])
    #return np.array(output_X), np.array(output_y)
    return np.squeeze(np.array(output_X)), np.array(output_y)






class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon




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
    trainDataEndIndex = int(totalRecordCount*tsplit)
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

# This is the default import function for UrbanSound8K
# https://urbansounddataset.weebly.com/urbansound8k.html
# Please download the URBANSOUND8K and not URBANSOUND
def buildModel(fineTrain,coarseTrain,fineTest,coarseTest, y_train, y_test):


    print('building Total training data:{}'.format(len(fineTrain)))
    print('Total test data:{}'.format(len(fineTest)))

    # Get the data (128, 128) and label from tuple
    #print("preProc is ", preProc[0])

    print("trainCoarse shape is ", coarseTrain.shape)
    print("trainFine shape is ", fineTrain.shape)

    #X_trainFine, y_trainFine = zip(*trainFine)
    #X_testFine, y_testFine = zip(*testFine)
    
    #X_trainCoarse, y_trainCoarse = zip(*trainCoarse)
    #X_testCoarse, y_testCoarse= zip(*testCoarse)

    # Reshape for CNN input
    #X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    #X_test = np.array([x.reshape( (128, 128, 1) ) for x in X_test])
    
    X_fineTrain = np.array([x.reshape( (64, 32, 1) ) for x in fineTrain])
    X_fineTest = np.array([x.reshape( (64, 32, 1 ) ) for x in fineTest])
  

    X_coarseTrain = np.array([x.reshape( (64, 32, 1) ) for x in coarseTrain])
    X_coarseTest = np.array([x.reshape( (64, 32, 1 ) ) for x in coarseTest])

  
    

    # One-Hot encoding for classes
    #y_train = np.array(keras.utils.to_categorical(y_train, totalLabel))
    #y_test = np.array(keras.utils.to_categorical(y_test, totalLabel))

    #fineEncoder.begin()
    model_a = Sequential()

    # Model Input

    l_input_shape_a=(64, 32, 1,)
    input_shape_a=(64, 32, 1)
    model_a_in = Input(shape=input_shape_a)
    re_1a = Reshape(target_shape=(64,32),input_shape=(64,32,1))(model_a_in)
 
    l1a = layers.LSTM(input_dim//latent_dim, return_sequences=True)(re_1a)    
    b1a = layers.Bidirectional(layers.LSTM(input_dim//2, return_sequences=True))(l1a)
    re_2a = Reshape(target_shape=(64,128,1),input_shape=(64,128))(b1a)
    '''
    conv_2a = Conv2D(1, (2,2), strides=(1, 1), input_shape=input_shape_a)(re_2a)
    # Using CNN to build model
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24
    print('conv_2a', conv_2a.shape)
    
    pool_3a = MaxPooling2D((3,3), strides=(2, 4))(conv_2a)
    act_4a =Activation('relu')(pool_3a)

    conv_2a = Conv2D(1, (3,3), strides=(1, 1), input_shape=input_shape_a)(act_4a)
    #31 x 62 x 24#    
    pool_4a = MaxPooling2D((3,3), strides=(2,2))(conv_2a)
    act_5a =Activation('relu')(pool_4a)
    # 27 x 58 x 48
    conv_3a = Conv2D(1, (3,3), padding="valid")(act_5a)
    '''
    #re_10a = Reshape(target_shape=(4,4),input_shape=(2,24,48))(act_9a)
    #re_10a = Reshape(target_shape=(12, 12),input_shape=(12, 12,1))(conv_3a)
    
    #fineEncoder.end()
    #ls11a= LSTM(32, return_sequences=True,unit_forget_bias=1.0,dropout=0.1)(re_10a)
    #ls11a   = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(re_10a)
    #seqa=SeqSelfAttention(attention_activation='sigmoid')(ls11a)
    '''
    #merge

    #coarseEncoder.begin()

    model_b = Sequential()

    # Model Input


    l_input_shape_b=(128,128,1,1)
    input_shape_b=(128,128,1)
    model_b_in = Input(shape=input_shape_b)
    
    conv_1b = Conv2D(1, (2,2), strides=(1, 1), input_shape=input_shape_b)(model_b_in)
    # Using CNN to build model
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24
    print('conv_1a', conv_1b.shape)
    
    pool_3b = MaxPooling2D((3,3), strides=(2,2))(conv_1b)
    act_4b =Activation('relu')(pool_3b)

    conv_2b = Conv2D(1, (3,3), strides=(2,2))(act_4b)
    #31 x 62 x 24#    
    pool_4b = MaxPooling2D((2,2), strides=(1,1))(conv_2b)
    act_5b =Activation('relu')(pool_4b)
    conv_5b = Conv2D(1, (3,3), strides=(1, 1))(act_5b)

    pool_6b = MaxPooling2D((3,3), strides=(2,2))(conv_5b)
    conv_6b =Activation('relu')(pool_6b)
    
    re_10b = Reshape(target_shape=(4, 4),input_shape=(4, 4 ,1))(conv_6b)
    print('re_10b.shape is', re_10b.shape)    
    print('re_10.shape is', re_10a.shape)  
    #fineEncoder.end()
    ls11b= LSTM(32, return_sequences=True,unit_forget_bias=1.0,dropout=0.1)(re_10a)
    #ls11b   = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(re_10b)
    #print('ls11a.shape is', ls11a.shape)    
    print('ls11b.shape is', ls11b.shape)  
    '''
    #merged = Concatenate(axis=1)([re_4b,re_4b])
    #merged = Concatenate(axis=1)([conv_1a,conv_1b])
    #merged = Concatenate(axis=1)([ls_5b,ls_5b])
    #merged = Concatenate(axis=1)([ls11a,ls11b])
    #attnLyr= Attention()([ls11a, ls_5b])
    #print('ls11a.shape is', ls11a.shape)    
    #print('ls_5b.shape is', ls_5b.shape)
    #print('merged.shape is', merged.shape)
    #flat12 = Flatten()(merged)
    #flat12 = Flatten()(conv_3a)
    flat12 = Flatten()(re_2a)
    #d12 = Dense(144, activation ="relu")(conv_3a)
    #print('flat.shape is', flat12.shape)
    
    dense13=Dense(50)(flat12)
    #dense14=Dense(1)(dense13)
    out = Activation('softmax')(dense13)
    print('out is ', out.shape)
    #model = Model(inputs=[model_a_in, model_b_in], outputs=out)
    model = Model(inputs= model_a_in, outputs=out)

    model.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=['accuracy'])
    
    initial_learning_rate = 0.005
    #epochs = 100
    decay = initial_learning_rate / epochs
    drop = 0.5
    epochs_drop = 10.0
    def lr_time_based_decay(epoch, lr):
      if True:#epoch < 5:
          #return decay *epochs
          lrate = initial_learning_rate * math.pow(drop,  
             math.floor((1+epoch)/epochs_drop))
          return lrate

      else:
        #return decay*lr#epochs
        #return lr * epoch / (epoch + decay * epoch)
        #return initial_learning_rate / (1 + decay * epoch)
        lrate = initial_learning_rate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
        return lrate




    #opt = keras.optimizers.Adam(learning_rate=0.01)
    #model.compile(optimizer=opt,loss="categorical_crossentropy", metrics=['accuracy'])
    print(model.summary())
    indata = [X_fineTrain,X_coarseTrain]
    #print ('xtrainFine shape is ',X_trainFine.shape)
    #print ('xtrainCoarse shape is ',Xb_trainCoarse.shape)
    #print ('indata[0] shape is ',indata[0].shape, '1', indata[1].shape,)
    #print ('ytrain shape is ',y_train.shape)
    

    early_stopping_monitor = EarlyStopping(
       monitor='val_loss',
       min_delta=0,
       patience=50,
       verbose=0,
       mode='auto',
       baseline=None,
       restore_best_weights=True
)
    print("indata0 shape is ", indata[0].shape)
    print("indata1 shape is ", indata[1].shape)
    print("ytrain shape is ", y_train.shape)
    
    print("X_fineTest shape is ", X_fineTest.shape)
    print("X_coarseTest shape is ", X_coarseTest.shape)
    print("y_test shape is ", y_test.shape)
    
    
    
    #print("preProcTrain shape is ", preProcTrain[0][0].shape)
    #print("preProcTest shape is ", preProcTest[0].shape)
    #model.fit(indata,
    model.fit(X_fineTrain,
        y=y_train,
        epochs=epochs,
        #batch_size=1,#batchSize,
        #validation_data= ([X_fineTest,X_coarseTest], y_test)#,
        validation_data= (X_fineTest, y_test),#,
        verbose = 1,
        #callbacks=[early_stopping_monitor]
        #)#,
        callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)]
    )

    score = model.evaluate(X_fineTest,
        y=y_test)
   #score = model.evaluate([X_fineTest,X_coarseTest],
   #     y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'esc50-sound-classification-{}.h5'.format(timestr)
    model.save('models/{}'.format(modelName))

    print('Model exported and finished')

if __name__ == '__main__':
    #dataSet = importData()
    
    fineEncoder = load_model('encoder8.hdf5')
    #)#   , custom_objects={'Sampling': Sampling}, compile=False)
    coarseEncoder = load_model('encoder8.hdf5')#, custom_objects={'Sampling': Sampling}, compile =False)
    fineEncoder.built=True
    coarseEncoder.built=True
    #coarseEncoder.load_weights('coarse-model-15.hdf5')
    (x_train, y_train), (x_test, y_test) = importData()#keras.datasets.mnist.load_data()
    #preProc = np.concatenate([x_train, x_test], axis=0)
    #preProcTrain=x_train[0]
    #preProcTest=x_test[0]
    #preProcTrainx = np.array(x_train).astype("float32") / np.max(x_train)
    #preProcTestx = np.array(x_test).astype("float32") / np.max(x_test)
    preProcTrainx = np.expand_dims(x_train, -1).astype("float32") /  np.max(x_train)
    preProcTestx = np.expand_dims(x_test, -1).astype("float32") /  np.max(x_train)
    #preProcTrainx, preProcTrainy = preProcess(x_train, y_train)
    #preProcTestx, preProcTesty = preProcess(x_test, y_test)
    #preProcTrain = np.reshape(preProcTrain,(preProcTrain.shape[0],preProcTrain.shape[1],preProcTrain.shape[2],preProcTrain.shape[3]))
    #preProcTest = np.reshape(preProcTest,(preProcTest.shape[0],preProcTest.shape[1],preProcTest.shape[2],preProcTest.shape[3]))
    
    X_train_fine_encoded = fineEncoder.predict(preProcTrainx)
    X_train_coarse_encoded = coarseEncoder.predict(preProcTrainx)
    
    X_test_fine_encoded = fineEncoder.predict(preProcTestx)
    X_test_coarse_encoded = coarseEncoder.predict(preProcTestx)
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
    
    X_train_fine_encoded = np.expand_dims(X_train_fine_encoded, -1).astype("float32") / 255
    X_train_coarse_encoded = np.expand_dims(X_train_coarse_encoded, -1).astype("float32") / 255
    X_test_fine_encoded = np.expand_dims(X_test_fine_encoded, -1).astype("float32") / 255
    X_test_coarse_encoded = np.expand_dims(X_test_coarse_encoded, -1).astype("float32") / 255
    
    preProcTrainy = np.array(y_train)
    preProcTesty = np.array(y_test)
    y_traincat = to_categorical(preProcTrainy)
    y_testcat = to_categorical(preProcTesty)
    #x_train = np.array(x_train)
    print('ytrain', preProcTrainy.shape)
    print('ytraincat', y_traincat.shape)
    print('ytestcat', y_testcat.shape)
    print('xtrain', X_train_fine_encoded.shape)
    #X_train_fine_encoded = np.reshape(X_train_fine_encoded,(X_train_fine_encoded.shape[1],X_train_fine_encoded.shape[2],X_train_fine_encoded.shape[0],X_train_fine_encoded.shape[3]))
    #X_test_fine_encoded = np.reshape(X_test_fine_encoded,(X_test_fine_encoded.shape[1],X_test_fine_encoded.shape[2],X_test_fine_encoded.shape[0],X_test_fine_encoded.shape[3]))
 
    buildModel(X_train_fine_encoded,X_train_coarse_encoded, X_test_fine_encoded,X_test_coarse_encoded, y_traincat,y_testcat)
