import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, Conv1D, Lambda, \
                         Flatten, MaxPooling2D,MaxPooling1D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input
from tensorflow.keras.models import Sequential, Model
#from attention import Attention
from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
from tensorflow.signal import fft
#from keras.callbacks import TensorBoard

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
import math
warnings.filterwarnings('ignore')

# Your data source for wav files
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'

#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
dataSourceBase = '/home/paul/Downloads/ESC-50-tst2b/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0

# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 50

# model parameters for training
batchSize = 128
epochs = 200
latent_dim=8
dataSize=128

timesteps = 128 # Length of your sequences
input_dim = 128 


def preprocess(array, labels):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    lookback = 1#latent_dim
    array=np.array(array)
    maxi=0
    #for i in range(array.shape[0]):
    #   if (maxi<np.max(array[i]):
    #       maxi= np.max(array[i])
    print("arrshape1:", array.shape)
    #print("labshape:", labels)
    #array, labels =  temporalize(array, labels, lookback)
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
        print('look', i, len(output_X), len(output_y))
        t=[]
        for j in range(1, lookback + 1):
            # Gather the past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
            output_y.append(y[i + lookback + 1])
        output_X.append(t)
    #return np.array(output_X), np.array(output_y)
    return np.squeeze(np.array(output_X)), np.array(output_y)





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

# This is the default import function for UrbanSound8K
# https://urbansounddataset.weebly.com/urbansound8k.html
# Please download the URBANSOUND8K and not URBANSOUND
#def buildModel(dataset):
def buildModel(X_train, X_test, y_train, y_test):
 
    ''' 
    # Get the data (128, 128) and label from tuple
    print("train 0 shape is ",train[0][0].shape)
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    # Reshape for CNN input
    #X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    #X_test = np.array([x.reshape( (128, 128, 1) ) for x in X_test])
    
    X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    X_test = np.array([x.reshape( (128, 128, 1 ) ) for x in X_test])
    '''
    Xb_train = X_train.copy()#np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    Xb_test = X_test.copy()#np.array([x.reshape( (128, 128, 1 ) ) for x in X_test])

    

    # One-Hot encoding for classes
    y_train = np.array(keras.utils.to_categorical(y_train, totalLabel))
    y_test = np.array(keras.utils.to_categorical(y_test, totalLabel))

    model_a = Sequential()

    # Model Input

    l_input_shape_a=(128, 128,1,1)
    input_shape_a=(128, 128,1)
    model_a_in = Input(shape=input_shape_a)
    re0a = Reshape(target_shape=(128*128,1),input_shape=(128,128,1))(model_a_in)
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
    print('ls11 a shape is ', ls5a.shape) 
    
    #re5a = Reshape(target_shape=(latent_dim*latent_dim ,latent_dim*latent_dim ))(ls5a)
    #merge


    # 27 x 58 x 48
    conv_5a = Conv1D(48, kernel_size=latent_dim,  activation='relu')(ls5a)

    # 6 x 29 x 48
    pool_6a=MaxPooling1D(pool_size=latent_dim)(conv_5a)
    act_7a = Activation('relu')(pool_6a)
    print(act_7a.shape)
    ift7a = Lambda(lambda v: tf.to_float(tf.spectral.irfft(tf.cast(v, dtype=tf.complex64))))(act_7a)

    #ift7a = Lambda(tf.signal.irfft)(act_7a)
    print(ift7a.shape)
    re_7a = Reshape(target_shape=(510,94,1))(ift7a)
    # 2 x 25 x 48
    conv_8a = Conv2D(48, (latent_dim//2,latent_dim//2), padding="valid")(re_7a)

    act_9a = Activation('relu')(conv_8a)    # 2 x 25 x 48
    conv_9a = Conv2D(48, (latent_dim//2,latent_dim//2), padding="valid")(act_9a)

    act_10a = Activation('relu')(conv_9a)
 
  
    model_b = Sequential()

    # Model Input

    l_input_shape_b=(128, 128,1,1)
    input_shape_b=(128, 128,1)

    model_b_in = Input(shape=input_shape_b)
    print(model_b_in.shape)

    conv_1b = Conv2D(24, (latent_dim,latent_dim), strides=(1, 1), input_shape=input_shape_a)(model_b_in)
    print(conv_1b.shape)
    # Using CNN to build model
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24   
    # 98x98x24    

    pool_2b = MaxPooling2D((latent_dim,latent_dim), strides=(latent_dim,latent_dim))(conv_1b)
    print(pool_2b.shape)
    conv_3b = Conv2D(48, (latent_dim,latent_dim), strides=(1, 1), input_shape=input_shape_a)(pool_2b)
    print(conv_3b.shape)
  
    act_3b =Activation('relu')(conv_3b)
    print(act_3b.shape)

    print('inshape b', act_3b.shape) 

    re_4b = Reshape(target_shape=(latent_dim*latent_dim,48),input_shape=(latent_dim,latent_dim,48))(act_3b)
    
    ls_5b= LSTM(latent_dim*latent_dim,return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(re_4b)
    print('ls11 a shape is ', ls_5b.shape) 
    
    #at15 = Attention(latent_dim)(ls_5b)
    #print('at15 shape is ', at15.shape) 

    flat12 = Flatten()(ls_5b)
    drop13 = Dropout(rate=0.5)(flat12)
    dense14 =  Dense(64)(drop13)
    act15 = Activation('relu')(dense14)
    drop16=Dropout(rate=0.5)(act15)
    dense17=Dense(totalLabel)(drop16)
    out = Activation('softmax')(dense17)
    model = Model(inputs=[model_a_in, model_b_in], outputs=out)
    
    
    model.summary()
    model.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=['accuracy'])
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
       


    #opt = keras.optimizers.Adam(learning_rate=0.01)
    #model.compile(optimizer=opt,loss="categorical_crossentropy", metrics=['accuracy'])
    #print(model.summary())
    indata = [X_train,Xb_train]
    print ('xtrain shape is ',X_train.shape)
    print ('xbtrain shape is ',Xb_train.shape)
    print ('indata[0] shape is ',indata[0].shape, '1', indata[1].shape,)
    print ('ytrain shape is ',y_train.shape)
    
    ''' 
    early_stopping_monitor = EarlyStopping(
       monitor='val_loss',
       min_delta=0,
       patience=50,
       verbose=0,
       mode='auto',
       baseline=None,
       restore_best_weights=True
)
   '''
    model.fit(indata,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= ([X_test,Xb_test], y_test),#,
        #callbacks=[early_stopping_monitor]

        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    )

    score = model.evaluate([X_test,Xb_test],
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'esc50-sound-classification-{}.h5'.format(timestr)
    model.save('models/{}'.format(modelName))

    print('Model exported and finished')

if __name__ == '__main__':
    (train_data,train_labels), (test_data, test_labels) = importData()#.load_data()

    # Normalize and reshape the data
    train_data, train_labels = preprocess(train_data,train_labels)
    print(train_data.shape)
    test_data, test_labels = preprocess(test_data,test_labels)
    print(test_data.shape)

    buildModel(train_data, test_data, train_labels, test_labels)
    
