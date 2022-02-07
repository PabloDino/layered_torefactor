import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input
from keras.models import Sequential, Model

from keras.callbacks import LearningRateScheduler,EarlyStopping

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
from keras import backend as K
warnings.filterwarnings('ignore')

# Your data source for wav files
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-base50/'
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

dataSourceBase=lastFolder
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 10

# model parameters for training
batchSize = 128
epochs = 100
latent_dim=8
dataSize=128

timesteps = 128 # Length of your sequences
input_dim = 128 


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

    model_b = Sequential()

    # Model Input

    l_input_shape=(128, 128,1,1)
    input_shape=(128, 128,1)


    model_b_in = Input(shape=input_shape)
    print(model_b_in.shape)

    conv_1b = Conv2D(24, (latent_dim,latent_dim), strides=(1, 1), input_shape=input_shape)(model_b_in)
    print(conv_1b.shape)
    # Using CNN to build model
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24   
    # 98x98x24    

    pool_2b = MaxPooling2D((latent_dim,latent_dim), strides=(latent_dim,latent_dim))(conv_1b)
    print(pool_2b.shape)
    conv_3b = Conv2D(48, (latent_dim,latent_dim), strides=(1, 1), input_shape=input_shape)(pool_2b)
    print(conv_3b.shape)
  
    act_3b =Activation('relu')(conv_3b)
    print(act_3b.shape)

    print('inshape b', act_3b.shape) 

    re_4b = Reshape(target_shape=(latent_dim*latent_dim,48),input_shape=(latent_dim,latent_dim,48))(act_3b)
    
    ls_5b= LSTM(latent_dim*latent_dim,return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(re_4b)
    #print('ls11 a shape is ', ls_5b.shape) 

      
    #at15 = Attention(latent_dim)(ls_5b)
    #print('at15 shape is ', at15.shape) 

    flat12 = Flatten()(ls_5b)
    drop13 = Dropout(rate=0.5)(flat12)
    dense14 =  Dense(64)(drop13)
    act15 = Activation('relu')(dense14)
    drop16=Dropout(rate=0.5)(act15)
    dense17=Dense(totalLabel)(drop16)
    out = Activation('softmax')(dense17)
    model = Model(inputs=model_b_in, outputs=out)
    
    
    model.summary()
    #model.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
    initial_learning_rate = 0.01
    #epochs = 100
    drop = 0.5
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
    indata = [X_train]
    print ('xtrain shape is ',X_train.shape)
    print ('ytrain shape is ',y_train.shape)
    #dot_img_file = 'lstm.png'
    #keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    filepath = "lstm.{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False,     save_weights_only=False, mode='auto', period=1)

    model.fit(X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= (X_test, y_test),#,
        #callbacks=[early_stopping_monitor]
        #callbacks=[checkpoint]
        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    )
    score = model.evaluate(X_test,
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'Incremental/20p/Model.1.'.format(timestr)
    modelName =modelName+".hdf5"
    model.save('{}'.format(modelName))

    print('Model exported and finished')

if __name__ == '__main__':
    (train_data,train_labels), (test_data, test_labels) = importData()#.load_data()

    # Normalize and reshape the data
    train_data, train_labels = preprocess(train_data,train_labels)
    print(train_data.shape)
    test_data, test_labels = preprocess(test_data,test_labels)
    print(test_data.shape)

    buildModel(train_data, test_data, train_labels, test_labels)
    
