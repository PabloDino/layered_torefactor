import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Conv1D, Lambda, Conv2DTranspose, \
                         Flatten, MaxPooling2D, MaxPooling1D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input     
from keras.models import Sequential, Model
from keras.models import Sequential, Model
import tensorflow as tf
from keras.callbacks import LearningRateScheduler,EarlyStopping#,ModelCheckPoint
from keras.utils import to_categorical
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
import datetime
import math
warnings.filterwarnings('ignore')

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
baseFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-Next30p/'
#nextFolder = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-next30p/'
#baseFolder = '/home/paul/Downloads/ESC-50-tst2b/'

dataSourceBase =baseFolder

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0

# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 0#50

# model parameters for training
batchSize = 128
epochs = 100
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


#filepath = "Model.1.-model-{epoch:02d}-{loss:.2f}.hdf5"
filepath = 'Incremental/30p/3/Model.3.{epoch:02d}-{loss:.2f}.'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


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
        totalLabel+=15#25
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

# This is the default import function for UrbanSound8K
# https://urbansounddataset.weebly.com/urbansound8k.html
# Please download the URBANSOUND8K and not URBANSOUND
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
    
    #Xb_train = X_train.copy()#np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    #Xb_test = X_test.copy()#np.array([x.reshape( (128, 128, 1 ) ) for x in X_test])


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
    print('ls5 a shape is ', ls5a.shape) 
    
    #re5a = Reshape(target_shape=(latent_dim*latent_dim ,latent_dim*latent_dim ))(ls5a)
    #merge


    # 27 x 58 x 48
    conv_5a = Conv1D(48, kernel_size=latent_dim,  activation='relu')(ls5a)

    # 6 x 29 x 48
    pool_6a=MaxPooling1D(pool_size=latent_dim)(conv_5a)
    act_7a = Activation('relu')(pool_6a)
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
    re_7a = Reshape(target_shape=(255,94,1))(ift7a)
    # 2 x 25 x 48
    #conv_8a = Conv2D(48, (latent_dim//2,latent_dim//2), padding="valid")(re_7a)
    pool_8a = MaxPooling2D((latent_dim//2,2))(re_7a)
    print('pool8b',pool_8a.shape)
    tr8a = Conv2DTranspose(1, kernel_size=(2,latent_dim), activation='relu', padding='valid')(pool_8a) 
    print('tr8a',tr8a.shape)
    act_9a = Activation('relu')(tr8a)    # 2 x 25 x 48

    tr9a = Conv2DTranspose(1, kernel_size=(1,latent_dim), activation='relu', padding='valid')(act_9a) 
    print('tr9a',tr9a.shape)
    act_9aa = Activation('relu')(tr9a)    # 2 x 25 x 48

    #conv_9a = Conv2D(1, (1,latent_dim), padding="valid")(act_9aa)
    #print('conv_9a',conv_9a.shape)
    tr10a = Conv2DTranspose(1, kernel_size=(1,latent_dim//2), activation='relu', padding='valid')(act_9aa)     
    act_10aa = Activation('relu')(tr10a)
    #************************************************************

    print('tr10a',tr10a.shape)
   
    re_10aa = Reshape(target_shape=(latent_dim*latent_dim, latent_dim*latent_dim))(act_10aa)
    #merge
    flat12 = Flatten()(re_10aa)
    drop13 = Dropout(rate=0.5)(flat12)
    dense14 =  Dense(64)(drop13)
    act15 = Activation('relu')(dense14)
    drop16=Dropout(rate=0.5)(act15)
    dense17=Dense(totalLabel)(drop16)
    out = Activation('softmax')(dense17)
    model = Model(inputs=model_a_in, outputs=out)

        
    
    #model.summary()
    model.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=['accuracy'])
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
    
    #''' 
    early_stopping_monitor = EarlyStopping(
       monitor='val_loss',
       min_delta=0,
       patience=50,
       verbose=0,
       mode='auto',
       baseline=None,
       restore_best_weights=True
)
   #'''
    print('FITTING2')
    model.fit(X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= (X_test, y_test),#,
        #callbacks=[early_stopping_monitor,checkpoint]

        callbacks=[checkpoint],
        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1), checkpoint],
    )

    score = model.evaluate(X_test,
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'SingleModel.3.30p.'.format(timestr)
    model.save('models/{}'.format(modelName))

    print('Model exported and finished')
    
if __name__ == '__main__':
    dataset =  importData('base')#(train, test) =  
    print('total recs =', totalRecordCount, '; Total Labels=', totalLabel, lblmap)
    #nextdataset =   importData('next')#(nextTrain,nextTest) = 
    #print('total recs =', totalRecordCount, '; Total Labels=', totalLabel, lblmap)
    #dataset = mergeSets2(dataset, nextdataset)#train, test, nextTrain, nextTest)

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
    

    # One-Hot encoding for classes
    #y_train = np.array(keras.utils.to_categorical(y_train, totalLabel))
    #y_test = np.array(keras.utils.to_categorical(y_test, totalLabel))

    buildModel(x_train, x_test, y_traincat, y_testcat)
    
