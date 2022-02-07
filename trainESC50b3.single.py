import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input
from keras.models import Sequential, Model
from keras.callbacks import LearningRateScheduler,EarlyStopping
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
warnings.filterwarnings('ignore')

# Your data source for wav files
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'

#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0

# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 50

# model parameters for training
batchSize = 128
epochs = 100
latent_dim=4
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
    return dataSet

# This is the default import function for UrbanSound8K
# https://urbansounddataset.weebly.com/urbansound8k.html
# Please download the URBANSOUND8K and not URBANSOUND
def buildModel(dataset):
    print('TotalCount: {}'.format(totalRecordCount))
    trainDataEndIndex = int(totalRecordCount*0.8)
    random.shuffle(dataset)

    train = dataset[:trainDataEndIndex]
    test = dataset[trainDataEndIndex:]

    print('Total training data:{}'.format(len(train)))
    print('Total test data:{}'.format(len(test)))

    # Get the data (128, 128) and label from tuple
    print("train 0 shape is ",train[0][0].shape)
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    # Reshape for CNN input
    #X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    #X_test = np.array([x.reshape( (128, 128, 1) ) for x in X_test])
    
    X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    X_test = np.array([x.reshape( (128, 128, 1 ) ) for x in X_test])

    Xb_train = X_train.copy()#np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    Xb_test = X_test.copy()#np.array([x.reshape( (128, 128, 1 ) ) for x in X_test])

    

    # One-Hot encoding for classes
    y_train = np.array(keras.utils.to_categorical(y_train, totalLabel))
    y_test = np.array(keras.utils.to_categorical(y_test, totalLabel))

    model_a = Sequential()

    # Model Input

    l_input_shape_a=(128, 128,1,1)
    input_shape_a=(128, 128,1)
    model_a_in = keras.layers.Input(shape=input_shape_a)
    
    conv_1a = keras.layers.Conv2D(24, latent_dim//2)(model_a_in)
    # Using CNN to build model
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24
    
    #conv_2a = Conv2D(24, (latent_dim//2,latent_dim//2), strides=((latent_dim//2,latent_dim//2)), input_shape=input_shape_a)(conv_1a)
    # 31 x 62 x 24
  
    pool_3a = MaxPooling2D((latent_dim//2,latent_dim//2), strides=(latent_dim//2,latent_dim//2))(conv_1a)
    act_4a =Activation('relu')(pool_3a)

    # 27 x 58 x 48
    conv_5a = Conv2D(48, (latent_dim//2,latent_dim//2), padding="valid")(act_4a)

    # 6 x 29 x 48
    pool_6a=MaxPooling2D((latent_dim,latent_dim), strides=(latent_dim,latent_dim))(conv_5a)
    act_7a = Activation('relu')(pool_6a)
  
    # 2 x 25 x 48
    conv_8a = Conv2D(48, (latent_dim//2,latent_dim//2), padding="valid")(act_7a)

    act_9a = Activation('relu')(conv_8a)    # 2 x 25 x 48
    conv_9a = Conv2D(48, (latent_dim//2,latent_dim//2), padding="valid")(act_9a)

    act_10a = Activation('relu')(conv_9a)

    print('inshape a', act_10a.shape) 

    # 27 x 58 x 48
    conv_11a = Conv2D(48, (latent_dim//2,latent_dim//2), padding="valid")(act_10a)

    # 6 x 29 x 48
    pool_12a=MaxPooling2D((latent_dim//2,latent_dim//2), strides=(latent_dim//2,latent_dim//2))(conv_11a)
    act_13a = Activation('relu')(pool_12a)
  

    print('inshape a', act_13a.shape) 
    re_14a = Reshape(target_shape=(((latent_dim+2)*(latent_dim+2), 48)),input_shape=((latent_dim+2),(latent_dim+2) ,48))(act_13a)
    ls14a= LSTM((latent_dim+2)*(latent_dim+2),return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(re_14a)
    print('ls11 a shape is ', ls14a.shape) 
    re_15a = Reshape(target_shape=(((latent_dim+2),(latent_dim+2), 48)))(ls14a)
   
    drop13 = Dropout(rate=0.5)(re_15a)
    dense14 =  Dense(64)(drop13)
    act15 = Activation('relu')(dense14)
    drop16=Dropout(rate=0.5)(act15)
    

    dense17=Dense(totalLabel)(drop16)
    out = Activation('softmax')(dense17)
    model = Model(inputs=model_a_in, outputs=out)


    
    
    model.summary()
     
    

    model.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=['accuracy'])
    initial_learning_rate = 0.01
    #epochs = 100
    decay = initial_learning_rate / epochs
    def lr_time_based_decay(epoch, lr):
       if epoch < 50:
            return decay *epochs
       else:
            return lr * epoch / (epoch + decay * epoch)


    #opt = keras.optimizers.Adam(learning_rate=0.01)
    #model.compile(optimizer=opt,loss="categorical_crossentropy", metrics=['accuracy'])
    #print(model.summary())
    indata = X_train#[X_train,Xb_train]
    print ('xtrain shape is ',X_train.shape)
    print ('xbtrain shape is ',Xb_train.shape)
    print ('indata[0] shape is ',indata[0].shape, '1', indata[1].shape,)
    print ('ytrain shape is ',y_train.shape)
    

    model.fit(indata,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= (X_test, y_test)#,
        #callbacks=[early_stopping_monitor]
    )#,
    #   callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    #)

    score = model.evaluate(X_test,
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'SingleModel.b3.h5'.format(timestr)
    model.save('models/{}'.format(modelName))

    print('Model exported and finished')

if __name__ == '__main__':
    dataSet = importData()
    buildModel(dataSet)
