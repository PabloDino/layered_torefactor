
import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input
from keras.models import Sequential, Model
from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.callbacks import TensorBoard

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
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'

#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
dataSourceBase = '/home/paul/Downloads/ESC-50-tst2b/'

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
    model_a_in = Input(shape=input_shape_a)
    
    conv_1a = Conv2D(24, (latent_dim//2,latent_dim//2), strides=(1, 1), input_shape=input_shape_a)(model_a_in)
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
    ls11a= LSTM((latent_dim+2)*(latent_dim+2),return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(re_14a)
    print('ls11 a shape is ', ls11a.shape) 
    
    #re_11a = Reshape(target_shape=(latent_dim*latent_dim ,latent_dim*latent_dim ))(ls11a)
    #merge

  
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


    # 27 x 58 x 48
    conv_4b = Conv2D(48, (latent_dim//2,latent_dim//2), padding="valid")(act_3b)

    # 6 x 29 x 48
    pool_4b=MaxPooling2D((latent_dim//2,latent_dim//2), strides=(latent_dim//2,latent_dim//2))(conv_4b)
    act_4b = Activation('relu')(pool_4b)
  

    print('inshape a2', act_4b.shape) 


   # 27 x 58 x 48
    conv_5b = Conv2D(48, (latent_dim//2,latent_dim//2), padding="valid")(act_4b)

    # 6 x 29 x 48
    pool_5b=MaxPooling2D((latent_dim//2,latent_dim//2), strides=(latent_dim//2,latent_dim//2))(conv_5b)
    act_5b = Activation('relu')(pool_5b)
  

    print('inshape a3', act_5b.shape) 


    #re_4b = Reshape(target_shape=(latent_dim*latent_dim*7*7,48),input_shape=(latent_dim*7,latent_dim*7,48))(act_4b)
    re_5b = Reshape(target_shape=(((latent_dim+2)*(latent_dim+2), 48)),input_shape=((latent_dim+2),(latent_dim+2) ,48))(act_5b)
    ls_5b= LSTM((latent_dim+2)*(latent_dim+2),return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(re_5b)
    #merge
    print('ls 5b shape is ', ls_5b.shape) 
    #re_5b = Reshape(target_shape=(latent_dim*15 ,latent_dim*15 ))(ls_5b)
  
 
    merged = Concatenate(axis=1)([ls11a,ls_5b])

    flat12 = Flatten()(merged)
    drop13 = Dropout(rate=0.5)(flat12)
    dense14 =  Dense(64)(drop13)
    act15 = Activation('relu')(dense14)
    drop16=Dropout(rate=0.5)(act15)
    dense17=Dense(totalLabel)(drop16)
    out = Activation('softmax')(dense17)
    model = Model(inputs=[model_a_in, model_b_in], outputs=out)
    fineModel1 = Model(inputs=model_a_in, outputs=ls11a)
    fineModel2 = Model(inputs=model_b_in, outputs=ls_5b)
    
    
    model.summary()
    fineModel1.summary()
    fineModel2.summary()
    
    

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
    indata = [X_train,Xb_train]
    print ('xtrain shape is ',X_train.shape)
    print ('xbtrain shape is ',Xb_train.shape)
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
        validation_data= ([X_test,Xb_test], y_test)#,
        #callbacks=[early_stopping_monitor]
    )#,
    #   callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    #)

    score = model.evaluate([X_test,Xb_test],
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'esc50-sound-classification-{}.h5'.format(timestr)
    model.save('models/{}'.format(modelName))
    fineModel1.save('lsfine.'+format(latent_dim)+'hdf5')
    fineModel2.save('lscoarse.'+format(latent_dim)+'hdf5')

    #'''
    parts_input1 = Input((36,36))
    parts_input2 = Input((36,36))
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
    
    partsModel.summary() 




    partsModel.save('lsparts.'+format(latent_dim)+'.hdf5')
    #'''

    print('Model exported and finished')

if __name__ == '__main__':
    dataSet = importData()
    buildModel(dataSet)
