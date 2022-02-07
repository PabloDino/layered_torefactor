import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, LSTM, ConvLSTM2D, Reshape, Concatenate
from keras.models import Sequential
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
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
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0

# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 50

# model parameters for training
batchSize = 128
epochs = 100

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

    # One-Hot encoding for classes
    y_train = np.array(keras.utils.to_categorical(y_train, totalLabel))
    y_test = np.array(keras.utils.to_categorical(y_test, totalLabel))

    model_a = Sequential()

    # Model Input

    l_input_shape_a=(128, 128,1,1)
    input_shape_a=(128, 128,1)
    model_a_in = Input(shape=input_shape)
    
    model_a.add(Conv2D(24, (5,5), strides=(1, 1), input_shape=input_shape_a))
    # Using CNN to build model
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24
    
    model_a.add(Conv2D(24, (5,5), strides=(1, 1), input_shape=input_shape_a))
    # 31 x 62 x 24
  
    model_a.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model_a.add(Activation('relu'))

    # 27 x 58 x 48
    model_a.add(Conv2D(48, (5, 5), padding="valid"))

    # 6 x 29 x 48
    model_a.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model_a.add(Activation('relu'))
  
    # 2 x 25 x 48
    model_a.add(Conv2D(48, (5, 5), padding="valid"))

    model_a.add(Activation('relu'))
    # model_a.add(Reshape(target_shape=(48,48),input_shape=(2,24,48)))
   # model_a.add(LSTM(64,return_sequences=True,unit_forget_bias=1.0,dropout=0.2))
    #merge

    conc0 =[]
    conc0.append(model_a)
    model_b = Sequential()

    # Model Input

    l_input_shape_b=(128, 128,1,1)
    input_shape_b=(128, 128,1)

    model_b.add(Conv2D(24, (5,5), strides=(1, 1), input_shape=input_shape_b))
    # Using CNN to build model
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24
    
    model_b.add(Conv2D(24, (5,5), strides=(1, 1), input_shape=input_shape_b))
    # 31 x 62 x 24
  
    model_b.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model_b.add(Activation('relu'))

    # 27 x 58 x 48
    model_b.add(Conv2D(48, (5, 5), padding="valid"))

    # 6 x 29 x 48
    model_b.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model_b.add(Activation('relu'))
    
    # 2 x 25 x 48
    model_b.add(Conv2D(48, (5, 5), padding="valid"))
    model_b.add(Activation('relu'))

    model = Sequential()
    model_c=Concatenate(axis=-1)([model_a.output, model_b.output])




    #//model_b.add(Reshape(target_shape=(48,48),input_shape=(2,24,48)))
    #model_b.add(LSTM(64,return_sequences=True,unit_forget_bias=1.0,dropout=0.2))
    #merge
 
 
   
    model = Sequential()
    model.add(Flatten())(model_c)
    model.add(Dropout(rate=0.5))


    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(totalLabel))
    model.add(Activation('softmax'))

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
    print ('xtrain shape is ',X_train.shape)
    print ('ytrain shape is ',y_train.shape)
    
    model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= (X_test, y_test),
        callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)],
    )

    score = model.evaluate(
        x=X_test,
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'esc50-sound-classification-{}.h5'.format(timestr)
    model.save('models/{}'.format(modelName))

    print('Model exported and finished')

if __name__ == '__main__':
    dataSet = importData()
    buildModel(dataSet)
