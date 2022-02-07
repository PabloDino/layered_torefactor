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
#from keras import layers
from sklearn.preprocessing import MinMaxScaler


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
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0

# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 50

# model parameters for training
batchSize = 128
epochs = 100
viewBatch=2




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

# This is the default import function for UrbanSound8K
# https://urbansounddataset.weebly.com/urbansound8k.html
# Please download the URBANSOUND8K and not URBANSOUND
def buildModel(fineTrain,coarseTrain,fineTest,coarseTest, y_train, y_test,preProcTrain,preProcTest):


    print('Total training data:{}'.format(len(fineTrain)))
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
    
    X_fineTrain = np.array([x.reshape( (16,16, 1) ) for x in fineTrain])
    X_fineTest = np.array([x.reshape( (16,16, 1 ) ) for x in fineTest])
  

    X_coarseTrain = np.array([x.reshape( (8, 8, 1) ) for x in coarseTrain])
    X_coarseTest = np.array([x.reshape( (8,8, 1 ) ) for x in coarseTest])

  
    

    # One-Hot encoding for classes
    y_train = np.array(keras.utils.to_categorical(y_train, totalLabel))
    y_test = np.array(keras.utils.to_categorical(y_test, totalLabel))

    #fineEncoder.begin()
    model_a = Sequential()

    # Model Input

    l_input_shape_a=(16,16,1,1)
    input_shape_a=(16,16,1)
    model_a_in = Input(shape=input_shape_a)
    
    conv_1a = Conv2D(1, (2,2), strides=(1, 1), input_shape=input_shape_a)(model_a_in)
    # Using CNN to build model
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24
    print('conv_1a', conv_1a.shape)
    
    pool_3a = MaxPooling2D((3,3), strides=(1,1))(conv_1a)
    act_4a =Activation('relu')(pool_3a)

    conv_2a = Conv2D(1, (3,3), strides=(1, 1), input_shape=input_shape_a)(act_4a)
    #31 x 62 x 24#    
    pool_4a = MaxPooling2D((3,3), strides=(1,1))(conv_2a)
    act_5a =Activation('relu')(pool_4a)
    # 27 x 58 x 48
    conv_3a = Conv2D(1, (3,3), padding="valid")(act_5a)

    # 6 x 29 x 48
    pool_6a=MaxPooling2D((3,3), strides=(1,1))(conv_3a)
    act_7a = Activation('relu')(pool_6a)
  
    # 2 x 25 x 48
    conv_8a = Conv2D(1, (2,2), padding="valid")(act_7a)

    act_9a = Activation('relu')(conv_8a)
    #re_10a = Reshape(target_shape=(4,4),input_shape=(2,24,48))(act_9a)
    re_10a = Reshape(target_shape=(4,4),input_shape=(4,4,1))(act_9a)
    
    #fineEncoder.end()
    ls11a= LSTM(4, return_sequences=True,unit_forget_bias=1.0,dropout=0.1)(re_10a)
    #seqa=SeqSelfAttention(attention_activation='sigmoid')(ls11a)

    #merge

    #coarseEncoder.begin()
    model_b = Sequential()

    # Model Input

    l_input_shape_b=(8,8 ,1,1)
    input_shape_b=(8,8,1)

    model_b_in = Input(shape=input_shape_b)
    print(model_b_in.shape)

    conv_1b = Conv2D(1, (1,1), strides=(2, 2), input_shape=input_shape_a)(model_b_in)#placeholder
    print('conv1b', conv_1b.shape)
    # Using CNN to build model
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24   
    # 98x98x24    
    '''
    pool_2b = MaxPooling2D((4,4), strides=(4,4))(conv_1b)
    print(pool_2b.shape)
    conv_3b = Conv2D(48, (3,3), strides=(1, 1), input_shape=input_shape_a)(pool_2b)
    print(conv_3b.shape)
  
    act_3b =Activation('relu')(conv_3b)
    print(act_3b.shape)
    '''

    re_4b = Reshape(target_shape=(4,4),input_shape=(4,4,1))(conv_1b)
    #coarseEncoder.end()
    ls_5b= LSTM(4,return_sequences=True,unit_forget_bias=1.0,dropout=0.1)(re_4b)
    #seqb=SeqSelfAttention(attention_activation='sigmoid')(ls_5b)
    #merge
 
   
    merged = Concatenate(axis=1)([ls11a,ls_5b])
    #attnLyr= Attention()([ls11a, ls_5b])
    print('ls11a.shape is', ls11a.shape)    
    print('ls_5b.shape is', ls_5b.shape)
    print('merged.shape is', merged.shape)
    flat12 = Flatten()(merged)
    print('flat.shape is', flat12.shape)
    '''
    #flat12 = Flatten()(attnLyr)
    drop13 = Dropout(rate=0.5)(flat12)
    dense14 =  Dense(64)(drop13)
    act15 = Activation('relu')(dense14)
    drop16=Dropout(rate=0.5)(act15)
    dense17=Dense(totalLabel)(drop16)
    
    '''
    dense13=Dense(50)(flat12)
    out = Activation('softmax')(dense13)
    print('out is ', out.shape)
    model = Model(inputs=[model_a_in, model_b_in], outputs=out)

    model.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=['accuracy'])
    initial_learning_rate = 0.001
    #epochs = 100
    decay = initial_learning_rate / epochs
    def lr_time_based_decay(epoch, lr):
       if epoch < 50:
            return initial_learning_rate#decay *epochs
       else:
            return (50.0/epoch)*initial_learning_rate#lr * epoch / (epoch + decay * epoch)


    #opt = keras.optimizers.Adam(learning_rate=0.01)
    #model.compile(optimizer=opt,loss="categorical_crossentropy", metrics=['accuracy'])
    #print(model.summary())
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
    #print("preProcTrain shape is ", preProcTrain[0][0].shape)
    #print("preProcTest shape is ", preProcTest[0].shape)
    model.fit(indata,
        y=y_train,
        epochs=epochs,
        #batch_size=batchSize,
        validation_data= ([X_fineTest,X_coarseTest], y_test)#,
        #callbacks=[early_stopping_monitor]
        #)#,
        #callbacks=[LearningRateScheduler(lr_time_based_decay, verbose=1)]
    )

    score = model.evaluate([X_fineTest,X_coarseTest],
        y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    timestr = time.strftime('%Y%m%d-%H%M%S')
    modelName = 'esc50-sound-classification-{}.h5'.format(timestr)
    model.save('models/{}'.format(modelName))

    print('Model exported and finished')

if __name__ == '__main__':
    #dataSet = importData()
    
    fineEncoder = load_model('encoder_latent_dim_16.h5', custom_objects={'Sampling': Sampling}, compile=False)
    coarseEncoder = load_model('encoder_latent_dim_256.h5', custom_objects={'Sampling': Sampling}, compile =False)
    #coarseEncoder.load_weights('coarse-model-15.hdf5')
    (x_train, y_train), (x_test, y_test) = importData()#keras.datasets.mnist.load_data()
    #preProc = np.concatenate([x_train, x_test], axis=0)
    
    preProcTrain = np.expand_dims(x_train, -1).astype("float32") / 255
    preProcTest = np.expand_dims(x_test, -1).astype("float32") / 255
    preProcTrain = np.reshape(preProcTrain,(preProcTrain.shape[0],preProcTrain.shape[1]*preProcTrain.shape[2],preProcTrain.shape[3]))
    preProcTest = np.reshape(preProcTest,(preProcTest.shape[0],preProcTest.shape[1]*preProcTest.shape[2],preProcTest.shape[3]))
    
    #X_train_fine_encoded = fineEncoder.predict([preProcTrain,preProcTrain])
    ### statements like this broken up because of OOM
    numrows = preProcTrain.shape[0]
    for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample = preProcTrain[i*viewBatch:i*viewBatch+viewBatch,]
      fineBatch, _, _ = fineEncoder.predict([[sample, sample]])
      if (i==0):
        fineCoded=fineBatch
      else:
        fineCoded = np.concatenate((fineCoded,fineBatch), axis=0)
      print('coding train fine', fineCoded.shape)
    X_train_fine_encoded= fineCoded
    
    #X_train_coarse_encoded = coarseEncoder.predict([preProcTrain,preProcTrain])
    numrows = preProcTrain.shape[0]
    for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample2 = preProcTrain[i*viewBatch:i*viewBatch+viewBatch,]
      coarseBatch, _, _ = coarseEncoder.predict([[sample2, sample2]])
      if (i==0):
        coarseCoded=coarseBatch
      else:
        coarseCoded = np.concatenate((coarseCoded,coarseBatch), axis=0)
      print('coding train coarse', fineCoded.shape)
    X_train_coarse_encoded= coarseCoded
    
    #X_test_fine_encoded = fineEncoder.predict([preProcTest,preProcTest])
    numrows = preProcTest.shape[0]
    for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample3 = preProcTest[i*viewBatch:i*viewBatch+viewBatch,]
      fineBatch, _, _ = fineEncoder.predict([[sample3, sample3]])
      if (i==0):
        fineCoded=fineBatch
      else:
        fineCoded = np.concatenate((fineCoded,fineBatch), axis=0)
      print('coding test fine', fineCoded.shape)
    X_test_fine_encoded= fineCoded
    
    #X_test_coarse_encoded = coarseEncoder.predict([preProcTest,preProcTest])
    numrows = preProcTest.shape[0]
    for i in range(0,int((numrows/viewBatch))):#print(x_train.shape)
      sample4 = preProcTest[i*viewBatch:i*viewBatch+viewBatch,]
      coarseBatch, _, _ = coarseEncoder.predict([[sample4, sample4]])
      if (i==0):
        coarseCoded=coarseBatch
      else:
        coarseCoded = np.concatenate((coarseCoded,coarseBatch), axis=0)
      print('coding test coarse', fineCoded.shape)
    X_test_coarse_encoded= coarseCoded
    print("got here")
    
    X_train_fine_encoded = np.expand_dims(X_train_fine_encoded, -1).astype("float32") / 255
    X_train_coarse_encoded = np.expand_dims(X_train_coarse_encoded, -1).astype("float32") / 255
    X_test_fine_encoded = np.expand_dims(X_test_fine_encoded, -1).astype("float32") / 255
    X_test_coarse_encoded = np.expand_dims(X_test_coarse_encoded, -1).astype("float32") / 255
    
    buildModel(X_train_fine_encoded,X_train_coarse_encoded, X_test_fine_encoded,X_test_coarse_encoded, y_train, y_test,preProcTrain,preProcTest)
