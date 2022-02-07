'''
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
from tensorflow.keras import layers

'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input
from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
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


def dolEncoder(in_shape, latent_dim, conv_params):
    kernel_size = (conv_params[0], conv_params[1])
    n_filters = conv_params[2]
    dft_dim = in_shape[1]
    inp = layers.Input(in_shape)
    loc = layers.Conv2D(n_filters, strides = (1, 1), kernel_size=kernel_size, activation='relu', padding='same')(inp) 
    loc = layers.MaxPool2D(pool_size=(1, dft_dim))(loc) 
    loc = layers.Reshape((in_shape[0], n_filters))(inp) 
    x   = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True))(loc)
    encoded   = layers.LSTM(latent_dim)(x)            
    return keras.Model(inputs =[inp], outputs=[encoded])



def dolDecoder(length, latent_dim, output_dim, conv_params):
    kernel_size = (conv_params[0], conv_params[1])
    n_filters = conv_params[2]

    inp = layers.Input((latent_dim))
    x   = layers.Reshape((1, latent_dim))(inp)
    x   = layers.ZeroPadding1D((0, length - 1))(x)
    x   = layers.LSTM(latent_dim, return_sequences=True)(x)    
    x   = layers.Bidirectional(layers.LSTM(output_dim // 2, return_sequences=True))(x)
    x   = layers.Reshape((length, output_dim, 1))(x)
    x   = layers.Conv2DTranspose(n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x) 
    decoded   = layers.Conv2DTranspose(1, kernel_size=(1, 1), activation='linear', padding='same')(x) 
    return keras.Model(inputs = [inp], outputs = [decoded])
    
    
def dolauto_encoder(in_shape, latent_dim, conv_params):
    enc = dolEncoder(in_shape, latent_dim, conv_params)
    dec = dolDecoder(in_shape[0], latent_dim, in_shape[1], conv_params)
    inp = layers.Input(in_shape)
    x   = enc(inp) 
    x   = dec(x) 
    model = keras.Model(inputs = [inp], outputs = [x])
    return model, enc, dec


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
      #z_mean8, _, _ = enc.predict([[sample, sample]])
      z_mean8 = enc.predict(sample)
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
      z_mean8 = dec.predict(sample)
      #z_mean8, _, _ = dec.predict([[sample, sample]])
      if (i==0):
        z_mean=z_mean8
      else:
        z_mean = np.concatenate((z_mean,z_mean8), axis=0)
      if (i%200==0):  
        print("dec stat",z_mean.shape)
   return z_mean



if __name__ == '__main__':
    #dataSet = importData()
    model256, enc256, dec256 = dolauto_encoder(input_shape_b, 256,conv_param)
    model32, enc32, dec32 = dolauto_encoder(input_shape_b, 32,conv_param)

    fineEncoder = model256.load_weights('modelblstm1_latent_dim_256.h5')
    #fineEncoder = load_model('encoder_latent_dim_256.h5')
    #)#   , custom_objects={'Sampling': Sampling}, compile=False)
    coarseEncoder = model32.load_weights('modelblstm1_latent_dim_32.h5')#, custom_objects={'Sampling': Sampling}, compile =False)
    #coarseEncoder = load_model('encoder_latent_dim_32.h5')#, custom_objects={'Sampling': Sampling}, compile =False)
    #coarseEncoder.load_weights('coarse-model-15.hdf5')
    (x_train, y_train), (x_test, y_test) = importData()#keras.datasets.mnist.load_data()
    #preProc = np.concatenate([x_train, x_test], axis=0)
    #preProcTrain=x_train[0]
    #preProcTest=x_test[0]
    #preProcTrain = np.expand_dims(x_train, -1).astype("float32") / 255
    #preProcTest = np.expand_dims(x_test, -1).astype("float32") / 255
    #preProcTrain = np.reshape(preProcTrain,(preProcTrain.shape[0],preProcTrain.shape[1],preProcTrain.shape[2],preProcTrain.shape[3]))
    #preProcTest = np.reshape(preProcTest,(preProcTest.shape[0],preProcTest.shape[1],preProcTest.shape[2],preProcTest.shape[3]))
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    

    x_train = tf.expand_dims(x_train,-1)
    x_test = tf.expand_dims(x_test,-1)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
      
    print ("about to encode fine train")
    X_train_fine_encoded = encPredict(enc256,x_train)
    print ("about to encode coarse test")
    X_train_coarse_encoded = encPredict(enc32,x_train)
    
    print ("about to encode fine test")
    X_test_fine_encoded = encPredict(enc256,x_test)# enc32(x_test)
    print ("about to encode coarse test")
    X_test_coarse_encoded = encPredict(enc32, x_test)
    
    
    print ("about to decode fine train")
    X_train_fine_decoded = decPredict(dec256,X_train_fine_encoded)#.astype("float32") / 255
    print ("about to decode coarse train")
    X_train_coarse_decoded = decPredict(dec32, X_train_coarse_encoded)#.astype("float32") / 255
    
    print ("about to decode coarse train")
    X_test_fine_decoded = decPredict(dec256, X_test_fine_encoded)#.astype("float32") / 255
    print ("about to decode coarse test")
    X_test_coarse_decoded = decPredict(dec32, X_test_coarse_encoded)#.astype("float32") / 255    
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
    X_train_fine_decoded = np.array(X_train_fine_decoded)
    X_train_coarse_decoded = np.array(X_train_coarse_decoded)
    X_test_fine_decoded = np.array(X_test_fine_decoded)
    X_test_coarse_decoded = np.array(X_test_coarse_decoded)
     
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
