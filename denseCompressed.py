import keras
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, concatenate, AveragePooling2D
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, LSTM, ConvLSTM2D, Reshape, Concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.optimizers import Adam
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels



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
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-base50/'
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-base50p/'
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-Next30p/'
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-last20p/'

#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#dataSourceBase = '/home/paul/Downloads/ESC-50-tst2b/'


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


def preProcess(dataset):
    print('TotalCount: {}'.format(totalRecordCount))
    trainDataEndIndex = int(totalRecordCount*0.8)
    totalLabel = 10

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
    return X_train, X_test, y_train, y_test

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



class DenseNet:
    def __init__(self, input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None,
                 dropout_rate=None, bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):

        # Checks
        if nb_classes == None:
            raise Exception(
                'Please define number of classes (e.g. num_classes=10). This is required for final softmax.')

        if compression <= 0.0 or compression > 1.0:
            raise Exception('Compression have to be a value between 0.0 and 1.0.')

        if type(dense_layers) is list:
            if len(dense_layers) != dense_blocks:
                raise AssertionError('Number of dense blocks have to be same length to specified layers')
        elif dense_layers == -1:
            dense_layers = int((depth - 4) / 3)
            if bottleneck:
                dense_layers = int(dense_layers / 2)
            dense_layers = [dense_layers for _ in range(dense_blocks)]
        else:
            dense_layers = [dense_layers for _ in range(dense_blocks)]

        self.dense_blocks = dense_blocks
        self.dense_layers = dense_layers
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.bottleneck = bottleneck
        self.compression = compression
        self.nb_classes = nb_classes
        
    def build_model(self):
        #img_input = Input(shape=self.input_shape, name='img_input')
        nb_channels = self.growth_rate
        latent_dim=4
        ############################
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
        '''
        # 27 x 58 x 48
        conv_5a = Conv2D(48, (latent_dim//2,latent_dim//2), padding="valid")(act_4a)

        # 6 x 29 x 48
        pool_6a=MaxPooling2D((latent_dim,latent_dim), strides=(latent_dim,latent_dim))(conv_5a)
        act_7a = Activation('relu')(pool_6a)
        '''
        # 2 x 25 x 48
        conv_8a = Conv2D(48, (latent_dim//2,latent_dim//2), padding="valid")(act_4a)

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
        
        
        x = Conv2D(2*self.growth_rate, (3,3), 
                   padding='same', strides = (1,1), 
                   kernel_regularizer=keras.regularizers.l2(self.weight_decay))(act_13a)
        
        for block in range(self.dense_blocks-1):
            x, nb_channels = self.dense_block(x, self.dense_layers[block], nb_channels, self.growth_rate,
                                              self.dropout_rate, self.bottleneck, self.weight_decay)
            
            x = self.transition_layer(x, nb_channels, self.dropout_rate, self.compression, self.weight_decay)
            nb_channels = int(nb_channels*self.compression)
            
        x, nb_channels = self.dense_block(x, self.dense_layers[-1], nb_channels, self.growth_rate, self.dropout_rate, self.weight_decay)
        
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        prediction = Dense(self.nb_classes, activation='softmax')(x)
        
        model_c= Model(inputs=model_a_in, outputs=act_13a, name='comp')
        
        return Model(inputs=model_a_in, outputs=prediction, name='densenet'), model_c
        
    def dense_block(self, x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
        for i in range(nb_layers):
            cb = self.convolution_block(x, growth_rate, dropout_rate, bottleneck)
            nb_channels += growth_rate
            x = concatenate([cb,x])
            
        return x, nb_channels
    
    def convolution_block(self, x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):       

        # Bottleneck
        if bottleneck:
            bottleneckWidth = 4
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(nb_channels * bottleneckWidth, (1, 1),
                                     kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
            # Dropout
            if dropout_rate:
                x = Dropout(dropout_rate)(x)

        # Standard (BN-ReLU-Conv)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_channels, (3, 3), padding='same')(x)

        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x

    def transition_layer(self, x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(int(nb_channels * compression), (1, 1), padding='same',
                                 kernel_regularizer=keras.regularizers.l2(weight_decay))(x)

        # Adding dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        return x
    
    
    
    
    

print('creating net')    
densenet = DenseNet((128,128,1), nb_classes=10, depth=25)
print('building model')    

model, model_comp = densenet.build_model()
 


model_optimizer = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
print('compiling model')    

#model.compile(loss='categorical_crossentropy', optimizer=model_optimizer, metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

'''
with open('my/directory/train-images-idx3-ubyte.gz', 'rb') as f:
  train_images = extract_images(f)
with open('my/directory/train-labels-idx1-ubyte.gz', 'rb') as f:
  train_labels = extract_labels(f)


idx = np.random.permutation(len(train_images))
x,y = train_images[idx], train_labels[idx]

trainDataEndIndex = int(len(train_images)*0.8)

totalLabel = 10


X_train = x[:trainDataEndIndex]
X_test = x[trainDataEndIndex:]

X_train = np.array([x.reshape( (28, 28, 1) ) for x in X_train])
X_test = np.array([x.reshape( (28, 28, 1 ) ) for x in X_test])
 

y_train = y[:trainDataEndIndex]
y_test = y[trainDataEndIndex:]

y_train = np.array(keras.utils.to_categorical(y_train, totalLabel))
y_test = np.array(keras.utils.to_categorical(y_test, totalLabel))

print(X_train[0])
print(y_train[0])
'''




filepath = "Incremental/20p/5/Dense25.20p.{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

dataset= importData()
X_train, X_test, y_train, y_test = preProcess(dataset)
batchSize=128
epochs=100
#dot_img_file = 'dense.png'
#keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

model.fit(X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batchSize,
        validation_data= (X_test, y_test),
        #callbacks=[checkpoint]
        )
model.save('Incremental/20p/Model.5.final.hdf5')        
#model.save('models/dense50.hdf5')
