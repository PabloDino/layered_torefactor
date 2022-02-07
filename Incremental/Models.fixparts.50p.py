
import keras
import tensorflow as tf
from keras.layers import Activation, Dense, Dropout, Conv2D, Conv1D, Lambda, Conv2DTranspose, InputLayer,  \
                         Flatten, MaxPooling2D, MaxPooling1D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input
from keras.models import Sequential, Model
from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.callbacks import TensorBoard
from keras.models import clone_model

# stacked generalization with linear meta model on blobs dataset
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.utils import to_categorical
from numpy import dstack




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
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-aug-base50/'

#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC50-Base50p/'
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-tst-base50p/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0

# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 25

# model parameters for training
batchSize = 128
epochs = 5
latent_dim=8
# This function will import wav files by given data source path.
# And will extract wav file features using librosa.feature.melspectrogram.
# Class label will be extracted from the file name
# File name pattern: {WavFileName}-{ClassLabel}
# e.g. 0001-0 (0001 is the name for the wav and 0 is the class label)
# The program only interested in the class label and doesn't care the wav file name
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
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


# load models from file
def load_all_models(n_models):
        all_models = list()
        for i in range(n_models):
            if i  > 2:#not in [1]:
            #if i not in [1,2]:
                # define filename for this ensemble
                filename = 'SingleModel.' + str(i + 1) + '.h5'
                #filename = 'model' + str(i + 1) + '.h5'
                # load model from file
                print('loading ', filename)
                model = load_model(filename,  custom_objects={'tf': tf})
                # add to list of members
                all_models.append(model)
                print('>loaded %s' % filename)
        return all_models

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
	
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))

	return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# fit standalone model
	model = LogisticRegression()
	model.fit(stackedX, inputy)
	return model

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat

# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# split into train and test


'''#
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
'''#
#'''

dataset = importData()
random.shuffle(dataset)
trainDataEndIndex = int(totalRecordCount*0.6)
train = dataset[:trainDataEndIndex]
test = dataset[trainDataEndIndex:]

print('Total training data:{}'.format(len(train)))
print('Total test data:{}'.format(len(test)))

# Get the data (128, 128) and label from tuple
print("train 0 shape is ",train[0][0].shape)
trainX, origy = zip(*train)
testX, origtesty = zip(*test)
trainX = np.array([x.reshape( (128, 128, 1) ) for x in trainX])
testX = np.array([x.reshape( (128, 128, 1 ) ) for x in testX])


# One-Hot encoding for classes
trainy = np.array(keras.utils.to_categorical(origy, totalLabel))
testy = np.array(keras.utils.to_categorical(origtesty, totalLabel))

#print(trainX.shape, testX.shape)
# load all models
n_members = 4
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset
dx=4
for model in members:
        if dx <=3:
         if dx!=2:
           newModel = Sequential()
           earlyModel = Sequential()
           lateModel = Sequential()
           settingEarly=True
           reshape_2=None
           late_input = Flatten(input_shape = ( 64,64), name ='inflat')
           lateModel.add(late_input)
           for layer in model.layers[:-2]:
              newModel.add(layer)
              if settingEarly:
                 earlyModel.add(layer) 
                 print(dx, layer.name)
                 #if layer.name=='lstm_1':
                 if (dx==1) and (layer.name=='reshape_2'):
                    reshape_2 = layer.get_output_at(-1)
                    lateModel.add(layer)
                    settingEarly= False
                 if False:#(dx==2) and (layer.name=='flatten_1'):
                    reshape_2 = layer.get_output_at(0)
                    lateModel.add(layer)
                    settingEarly= False
                 if (dx==3) and (layer.name=='reshape_3'):
                    reshape_2 = layer.get_output_at(-1)
                    lateModel.add(layer)
                    settingEarly= False
              else:
                 lateModel.add(layer) 
              
           denseout = Dense(totalLabel, name='denseout')(model.layers[-1].output)
           out = Activation('softmax', name='out')(denseout)
           newModel = Model(input=model.input, output=[out])   
           earlyModel = Model(input=model.input, outputs=reshape_2)
           #lateModel = Model(input=reshape_2, output=[out])
           
        else:
           newModel0 =  clone_model(model)
           earlyModel = Sequential()
           lateModel =  clone_model(model)
           settingEarly=True
           modelPos =0
           model.summary()
           #late_preinput = Flatten(input_shape = ( 30,30,12), name ='inflat')
           late_preinput = None
           
           #late_preinput = Input((30,30,12))
           #lateModel.add(late_input)

           for layer in model.layers:
              if settingEarly:
                 earlyModel.add(layer) 
                 print(dx, layer.name)
                 lateModel.layers.pop(0)
                 if layer.name=='conv2d_4':
                     earlyout = layer.get_output_at(0)
                     print(str(layer.get_output_at(0).shape))
                     #print(lateModel.layers[0].strides) 
                     print(str(lateModel.layers[0].get_output_at(0).shape) )
                     #lateModel.layers.pop(0) 
                     lateModel.layers.pop()
                     print(lateModel.layers[0].get_output_at(0).shape) 
                     #print(lateModel.layers[0].strides) 
                     
                     #lateModel.layers.pop(0)
                     late_preinput = Input(shape = lateModel.layers[0].get_output_at(0).shape, name ='inflat')
                     #late_next = Conv2D(48, kernel_size=(1,1),padding = "same", activation="relu")(late_preinput)
                     
                     late_output = lateModel(late_preinput)
                     settingEarly= False
                     #if modelPos >0:
                 #else:              
                    
              modelPos+=1 
         
           prediction = Dense(totalLabel, activation='softmax', name ='pred')(newModel0.layers[-1].output)
           predictionLate = Dense(totalLabel, activation='softmax', name ='pred')(late_output)
           newModel = Model(input=newModel0.input, output=[prediction])
           earlyModel = Model(input=model.input, outputs=earlyout)
           lateModel = Model(input=earlyout, outputs=predictionLate)
           #lateModel = Model(newInput, newOutputs)

           
        newModel.summary() 
        newModel.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=['accuracy'])
        newModel.fit(trainX,
           y=trainy,
           epochs=epochs,
           batch_size=batchSize,
           validation_data= (testX,testy),#,
           )
           
        newModel.save('model.base.'+str(dx)+'.50p.h5')
        earlyModel.save('earlymodel.base.'+str(dx)+'.50p.h5')
        lateModel.save('latemodel.base.'+str(dx)+'.50p.h5')
        
        _, score = newModel.evaluate(testX, y=testy)
        print('Model Accuracy: %.3f' % score)
        dx+=1
       
    
