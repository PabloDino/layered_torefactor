
import keras
import tensorflow as tf
from keras.layers import Activation, Dense, Dropout, Conv2D, Conv1D, Lambda, Conv2DTranspose, \
                         Flatten, MaxPooling2D, MaxPooling1D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input
from keras.models import Sequential, Model
from tensorflow.keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.callbacks import TensorBoard


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
from keras import backend as K
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
epochs = 150
latent_dim=8
# This function will import wav files by given data source path.
# And will extract wav file features using librosa.feature.melspectrogram.
# Class label will be extracted from the file name
# File name pattern: {WavFileName}-{ClassLabel}
# e.g. 0001-0 (0001 is the name for the wav and 0 is the class label)
# The program only interested in the class label and doesn't care the wav file name
from itertools import chain, combinations




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
            #if i not in [1]:
            #if i not in [1,2]:
                # define filename for this ensemble
                filename = '../models/Model.' + str(i + 1) + '.hdf5'
                #filename = '../models/SingleModel.' + str(i + 1) + '.h5'
                # load model from file
                print('loading ', filename)
                model = load_model(filename,  custom_objects={'tf': tf, 'f1_m':f1_m, 'precision_m':precision_m, 'recall_m':recall_m})
                model.built=True
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
testX, origy = zip(*test)
trainX = np.array([x.reshape( (128, 128, 1) ) for x in trainX])
testX = np.array([x.reshape( (128, 128, 1 ) ) for x in testX])


# One-Hot encoding for classes
trainy = np.array(keras.utils.to_categorical(origy, totalLabel))
testy = np.array(keras.utils.to_categorical(origy, totalLabel))

#print(trainX.shape, testX.shape)
# load all models
n_members = 3
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset
for model in members:
        model.summary()
        #testy_enc = to_categorical(testy)
        #print('testX.shape is ', testX.shape)
        #print('testY_enc.shape is ', testy.shape)
        _, acc,f1,precision, recall = model.evaluate(testX, testy, verbose=0)
        #_, acc = model.evaluate(testX, testy, verbose=0)
        print('Model Accuracy: %.3f ' % acc, f1,precision, recall )
        #print (acc,f1,precision, recall)
# fit stacked model using the ensemble
#print('test x shape is ', testX.shape)
#print('test y shape is ', testy.shape)
ps =powerset([1,2,3])
for s in ps:
   tempModel =[]
   for num in s:
      print (s)
      tempModel.append(members[num-1])
   if len(tempModel) >1:
      stackedModel = fit_stacked_model(tempModel, testX, origy)
      # evaluate model on test set
      yhat = stacked_prediction(tempModel, stackedModel, testX)
      modelsuffix =""
      for i in range(len(tempModel)):
          modelsuffix=modelsuffix+str(tempModel[i])+"_"
      acc = accuracy_score(origy, yhat)
      print('Stacked Test Accuracy:',s,': %.3f' % acc)
      #dot_img_file = 'stack'+modelsuffix+'.png'
      #keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

