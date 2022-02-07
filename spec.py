from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from numpy import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

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


# Your data source for wav files
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'

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

#sc = SpectralClustering(n_clusters=4).fit(x)
#print(sc)

(x_train, y_train), (x_test, y_test) = importData()#keras.datasets.mnist.load_data())

tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=2000)
x2d = tsne.fit_transform(x_train)

f = plt.figure()
f.add_subplot(2, 2, 1)
for i in range(2, 6):
 sc = SpectralClustering(n_clusters=i).fit(x2d)
 f.add_subplot(2, 2, i-1)
 plt.scatter(x2d[:,0], x2d[:,1], s=5, c=sc.labels_, label="n_cluster-"+str(i))
 plt.legend()

plt.savefig("spec.png")

