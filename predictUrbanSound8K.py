import numpy as np
import pandas as pd
import librosa
import sys
import glob
import os
from keras.models import load_model

def predict():
    df = pd.read_csv('dict50.csv')
    #print('df',df)
    #model = load_model('models/urban-sound.h5')
    #model = load_model('models/sound-classification.h5')
    model = load_model('models/esc50-2.h5')

    #wavFiles = glob.glob("predict/*.wav")
    dirlist = os.listdir("/home/paul/Downloads/ESC-50-clone/")
    #print ("dirlist is ",dirlist)
    for dr in dirlist:
      #print("dr is ","/*.wav")
      wavFiles = glob.glob("/home/paul/Downloads/ESC-50-clone/"+dr+"/*.wav")

      for wavFile in wavFiles:
        y, sr = librosa.load(wavFile, duration=2.97)
        # exract features
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        if ps.shape != (128, 128):
            print('Skip wav file:{} because data shape is not (128, 128)'.format(wavFile))
            continue

        dataSet = []
        dataSet.append(ps)

        # reshape data to 128 x 128
        dataSet = np.array([data.reshape( (128, 128, 1) ) for data in dataSet])

        predictions = model.predict(dataSet)[0]
        #print('pred:',predictions,';df:',df)
        print('============= Predict wav {} ============='.format(wavFile))
        for index, predict in enumerate(predictions):

            resultStr = '{0} {1:.2f}%'.format(df.iloc[index,1], predict * 100)
            print(resultStr)
        predictClass = model.predict_classes(dataSet)
        print('Result for ',format(wavFile), ':',format(df.iloc[predictClass[0],1]))
        print('============= Predict End =============')

if __name__ == '__main__':
    predict()
