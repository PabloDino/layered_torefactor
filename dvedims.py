"""
Title: Variational AutoEncoder
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/03
Last modified: 2020/05/03
Description: Convolutional Variational AutoEncoder (VAE) trained on MNIST digits.
"""

"""
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from keras_self_attention import SeqSelfAttention
#from keras.layers import  Input, Dense,Activation, Conv2D,\
#	 MaxPooling2D, Reshape
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

import random
import keras.optimizers as ko
import librosa
import librosa.display
import pandas as pd
import warnings
import os
import time
"""
## Create a sampling layer
"""

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
epochs =100
digitSize = 124
dataSize = 128
latent_dim = 256

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



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


"""
## Build the encoder
"""

input_shape_b=(dataSize,dataSize,1)
#######################################################################
##FINE ENCODER/DECODER
#######################################################################
#'''
encoder_inputs = layers.Input(shape=input_shape_b)
#conv_1b = layers.Conv2D(1, (3,3), strides=(1,1), input_shape=input_shape_b)(encoder_inputs)
#print('conv1b', conv_1b.shape)
# Using CNN to build model
# 24 depths 128 - 5 + 1 = 124 x 124 x 24   
# 98x98x24    


#act_3b =layers.Activation('relu')(conv_3b)
re_4b = layers.Reshape(target_shape=(dataSize,dataSize),input_shape=(dataSize,dataSize,1))(encoder_inputs)
#################################################
ls_5b= layers.LSTM(64,return_sequences=True,unit_forget_bias=1.0,dropout=0.1)(re_4b)
####################################
#pool_2b = layers.MaxPooling2D((4,4), strides=(4,4))(conv_1b)
print(ls_5b.shape)
#re_5b = layers.Reshape(target_shape=(64,dataSize,1),input_shape=(64,dataSize))(ls_5b)

#print(re_5b.shape)
#conv_3b = layers.Conv2D(1, (2,2), strides=(1,1), input_shape=input_shape_b)(re_5b)
#print('conv3',conv_3b.shape)

#re_6b = layers.Reshape(target_shape=(dataSize,dataSize),input_shape=(dataSize,dataSize,1))(conv_3b)
#################################################
ls_6b= layers.LSTM(64,return_sequences=True,unit_forget_bias=1.0,dropout=0.1)(ls_5b)
seqa=SeqSelfAttention(attention_activation='sigmoid')(ls_6b)

####################################
#pool_2b = layers.MaxPooling2D((4,4), strides=(4,4))(conv_1b)
print('ls6b',ls_6b.shape)
re_7b = layers.Reshape(target_shape=(64,dataSize,1),input_shape=(64,dataSize))(seqa)

#x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(re_4b)
#x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(seqa)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(digitSize*digitSize, activation="relu")(latent_inputs)
x = layers.Reshape((digitSize, digitSize, 1))(x)
#x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
#x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(1, input_shape=(digitSize,digitSize),kernel_size=(3,3), activation="relu", strides=1, padding="valid")(x)
decoder_outputs = layers.Conv2DTranspose(1, kernel_size=(3,3),  input_shape=(digitSize+2,digitSize+2),  activation="relu", strides=1, padding="valid")(x)
#'''
##########################################



#######################################################################
##COARSE ENCODER/DECODER
#######################################################################
'''
encoder_inputs = layers.Input(shape=input_shape_b)
print('encoder_inputs', encoder_inputs.shape)
re_1b = layers.Reshape(target_shape=(128,128),input_shape=(128,128,1))(encoder_inputs)
ls2a= layers.LSTM(32,return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(re_1b)
#rv2 = layers.RepeatVector(8)(ls2a)
#ls2b= layers.LSTM(32,return_sequences=True,unit_forget_bias=1.0,dropout=0.2)(ls2a)

re_2b = layers.Reshape(target_shape=(64,64,1),input_shape=(32,128))(ls2a)
print('re_2b', re_2b.shape)
conv_1b = layers.Conv2D(1, (7,7), strides=(2,2), input_shape=input_shape_b)(re_2b)
print('conv_1b', conv_1b.shape)									

# Using CNN to build model
# 24 depths 128 - 5 + 1 = 124 x 124 x 24   
# 98x98x24    

conv_3b = layers.Conv2D(1, (2,2), strides=(1,1), input_shape=input_shape_b)(conv_1b)
#print(conv_3b.shape)
#act_3b =layers.Activation('relu')(conv_3b)
re_4b = layers.Reshape(target_shape=(28,28,1),input_shape=(1,28,28))(conv_3b)

####################################

x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(re_4b)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
print('x:', x.shape)

print('z:',z.shape)
print('z_mean:',z_mean.shape)
print('z_log_var:',z_log_var.shape)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(1, input_shape=(28,28),kernel_size=(3,3), activation="relu", strides=2, padding="valid")(x)
x = layers.Conv2DTranspose(1, kernel_size=(17,17),  input_shape=(30,30),  activation="relu", strides=2, padding="valid")(x)
decoder_outputs = layers.Conv2D(1, (2,2), strides=(1,1))(x)
'''
##########################################

 
############################################
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

"""
## Define the VAE as a `Model` with a custom `train_step`
"""


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


"""
## Train the VAE
"""
#testdata = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = importData()#keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
print("about to init VAE")
early_stopping_monitor = callbacks.EarlyStopping(
       monitor='loss',
       min_delta=0,
       patience=20,
       verbose=0,
       mode='auto',
       baseline=None,
       restore_best_weights=True)
       
filepath = "vae-model-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)



vae = VAE(encoder, decoder)
print("about to compile")
vae.compile(optimizer=keras.optimizers.Adam())
print('compiled, about to fit')
vae.fit(mnist_digits, epochs=epochs, batch_size=32,callbacks=[early_stopping_monitor,checkpoint])#,validation_data=(x_test, None))
vae.save_weights('vae_mlp_mnist_latent_dim_%s.h5' %latent_dim)
encoder.save('encoder.fine.h5')
"""
## Display a grid of sampled digits
"""

import matplotlib.pyplot as plt


#pca_result = plot_latent_space(vae)

"""
## Display how the latent space clusters different digit classes
"""


def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean8, _, _ = vae.encoder.predict(data)
    ###################################################
    #pca = PCA(n_components=2)
    #z_mean = pca.fit_transform(z_mean8)
    ####################################################
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
    z_mean = tsne.fit_transform(z_mean8)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    #plt.show()
    plt.savefig("clusters.png")



#(x_train, y_train), _ = keras.datasets.mnist.load_data()
#(x_train, y_train), _ = importData()#keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(vae, x_train, y_train)
#plot_latent_space(vae)

