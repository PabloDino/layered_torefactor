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
#from keras.layers import  Input, Dense,Activation, Conv2D,\
#	 MaxPooling2D, Reshape
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import math
import random
import keras.optimizers as ko
import librosa
import librosa.display
import pandas as pd
import warnings
import os

"""
## Create a sampling layer
"""
conv_param   = (8, 8, 128)
# Your data source for wav files
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0
input_dim =4096
latent_dim=32
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 50

# model parameters for training
batchSize = 1#32
epochs = 10
digitSize = 124

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
dataSize = 128
#latent_dim = 256
input_shape_b=(dataSize,dataSize,1)
#######################################################################
##FINE ENCODER/DECODER
#######################################################################
#'''
def dolEncoder(in_shape, latent_dim, conv_params):
    kernel_size = (conv_params[0], conv_params[1])
    n_filters = conv_params[2]
    dft_dim = in_shape[1]
    inp = layers.Input(in_shape)
    loc = layers.Conv2D(n_filters, strides = (1, 1), kernel_size=kernel_size, activation='relu', padding='same')(inp) 
    loc = layers.MaxPool2D(pool_size=(1, dft_dim))(loc) 
    loc = layers.Reshape((in_shape[0], n_filters))(inp) 
    x   = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True))(loc)
    x   = layers.LSTM(latent_dim)(x)            
    return keras.Model(inputs =[inp], outputs=[x])



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
    x   = layers.Conv2DTranspose(1, kernel_size=(1, 1), activation='linear', padding='same')(x) 
    return keras.Model(inputs = [inp], outputs = [x])
    
    
def dolauto_encoder(in_shape, latent_dim, conv_params):
    enc = dolEncoder(in_shape, latent_dim, conv_params)
    dec = dolDecoder(in_shape[0], latent_dim, in_shape[1], conv_params)
    inp = layers.Input(in_shape)
    x   = enc(inp) 
    x   = dec(x) 
    model = keras.Model(inputs = [inp], outputs = [x])
    return model, enc, dec

    
def auto_encoder(in_shape, latent_dim):
    enc = origEncoder(in_shape)
    dec = origDecoder(latent_dim)
    inp = layers.Input(in_shape)
    x   = enc(inp) 
    x   = dec(x) 
    model = keras.Model(inputs = [inp], outputs = [x])
    return model, enc, dec


def code64(in_shape, latent_dim, conv_params):
    
    inputLayer = layers.Input(shape=(input_dims))
    #print('INDIMS in',inputLayer.shape)
    #h = layers.Dense(latent_dim, activation="relu")(inputLayer)
    #print('HCODEEEEEEEEEEEEEEEEEE in',h.shape)
    #h = layers.Dense(64, activation="relu")(h)
    #h = layers.Dense(8, activation="relu")(h)
    #print('HCODEEEEEEEEEEEEEEEEEE out',h.shape)
    #h = layers.Flatten()(h)
    #x = layers.Dense(16, activation="relu")(x)
    kernel_size = (conv_params[0], conv_params[1])
    n_filters = conv_params[2]
    dft_dim = in_shape[1]
    h   = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True))(inputLayer)

    #h = layers.Dense(latent_dim, activation="relu")(h)
    return keras.Model(inputs =[inputLayer], outputs=[h]),h
    
def decode64(input_dims,decode_dims):
    #print('HINDECODEEEEEEEEEEEEEEEEEE out',decode_dims)
    
    
    latent_inputs = keras.Input(shape=(decode_dims))
    h   = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True))(latent_inputs)

    #inputLayer = layers.Input(shape=(decode_dims[1],decode_dims[2],decode_dims[3]))
    #print('HOINLyr',inputLayer.shape)

    #h = layers.Dense(latent_dim, activation="relu")(latent_inputs)

    #h = layers.Dense(64, activation="relu")(latent_inputs)
    #h = layers.Dense(64, activation="relu")(h)
    #print('HOUTED64',h.shape)

    #h = layers.Dense(latent_dim, activation="relu")(h)
    #h = layers.Flatten()(h)
    #h   = layers.Reshape((128,128,1))(h)
    #h   = layers.Reshape(input_dims)(h)

    #print('HOUTECODEEEEEEEEEEEEEEEEEE out',h.shape)

    return keras.Model(inputs =[latent_inputs], outputs=[h])

def auto64(input_dims):
    enc, h = code64(input_dims)
    dec = decode64(input_dims,h.shape)
    inp = layers.Input(input_dims)
    print('INP SHAPE IS ', inp)
    x   = enc(inp) 
    print('x1 SHAPE IS ', x)
    x   = dec(x) 
    print('x2 SHAPE IS ', x)
    model = keras.Model(inputs = [inp], outputs = [x])
    return model, enc, dec


def origEncoder(input_shape_b):    
   encoder_inputs = layers.Input(shape=input_shape_b)
   #token_embedding = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=latent_dim)
   #query_embeddings = token_embedding(encoder_inputs)
   #conv_0b = layers.Conv2D(1, (1,1), strides=(1,1), input_shape=input_shape_b)(query_embeddings)
   #print('qe', query_embeddings.shape)
   re_0b = layers.Reshape(target_shape=(dataSize,dataSize,1),input_shape=(1,latent_dim))(encoder_inputs)
   mx1 = layers.MaxPool2D(pool_size=(2, 2))(re_0b) 
   
   #conv_1b = layers.Conv2D(1, (3,3), strides=(1,1), input_shape=input_shape_b, padding="same")(conv_0b)
   #conv_1b = layers.Conv2D(1, (3,3), strides=(1,1), input_shape=input_shape_b)(encoder_inputs)
   #print('conv1b', conv_1b.shape)
   #conv_3b = layers.Conv2D(1, (3,3), strides=(1,1), input_shape=input_shape_b)(conv_1b)
   print('mx',mx1.shape)
   act_3b =layers.Activation('relu')(mx1)
   print('conv3b',act_3b.shape)
   #re_4b = layers.Reshape(target_shape=(int(digitSize),int(digitSize)),input_shape=(1,int(digitSize),int(digitSize)))(act_3b)
   re_4b = layers.Reshape(target_shape=(int(dataSize/2),int(dataSize/2)),input_shape=(1,int(dataSize/2),int(dataSize/2)))(act_3b)
   
   x   = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True,dropout =0.4))(re_4b)
   #x   = layers.LSTM(16, return_sequences=True,dropout =0.2)(x)    
 
   x = layers.Flatten()(x)
   #x = layers.Dense(16, activation="relu")(x)
   x = layers.Dense(latent_dim, activation="relu")(x)
   #z_mean = layers.Dense(latent_dim, name="z_mean")(x)
   #z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
   #z = Sampling()([z_mean, z_log_var])
   #encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
   encoder = keras.Model(encoder_inputs, x, name="encoder")
   encoder.summary()
   return encoder
   
def origDecoder(latent_dim):
   print('latent dim =', latent_dim)
   latent_inputs = keras.Input(shape=(latent_dim,))
  
   x = layers.Dense(digitSize*digitSize, activation="relu")(latent_inputs)
   x = layers.Reshape(target_shape=(digitSize,digitSize),input_shape=((digitSize*digitSize,None)))(x)
   #x   = layers.LSTM(16, return_sequences=True,dropout =0.2)(x)
   
   x   = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True, dropout=0.4))(x)
   print('blout',x.shape)
   
   x = layers.Reshape((digitSize,latent_dim*2, 1))(x)
    
   conv_1b = layers.Conv2D(1, (1,1), strides=(1,2), input_shape=input_shape_b)(x)
   print('conv1bd', conv_1b.shape)
   conv_3b = layers.Conv2D(1, (1,1), strides=(1,2), input_shape=input_shape_b)(conv_1b)
   print('conv3bd',conv_3b.shape)

   x = layers.Conv2DTranspose(1, input_shape=(digitSize,digitSize),kernel_size=(3,1), activation="relu", strides=1, padding="valid")(conv_3b)
   print('c2d',x.shape)
  
   decoder_outputs = layers.Conv2DTranspose(1, kernel_size=(3,1),  input_shape=(digitSize+2,digitSize+2),  activation="relu", strides=1, padding="valid")(x)
   decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
   decoder.summary()
   return decoder
   
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
            #print(type(data))
            #print(data.shape)
            #data = np.array(data)
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

filepath = "dve-model-{epoch:02d}-{loss:.2f}.hdf5"
#checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


"""
## Train the VAE
"""
#testdata = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = importData()#keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
mnist_digits = np.expand_dims(mnist_digits,axis=0).astype("float32") / 255
#mnist_digits = np.expand_dims(mnist_digits)
print("about to init VAE")

#encoder = origEncoder(input_shape_b)
#decoder = origDecoder(latent_dim)

#encoder = dolEncoder(input_shape_b, latent_dim, conv_param)#origEncoder(input_shape_b)
#decoder = dolDecoder(latent_dim, latent_dim,latent_dim, conv_param)#origDecoder(latent_dim)
#'''



x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test= np.array(y_test)
x_train = tf.expand_dims(x_train,-1)
x_test = tf.expand_dims(x_test,-1)
#print('XTRAIN shape is ', x_train.shape)
#model, enc, dec = auto_encoder(input_shape_b, latent_dim)
model, enc, dec = dolauto_encoder(input_shape_b, latent_dim,conv_param)
#model, enc, dec = auto64(input_shape_b)
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
#hist = model.fit(x=np.array(x_train), y=np.array(y_train), validation_data=(x_test, y_test), batch_size=batchSize, epochs=epochs, shuffle=True)

hist = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=batchSize, epochs=epochs, shuffle=True)
model.save_weights('modelblstm1_latent_dim_%s.h5' %latent_dim)
enc.save('encoder_latent_dim_%s.h5' %latent_dim)

import matplotlib.pyplot as plt


def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 128
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    pca = PCA(n_components=2)
    #pca_result = pca.fit_transform(df[feat_cols].values)

    '''
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            
    '''        
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit
    #'''
    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    #plt.show()
    filename="vae.png" 
    plt.savefig(filename)



#plot_latent_space(vae)

"""
## Display how the latent space clusters different digit classes
"""


def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    #plt.show()
    plt.savefig("clusters.png")

def plot_enc_clusters(enc, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean = enc.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    #plt.show()
    plt.savefig("clusters.32.blstm.png")



#(x_train, y_train), _ = keras.datasets.mnist.load_data()
(x_train, y_train), _ = importData()#keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

#plot_label_clusters(model, x_train, y_train)
plot_enc_clusters(enc, x_train, y_train)
#plot_latent_space(vae)

