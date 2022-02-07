import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
#from keras.layers import  Input, Dense,Activation, Conv2D,\
#	 MaxPooling2D, Reshape
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

# Your data source for wav files
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-aug/'
#dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'

# Total wav records for training the model, will be updated by the program
totalRecordCount = 0
input_dim =4096
latent_fine_dim=256

latent_coarse_dim=32
# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 50

# model parameters for training
batchSize = 128
epochs = 50#00
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
input_mbed_shape=(dataSize,dataSize,1,latent_dim)
#######################################################################
##FINE ENCODER/DECODER
#######################################################################
#'''
encoder_inputs = layers.Input(shape=input_shape_b)
token_embedding = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=latent_dim)
query_embeddings = token_embedding(encoder_inputs)
#conv_0b = layers.Conv2D(1, (1,1), strides=(1,1), input_shape=input_shape_b)(query_embeddings)
print('qe', query_embeddings.shape)
re_0b = layers.Reshape(target_shape=input_mbed_shape,input_shape=(1,latent_dim))(query_embeddings)

#conv_1b = layers.Conv2D(1, (3,3), strides=(1,1), input_shape=input_shape_b, padding="same")(conv_0b)
conv_1b = layers.Conv2D(1, (3,3), strides=(1,1), input_shape=input_shape_b)(encoder_inputs)
print('conv1b', conv_1b.shape)
# Using CNN to build model
# 24 depths 128 - 5 + 1 = 124 x 124 x 24   
# 98x98x24    

#pool_2b = layers.MaxPooling2D((4,4), strides=(4,4))(conv_1b)
#print(pool_2b.shape)
conv_3b = layers.Conv2D(1, (3,3), strides=(1,1), input_shape=input_shape_b)(conv_1b)
print(conv_3b.shape)
act_3b =layers.Activation('relu')(conv_3b)
re_4b = layers.Reshape(target_shape=(digitSize,digitSize,1),input_shape=(1,digitSize,digitSize))(act_3b)

####################################

#x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(re_4b)
#x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(re_4b)
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
conv_1b = layers.Conv2D(1, (11,11), strides=(4,4), input_shape=input_shape_b)(encoder_inputs)
print('conv1b', conv_1b.shape)
# Using CNN to build model
# 24 depths 128 - 5 + 1 = 124 x 124 x 24   
# 98x98x24    

#pool_2b = layers.MaxPooling2D((4,4), strides=(4,4))(conv_1b)
#print(pool_2b.shape)
conv_3b = layers.Conv2D(1, (3,3), strides=(1,1), input_shape=input_shape_b)(conv_1b)
print(conv_3b.shape)
act_3b =layers.Activation('relu')(conv_3b)
re_4b = layers.Reshape(target_shape=(28,28,1),input_shape=(1,28,28))(act_3b)

####################################

x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(re_4b)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
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

import matplotlib.pyplot as plt

def plot_label_clusters(vae, data, labels,f):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    #plt.show()
    plt.savefig(f+".png")

## Train the VAE
(x_train, y_train), (x_test, y_test) = importData()#keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

vae = VAE(encoder, decoder)
print("about to compile")
#vae.compile(optimizer=keras.optimizers.Adam())
print('compiled, about to fit')
vae.built=True
lst = os.listdir()
mList=[]
for f in lst:
   if ((f.endswith("hdf5")) and (f.startswith("dve-model"))):
      mList.append(f)
     
for f in mList:
    print("loading for ", f)
    vae.load_weights(f) 
    plot_label_clusters(vae, x_train, y_train,f)

