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
dataSourceBase = '/home/paul/Downloads/ava_vidprep_supportingModels/ESC-50-clone/'
#dataSourceBase = '/home/paul/Downloads/ESC-50-tst2/'

# Total wav records for training the model, will22 be updated by the program
totalRecordCount = 0

# Total classification class for your model (e.g. if you plan to classify 10 different sounds, then the value is 10)
totalLabel = 50

# model parameters for training
batchSize = 128
epochs =10
dataSize = 128
dataSize2 = 256
latent_dim = 256
digitSize = dataSize-2#124

#'''
### FINE SETTINGS###
input_dim = 16384
latent_dim=256
###################
#digitSize = 124

'''
### COARSE SETTINGS###
input_dim = 4096
latent_dim=64
###################
'''

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
            ps = ps[0:dataSize,0:dataSize]
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
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
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

input_shape_b=(dataSize*dataSize)
#input_shape_b=(1024)
input_shape_v=(200)
time_steps =64



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


#***********************************************************************************************************
'''
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
'''
#**********************************************************************************************************


query_input = tf.keras.Input(shape=(input_shape_b,), dtype='int32')
value_input = tf.keras.Input(shape=(input_shape_b,), dtype='int32')
#query_input = tf.keras.Input(shape=input_shape_b)
#value_input = tf.keras.Input(shape=input_shape_b)


#query_input1 = layers.Reshape(target_shape=(128,128),input_shape=(1,dataSize*dataSize))(query_input)
#value_input1 = layers.Reshape(target_shape=(128,128),input_shape=(1, dataSize*dataSize))(value_input)
print('Q,V INPUT SHAPE IS ', query_input.shape, value_input.shape)
###****************************************************
# Embedding lookup.
token_embedding = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=latent_dim)
#token_embedding = tf.keras.layers.Embedding(input_dim=4096, output_dim=64)
# Query embeddings of shape [batch_size, Tq, dimension].
query_embeddings = token_embedding(query_input)
# Value embeddings of shape [batch_size, Tv, dimension].
value_embeddings = token_embedding(value_input)
print('q-embed', query_embeddings.shape)
# CNN layer.
cnn_layer = tf.keras.layers.Conv1D(
    filters=1,
    kernel_size=4,
    # Use 'same' padding so outputs have the same shape as inputs.
    padding='same')
# Query encoding of shape [batch_size, Tq, filters].
query_seq_encoding = cnn_layer(query_embeddings)
print('q-seq-encod', query_seq_encoding.shape)
# Value encoding of shape [batch_size, Tv, filters].
value_seq_encoding = cnn_layer(value_embeddings)


#attention_output, weights = \
#    tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=5)(query=query_embeddings,
#                                                                 value=value_embeddings,
                                                                 #return_attention_scores=True)

#model = tf.keras.Model(inputs=[query_input, value_input],
#                       outputs=[query_embeddings, attention_output])
#names = ('query_embeddings', 'attention_output')
#model.summary()
# Query-value attention of shape [batch_size, Tq, filters].
query_value_attention_seq = tf.keras.layers.Attention()(
    [query_seq_encoding, value_seq_encoding])
print('Q-VAL-ATTN-SQE', query_value_attention_seq.shape)

re_3b = layers.Reshape(target_shape=(dataSize, dataSize,1),input_shape=(None, dataSize, dataSize, 1))(query_value_attention_seq)
print('re_3b', re_3b.shape)

pool_3b = layers.MaxPooling2D((2,2), strides=(1,1))(re_3b)
#pool_3b = layers.MaxPooling2D((2,2), strides=(2,2))(re_3b)
print('pool_3b',pool_3b.shape)

re_3b2 = layers.Reshape(target_shape=(dataSize,dataSize),input_shape=( dataSize, dataSize, 1))(query_value_attention_seq)
print('re_3b', re_3b.shape)

#conv_3b2 = layers.Conv2D(1, (1,1), strides=(2,1), input_shape=input_shape_b)(re_5b)
#print('conv3',conv_3b.shape)

# Reduce over the sequence axis to produce encodings of shape
# [batch_size, filters].
query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
    re_3b2)
query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
    re_3b2)
print('query_encoding is ', query_encoding.shape)
# Concatenate query and document encodings to produce a DNN input layer.
input_layer = tf.keras.layers.Concatenate()(
    [query_encoding, query_value_attention])





re_3ba = layers.Reshape(target_shape=(int(dataSize/16),32),input_shape=(None, dataSize*2))(input_layer)
#re_3ba = layers.Reshape(target_shape=(8,16),input_shape=(None, 128))(input_layer)

###****************************************************
pooled1d = tf.keras.layers.GlobalAveragePooling1D()(
    re_3ba)
print('pooled1d shape is ', re_3ba.shape)

#act_3b =layers.Activation('relu')(conv_3b)
#re_4b = layers.Reshape(target_shape=(10, 10,1),input_shape=(None, 1024))(input_layer)
#################################################
#ls_5b= layers.LSTM(64,return_sequences=True,unit_forget_bias=1.0,dropout=0.1)(re_4b)
####################################
#pool_2b = layers.MaxPooling2D((4,4), strides=(4,4))(conv_1b)
#print(ls_5b.shape)
#re_5b = layers.Reshape(target_shape=(16,16,1),input_shape=(8,32))(re_3ba)
re_5b = layers.Reshape(target_shape=(int(dataSize/8),16,1),input_shape=(int(dataSize/8),32))(re_3ba)

#print(re_5b.shape)
#conv_3b = layers.Conv2D(1, (1,1), strides=(2,1), input_shape=(64,64,1))(re_5b)
#print('conv3',conv_3b.shape)

#re_6b = layers.Reshape(target_shape=(dataSize,dataSize),input_shape=(dataSize,dataSize,1))(conv_3b)
#################################################
#ls_6b= layers.LSTM(64,return_sequences=True,unit_forget_bias=1.0,dropout=0.1)(conv_3b)
#seqa=SeqSelfAttention(attention_activation='sigmoid')(ls_6b)

####################################
#pool_2b = layers.MaxPooling2D((4,4), strides=(4,4))(conv_1b)
#print('ls6b',ls_6b.shape)
#re_7b = layers.Reshape(target_shape=(64,dataSize,1),input_shape=(64,dataSize))(ls_6b)

#x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(re_4b)
#x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(re_5b)
x = layers.Dense(16, activation="relu")(x)
#x = layers.Dense(16, activation="relu")(query_input)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
print ("query input shape is ", query_input.shape, value_input.shape)

encoder = keras.Model([query_input,value_input], [z_mean, z_log_var, z], name="encoder")
#encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
print('latent shape is',latent_inputs.shape)
x = layers.Dense(digitSize*digitSize, activation="relu")(latent_inputs)
x = layers.Reshape((digitSize, digitSize, 1))(x)
#x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
#x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, input_shape=(digitSize,digitSize),kernel_size=(3,3), activation="relu", strides=1, padding="valid")(x)
print('DECODER SHAPE IS :',x.shape)
x = layers.Conv2DTranspose(2, kernel_size=(3,3),  input_shape=(digitSize+2,digitSize+2),  activation="relu", strides=1, padding="valid")(x)
#print('xshp2:',x.shape)
#x = layers.Conv2DTranspose(x.shape[3], input_shape=(dataSize,dataSize),kernel_size=(1,1), activation="relu", strides=(1,1), padding="valid")(x)
print('DECODER2 SHAPE IS :',x.shape)

#decoder_outputs= layers.Reshape((None,dataSize*dataSize))(x)
x= layers.Reshape((x.shape[3],dataSize*dataSize))(x)
decoder_outputs= layers.Reshape((2,1,dataSize*dataSize))(x)
print('DECODER3 SHAPE IS :',decoder_outputs.shape)

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
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)#
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(1, input_shape=(28,28),kernel_size=(3,3), activation="relu", strides=2, padding="valid")(x)
x = layers.Conv2DTranspose(1, kernel_size=(17,17),  input_shape=(30,30),  activation="relu", strides=2, padding="valid")(x)
decoder_outputs = layers.Conv2D(1, (2,2), strides=(1,1))(x)
#'''
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
        print("CONSTRUCTED VAE")

    @property
    def metrics(self):
        print("METRICS DEFINED")
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        #print("IN TRAIN STEP")
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            #print('data , recon = ', np.array(list(data)).shape, np.array(list(reconstruction)).shape)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    #keras.losses.binary_crossentropy(data[0][0], reconstruction)#), axis=(1, 2)
                    #keras.losses.mse(data[0][0], reconstruction), axis=(1, 2)
                    
                    keras.losses.categorical_crossentropy(data[0], reconstruction[0])#, axis=(1, 2)
                    #keras.losses.kl_divergence(data[0][0], reconstruction)#, axis=(1)#, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
                 
        grads =  tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        #print("COMPLETED TRAIN STEP")
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
x_train = np.array(x_train)
x_test = np.array(x_test)
# One-Hot encoding for classes
#y_train = np.array(keras.utils.to_categorical(y_train, totalLabel))#.reshape(1,-1)
#y_test = np.array(keras.utils.to_categorical(y_test, totalLabel))#.reshape(1,-1)
print('x_train, y_train shape is ',x_train.shape, y_train.shape)
x_train *= int(1.0*input_dim/x_train.max())
x_test *= int(1.0*input_dim/x_test.max())
#x_train *= int(1.0/x_train.max())
#x_test *= int(1.0/x_test.max())

mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
print("about to init VAE")
early_stopping_monitor = callbacks.EarlyStopping(
       monitor='loss',
       min_delta=0,
       patience=20,
       verbose=0,
       mode='auto',
       baseline=None,
       restore_best_weights=True)

#filepath = "coarse-model-{epoch:02d}.hdf5"
filepath = "fine3-model-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

print("encoder=",encoder)
print("decoder=",decoder)
vae = VAE(encoder, decoder)
print("about to compile",vae)
numItems = x_train.shape[0]
x_train = x_train.reshape(1,-1)
numvars =x_train.shape[1]
x_train = x_train.reshape(numItems,int(numvars/numItems))
#y_train = y_train.reshape(1,-1)#.reshape(1,-1)
#np.pad(y_train, 128)
#x_train = x_train.reshape( 256, 256, 1)





q1 = np.array([[1, 2, 0]])
q = x_train#np.array([[1, 2, 0]])
print("Q SHAPE IS",q.shape)
'''
print(q1.shape, q.shape)
prediction = model.predict([q, q])  # self-attention

print('\nWITH PADDING')
for n, v in zip(names, prediction):
    print(f'\n{n}:\n{v}')

q = q[:, :-1]  # remove the padding column in this example
prediction = model.predict([q, q])  # self-attention
print('\nWITHOUT PADDING')
for n, v in zip(names, prediction):
    print(f'\n{n}:\n{v}')
'''

vae.compile(optimizer=keras.optimizers.Adam())#, run_eagerly=True)#, loss="categorical_crossentropy")
#vae.compile(loss="categorical_crossentropy")
print('compiled, about to fit')
#vae.fit([q,q], epochs=epochs, batch_size=32, callbacks=[early_stopping_monitor],steps_per_epoch=64) #,validation_data=(x_test, None))

#vae.fit([[query_input,value_input],decoder_outputs], epochs=epochs, batch_size=32, callbacks=[early_stopping_monitor],steps_per_epoch=64) #,validation_data=(x_test, None))
#qroll = np.roll(q,1)
vae.fit([q,q], epochs=epochs , batch_size=1,callbacks=[early_stopping_monitor, checkpoint])#,validation_data=(x_test, None)) #,validation_data=(x_test, None))
#vae.fit(mnist_digits, epochs=epochs, batch_size=32,callbacks=[early_stopping_monitor])#,validation_data=(x_test, None))
vae.save_weights('vae_mlp_mnist_latent_dim_%s.h5' %latent_dim)
#vae.save('vae_coarse_model.h5')
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
    z_mean8, _, _ = vae.encoder.predict([[data, data]])
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
print(y_train.shape)#, y_train)
plot_label_clusters(vae, x_train, y_train)
#plot_latent_space(vae)

