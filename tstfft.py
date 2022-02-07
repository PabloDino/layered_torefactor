import numpy
import tensorflow as tf
K = tf.keras

inputs = K.Input(shape=(10, 8), name='main_input')
x = K.layers.Lambda(tf.spectral.rfft)(inputs)
decoded = K.layers.Lambda(tf.spectral.irfft)(x)
model = K.Model(inputs, decoded)
fqmodel = K.Model(inputs, x)
output = model(tf.ones([10, 8]))
fqval= fqmodel(tf.ones([10, 8]))
with tf.Session():
  print(fqval.eval())
  print(output.eval())
