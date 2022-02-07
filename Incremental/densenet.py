from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, LSTM, ConvLSTM2D, Reshape, Concatenate, Input

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
        #print('x shape is ', x.shape) 
       
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

