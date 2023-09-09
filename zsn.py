# -*- coding: utf-8 -*-
"""
@author: Anjing Guo
"""

import numpy as np
from keras.models import Model
from keras.layers import Concatenate, UpSampling2D, Conv2D, Input, Layer, Add, Activation, Reshape,LeakyReLU, Lambda, Multiply, Conv2DTranspose
from keras.layers import Subtract, SeparableConv2D, DepthwiseConv2D
from keras.optimizers import Adam, SGD
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import tensorflow as tf
from keras import initializers
#from keras.initializers import RandomNormal
from keras.utils import plot_model

###############################################################################

def psnr(y_true, y_pred):
    """Peak signal-to-noise ratio averaged over samples and channels."""
    mse = K.mean(K.square(K.cast(K.cast(K.clip(y_true*255, 0, 255), 'int32'), 'float64') - K.cast(K.cast(K.clip(y_pred*255, 0, 255), 'int32'), 'float64') ), axis=(-3, -2, -1))
    return K.mean(20 * K.log(255 / K.sqrt(mse)) / np.log(10))

def psnr_ac(y_true, y_pred):
    """Peak signal-to-noise ratio averaged over samples and channels."""
    mse = K.mean(K.square(K.cast(K.cast(K.clip(y_true*255, 0, 255), 'int32'), 'float64') - K.cast(K.cast(K.clip(y_pred*255, 0, 255), 'int32'), 'float64') ), axis=(-2, -1))
    return K.mean(K.mean(20 * K.log(255 / K.sqrt(mse)) / np.log(10), axis=-1), axis=0)
    
class resize_s(Layer):
    def __init__(self, target_size,
                 **kwargs):
        self.target_size = (target_size[0], target_size[1])
        super(resize_s, self).__init__(**kwargs)

    def call(self, inputs):

        temp = tf.image.resize_bilinear(inputs, self.target_size, align_corners=True)

        return temp

    def compute_output_shape(self, input_shape):
#        return self.out_shape
        return (input_shape[0], self.target_size[0], self.target_size[1], input_shape[3])
    
    def get_config(self):
        config = super(resize_s, self).get_config()
        return config
    
class resize_c(Layer):
    def __init__(self, target_size,
                 **kwargs):
        self.target_size = target_size[2]
        super(resize_c, self).__init__(**kwargs)

    def call(self, inputs):        
        _, self.h, __, ___ = inputs.get_shape().as_list()
        
        inputs = tf.transpose(inputs, (0, 3, 1, 2))
        temp = tf.image.resize_bilinear(inputs, (self.target_size, self.h), align_corners=True)
        temp = tf.transpose(temp, (0, 2, 3, 1))
        return temp

    def compute_output_shape(self, input_shape):
#        return self.out_shape
        return (input_shape[0], input_shape[1], input_shape[2], self.target_size)
    
    def get_config(self):
        config = super(resize_c, self).get_config()
        return config


def upsample(x, scale):
    def upsample_1(x, factor, **kwargs):
        x = Lambda(lambda b: tf.depth_to_space(b, factor))(x)
        return x

    if scale == 2:
        x = upsample_1(x, 2)
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')
        
    return x

def downsample(x, scale):
    def upsample_1(x, factor, **kwargs):
        x = Lambda(lambda b: tf.space_to_depth(b, factor))(x)
        return x

    if scale == 2:
        x = upsample_1(x, 2)
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')
        
    return x

def conv_block1(inputs, nf=128, block_name='1'):
    
    conv1 = Conv2D(nf, (3, 3), strides=(1, 1), padding='same', name=block_name+'_1_1')(inputs)
    
    conv1 = Activation('relu')(conv1)
    
    conv2 = Conv2D(nf, (3, 3), strides=(1, 1), padding='same', name=block_name+'_2')(conv1)

    outputs = Add()([inputs, conv2])
    return outputs

class pc(Layer):
    def __init__(self, kernel_initializer='glorot_uniform',
                 **kwargs):
        super(pc, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]
        self.input_c = input_shape[3]

        self.pc = self.add_weight(shape=[self.input_h, self.input_w, self.input_c],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs):
        
        shape = tf.shape(inputs)
        
        pc = K.expand_dims(self.pc, 0)
        
        F = K.tile(pc, (shape[0], 1, 1, 1))

        new_temp = inputs+F
        
        self.out_s = new_temp.get_shape().as_list()
        
        return new_temp
        
    def compute_output_shape(self, input_shape):
        return tuple(self.out_s)
    
    def get_config(self):
        config = super(pc, self).get_config()
        return config
    

def DFMFf(lrhs_size=(8, 8, 31), hrms_size = (64, 64, 3), include_top=True):
    
    lrhs_inputs = Input(lrhs_size)
    hrms_inputs = Input(hrms_size)
    
    lrhs_inputs1 = resize_s(hrms_size)(lrhs_inputs)
    hrms_inputs1 = resize_c(lrhs_size)(hrms_inputs)
    
    mixed1 = Concatenate()([lrhs_inputs1, hrms_inputs1])
    
    mixed1 = Conv2D(80, (1, 1), strides=(1, 1), padding='same', activation='relu')(mixed1)
    
    a = pc()
    b = pc()
       
    mixed1 = downsample(mixed1, 2)
#        
    mixed1 = a(mixed1)
 
    sc = 80*4
    
    mixed1 = conv_block1(mixed1, nf=sc, block_name='1')
    
    mixed1 = conv_block1(mixed1, nf=sc, block_name='2')
    
    mixed1 = conv_block1(mixed1, nf=sc, block_name='3')
    
    mixed1 = conv_block1(mixed1, nf=sc, block_name='4')

    mixed1 = conv_block1(mixed1, nf=sc, block_name='5')
    
    mixed1 = b(mixed1)
    
    mixed1 = upsample(mixed1, 2)

    mixed2 = Conv2D(lrhs_size[2], (3, 3), strides=(1, 1), padding='same')(mixed1)
    
    c6 = Add()([lrhs_inputs1, mixed2])
    
    model = Model(inputs = [lrhs_inputs, hrms_inputs], outputs = c6)

    model.compile(optimizer = Adam(lr = 5e-4), loss = 'mae', metrics=[psnr])
    
    model.summary()

    return model

################################################################################
if __name__=="__main__":
    pup = DFMFf(lrhs_size=(15, 15, 48), hrms_size = (60, 60, 4))
    plot_model(pup, to_file='DFMFf.png', show_shapes=True)
