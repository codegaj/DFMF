#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:18:05 2023

@author: ubuntu
"""
import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, Input, Layer, LeakyReLU, GlobalAveragePooling2D, ReLU

def tf_flatten(a):
    """Flatten tensor"""
    return tf.reshape(a, [-1])


def tf_repeat(a, repeats, axis=0):
    """TensorFlow version of np.repeat for 1D"""
    assert len(a.get_shape()) == 1

    a = tf.expand_dims(a, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a


def tf_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.get_shape()) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a


def tf_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates

    Note that coords is transposed and only 2D is supported

    Parameters
    ----------
    input : tf.Tensor. shape = (s, s)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1

    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[:, 0], coords_rb[:, 1]], axis=1)
    coords_rt = tf.stack([coords_rb[:, 0], coords_lt[:, 1]], axis=1)

    vals_lt = tf.gather_nd(input, coords_lt)
    vals_rb = tf.gather_nd(input, coords_rb)
    vals_lb = tf.gather_nd(input, coords_lb)
    vals_rt = tf.gather_nd(input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]

    return mapped_vals


def tf_batch_map_coordinates(input, coords, order=1):
    """Batch version of tf_map_coordinates

    Only supports 2D feature maps

    Parameters
    ----------
    input : tf.Tensor. shape = (b, s, s, c)
    coords : tf.Tensor. shape = (b, n_points, 2)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]
    n_c = input_shape[3]
    n_coords = tf.shape(coords)[1]

    coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)
    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.ceil(coords), 'int32')

#    coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
#    coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)
    #new
    coords_lb = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)
    coords_rt = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)

    idx = tf_repeat(tf.range(batch_size), n_coords)

    def _get_vals_by_coords(input, coords):
        indices = tf.stack([
            idx, tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1])
        ], axis=-1)
        vals = tf.gather_nd(input, indices)
        vals = tf.reshape(vals, (batch_size, n_coords, n_c))
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt)
    vals_rb = _get_vals_by_coords(input, coords_rb)
    vals_lb = _get_vals_by_coords(input, coords_lb)
    vals_rt = _get_vals_by_coords(input, coords_rt)
    
    print(vals_rt.get_shape().as_list())

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    print(coords_offset_lt.get_shape().as_list())
    
    coords_offset_lt = tf.expand_dims(coords_offset_lt, -2)
    coords_offset_lt = tf.tile(coords_offset_lt, (1, 1, 1, 1))
    
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 1]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 1]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 0]
    
#    print(mapped_vals.get_shape().as_list())

    return mapped_vals


def tf_batch_map_offsets(input, offsets, order=1):
    """Batch map offsets into input

    Parameters
    ---------
    input : tf.Tensor. shape = (b, s, s, c)
    offsets: tf.Tensor. shape = (b, s, s, 2)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]

    offsets = tf.reshape(offsets, (batch_size, -1, 2))
    grid = tf.meshgrid(
        tf.range(input_size), tf.range(input_size), indexing='ij'
    )
    grid = tf.stack(grid, axis=-1)
    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 2))
    grid = tf_repeat_2d(grid, batch_size)
    coords = offsets + grid

    mapped_vals = tf_batch_map_coordinates(input, coords)
    return mapped_vals

class resample(Layer):
    """offset"""

    def __init__(self, **kwargs):
        """Init"""

        super(resample, self).__init__(**kwargs)

    def call(self, inputs):

        offsets = inputs[0]
        inputs = inputs[1]
        

        b,h,w,c = inputs.get_shape().as_list()
        x = tf.reshape(inputs, (-1, h, w, c))
        
        x_offset = tf_batch_map_offsets(x, offsets)

        x_offset = tf.reshape(x_offset, (-1, h, w, c))
        self.out_size = x_offset.get_shape().as_list()
        
        return x_offset

    def compute_output_shape(self, input_shape):
        return tuple(self.out_size)
