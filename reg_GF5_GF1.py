# -*- coding: utf-8 -*-
"""
@author: gaj
"""

import numpy as np
import tensorflow as tf
import cv2
import keras.backend as K
import scipy.io as sio
import time
from keras.optimizers import Adam,  SGD
import random
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, Input, Layer, LeakyReLU, GlobalAveragePooling2D, ReLU
from keras.models import Model
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from Resample import resample
e = 1e-8

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

###############################################################################

def tv_loss1(image):
    shape = tuple(image.get_shape().as_list())
    tv_loss = 2*(K.mean(K.abs(image[:,1:,:,:] - image[:,:shape[1]-1,:,:])) + K.mean(K.abs(image[:,:,1:,:] - image[:,:,:shape[2]-1,:])))

    return tv_loss

###############################################################################
import tifffile as tif
lrhs = np.float64(tif.imread('./GF5-GF1/GF5-2018-HHK.tif'))
lrhs = lrhs/np.max(lrhs)
lrhs = np.transpose(lrhs, (1, 2, 0))
lrhs = np.expand_dims(lrhs, 0)
hrms = np.float64(tif.imread('./GF5-GF1/GF1-2018-HHK.tif'))
hrms = hrms/np.max(hrms)
hrms = np.transpose(hrms, (1, 2, 0))
hrms = cv2.resize(hrms, (2322, 2258))
hrms = np.expand_dims(hrms, 0)

ch_h = 150
ch_m = 4

stride = 16#

lrhs = np.pad(lrhs, ((0, 0), (5, 5), (5, 5), (0, 0)), 'constant')
train_lrhs_all=[]
train_hrms_all=[]
for ji, j in enumerate(range(0, hrms.shape[1]-60, stride)):
    for ki, k in enumerate(range(0, hrms.shape[2]-60, stride)):
        temp_hrms = hrms[0, ji*stride:ji*stride+60, ki*stride:ki*stride+60, :]
        temp_lrhs = lrhs[0, ji*int(stride/2):ji*int(stride/2)+40, ki*int(stride/2):ki*int(stride/2)+40, :]

        train_lrhs_all.append(temp_lrhs)
        train_hrms_all.append(temp_hrms)
        
train_lrhs_all = np.array(train_lrhs_all, dtype='float16')
train_hrms_all = np.array(train_hrms_all, dtype='float16')

###############################################################################
  
def spa_k(sigma, k_size=(7,7)):
    
    l = []
    for i in range(k_size[0]):
        for j in range(k_size[1]):
            temp_x = i-k_size[0]//2
            temp_y = j-k_size[1]//2
            temp = 1/(2*np.pi*(sigma)**2)*tf.exp(-(temp_x**2+temp_y**2)/(2*sigma**2))
            l.append(temp)
    l = K.concatenate(l, -1)
    l = K.reshape(l, k_size)
    return l

sigma = K.variable([[1.0]])
half_k = K.variable(1.5)

a = np.random.random((ch_h, ch_m))
a = a/np.sum(a, axis=0)
spek = K.variable(a)

spek1 = K.abs(spek)
spek1 = spek1/K.sum(spek1, axis=0)
#spek1 = spek

mapm = np.random.normal(1, 0.005, (ch_h,))
mapm = K.variable(mapm)

mapa = np.random.normal(0, 0.005, (ch_h,))
mapa = K.variable(mapa)

#######################################################################

all_spak = spa_k(sigma, k_size=(25, 25))

f_hk = tf.cast(tf.floor(half_k), 'int32')
c_hk = tf.cast(tf.ceil(half_k), 'int32')

weight = half_k - tf.cast(f_hk, 'float32')

f_k = all_spak[12-f_hk:12+f_hk+1, 12-f_hk:12+f_hk+1]
c_k = all_spak[12-c_hk:12+c_hk+1, 12-c_hk:12+c_hk+1]

f_k = tf.pad(f_k, [[1, 1], [1, 1]])

spak = f_k*(1-weight)+weight*c_k

spak = spak/K.sum(spak)

spak1 = K.expand_dims(spak, -1)
spak1 = K.expand_dims(spak1, -1)
spak1 = K.tile(spak1, (1, 1, ch_m, 1))

ilrhs = K.placeholder((None, 40, 40, ch_h))
ihrms = K.placeholder((None, 60, 60, ch_m))

inloc = tf.pad(ihrms, ((0, 0), (10, 10), (10, 10), (0, 0)), mode='CONSTANT')
inloc = tf.image.resize_bicubic(inloc, (40, 40))
inloc = K.concatenate((K.mean(ilrhs, axis=-1, keepdims=True), inloc))

def locnet(use_bias=False, trainable=True):
    inloc1 = Input(shape=(40, 40, 5))
    
    fo1 = Conv2D(32, (5, 5), dilation_rate=2, use_bias=use_bias, padding='same', trainable=trainable)(inloc1)
    fo1 = ReLU()(fo1)
    fo1 = Conv2D(32, (5, 5), dilation_rate=2, use_bias=use_bias, padding='same', trainable=trainable)(fo1)
    fo1 = ReLU()(fo1)
    
    fo1 = MaxPool2D()(fo1)
    
    fo1 = Conv2D(32, (5, 5), dilation_rate=2, use_bias=use_bias, padding='same', trainable=trainable)(fo1)
    fo1 = ReLU()(fo1)
    fo1 = Conv2D(32, (5, 5), dilation_rate=2, use_bias=use_bias, padding='same', trainable=trainable)(fo1)
    fo1 = ReLU()(fo1)
    
    of = Conv2D(2, (5, 5), padding='same', use_bias=use_bias, trainable=trainable)(fo1)
    
    return Model(inputs = inloc1, outputs = of)

locnet = locnet()
offset1 = locnet(inloc)
offset1 = tf.image.resize_bicubic(offset1, (40, 40))

meanx = K.mean(K.mean(offset1[:, 5:-5, 5:-5, 0:1], axis=(1,2)))
meany = K.mean(K.mean(offset1[:, 5:-5, 5:-5, 1:], axis=(1,2)))

tv = tv_loss1(offset1[:, 5:-5, 5:-5, :])
o_lrhs = resample()([offset1, ilrhs])
o_lrhs = o_lrhs[:, 5:-5, 5:-5, :]

o_lrhs1 = o_lrhs*mapm+mapa#Map
#o_lrhs1 = o_lrhs

pre_lrhs = K.dot(o_lrhs1, spek1)#FC

pre_hrms = K.depthwise_conv2d(ihrms, spak1, strides=(1, 1), padding='same')#Conv
pre_hrms = pre_hrms[:, 0::2, 0::2, :]

loss = (1-K.mean(tf.image.ssim(pre_lrhs, pre_hrms, max_val=1.0)))+0.0005*(half_k+sigma)+0.01*tv

updates = locnet.trainable_weights
updates.append(sigma)
updates.append(half_k)
updates.append(spek)
updates.append(mapm)
updates.append(mapa)

training_updates = Adam(lr=0.005, beta_1=0.0, beta_2=0.9).get_updates(loss, updates)
    
f = K.function([ilrhs, ihrms], [loss, half_k, sigma, spak, spek1, offset1, meanx, meany, tv], training_updates)

f1 = K.function([ilrhs, ihrms], [spek1, spak, sigma, meanx, o_lrhs, offset1[:, 5:-5, 5:-5, :], mapm, mapa])

###############################################################################

bs = 32
epoch=20
ks = []
for i in range(epoch):
    tl = []
    index = [i for i in range(train_lrhs_all.shape[0])]
    random.shuffle(index)
    for j in range(train_hrms_all.shape[0]//bs):
        
        start_time = time.time()
        l, hfk, sigma, spak, spe, x1, meanx, meany, u = f([train_lrhs_all[index[j*bs:(j+1)*bs], ...], train_hrms_all[index[j*bs:(j+1)*bs], ...]])
            
        if j%20==0:
            print('kernel: ', hfk, 'sigma: ', sigma, 'meanx: ', meanx, 'meany: ', meany, 'tv: ', u)

        tl.append(l)
        end_time = time.time()
#        print('Iteration %d completed in %ds' % (j, end_time - start_time))
        
    ks.append(spak)
    
    print('epoch: %d, train_loss: %f, kernel size: %d'%(i, sum(tl)/(train_hrms_all.shape[0]//bs), hfk))

    if i == epoch-1:
        spek, spak, sigma, _, __, ___, mapm, mapa = f1([train_lrhs_all[(j-2)*bs:(j-1)*bs, ...], train_hrms_all[(j-2)*bs:(j-1)*bs, ...]])
        
        save_dir='./reg_results/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        print('sigma: ', sigma[0])
        np.save(save_dir+'C', spak)
        np.save(save_dir+'R', spek)
        np.save(save_dir+'A', mapm)
        np.save(save_dir+'B', mapa)
        
        rec_size = 60
#        rec_size = 24
        
        rec_h = len([i for i in range(0, hrms.shape[1]-60, rec_size)])
        rec_w = len([i for i in range(0, hrms.shape[2]-60, rec_size)])
        
        rec_pan = np.zeros((1, rec_h*rec_size, rec_w*rec_size, ch_m))
        rec_msi = np.zeros((1, rec_h*int(rec_size/2), rec_w*int(rec_size/2), ch_h))
        
#        here register the lrhs image
        for ji, j in enumerate(tqdm(range(0, hrms.shape[1]-60, rec_size))):
            for ki, k in enumerate(range(0, hrms.shape[2]-60, rec_size)):
                temp_hrms = hrms[:, ji*rec_size:ji*rec_size+60, ki*rec_size:ki*rec_size+60, :]
                temp_lrhs = lrhs[:, ji*int(rec_size/2):ji*int(rec_size/2)+40, ki*int(rec_size/2):ki*int(rec_size/2)+40, :]
                
                _, __, ___, ____, temp, temp_off, mapm, mapa = f1([temp_lrhs, temp_hrms])
                
                rec_pan[:, ji*rec_size:ji*rec_size+rec_size, ki*rec_size:ki*rec_size+rec_size, :] = temp_hrms[:, int((60-rec_size)/2):60-int((60-rec_size)/2), int((60-rec_size)/2):60-int((60-rec_size)/2), :]
                rec_msi[:, ji*int(rec_size/2):ji*int(rec_size/2)+int(rec_size/2), ki*int(rec_size/2):ki*int(rec_size/2)+int(rec_size/2), :] = temp[:, int((60-rec_size)/2/2):30-int((60-rec_size)/2/2), int((60-rec_size)/2/2):30-int((60-rec_size)/2/2), :]
                
        np.save(save_dir+'reg_pan', rec_pan[0, :, :, :])
        np.save(save_dir+'reg_msi', rec_msi[0, :, :, :])
#        
        cv2.imwrite(save_dir+'pan_ori.tif', np.uint8(rec_pan[0,:,:,:3]*255))
        cv2.imwrite(save_dir+'reg_msi_resize.tif', cv2.resize(np.uint8(rec_msi[0,:,:,15]*255), (rec_pan.shape[2], rec_pan.shape[1]))) 
