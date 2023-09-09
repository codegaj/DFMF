# -*- coding: utf-8 -*-
"""
@author: Anjing Guo
"""

import numpy as np
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import cv2
import scipy.io as sio
import random
import os
from tqdm import tqdm
from zsn import DFMFf
import h5py
from scipy import signal

def fusion(hrms, lrhs, sk, scale=None, epoch=50, stride=16, horm='M', fast_mode=True, verbose=1):
    """
    this is an zero-shot learning method with deep learning
    hrms: numpy array with MXNXc
    lrhs: numpy array with mxnxC
        assert: M>>m, N>>n, C>>c
    scale: fusion scale, if None, scale will be set to M//m
    stride: crop stride
    horm: this parameter means the fusion kinds, if 'H', means HSI sharpening, if 'M', means MSI pansharpening
    fast_mode: if True, the deep model is effcient, if False, the deep model can deel with images when they are not registered very well
                but can be a little slow 
    verbose: training verbose, 1 means showing logs, 0 means not showing logs
    """
    
    M, N, c = hrms.shape
    m, n, C = lrhs.shape
    
    start_point = 0
    
    hr_patch_size=60#training patch size
    re_patch_size=48#reconstructing patch size
    left_pad = (hr_patch_size-re_patch_size)//2#pad_size
    
    if scale==None:
        scale = int(np.round(M/m))
        
    print('get sharpening scale: ', scale)
    assert int(np.round(M/m)) == int(np.round(N/n))
    
    train_hrhs_all = []
    train_hrms_all = []
    train_lrhs_all = []
    
    valid_hrhs_all = []
    valid_hrms_all = []
    valid_lrhs_all = []
    
    valid_hrhs = lrhs
    
#    #estimated kernel
    B = sk
#    B = np.multiply(cv2.getGaussianKernel(7,3), cv2.getGaussianKernel(7,3).T)

    used_hrms=[]
    print('downsampling hrms')
    for i in tqdm(range(c)):
        temp_hrms = signal.convolve2d(hrms[:, :, i], B, boundary='symm',mode='same')
        temp_hrms = np.expand_dims(temp_hrms, -1)
        used_hrms.append(temp_hrms)
        
    valid_hrms = np.concatenate(used_hrms, axis=-1)
    valid_hrms = valid_hrms[start_point::scale, start_point::scale]
    
    used_lrhs=[]
    print('downsampling lrhs')
    for i in tqdm(range(C)):
        temp_lrhs = signal.convolve2d(lrhs[:, :, i], B, boundary='symm',mode='same')
        temp_lrhs = np.expand_dims(temp_lrhs, -1)
        used_lrhs.append(temp_lrhs)
        
    valid_lrhs = np.concatenate(used_lrhs, axis=-1)
    valid_lrhs = valid_lrhs[start_point::scale, start_point::scale]
    
    train_hrhs = valid_lrhs.copy()
    
    used_hrms=[]
    for i in tqdm(range(c)):
        temp_hrms = signal.convolve2d(valid_hrms[:, :, i], B, boundary='symm',mode='same')
        temp_hrms = np.expand_dims(temp_hrms, -1)
        used_hrms.append(temp_hrms)
        
    train_hrms = np.concatenate(used_hrms, axis=-1)
    train_hrms = train_hrms[start_point::scale, start_point::scale]
    
    used_lrhs=[]
    print('downsampling lrhs')
    for i in tqdm(range(C)):
        temp_lrhs = signal.convolve2d(valid_lrhs[:, :, i], B, boundary='symm',mode='same')
        temp_lrhs = np.expand_dims(temp_lrhs, -1)
        used_lrhs.append(temp_lrhs)
        
    train_lrhs = np.concatenate(used_lrhs, axis=-1)
    train_lrhs = train_lrhs[start_point::scale, start_point::scale]

    print(train_hrhs.shape, train_lrhs.shape, train_hrms.shape, valid_hrhs.shape, valid_lrhs.shape, valid_hrms.shape)
    
    """crop images"""
    print('croping images...')
    
    for j in range(0, train_hrhs.shape[0]-hr_patch_size, stride):
        for k in range(0, train_hrhs.shape[1]-hr_patch_size, stride):
            
            temp_hrhs = train_hrhs[j:j+hr_patch_size, k:k+hr_patch_size, :]
            temp_hrms = train_hrms[j:j+hr_patch_size, k:k+hr_patch_size, :]
            temp_lrhs = train_lrhs[int(j/scale):int((j+hr_patch_size)/scale), int(k/scale):int((k+hr_patch_size)/scale), :]
            
            train_hrhs_all.append(temp_hrhs)
            train_hrms_all.append(temp_hrms)
            train_lrhs_all.append(temp_lrhs)
            
    train_hrhs_all = np.array(train_hrhs_all, dtype='float16')
    train_hrms_all = np.array(train_hrms_all, dtype='float16')
    train_lrhs_all = np.array(train_lrhs_all, dtype='float16')
    
    index = [i for i in range(train_hrhs_all.shape[0])]
    random.seed(2009)
    random.shuffle(index)
    train_hrhs_all = train_hrhs_all[index, :, :, :]
    train_hrms_all = train_hrms_all[index, :, :, :]
    train_lrhs_all = train_lrhs_all[index, :, :, :]
    
    for j in range(0, valid_hrhs.shape[0]-hr_patch_size, re_patch_size):
        for k in range(0, valid_hrhs.shape[1]-hr_patch_size, re_patch_size):
            
            temp_hrhs = valid_hrhs[j:j+hr_patch_size, k:k+hr_patch_size, :]
            temp_hrms = valid_hrms[j:j+hr_patch_size, k:k+hr_patch_size, :]
            temp_lrhs = valid_lrhs[int(j/scale):int((j+hr_patch_size)/scale), int(k/scale):int((k+hr_patch_size)/scale), :]
            
            valid_hrhs_all.append(temp_hrhs)
            valid_hrms_all.append(temp_hrms)
            valid_lrhs_all.append(temp_lrhs)
            
    valid_hrhs_all = np.array(valid_hrhs_all, dtype='float16')
    valid_hrms_all = np.array(valid_hrms_all, dtype='float16')
    valid_lrhs_all = np.array(valid_lrhs_all, dtype='float16')
    
    index = [i for i in range(valid_hrhs_all.shape[0])]
    random.seed(2009)
    random.shuffle(index)
    valid_hrhs_all = valid_hrhs_all[index, :, :, :]
    valid_hrms_all = valid_hrms_all[index, :, :, :]
    valid_lrhs_all = valid_lrhs_all[index, :, :, :]
    
    """train net"""
    print('training...')
    
    def lr_schedule(epoch):
        """Learning Rate Schedule
    
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
    
        # Arguments
            epoch (int): The number of epochs
    
        # Returns
            lr (float32): learning rate
        """
        lr = 1e-4
        if epoch > 80:
            lr *= 0.5e-3
        elif epoch > 60:
            lr *= 1e-3
        elif epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return lr
    
    save_dir='./fus_results/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=verbose)
    log = CSVLogger('./log1.csv', separator=',', append=False)
    checkpoint = ModelCheckpoint(filepath=save_dir+'best_models.h5',
                             monitor='val_psnr',
                             mode='max',
                             verbose=verbose,
                             save_best_only=True)
    callbacks = [log, lr_scheduler, checkpoint]
    
    if fast_mode:
        model = DFMFf(lrhs_size=(int(hr_patch_size/scale), int(hr_patch_size/scale), C), hrms_size=(hr_patch_size, hr_patch_size, c))

    model.fit( x=[train_lrhs_all, train_hrms_all],
                y=train_hrhs_all,
                validation_data=[[valid_lrhs_all, valid_hrms_all], valid_hrhs_all],
                batch_size=32,
                epochs=epoch,
                verbose=verbose,
                callbacks=callbacks)
    
    model.load_weights(save_dir+'best_models.h5')
    
    del train_hrhs_all
    del train_hrms_all
    del train_lrhs_all
    
    """eval"""
    print('evaling...')
    
    #test
    used_hrms = valid_hrms
    used_lrhs = valid_lrhs

    new_M = min(used_hrms.shape[0], used_lrhs.shape[0]*scale)
    new_N = min(used_hrms.shape[1], used_lrhs.shape[1]*scale)
    
    print(used_lrhs.shape, used_hrms.shape, new_M, new_N)
    
    used_lrhs = np.expand_dims(used_lrhs, 0)
    used_hrms = np.expand_dims(used_hrms, 0)
    
    test_label = np.zeros((new_M, new_N, C), dtype = 'uint8')
    
    used_lrhs = used_lrhs[:, :new_M//scale, :new_N//scale, :]
    used_hrms = used_hrms[:, :new_M, :new_N, :]
    
    print(used_lrhs.shape, used_hrms.shape)
    
    used_lrhs = np.pad(used_lrhs, ((0, 0), (left_pad//scale, hr_patch_size//scale), (left_pad//scale, hr_patch_size//scale), (0, 0)), mode='symmetric')
    used_hrms = np.pad(used_hrms, ((0, 0), (left_pad, hr_patch_size), (left_pad, hr_patch_size), (0, 0)), mode='symmetric')
    
    for h in tqdm(range(0, new_M, re_patch_size)):
        for w in range(0, new_N, re_patch_size):
            temp_lrhs = used_lrhs[:,int(h/scale):int((h+hr_patch_size)/scale), int(w/scale):int((w+hr_patch_size)/scale), :]
            temp_hrms = used_hrms[:, h:h+hr_patch_size, w:w+hr_patch_size, :]
            
            fake = model.predict([temp_lrhs, temp_hrms])
            fake = np.clip(fake, 0, 1)
            fake.shape=(hr_patch_size, hr_patch_size, C)
            fake = fake[left_pad:-left_pad, left_pad:-left_pad]
            fake = np.uint8(fake*255)
            
            if h+hr_patch_size>new_M:
                fake = fake[:new_M-h, :, :]
                
            if w+hr_patch_size>new_N:
                fake = fake[:, :new_N-w, :]
            
            test_label[h:h+re_patch_size, w:w+re_patch_size]=fake
            
    if horm == 'H':
        ci_dir=save_dir+'./GF5-GF1/'
        if not os.path.isdir(ci_dir):
            os.makedirs(ci_dir)
            
        cv2.imwrite(ci_dir+'rec_hsi.bmp', np.uint8(test_label[:, :, [14, 38, 61]]))
        cv2.imwrite(ci_dir+'gt_hsi.bmp', np.uint8(valid_hrhs[:, :, [14, 38, 61]]*255))
        sio.savemat(ci_dir+'rec_hrhs.mat', {'hrhs': np.uint8(test_label)})
        sio.savemat(ci_dir+'val_hrhs.mat', {'hrhs':np.uint8(valid_hrhs*255)})
        
    print('image saved')
        
if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    used_hs = np.load('./reg_results/reg_msi.npy')[4:-4, 4:-4, :]
    used_rgb = np.load('./reg_results/reg_pan.npy')[8:-8, 8:-8, :]
    sk = np.load('./reg_results/C.npy')

    print(used_hs.shape, used_rgb.shape)
    
    '''
    Here, we must give right arguments to the fusion function.
    '''
    fusion(used_rgb, used_hs, sk, horm='H', epoch=50, stride=4, fast_mode=True, verbose=1)