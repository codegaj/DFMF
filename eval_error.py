# -*- coding: utf-8 -*-
"""
@author: gaj
"""

import numpy as np
from scipy import signal
e = 1e-8

lrhs = np.load('./reg_results/reg_msi.npy')
hrms = np.load('./reg_results/reg_pan.npy')

start_point = 0
scale = 2

A = np.load('./reg_results/A.npy')
B = np.load('./reg_results/B.npy')
C = np.load('./reg_results/C.npy')
R = np.load('./reg_results/R.npy')

new_hrms = []
for i in range(hrms.shape[-1]):
    temp = signal.convolve2d(hrms[:,:, i], C, boundary='fill',mode='same')
    temp = np.expand_dims(temp, -1)
    new_hrms.append(temp)
new_hrms = np.concatenate(new_hrms, axis=-1)
used_hrms = new_hrms[start_point::scale, start_point::scale, :]

used_lrhs = lrhs*A + B
#used_lrhs = lrhs.copy()
used_lrhs = np.dot(used_lrhs, R)

print(np.min(used_lrhs), np.max(used_lrhs), np.min(used_hrms), np.max(used_hrms))

used_hrms = np.float64(np.uint8(np.clip(used_hrms[4:-4, 4:-4, :], 0, 1)*255))
used_lrhs = np.float64(np.uint8(np.clip(used_lrhs[4:-4, 4:-4, :], 0, 1)*255))

error = np.sqrt(np.mean(np.square(used_hrms-used_lrhs)))
print('error: ', error)

from skimage.metrics import structural_similarity as SSIM

error = SSIM(used_hrms, used_lrhs, data_range=255.0, multichannel=True)
print('error: ', error)
