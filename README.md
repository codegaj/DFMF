#Image registration and blur kernel learning

(1) Download GF5-GF1-HHK.zip from https://pan.baidu.com/s/1KOiKHKXxeiYEnBrObf1rMg?pwd=dfmf, and unzip them to  "./"

(2) Run reg_GF5_GF1.py, then check if there are A.npy, B.npy, C.npy, R.npy, reg_pan.npy, and reg_msi.npy in "./reg_results/"

#Image fusion

(1) Run simu_fusion.py, and the results are saved at "./fus_results/GF5_GF1/"

#Note

(1) If you only need the registered HSI and MSI for evaluating the HSI fusion performance of your methods, you can also download an advanced
version of the registered images at https://pan.baidu.com/s/1m1e0Nf7alBeEERezHeGDnA?pwd=dfmf, which are obtained by a few latest algorithm improvements and can be better suited for fusion tasks.

--A.npy and B.npy: two mapping matrix

--C.npy: spatial kernel

--R.npy: spectral kernel

--reg_msi.npy: registered hsi

--reg_pan.npy: registered msi

(2) The reg_GF5_GF1.py can occasionally crash due to excessive initial gradients.

(3) If you have downloading troubles about the above-mentioned links, please email me: anjing_guo@hnu.edu.cn.

(4) We appreciate the original providers of the datasets, and the above dataset can only be used for academic purposes.

#Device

Nvidia 1080ti GPU + 64GB RAM

#Enviroments

ubuntu 16.04 + cuda 9.0 + python 3.6

#Packages

numpy==1.16.4

tensorflow-gpu==1.12.0

keras==2.2.4

opencv==3.4.2

tqdm==4.31.1

matplotlib==3.0.3

scipy==0.19.1

h5py==2.9.0
