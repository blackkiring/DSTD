#-*- coding :GBK-*-
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from libtiff import TIFF
import torch.nn.functional as F
import torch
import numpy as np
############# Ҫ�õ��ĺ��� #############
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


def split(pan, size):
    st = []
    for i in range(size):
        for j in range(size):
            st.append(pan[i::size,j::size])
    return np.array(st)


def image_gradient(img):
    H, W = img.shape
    gradient = np.zeros([H, W])
    gx = 0
    gy = 0
    for i in range(H - 1):
        for j in range(W - 1):
            gx = img[i + 1, j] - img[i, j]
            gy = img[i, j + 1] - img[i, j]
            gradient[i, j] = np.sqrt(gx**2 + gy**2)

    return gradient


def edge_dect(img):
    nam=1e-9
    apx=1e-10
    return np.exp( -nam / ( (image_gradient(img)**4)+apx ) )


def get_gram(msf, pan):
    res = np.array(range(16)).reshape((4,4)).astype('float32')
    for i in range(4):
        for j in range(4):
            res[i][j] = torch.sum(msf[i]*pan[j])
    return res

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def sign(x):
    x[x>=0] =  1
    x[x< 0] = -1
    return x 


chk = np.zeros(128).astype('int32')
ans = np.zeros(128).astype('int32')
plzh= []
def find_plzh(x):
    if x>3:
        plzh.append( [ans[0],ans[1],ans[2],ans[3]] )
        return 
    
    for i in range(4):
        if chk[i]==0:
            ans[x] = i
            chk[i] = 1
            find_plzh(x+1)
            chk[i] = 0

