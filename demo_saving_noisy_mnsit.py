import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import cv2

batch_size = 64
learning_rate = 1e-3  # 0.001
num_epoches = 3
image_size = 28

# Read Matlab file
x = loadmat("mnist-with-motion-blur.mat")
train_x = x["train_x"]
train_y = x["train_y"]

## Train Code
train_imgs = np.asfarray(train_x)
train_gts = np.asfarray(train_y)
# for i in range(1):
for i in range (train_imgs.shape[0]):
     img = train_imgs[i].reshape((28, 28))
     img_cls = np.where(train_gts[i]==1)[0][0]
     cv2.imwrite("/Users/daviddai/Downloads/NMNIST_Combine/trainSet/{}/{}.png".format(img_cls,i), img) 
     print("Training: image saved: {} of class {}".format(i, img_cls))

## Test Code 
train_x = x["test_x"]
train_y = x["test_y"]

train_imgs = np.asfarray(train_x)
train_gts = np.asfarray(train_y)
# for i in range(1):
for i in range (train_imgs.shape[0]):
     img = train_imgs[i].reshape((28, 28))
     img_cls = np.where(train_gts[i]==1)[0][0]
     cv2.imwrite("/Users/daviddai/Downloads/NMNIST_Combine/testSet/{}/{}.png".format(img_cls,i), img)
     print("Testing: image saved: {} of class {}".format(i, img_cls))