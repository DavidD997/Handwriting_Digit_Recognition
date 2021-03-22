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

## Read Matlab file
x = loadmat("SVHN-test_32x32.mat")


# ## train code  
# train_x = x["X"]
# train_y = x["y"]
# train_imgs = np.asfarray(train_x)
# train_gts = np.asfarray(train_y)
# # for i in range(1):
# for i in range (train_imgs.shape[3]):
#      img = train_imgs[i].reshape((32, 32,3))
#      img = img.astype(np.uint8)
#      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#      img = cv2.resize(img, (28, 28))

#      img_cls = int(train_gts[i])
#      if img_cls == 10:
#           img_cls = 0
#      cv2.imwrite("./SVHN/trainSet/{}/{}.png".format(img_cls,i), img)
#      print("Training: image saved: {} of class {}".format(i, img_cls))

## test code  
train_x = x["X"]
train_y = x["y"]
train_imgs = np.asfarray(train_x)
train_gts = np.asfarray(train_y)
# for i in range(1):
for i in range (train_imgs.shape[3]):
     img = train_imgs[:,:,:,i].reshape((32, 32,3))
     img = img.astype(np.uint8)
     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     img = cv2.resize(img, (28, 28))

     img_cls = int(train_gts[i])
     if img_cls == 9:
          img_cls = 9
          cv2.imwrite("./SVHN/testSet//{}/{}.png".format(img_cls,i), img)
          print("Testing: image saved: {} of class {}".format(i, img_cls))