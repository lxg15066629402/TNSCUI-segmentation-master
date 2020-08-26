# # coding: utf-8
# import torch.utils.data as data
# import torch
#
# from scipy.ndimage import imread
# import os
# import os.path
# import glob
#
# import numpy as np
#
# from torchvision import transforms
#
#
# def make_dataset(path, train=True):
#
#   dataset = []
#
#   if train:
#     # path = "/Users/Jone/Downloads/Code/data/TNSCUI2020/TNSCUI2020_train"
#     data = "image"
#     label = "mask"
#
#     data_path = os.path.join(path, data)
#     label_path = os.path.join(label, data)
#
#     for file in os.listdir(data_path):
#
#       f_data = os.path.basename(file)
#       f_label = f_data + '.PNG'
#
#       dataset.append([os.path.join(data_path, file), os.path.join(label_path, f_label)])
#
#   return dataset
#
#
# class Data_Procss(data.Dataset):
#   """
#   前列腺数据集
#   """
#
#   def __init__(self, root, transform=None, train=True):
#     self.train = train
#
#     # cropped the image
#     self.nRow = 400
#     self.nCol = 560
#
#     if self.train:
#       self.train_set_path = make_dataset(root, train)
#
#   def __getitem__(self, idx):
#     if self.train:
#       img_path, gt_path = self.train_set_path[idx]
#
#       img = imread(img_path)
#       img = img[0:self.nRow, 0:self.nCol]
#       img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
#       img = (img - img.min()) / (img.max() - img.min())
#       img = torch.from_numpy(img).float()
#
#       gt = imread(gt_path)[0:self.nRow, 0:self.nCol]
#       gt = np.atleast_3d(gt).transpose(2, 0, 1)
#       gt = gt / 255.0
#       # gt = torch.from_numpy(gt).float()
#
#       return img, gt
#
#   def __len__(self):
#     if self.train:
#       return 3360
#     else:
#       return 0

import torch.utils.data as data
import PIL.Image as Image
import os
from skimage import transform
import numpy as np
import cv2


# 获取数据
def make_dataset(img, label):
  data = []
  for i in sorted(os.listdir(img)):

    image = os.path.join(img, i)
    mask = os.path.join(label, i)
    data.append((image, mask))

  return data


# def get_data(imgs):
#     for index in range(len(imgs)):
#
#         x_path, y_path = imgs[index]
#         img_x = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
#         img_y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
#         # 数据归一化处理
#         img, mask = normalize(img_x, img_y)


def normalize(image, label):


    # image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image = transform.resize(image.astype(np.float32), (512, 512))  # size need to set up d

    label = transform.resize(label.astype(np.uint8), (512, 512))
    #
    image = (image - np.mean(image)) / (np.std(image))

    image = np.array(image)
    label = np.array(label)

    image = image[np.newaxis, :]
    label = label[np.newaxis, :]

    label = (label).astype(np.uint8)

    # print(label)
    return image, label


# 使用 pytorch 深度学习框架方式读取数据
class TNSCUIDataset(data.Dataset):
  def __init__(self, img, label):
    imgs = make_dataset(img, label)
    self.imgs = imgs
    # print(self.imgs)

  def __getitem__(self, index):
    x_path, y_path = self.imgs[index]
    img_x = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
    img_y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
    # 数据归一化处理
    img, mask = normalize(img_x, img_y)

    return img, mask

  def __len__(self):
    return len(self.imgs)


# data = "/dicom/Jone/data/TNSCUI2020/train/data"
# label = "/dicom/Jone/data/TNSCUI2020/train/seg"
#
# # TNSCUIDataset(data, label)
# imgs = make_dataset(data, label)
# get_data(imgs)