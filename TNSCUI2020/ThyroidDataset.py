# coding=utf-8
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import cv2
import os


########################load dataset ###########################
################################################################


def load_data(data, label):
    # all_data = []
    # all_label = []
    all = []
    for file in sorted(os.listdir(data)):
        f_data = os.path.basename(file)
        file_name = os.path.join(data, file)
        # label_name = os.path.join(label, f_data)
        label_name = os.path.join(label, file)

        a = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

        # all_data.append(a)
        # all_label.append(b)
        all.append((a, b))

    # return all_data, all_label
    return all


train_data = "/dicom/Jone/data/TNSCUI2020/TNSCUI2020_train/image"
train_label = "/dicom/Jone/data/TNSCUI2020/TNSCUI2020_train/mask"

data = load_data(train_data, train_label)

# random.shuffle(data)

# 将数据划分为训练集和测试集
train_dataset = data[: int(len(data)*0.9)]
test_dataset = data[int(len(data)*0.9)]


class ThyroidDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # 这里可以对数据进行处理
        data = self.data[index]
        image = data[0]
        label = data[1]
        return image, label

    def __len__(self):
        return len(self.data)

train_dataset = ThyroidDataset(train_dataset)
test_dataset = ThyroidDataset(test_dataset)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True)


for data_ in train_loader:
    print(data_[0])
    break