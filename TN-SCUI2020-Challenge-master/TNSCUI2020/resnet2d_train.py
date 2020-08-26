# coding: utf-8

import os
from tensorflow.python.client import device_lib
from Resnet2d.model_resNet2d import ResNet2dModule
import numpy as np
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    # csv file should have the type:
    # label,data_npy
    # label,data_npy
    # ....
    #
    train_augcvs = pd.read_csv(r'D:\cjq\project\python\TNSCUI2020\dataprocess\classifyaugtraindata.csv')
    train_csv = pd.read_csv(r'D:\cjq\project\python\TNSCUI2020\dataprocess\classifytraindata.csv')

    train_aug = train_augcvs.iloc[:, :].values
    train = train_csv.iloc[:, :].values
    train_Data = np.concatenate((train_aug, train), axis=0)
    np.random.shuffle(train_Data)
    # For Image
    trainimages = train_Data[:, 1]
    trainlabels = train_Data[:, 0]
    ResNet3d = ResNet2dModule(512, 512, channels=1, n_class=2, costname="cross_entropy")
    ResNet3d.train(trainimages, trainlabels, "resnet.pd", "log\\classify\\resnet\\", 0.001, 0.5, 10, 16)


if __name__ == "__main__":
    train()
