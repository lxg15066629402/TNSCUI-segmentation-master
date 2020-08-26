# coding: utf-8
"""
数据的获取过程，数据的处理通过 pytorch 框架的 Dataloader 类实现
"""

from os.path import exists, join
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale
import torch.utils.data as data
from os import listdir
from PIL import Image


# 获取数据信息
def bsd500(dest="/dicom/Jone/data/TNSCUI2020"):

    if not exists(dest):
        print("dataset not exist ")
    return dest


# 数据的转化过程
def input_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor()
    ])


# 获取训练集数据
def get_training_set(size, target_mode='seg', colordim=1):
    root_dir = bsd500()
    train_dir = join(root_dir, "train")
    return DatasetFromFolder(train_dir, target_mode, colordim,
                             input_transform=input_transform(size),
                             target_transform=input_transform(size))


# 获取测试集数据
def get_test_set(size, target_mode='seg', colordim=1):
    root_dir = bsd500()
    test_dir = join(root_dir, "test")
    return DatasetFromFolder(test_dir, target_mode, colordim,
                             input_transform=input_transform(size),
                             target_transform=input_transform(size))


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", '.PNG'])


def load_img(filepath, colordim):
    if colordim == 1:
        img = Image.open(filepath).convert('L')
    else:
        img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, target_mode, colordim, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [x for x in listdir(join(image_dir, 'data')) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.image_dir = image_dir
        self.target_mode = target_mode
        self.colordim = colordim

    def __getitem__(self, index):

        input = load_img(join(self.image_dir, 'data', self.image_filenames[index]), self.colordim)
        if self.target_mode == 'seg':
            target = load_img(join(self.image_dir, 'seg', self.image_filenames[index]), 1)
        else:
            target = load_img(join(self.image_dir, 'bon', self.image_filenames[index]), 1)

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)