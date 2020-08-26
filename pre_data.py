# coding: utf-8
# 数据预处理
import os
import cv2

all_data = []
all_label = []


def get_data(data, label):
    for i in sorted(os.listdir(data)):
        # print(len(os.listdir(data)))
        file_name = os.path.join(data, i)
        # print(i)
        a = os.path.basename(i)
        # print(a)
        # exit()
        label_name = os.path.join(label, i)
        a = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        # print(a.shape)  # 数据尺度不同
        b = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        # print(b)
        print(type(b))
        print(type(b.max()))
        print((b/255).max())
        exit()
    all_data.append(a)
    all_label.append(b)
    return all_data, all_label


if __name__ == "__main__":
    train_data = "/dicom/Jone/data/TNSCUI2020/TNSCUI2020_train/image"
    train_label = "/dicom/Jone/data/TNSCUI2020/TNSCUI2020_train/mask"
    a, b = get_data(train_data, train_label)
