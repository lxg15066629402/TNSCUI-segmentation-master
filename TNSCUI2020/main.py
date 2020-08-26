# coding: utf-8

import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from net import Unet
from dataset import TNSCUIDataset
import matplotlib.pyplot as plt

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 把多个步骤整合到一起, channel=（channel-mean）/std, 因为是分别对三个通道处理
# x_transforms = transforms.Compose([
#     transforms.ToTensor(),  # -> [0,1]
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
# ])

x_transforms = transforms.ToTensor()
# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

# 参数解析器,用来解析从终端读取的命令
parse = argparse.ArgumentParser()


def train_model(model, criterion, optimizer, dataload, num_epochs=2):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            # x.type(torch.FloatTensor)
            # 转Float
            inputs = x.type(torch.FloatTensor).to(device)
            # labels = y.to(device)
            # print(labels)
            # exit()
            labels = y.type(torch.FloatTensor).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            # print(outputs)
            # print(labels)
            # exit()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model


# 训练模型
def train(data, label):
    model = Unet(1, 1).to(device)
    batch_size = args.batch_size
    criterion = torch.nn.BCELoss()  # 损失函数
    # criterion = torch.nn.BCELoss()  # 损失函数
    optimizer = optim.Adam(model.parameters())  # 优化器
    liver_dataset = TNSCUIDataset(data, label)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)


# 显示模型的输出结果
def test(test_data, test_label):
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    liver_dataset = TNSCUIDataset(test_data, test_label)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()

    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.pause(0.01)
        plt.show()


if __name__ == "__main__":
    data = "/dicom/Jone/data/TNSCUI2020/train/data"
    label = "/dicom/Jone/data/TNSCUI2020/train/seg"
    test_data = "/dicom/Jone/data/TNSCUI2020/test/data"
    test_label = "/dicom/Jone/data/TNSCUI2020/test/seg"
    parse = argparse.ArgumentParser()
    # parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=1)
    # parse.add_argument("--ckp", type=str, help="the path of model weight file", default="/dicom/Jone/data/TNSCUI2020/model")
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()

    # train
    train(data, label)

    # test()
    # args.ckp = "weights_19.pth"
    # test()