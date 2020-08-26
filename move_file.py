import os
import shutil


# 创建迭代的文件夹
def create_dir_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def move_train(src, dis):

    # print(sorted(os.listdir(src)))
    # print(sorted(os.listdir(src))[:3000])
    # exit()
    for file in sorted(os.listdir(src))[:3000]:
        file_name = os.path.join(src, file)  # 拼接得到文件路径
        shutil.copy(file_name, dis)  # 移动文件


def move_test(src, dis):

    for file in sorted(os.listdir(src))[3000:]:
        file_name = os.path.join(src, file)  # 拼接得到文件路径

        shutil.copy(file_name, dis)  # 移动文件


if __name__ == "__main__":
    data = "/dicom/Jone/data/TNSCUI2020/TNSCUI2020_train/image"
    mask = "/dicom/Jone/data/TNSCUI2020/TNSCUI2020_train/mask"

    train_data = "/dicom/Jone/data/TNSCUI2020/train/data"
    test_data = "/dicom/Jone/data/TNSCUI2020/test/data"

    train_seg = "/dicom/Jone/data/TNSCUI2020/train/seg"
    test_seg = "/dicom/Jone/data/TNSCUI2020/test/seg"

    create_dir_not_exists(train_data)
    create_dir_not_exists(test_data)
    create_dir_not_exists(test_seg)
    create_dir_not_exists(train_seg)

    move_train(data, train_data)
    move_train(mask, train_seg)
    move_test(data, test_data)
    move_test(mask, test_seg)