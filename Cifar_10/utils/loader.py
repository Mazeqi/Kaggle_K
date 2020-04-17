from torchvision import transforms
from torchvision import datasets
import os
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


# 图像增广
def get_data_provider(batch_size, data_dir, train_dir, test_dir, train_class_dir):
    train_T = transforms.Compose([
        transforms.Resize(40), # 40 * 40
        transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    # 保证输出的确定性只对图像做标准化
    test_T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, train_dir), train_T)
    test_data  = datasets.ImageFolder(os.path.join(data_dir, test_dir), test_T)
    train_class_data = datasets.ImageFolder(os.path.join(data_dir, train_class_dir), train_T)

    train_data = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
    test_data = DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)
    train_class_data = DataLoader(train_class_data, batch_size, shuffle=True, pin_memory=True)

    return (train_data, test_data, train_class_data)


# 显示图片
def Imgshow():
    plt.figure()
    i = 1
    unloader = transforms.ToPILImage()
    for img, label in train_data:
        img = img.squeeze(0)
        img = unloader(img)
        plt.subplot(4,8, i)
        plt.imshow(img)
        plt.axis('off')
        plt.title(str(label))
        i += 1
        if(i >= 33):break
    plt.show()


if __name__ == '__main__':

    data_dir = '../../Datasets/cifar-10'
    train_dir = 'train_data'
    train_class_dir = 'train_class'  # 将图片分成10类
    test_dir = 'valid_data'

    (train_data, test_data, train_class_data) = get_data_provider(1, data_dir, train_dir, test_dir, train_class_dir)
    Imgshow()