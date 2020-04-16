import os
import shutil


class Preprocessor:
    def __init__(self):
        self.data_dir = '../../Datasets/cifar-10'
        self.label_file = 'trainLabels.csv'
        self.train_dir = 'train'
        self.valid_ratio = 0.1  # 一部分做训练一部分做验证

        self.idx_label = None
        self.label_count = dict()
        self.num_train_turning_per_label = None

    def read_label_file(self):
        with open(os.path.join(self.data_dir,self.label_file),'r') as f:

            lines = f.readlines()[1:]  # 跳过文件头行
            tokens = [l.rstrip().split(',') for l in lines]  # 先用rstrip去掉多余的空格换行等
            self.idx_label = dict((int(idx),label) for idx, label in tokens)
            labels = set(self.idx_label.values())

            num_train = len(os.listdir(os.path.join(self.data_dir, self.train_dir)))  # 获取训练集的大小
            num_train_turning = int(num_train * (1 - self.valid_ratio))
            self.num_train_turning_per_label = num_train_turning // len(labels)
            # print(labels)
            # print(self.num_train_turning_per_label)

if __name__ == '__main__':
    test = Preprocessor()
    test.read_label_file()

