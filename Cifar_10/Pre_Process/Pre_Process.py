import os
import shutil


class Preprocessor:
    def __init__(self):
        self.data_dir = '../../Datasets/cifar-10'
        self.label_file = 'trainLabels.csv'
        self.train_dir = 'train'
        self.valid_ratio = 0.1  # 一部分做训练一部分做验证
        self.train_class_dir = 'train_class'  # 将图片分成10类

        self.idx_label = None  # 图片的下标和标签
        self.label_count = dict()
        self.num_train_turning_per_label = None

        # self.read_label_file()

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

    def mkdir_if_not_exist(self, path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # 根据调参的结果，将45000张照片放到trian_data中，将5000张验证图片放到valid_data中
    # 并且将每一类图片都整一个文件夹，十类图片分到十个文件夹中
    def reorg_train_valid(self):
        self.read_label_file() # 加载文件夹中的图片
        for train_file in os.listdir(os.path.join(self.data_dir, self.train_dir)):
            # print(train_file)
            idx = int(train_file.split('.')[0])
            label = self.idx_label[idx]
            self.mkdir_if_not_exist([self.data_dir, self.train_class_dir, label])
            shutil.copy(os.path.join(self.data_dir,self.train_dir, train_file),
                        os.path.join(self.data_dir,self.train_class_dir,label)
                        )

            if label not in self.label_count or self.label_count[label] < self.num_train_turning_per_label:
                self.mkdir_if_not_exist([self.data_dir, 'train_data', label])
                shutil.copy(os.path.join(self.data_dir, self.train_dir, train_file),
                            os.path.join(self.data_dir, 'train_data', label)
                            )
                self.label_count[label] = self.label_count.get(label, 0) + 1
            else:
                self.mkdir_if_not_exist([self.data_dir, 'valid_data', label])
                shutil.copy(os.path.join(self.data_dir, self.train_dir, train_file),
                            os.path.join(self.data_dir, 'valid_data', label)
                            )


if __name__ == '__main__':
    test = Preprocessor()
    # test.reorg_train_valid()

