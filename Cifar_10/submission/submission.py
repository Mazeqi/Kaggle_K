from utils.loader import get_test_provider
import argparse
from network.ResNet import ResNet18
from torch import  nn
import torch
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def submit(args):
    test_data, id_to_class = get_test_provider(args.batch_size, args.data_dir, args.test_dir, args.train_dir)
    model = ResNet18(num_class=10)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path)['state_dict'])
    model.eval()

    model = model.to(device)

    sub_label = list()
    sub_id = list()
    for i , data in enumerate(test_data):
        img, img_name = data
        img = img.to(device)

        with torch.no_grad():
            scores = model(img)

        label = scores.argmax(dim=1).cpu().numpy()
        labels = [id_to_class[i] for i in label]
        sub_label.extend(labels)
        sub_id.extend(img_name.numpy())

    Data = {'id':sub_id, 'label':sub_label}
    DataFrame = pd.DataFrame(Data)

    DataFrame.to_csv('../checkpoints/submission.csv', index=False)


def set_args():
    parser = argparse.ArgumentParser(description='the args of submission')
    parser.add_argument('--batch_size',type=int, default=128,help='the batch size of test')
    parser.add_argument('--data_dir',type=str, default='../../Datasets/cifar-10',help='the dir of all dataset')
    parser.add_argument('--test_dir',type=str, default='test',help='the dir of test data')
    parser.add_argument('--train_dir',type=str, default='train_data',help='the dir of train data')
    parser.add_argument('--model_path',type=str, default='../checkpoints/model_best.pth.tar')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    submit(args)