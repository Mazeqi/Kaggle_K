import sys
import argparse
from torch.backends import cudnn
from utils.loader import get_data_provider
from torch import optim
from torch import nn
from network.ResNet import ResNet18
from utils.solver import solver
import logging
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORMAT = '[%(levelname)s]: %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    stream=sys.stdout
)


def train(args):
    (train_data, valid_data, train_class_data) = get_data_provider(args.batch_size,args.data_dir,
                                                                   args.train_dir,args.valid_dir,
                                                                   args.train_class_dir
                                                                  )
    print('finish load data')
    print('use :', device)
    model = ResNet18(num_class=10)
    model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    trainer = solver(model, device)
    trainer.net_trian(train_data, valid_data, optimizer, criterion, args.num_epochs, args.print_interval,
                      args.eval_step, args.save_step, args.save_model_dir
                      )


def set_arg():
    parser = argparse.ArgumentParser(description='cifar10 model training')
    parser.add_argument('--batch_size',type=int, default=128, help='trainng batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='trainng learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='trainng weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum training')
    parser.add_argument('--num_epochs',type=int, default=100, help='training epochs')
    parser.add_argument('--print_interval', type=int, default=300, help='how many iterations to print')
    parser.add_argument('--eval_step',type=int, default=20, help='how many epochs to ecaluate')
    parser.add_argument('--save_step',type=int, default=20, help='how many epochs to save model')
    parser.add_argument('--save_model_dir', type=str, default='checkpoints', help='the directory to save model')
    parser.add_argument('--use_gpu',action='store_true',help='decide if user gpu trianing')
    parser.add_argument('--data_dir',type=str, default='../Datasets/cifar-10',help='the directory to save data')
    parser.add_argument('--train_dir',type=str,default='train_data', help='the directory to save train_data')
    parser.add_argument('--valid_dir',type=str,default='valid_data',help='the directory to save valid_data')
    parser.add_argument('--train_class_dir',type=str,default='train_class',help='the directory to save all data that have class')
    args = parser.parse_args()

    return args


if __name__ =='__main__':
    args = set_arg()
    train(args)




