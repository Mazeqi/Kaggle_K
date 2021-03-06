import math
import numpy as np
import time
import logging
import torch
from utils.serilization import save_checkpoint


class solver(object):
    def __init__(self,model, device):
        self.net = model
        self.loss_sum = AverageValueMeter()
        self.acc_sum  = AverageValueMeter()
        self.device = device
        self.net = self.net.to(device)

    def net_trian(self, train_data, valid_data, optimizer, criterion, num_epochs=100, print_interval=100,
                  eval_step=50, save_step=10, save_model_dir='checkpoints'):
        best_valid_acc = -np.inf

        for epoch in range(num_epochs):
            print('epoch:', epoch)
            self.loss_sum.reset()
            self.acc_sum.reset()

            tic = time.time()
            btic = time.time()
            for i, data in enumerate(train_data):
                imgs, labels = data
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                scores = self.net(imgs)
                loss = criterion(scores, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.loss_sum.add(loss.cpu().item())
                acc = (scores.argmax(dim=1) == labels.long()).cpu().float().mean()
                self.acc_sum.add(acc.cpu().item())

                if print_interval and not (i + 1) % print_interval:
                    loss_mean = self.loss_sum.value()[0]
                    acc_mean  = self.acc_sum.value()[0]

                    logging.info('Epoch[%d] Batch [%d]\t Speed: %f samples/sec\t loss=%f\t acc=%f'
                                 % (epoch, i+1, train_data.batch_size * print_interval / (time.time() - btic),
                                     loss_mean, acc_mean)
                                 )

            loss_mean = self.loss_sum.value()[0]
            acc_mean  = self.acc_sum.value()[0]
            throughput = int(train_data.batch_size * len(train_data) / (time.time() - tic))

            logging.info('[Epoch %d] training: loss=%f\t acc=%f' %
                         (epoch, loss_mean, acc_mean))
            logging.info('[Epoch %d] speed: %d samples/sec\t time cost: %f' %
                         (epoch, throughput, time.time()-tic))

            # 验证
            is_best = False
            if valid_data is not None and eval_step and not (epoch + 1) % eval_step:
                valid_acc = self.valid_test(valid_data)
                logging.info('[Epoch %d] valid_acc %f' %
                             (epoch, valid_acc))
                is_best = valid_acc > best_valid_acc
                if is_best:
                    best_valid_acc = valid_acc

            state_dict = self.net.state_dict()

            if not (epoch + 1) % save_step:
                save_checkpoint({
                    'state_dict':state_dict,
                    'epoch':epoch + 1
                }, is_best=is_best, save_dir=save_model_dir,
                filename='model.pth.tar')

    def valid_test(self, valid_data) -> float:
        num_correct = 0
        num_imgs = 0
        self.net.eval()

        for data in valid_data:
            imgs, labels = data
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                scores = self.net(imgs)
            num_correct += (scores.argmax(dim=1) == labels).cpu().float().sum().item()
            num_imgs += imgs.shape[0]

        return num_correct / num_imgs


class AverageValueMeter(object):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value

        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean, self.std = self.sum, np.inf
        else:
            self.mean = self.sum / self.n
            self.std = math.sqrt(
                (self.var - self.n * self.mean * self.mean) / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.std = np.nan
