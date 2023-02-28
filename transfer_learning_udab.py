import numpy as np
import pickle
import scipy.io
import torch
import sys
import matplotlib.pyplot as plt
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
sys.path.append('./')

#加载数据的类
class sourceDataSet(Dataset):
    def __init__(self, train, testIndex):
        EEG_X = scipy.io.loadmat('EEG_X.mat')['X'][0]
        EEG_y = scipy.io.loadmat('EEG_Y.mat')['Y'][0]
        EEG_X = np.stack(EEG_X)
        EEG_y = np.stack(EEG_y)
        with open('EEG_X.pickle', 'wb') as output_f:
            pickle.dump(EEG_X, output_f)
        with open('EEG_y.pickle', 'wb') as output_f:
            pickle.dump(EEG_y, output_f)

        with open('EEG_X.pickle', 'rb') as input_f:
            EEG_X = pickle.load(input_f)
        with open('EEG_Y.pickle', 'rb') as input_f:
            EEG_Y = pickle.load(input_f) + 1
            EEG_Y = np.squeeze(EEG_Y)
        print('data load complete!')

        if train:
            self.EEG_X = torch.from_numpy(np.concatenate(np.concatenate((EEG_X[:testIndex], EEG_X[testIndex + 1:])))).float()
            mean = self.EEG_X.mean(dim=0, keepdim=True)
            std = self.EEG_X.std(dim=0, unbiased=False, keepdim=True)
            self.EEG_X = (self.EEG_X - mean) / std
            self.EEG_Y = torch.from_numpy(np.concatenate(np.concatenate((EEG_Y[:testIndex], EEG_Y[testIndex + 1:])))).long()
            self.num_samples = len(self.EEG_X)
        else:
            self.EEG_X = torch.from_numpy(EEG_X[testIndex]).float()
            mean = self.EEG_X.mean(dim=0, keepdim=True)
            std = self.EEG_X.std(dim=0, unbiased=False, keepdim=True)
            self.EEG_X = (self.EEG_X - mean) / std
            self.EEG_Y = torch.from_numpy(EEG_Y[testIndex]).long()
            self.num_samples = len(self.EEG_X)

    def __getitem__(self, index):
        return self.EEG_X[index], self.EEG_Y[index]

    def __len__(self):
        return self.num_samples

#定义DAN网络
class DAN(nn.Module):
    #定义网络的形状
    def __init__(self, track_running_stats, momentum):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(310, 256),
            nn.BatchNorm1d(num_features=256, momentum=momentum, track_running_stats=track_running_stats),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(num_features=128, momentum=momentum, track_running_stats=track_running_stats),
            nn.Dropout(),
            nn.ReLU()
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(num_features=64, momentum=momentum, track_running_stats=track_running_stats),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(num_features=32, momentum=momentum, track_running_stats=track_running_stats),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.LogSoftmax(dim=1)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32, momentum=momentum, track_running_stats=track_running_stats),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )
    #定义传播函数
    def forward(self, input_data, alpha=0):
        feature = self.feature_extractor(input_data)
        reverse_feature = ReverseLayer.apply(feature, alpha)
        class_pred = self.class_classifier(feature)

        domain_pred = self.domain_classifier(reverse_feature)

        return class_pred, domain_pred

#定义反向层的类
class ReverseLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output, None


if __name__ == '__main__':
    learningRate = 1e-3
    NumofEpoch = 20
    CUDA = True
    batchSize = 256
    LOSS = 'NLL'
    MOMENTUM = 0.5
    LAMBDA = 0.5
    transferBase = 0
    testIndex = 4

    cudnn.benchmark = True

    acc_list = []
    for testIndex in range(15):
        print(f'-----------------以第{testIndex}组为测试集-----------')
        model = DAN(track_running_stats=True, momentum=MOMENTUM)
        optimizer = optim.Adam(model.parameters(), lr=learningRate)

        #计算 NLL loss
        class_loss = nn.NLLLoss()
        domain_loss = nn.NLLLoss()
        if CUDA:
            model = model.cuda()
        for p in model.parameters():
            p.requires_grad = True

        Xs = sourceDataSet(train=True, testIndex=testIndex)
        Xs_loader = DataLoader(dataset=Xs, batch_size=batchSize, shuffle=True, num_workers=3)
        Xt = sourceDataSet(train=False, testIndex=testIndex)
        Xt_loader = DataLoader(dataset=Xt, batch_size=batchSize, shuffle=True, num_workers=3)

        for epoch in range(NumofEpoch):

            loaderLen = min(len(Xs_loader), len(Xt_loader))
            Xs_iter = iter(Xs_loader)
            Xt_iter = iter(Xt_loader)

            model.train()
            for i in range(loaderLen):
                p = float(i + (epoch - transferBase) * loaderLen) / (NumofEpoch - transferBase) / loaderLen
                alpha = LAMBDA * (2. / (1. + np.exp(-10 * p)) - 1)

                if epoch < transferBase:
                    alpha = 0

                # 使用source domain的数据训练网络
                Xs = Xs_iter.next()
                s_input, s_label = Xs
                optimizer.zero_grad()
                batch_size = len(s_label)
                #计算 NLL loss
                source_domain_label = torch.zeros(batch_size).long()

                if CUDA:
                    s_input = s_input.cuda()
                    s_label = s_label.cuda()
                    source_domain_label = source_domain_label.cuda()

                source_class_pred, source_domain_pred = model(input_data=s_input, alpha=alpha)
                XsCLoss = class_loss(source_class_pred, s_label)
                XsDLoss = domain_loss(source_domain_pred, source_domain_label)

                # 使用target domain的数据训练网络
                data_target = Xt_iter.next()
                t_input, t_label = data_target
                batch_size = len(t_input)

                #计算 NLL loss
                target_domain_label = torch.ones(batch_size).long()

                if CUDA:
                    t_input = t_input.cuda()
                    target_domain_label = target_domain_label.cuda()

                XtClassPred, XtDomainPred = model(input_data=t_input, alpha=alpha)
                XtDLoss = domain_loss(XtDomainPred, target_domain_label)
                loss = XsCLoss + XsDLoss + XtDLoss
                loss.backward()
                optimizer.step()

            #测试当前模型的准确性
            model.eval()
            num_correct = 0
            test_input, test_label = Xt[:]
            # print(test_input.shape)
            if CUDA:
                test_input = test_input.cuda()
            with torch.no_grad():
                XtClassPred, _ = model(input_data=test_input, alpha=alpha)
            XtClassPred = XtClassPred.cpu()
            for j in range(len(Xt)):
                if (LOSS == 'MSE' and np.argmax(XtClassPred[j].numpy()) == np.argmax(test_label[j].numpy())) or (
                        LOSS == 'NLL' and np.argmax(XtClassPred[j].numpy()) == test_label[j]):
                    num_correct += 1
            acc = round(num_correct / len(Xt), 4)
            print(f'Epoch {epoch} 的准确率为 {acc* 100:.4f}%')
        acc_list.append(acc)

    print(f'每轮的准确率如下：:\n{acc_list}%')
    print(f'总准确度: {sum(acc_list) * 100 / len(acc_list):.4f}%')
