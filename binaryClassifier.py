import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

INPUT_LYR=0
WEIGHT=0 
BIAS=1



def outputSize(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)



class LogisticRegression(nn.Module):


    def __init__(self, args, filter_size=5):
        super(LogisticRegression, self).__init__()

        self.D = args.x_dim * args.t_dim
        self.y_dim = args.c_dim 
        self.hid_size = args.h_dim           #hid_size=[64]
        self.l1wd = args.l1wd
        self.l2wd = args.l2wd
        self.atype  = args.atype

        self.layer = nn.Linear(self.D, self.y_dim)


    def forward(self, X):

        N, D, T = X.shape
        X_ = X.view([N,T*D])
        y = self.layer(X_)
        return y


    def prediction(self, X):

        pred = self.forward(X).flatten()
        return F.sigmoid(pred)


    def accuracy(self, X, target):

        pred, _, _, _ = self.prediction(X)
        amaxpred = torch.argmax(pred, dim=-1)
        amaxtarg = torch.argmax(target, dim=-1)
        acc = torch.mean(torch.eq(amaxtarg, amaxpred).float())
        return acc
   

    def loss(self, X, target, utype=None, use_cuda=1, epoch=None):

        N = X.shape[0]
        pred = self.prediction(X)
        loss = target * torch.log(pred) + (1-target) * torch.log(1.0 - pred)
        loss = -torch.mean(loss)

        acc = torch.mean(((pred>0.5).float() == target).float())
        return loss, acc, pred.data.cpu().numpy()




class ConvNet(nn.Module):

    def __init__(self, input_size, output_size, args, filter_size=5):
        super(ConvNet, self).__init__()

        self.D = args.t_dim  * args.x_dim
        self.t_dim = args.t_dim
        self.num_class = output_size
        self.x_dim = args.x_dim           #hid_size=[64]
        self.hid_size = args.h_dim           #hid_size=[64]
        self.num_channels = args.ch_dim   #num_channels=[1, 16,32]
        self.nlayers = len(args.ch_dim)
        self.conv_layers, self.fc_layers = [], []
        self.conv_params, self.fc_params = [], []
        self.l1wd = args.l1wd
        self.l2wd = args.l2wd
        self.atype  = args.atype

        self.conv= torch.nn.Sequential(\
            torch.nn.Conv1d(  self.num_channels[0], \
                              self.num_channels[1], \
                              kernel_size=filter_size, \
                              stride=1, padding=1),\
            torch.nn.ReLU(),

            torch.nn.Conv1d(  self.num_channels[1], \
                              self.num_channels[2], \
                              kernel_size=filter_size, \
                              stride=1, padding=1),
            torch.nn.ReLU(),

            torch.nn.Conv1d(  self.num_channels[2], \
                              self.num_channels[3], \
                              kernel_size=filter_size, \
                              stride=1, padding=1),
            torch.nn.ReLU(),
        )


        conv_odim1 = outputSize(self.t_dim, filter_size, stride=1, padding=1)
        conv_odim2 = outputSize(conv_odim1, filter_size, stride=1, padding=1)
        conv_odim3 = outputSize(conv_odim2, filter_size, stride=1, padding=1)
    

        self.fc= torch.nn.Sequential(
            torch.nn.Linear(self.num_channels[3] * conv_odim3, self.hid_size),\
            #torch.nn.BatchNorm1d(args.h_dim),
            torch.nn.ReLU(),

            torch.nn.Linear(self.hid_size, self.hid_size),
            torch.nn.BatchNorm1d(args.h_dim),
            torch.nn.ReLU(),

            torch.nn.Linear(self.hid_size, output_size),
            #torch.nn.Sigmoid()
        )

        self.lossfn = nn.BCEWithLogitsLoss().cuda()

 
    def forward(self, X):

        N = X.shape[0]
        H = self.conv(X)
        H = H.view([N,-1])
        pred = self.fc(H)
        return pred 


    def prediction(self, X):

        pred = F.sigmoid(self.forward(X).flatten())
        return pred


    def accuracy(self, X, target):

        pred, _, _, _ = self.prediction(X)
        amaxpred = torch.argmax(pred, dim=-1)
        amaxtarg = torch.argmax(target, dim=-1)
        acc = torch.mean(torch.eq(amaxtarg, amaxpred).float())
        return acc
   

    def loss(self, X, target, utype=None, use_cuda=1, epoch=None, ltype='ce'):

        N = X.shape[0]
        logit = self.forward(X).flatten()
        pred = F.sigmoid(logit)
        #loss = target * torch.log(pred) + (1-target) * torch.log(1.0 - pred)
        #loss = -torch.mean(loss)
        if ltype == 'sq':
            loss = torch.mean(target * (pred-1.)**2) \
                    +  torch.mean((1-target) * (pred**2))
        else:
            #loss = self.lossfn(logit, target)
            loss = self.lossfn(logit, target)
        acc = torch.mean(((pred>0.5).float() == target).float())
        return loss, acc, pred.data.cpu().numpy()





