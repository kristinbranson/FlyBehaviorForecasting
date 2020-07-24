import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
#use_cuda=1
from util import *

class NeuralNetwork4(nn.Module):

    def __init__(self, args):
        super(NeuralNetwork4, self).__init__()

        self.t_dim = args.t_dim
        self.x_dim = args.x_dim
        self.y_dim = args.y_dim
        self.m_dim = args.m_dim
        self.h_dim = args.h_dim

        if args.visionHist:
            self.i_dim = self.t_dim * self.x_dim
        else:
            if args.visionOnly:
                self.i_dim = self.t_dim * (self.x_dim - self.m_dim)
            elif args.vision:
                self.i_dim = self.t_dim * self.m_dim + (self.x_dim - self.m_dim)
            else:
                self.i_dim = self.t_dim * self.m_dim 


        self.model = nn.Sequential(
            nn.Linear(self.i_dim, self.h_dim),
            #nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.h_dim, self.h_dim),
            #nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.h_dim, self.h_dim),
            #nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.h_dim, self.y_dim),
        )
       
        initialize_weights(self)
        self.BCE_loss = nn.BCEWithLogitsLoss().cuda()


    def forward(self, x):

        y = self.model(x)
        return y


    def bce_loss(self, x, t):
    
        logit = self.model(x)
        y = F.softmax(logit, dim=1)
        loss = self.BCE_loss(logit, t)
        return loss, y


    def mse_loss(self, x, t):
   
        y = self.model(x)
        loss_nm = torch.mean((y-t)**2, dim=0)
        loss = torch.sum(loss_nm, dim=0)
        return loss, loss_nm, y



class NeuralNetwork5(nn.Module):

    def __init__(self, args):
        super(NeuralNetwork5, self).__init__()

        self.t_dim = args.t_dim
        self.x_dim = args.x_dim
        self.y_dim = args.y_dim
        self.h_dim = args.h_dim

        if args.visionHist:
            self.i_dim = self.t_dim * self.x_dim
        else:
            if args.visionOnly:
                self.i_dim = self.t_dim * (self.x_dim - self.m_dim)
            elif args.vision:
                self.i_dim = self.t_dim * self.m_dim + (self.x_dim - self.m_dim)
            else:
                self.i_dim = self.t_dim * self.m_dim 



        self.model = nn.Sequential(
            nn.Linear(self.i_dim, self.h_dim),
            nn.Tanh(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.h_dim, self.y_dim),
        )
       

        initialize_weights(self)
        self.BCE_loss = nn.BCEWithLogitsLoss().cuda()


    def forward(self, x):

        y = self.model(x)
        return y


    def loss(self, x, t):
    
        y = self.model(x)
        loss = torch.mean(torch.sum((y-t)**2, dim=-1))
        return loss, y



WEIGHT=0 
BIAS=1

class ConvNet(nn.Module):

    def __init__(self, args, input_size, output_size, initbias=None, pool_sz=2):
        super(ConvNet, self).__init__()

        self.params = []
        self.atype  = args.atype 
        self.hdim = args.h_dim
        self.nhid = args.n_hid
        self.tau = args.t_dim
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        
        hids = [input_size] + [self.hdim] * self.nhid 
        hids2 = [input_size] + [self.hdim] * self.nhid 
        for i in range(self.nhid-1):

            if self.atype == 'sigmoid':
                layer = nn.Sequential(
                    nn.Conv1d(hids[i], hids[i+1], 5, stride=1),
	            nn.Sigmoid(),
                )
            elif self.atype == 'tanh':
                layer = nn.Sequential(
                    nn.Conv1d(hids[i], hids[i+1], 5, stride=1),
	            nn.Tanh(),
                )
            elif self.atype == 'relu':
                print('Relu activation')
                layer = nn.Sequential(
                    nn.Conv1d(hids[i], hids[i+1], 5, stride=1, padding=2),
	            nn.ReLU(),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv1d(hids[i], hids[i+1], 5, stride=1, padding=2),
                    nn.LeakyReLU(0.2),            
                )
            self.layers1.append(layer)
            #self.layers2.append(layer2) 

        last_dim = self.tau 
        if self.atype == 'relu':
            self.fclayer1 = nn.Sequential(
                nn.Linear(self.hdim*last_dim, self.hdim),
	        nn.ReLU(),
            )
        else:
            self.fclayer1 = nn.Sequential(
                nn.Linear(self.hdim*last_dim, self.hdim),
                nn.LeakyReLU(0.2),            
            )
        self.fclayer2 = nn.Sequential(
            nn.Linear(self.hdim, output_size),
        )

        #self.layers.append(layer)


    def forward(self, X):

        N, T, D = X.shape
        H, Hs = X.permute(0, 2, 1), []
        for i, layer in enumerate(self.layers1): 
            H = self.layers1[i](H)
            #H = self.layers2[i](H.permute(0,2,1)).permute(0,2,1)
            Hs.append(H)
        H = H.view([H.shape[0],-1])
        H = self.fclayer1(H)
        Hs.append(H)
        H = self.fclayer2(H)
        Hs.append(H)

        return H, Hs 


    def prediction(self, X):

        pred, hids = self.forward(X)
        return pred.flatten(), hids


    def loss(self, X, target, W, ignore_W=0 , utype=None, use_cuda=1, epoch=None):

        pred, hids = self.prediction(X)
        loss_l2 = (pred - target)**2
        #loss_l1 = torch.abs(pred - target)
        loss = loss_l2 #+ loss_l1 
        if not ignore_W: loss = loss * (1.0+W)
        loss = torch.mean(loss, dim=0)
        return loss, pred


    def mse_loss(self, x, t):
   
        y, hids = self.forward(x)
        loss_nm = torch.mean((y-t)**2, dim=0)
        loss = torch.sum(loss_nm, dim=0)
        return loss, loss_nm, y


