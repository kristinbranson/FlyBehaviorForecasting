import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#use_cuda=1




class FlyNetworkGRU(nn.Module):

    def __init__(self, args):
        super(FlyNetworkGRU, self).__init__()

        self.hidden_size = args.h_dim
        self.gru1 = nn.GRU(args.x_dim, args.h_dim)
        self.gru2 = nn.GRU(args.h_dim, args.h_dim)
        self.gru3 = nn.GRU(args.h_dim, args.h_dim)
        self.out  = nn.Linear(args.h_dim, args.y_dim)

    def forward(self, X, hidden, hiddenF=False):

        T, B, D = X.size()
        h1, hidden1 = self.gru1(X, hidden[0])
        h2, hidden2 = self.gru2(h1, hidden[1])
        h3, hidden3 = self.gru3(h2, hidden[2])  
        
        output = self.out(h3.view(T*B, h3.shape[-1]))
        if hiddenF:
            return output, [hidden1, hidden2, hidden3], [h1,h2,h3]

        return output, [hidden1, hidden2, hidden3]


    def initHidden(self, batch_sz, T=1, use_cuda=1):

        result1 = Variable(torch.zeros(T, batch_sz, self.hidden_size))
        result2 = Variable(torch.zeros(T, batch_sz, self.hidden_size))
        result3 = Variable(torch.zeros(T, batch_sz, self.hidden_size))

        if use_cuda:
            return [result1.cuda(), result2.cuda(), result3.cuda()]
        else:
            return [result1, result2, result3]



class FlyNetworkRegression(nn.Module):

    def __init__(self, args):
        super(FlyNetworkRegression, self).__init__()


        self.hidden_size = args.h_dim
        self.gru1 = nn.GRU(args.x_dim, args.h_dim)
        self.gru2 = nn.GRU(args.h_dim, args.h_dim)
        self.gru3 = nn.GRU(args.h_dim, args.h_dim)
        self.out  = nn.Linear(args.h_dim, args.y_dim)
        self.dOut = args.y_dim

    def forward(self, X, hidden, hiddenF=False):

        T, B, D = X.size()

        h1, hidden1 = self.gru1(X, hidden[0])
        h2, hidden2 = self.gru2(h1, hidden[1])
        h3, hidden3 = self.gru3(h2, hidden[2])  
        
        output = self.out(h3.view(T*B, h3.shape[-1]))
        if hiddenF:
            return output, [hidden1, hidden2, hidden3], [h1,h2,h3]

        return output, [hidden1, hidden2, hidden3]


    def initHidden(self, batch_sz, T=1, use_cuda=1):

        result1 = Variable(torch.zeros(T, batch_sz, self.hidden_size))
        result2 = Variable(torch.zeros(T, batch_sz, self.hidden_size))
        result3 = Variable(torch.zeros(T, batch_sz, self.hidden_size))

        if use_cuda:
            return [result1.cuda(), result2.cuda(), result3.cuda()]
        else:
            return [result1, result2, result3]




class FlyNetworkSKIP6(nn.Module):

    def __init__(self, args):
        super(FlyNetworkSKIP6, self).__init__()

        self.hidden_size = args.h_dim
        self.gru1 = nn.GRU(args.x_dim, args.h_dim)
        self.gru2 = nn.GRU(args.h_dim, args.h_dim)
        self.gru3 = nn.GRU(2*args.h_dim, args.h_dim)
        self.gru4 = nn.GRU(2*args.h_dim, args.h_dim)
        self.gru5 = nn.GRU(2*args.h_dim, args.h_dim)


        self.out  = nn.Linear(args.h_dim, args.y_dim)
        self.dOut = args.y_dim

    def forward(self, X, hidden, hiddenF=False):

        T, B, D = X.size()
        h1, hidden1 = self.gru1(X, hidden[0])
        h2, hidden2 = self.gru2(h1, hidden[1])
        h3, hidden3 = self.gru2(h2, hidden[2])
        

        h3_ = torch.cat([h2,h3],2)
        h4, hidden4 = self.gru3(h3_, hidden[3])  
        h4_ = torch.cat([h1,h4],2)
        h5, hidden5 = self.gru3(h4_, hidden[4])        

        output = self.out(h5.view(T*B, h5.shape[-1]))
        if hiddenF:
            return output, [hidden1, hidden2, hidden3, hidden4, hidden5], [h1, h2, h3, h4, h5]
        else:
            return output, [hidden1, hidden2, hidden3, hidden4, hidden5]


    def initHidden(self, batch_sz, T=1, use_cuda=1):

        result1 = Variable(torch.zeros(T, batch_sz, self.hidden_size))
        result2 = Variable(torch.zeros(T, batch_sz, self.hidden_size))
        result3 = Variable(torch.zeros(T, batch_sz, self.hidden_size))
        result4 = Variable(torch.zeros(T, batch_sz, self.hidden_size))
        result5 = Variable(torch.zeros(T, batch_sz, self.hidden_size))


        if use_cuda:
            return [result1.cuda(), result2.cuda(), result3.cuda(),\
                                    result4.cuda(), result5.cuda()]
        else:
            return [result1, result2, result3, result4, result5]





