import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class ResNetFFBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer=None):
        super(ResNetFFBlock, self).__init__()
        self.linear1 = nn.Linear(inplanes, planes)
        self.bn1 = None if norm_layer is None else norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(planes, planes)
        self.bn2 = None if norm_layer is None else norm_layer(planes)

    def forward(self, x):
        out = self.linear1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.linear2(out)
        if self.bn1 is not None:
            out = self.bn2(out)

        out += x
        out = self.relu(out)

        return out

class ResNetFF(nn.Module):
    def __init__(self, inplanes, planes, outputs, num_blocks, norm_layer=nn.BatchNorm1d):
        super(ResNetFF, self).__init__()
        self.linear1 = nn.Linear(inplanes, planes)
        blocks = [ResNetFFBlock(planes, planes, norm_layer) for i in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.linear2 = nn.Linear(planes, outputs)
                 
    def forward(self, x):
        out = self.linear1(x)
        for block in self.blocks:
            out = block(out)
        return self.linear2(out)
    

class FlyNetworkGRU2(nn.Module):
    def __init__(self, args):
        super(FlyNetworkGRU2, self).__init__()

        self.hidden_size = args.h_dim
        self.num_layers = args.num_layers
        self.gru = nn.GRU(args.x_dim, args.h_dim, args.num_layers)
        self.head = nn.Linear(args.h_dim, args.y_dim)
        self.head = ResNetFF(args.h_dim, args.r_dim, args.y_dim, args.num_blocks, norm_layer=None)
        self.initWeights(args.init_weights)

    def forward(self, X, hidden):
        hidden = hidden[0]
        T, B, D = X.size()
        n, hidden = self.gru(X, hidden)
        n = n.view(T, B, -1)
        output = torch.stack([self.head(n[t, ...]) for t in range(T)], 0)

        return output, [hidden]

    def initHidden(self, batch_sz, device="cpu"):
        return [Variable(torch.zeros(self.num_layers, batch_sz, self.hidden_size, device=device))]

    def initWeights(self, init_weights):
        if init_weights == 'kaiming':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, ResNetFFBlock):
                    if m.bn1 is not None:
                        nn.init.constant_(m.bn1.weight, 1)
                        nn.init.constant_(m.bn1.bias, 0)
                    if m.bn2 is not None:
                        nn.init.constant_(m.bn2.weight, 0)
                        nn.init.constant_(m.bn2.bias, 0)
                    nn.init.kaiming_normal_(m.linear1.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.linear1.bias, 0)
                    nn.init.kaiming_normal_(m.linear2.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.linear2.bias, 0)
                elif isinstance(m, nn.GRU):
                    for name, param in m.named_parameters():
                        if 'bias' in name:
                            nn.init.constant_(param, 0.0)
                        elif 'weight_ih' in name:
                            nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                        elif 'weight_hh' in name:
                            nn.init.orthogonal_(param)
    
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


    def initHidden(self, batch_sz, T=1, device="cpu"):

        result1 = Variable(torch.zeros(T, batch_sz, self.hidden_size, device=device))
        result2 = Variable(torch.zeros(T, batch_sz, self.hidden_size, device=device))
        result3 = Variable(torch.zeros(T, batch_sz, self.hidden_size, device=device))

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





