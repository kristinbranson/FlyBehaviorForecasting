import os, sys, time
import math
import torch
import torch.nn as nn
from torch import optim

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn as nn

RNG = np.random.RandomState(0)

"""checking arguments"""
def check_args(args):

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_sz >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


def np2pandas(trx, fly_ind=0):

    import pandas as pd
    new_trx = {}
    for trx_key in trx.keys():
        new_trx[trx_key] = trx[trx_key][:,fly_ind]

    df = pd.DataFrame(new_trx, columns=new_trx.keys())    
    return df


def normalize_outputs(data, mu, std):
    return (data - mu) / std


def multiclass_cross_entropy(pred, target, NB=8, dim=2):

    N = pred.shape[0] 
    lossM   = - target * torch.log(pred)
    lossM   = torch.sum(lossM, dim=dim)
    loss    = torch.mean(torch.mean(lossM, dim=1), dim=0)

    lossM2  = lossM.view([N, NB]) #T, bs, nb
    lossNM  = torch.squeeze(torch.mean(lossM2, dim=0)) 

    return loss, lossNM


def compute_inter_animal_distance(data_pos):

    T, F = data_pos.shape

    dists = []
    for t in range(T):
        pos = data_pos[t,:].reshape([F//2,2])
        dist_t = dist_np(pos, pos)
        np.fill_diagonal(dist_t, np.inf)
        min_dists = np.min(dist_t, axis=0)
        dists.append(min_dists)

    return np.asarray(dists)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()



def dist_np(x,y):
    '''Distance matrix computation
    '''
    d0 = -2.0*np.dot(x,y.T)
    d1 = np.sum(x*x, axis=1)[:,None]#.dimshuffle(0,'x')
    d2 = np.sum(y*y, axis=1)
    return np.sqrt(d0+d1+d2)



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def copy_hiddens(hiddens, params):

    if params['mtype']=='0rnn':
        hiddens0 = copy.deepcopy(hiddens)
    else:
        hiddens0 = []
        for hiddens_fly in hiddens:
            hiddens_fly0=[]
            for hidden in hiddens_fly:
                hiddens_fly0.append(hidden.clone())
            hiddens0.append(hiddens_fly0)

    return hiddens0


def save(model, save_path):
    torch.save([model.state_dict()], save_path+'.pkl')



def load(model, save_path):
    model.load_state_dict(torch.load(save_path+'.pkl', \
                                map_location=lambda storage, \
                                loc: storage)[0])

    return model


def lr_scheduler_init(optimizer, lrsch_type, gamma=0.95):

    if lrsch_type == 'lambda':
        # Assuming optimizer has two groups.
        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    elif lrsch_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)

    elif lrsch_type == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    elif lrsch_type == 'multstep':
         scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25*250,50*250,75*250,100*250], gamma=0.3)

    elif lrsch_type == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer)
    else:
        return None
    return scheduler


def chi_square_dist(hist1, hist2):

    return np.sum((hist1 - hist2) ** 2 / (hist1+1e-6))


def correlation_distance(hist1, hist2):
 
    num = np.sum((hist1 - np.mean(hist1)) * (hist2 - np.mean(hist2)))
    den = np.sqrt(np.sum((hist1 - np.mean(hist1))**2)) * np.sqrt(np.sum((hist2 - np.mean(hist2))**2))
    return 1.0 - num / den


def log_likelihood(pred, target, T=50, NB=8, dim=2):

    N = pred.size()[0] 
    amax_targ = torch.max(target, dim=2)[1]
    lossM     = - torch.log(pred.view([N*NB,-1])[np.asarray(N), amax_targ.view([-1])])
    loss      = torch.mean(lossM, dim=0) * T
    return loss


def load_rnn(   args, mtype='rnn',\
                num_hid=100, \
                model_epoch=100000, \
                num_epoch=100000, num_mfeat=8,\
                gender=2, cudaF=0, num_bin=51,
                load_path=None):

    mpath = '/groups/branson/home/imd/Documents/janelia/research/FlyTrajPred/pytorch'
    if 'rnn' in mtype:
        from flyNetwork_RNN import FlyNetworkGRU
        model = FlyNetworkGRU(args)
        save_path = mpath + '/models/flyNet4_'

    elif 'skip' in mtype:
        from flyNetwork_RNN import FlyNetworkSKIP6
        model = FlyNetworkSKIP6(args)
        save_path = './models/flyNetSKIP6_'

    if cudaF: model = model.cuda()
    print('Model Load %s' % load_path)
    model = load(model, load_path)
    return model
    


