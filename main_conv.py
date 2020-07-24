from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, time, string, random, argparse
import numpy as np
RNG = np.random.RandomState(0)

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from tqdm import tqdm
teacher_forcing_ratio = 1.0
from flyNetwork_MLP import ConvNet 

#from eval import *
from util import *
from gen_dataset import sample_batch_videos, test_batch_videos, \
            load_eyrun_data, load_videos, quick_sample_batch_videos


SOURCE=0
TARGET=1
MOTION=2
MALE=0
FEMALE=1
MFEAT=8
TRAIN=0
VALID=1
TEST=2

def eyruns_log_likelihood(pred, target, T=50, NB=8, dim=2):

    N = pred.size()[0] 
    amax_targ = torch.max(target, dim=2)[1]
    lossM     = -torch.log(pred.view([N*NB,-1])[np.asarray(N), amax_targ.view([-1])])
    loss      = torch.mean(lossM, dim=0) * T
    return loss


def feval(input_variable, target_variable, model, optimizer, 
                                            num_bins=51,\
                                            max_length=100, \
                                            mode='train', \
                                            N=100, \
                                            accuracyMetricF=0, \
                                            use_cuda=1,
                                            hiddenF=False):

    if mode=='train':
        model.train()
    else:
        model.eval()

    if mode == 'train': optimizer.zero_grad()
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    loss = 0
    batch_sz, T, D = input_variable.size()

    #target_variable = target_variable.contiguous()
    #target_variable = target_variable.view([batch_sz, T, MFEAT])
    logit, _ = model.forward(input_variable)
    logit = logit.view([batch_sz, MFEAT, -1])
    prediction = F.softmax(logit, dim=2)

    target_variable = target_variable.contiguous()
    target_variable = target_variable.view([batch_sz, MFEAT, -1])

    loss, lossNM = multiclass_cross_entropy(prediction, \
                                            target_variable, \
                                            NB=MFEAT)


    if mode=='train':
        loss.backward()
        optimizer.step()

    return loss.item(), prediction.data.cpu().numpy() 


def trainIters(args, print_every=1000, plot_every=100, learning_rate=0.01):

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    if not os.path.exists(args.save_dir+args.dtype): os.makedirs(args.save_dir+args.dtype)
    if not os.path.exists(args.save_dir+args.dtype+'/model'): os.makedirs(args.save_dir+args.dtype+'/model')


    ftag = 'gender%d_%s_'% (args.gender, args.vtype) 
    #save_file = args.save_dir+'/model/model_gender%d_%s_' % (args.gender, args.vtype) 
    if args.visionOnly:
        ftag = ftag+'visionOnly1'
    elif not args.vision: 
        ftag = ftag+'visionF0'


    if not args.visionOnly and args.vision:
        model = ConvNet(args, args.x_dim, args.y_dim)
    elif args.visionOnly:
        model = ConvNet(args, args.x_dim-args.y_dim, args.y_dim)
    else:
        model = ConvNet(args, args.y_dim, args.y_dim)
    model = model.cuda()

    #X_train, X_valid, X_test = dataset
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    video_data_list = load_videos(onehotF=0, vtype=args.vtype, \
            dtype=args.dtype, num_bin=args.num_bin, bin_type=args.btype)
    video_data_list[2] = None
    Nvl = 10000
    valid_batch = sample_batch_videos(video_data_list[VALID], K=Nvl, \
                        tau=args.t_dim+1, etype='vl', \
                        genderF=args.gender)


    losses = [[],[]] 
    losses_mf = [[],[]] 

    for iter in tqdm(range(1, args.epoch + 1)):
        
        ## Sample Data
        Ntr=5000
        train_batch = sample_batch_videos(video_data_list[TRAIN], \
                        K=Ntr, tau=args.t_dim+1, etype='tr', \
                        genderF=args.gender)

        YY= train_batch[TARGET][-1,:,:]
        X = train_batch[SOURCE][:-1,:,:].transpose(1,0,2)
        XX = X[:,:,-MFEAT:] 
        #XX = np.concatenate([XX, train_batch[SOURCE][:,-1,MFEAT:]], axis=1)
        if args.visionOnly:
            XX = X[:,:,:-MFEAT] 
        elif args.vision:
            XX = np.concatenate([XX, X[:,:,:-MFEAT]], axis=2)

        tr_losses, tr_losses_mf = [], []
        num_batches = int(Ntr/args.batch_sz)
        for i in range(num_batches):

            if args.use_cuda:
                input_variable  = Variable(torch.FloatTensor(XX[i*args.batch_sz:(i+1)*args.batch_sz]).cuda(), requires_grad=True)
                target_variable = Variable(torch.FloatTensor(YY[i*args.batch_sz:(i+1)*args.batch_sz]).cuda(), requires_grad=True)
            else:
                input_variable  = Variable(torch.FloatTensor(XX[i*args.batch_sz:(i+1)*args.batch_sz]), requires_grad=True)
                target_variable = Variable(torch.FloatTensor(YY[i*args.batch_sz:(i+1)*args.batch_sz]), requires_grad=True)
    
        
            tr_loss, pred = feval(input_variable, target_variable, model, optimizer, args, mode='train')
            tr_losses.append(tr_loss)

        if iter % args.print_every == 1 or iter == 0:

            ## Sample Data
            #valid_batch = sample_batch_linear_reg(X_valid, Nvl, args.t_dim, visionF=1)
            YY = valid_batch[TARGET][-1,:,:]
            X  = valid_batch[SOURCE][:-1,:,:].transpose(1,0,2)
            XX = X[:,:,-MFEAT:] 
            if args.visionOnly:
                XX = X[:,:,:-MFEAT] 
            elif args.vision:
                XX = np.concatenate([XX, X[:,:,:-MFEAT]], axis=2)

            #XX = valid_batch[SOURCE][:,:,:MFEAT] 
            #XX = XX.reshape([Nvl, -1])
            #XX = np.concatenate([XX, valid_batch[SOURCE][:,-1,MFEAT:]], axis=1)
            #YY = valid_batch[TARGET]


            ## Sample Data
            num_batch_vl = int(float(Nvl) / args.batch_sz)
            vl_losses, vl_losses_mf = [], []
            for jj in range(num_batch_vl):

                if use_cuda:
                    input_variable  = Variable(torch.FloatTensor(XX[jj*args.batch_sz:(jj+1)*args.batch_sz,:]).cuda(), requires_grad=False)
                    target_variable = Variable(torch.FloatTensor(YY[jj*args.batch_sz:(jj+1)*args.batch_sz,:]).cuda(), requires_grad=False)
                else:
                    input_variable  = Variable(torch.FloatTensor(XX[jj*args.batch_sz:(jj+1)*args.batch_sz,:]), requires_grad=False)
                    target_variable = Variable(torch.FloatTensor(YY[jj*args.batch_sz:(jj+1)*args.batch_sz,:]), requires_grad=False)


                #input_variable  =  input_variable.reshape([args.batch_sz, -1])
                #target_variable = target_variable.reshape([args.batch_sz, -1])

                vl_loss, pred = feval(input_variable, target_variable, \
                                model, optimizer, args, mode='eval')
                vl_losses.append(vl_loss)

            tr_loss = np.nanmean(tr_loss)
            vl_loss = np.nanmean(vl_loss)
            print('%d Tr Loss %f, Vl Loss %f' \
                    % (iter, tr_loss, vl_loss))
    
            losses[0].append(tr_loss)
            losses[1].append(vl_loss)
            losses_mf[0].append(vl_losses_mf)
            losses_mf[1].append(vl_losses_mf)

        if iter % 1000 == 0:
            save(model, args.save_dir+args.dtype+'/model/model_'+ftag + 'bs%d_epoch%d' % (args.batch_sz, iter))


    print(ftag + '_epoch'+str(iter))
    save(model, args.save_dir+args.dtype+'/model/model_'+ftag + 'bs%d_epoch%d' % (args.batch_sz, iter))
    np.save(args.save_dir+args.dtype+'/loss_%s_' % ftag, np.asarray(losses))
    np.save(args.save_dir+args.dtype+'/loss_mf_%s_' % ftag, np.asarray(losses_mf))
    return model


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of FlyNetwork collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--print_every', type=int, default=10, help='print every')#, required=True)
    parser.add_argument('--save_every' , type=int, default=50,help= 'save every')#, required=True)
    parser.add_argument('--epoch', type=int, default=5000, help='The number of epochs to run')
    parser.add_argument('--batch_sz', type=int, default=32, help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='./runs/conv4_cat50/', \
                        help='Directory name to save the model')
    parser.add_argument('--lrsch_type', type=str, default='step')
    parser.add_argument('--num_bin', type=int, default=101)
    parser.add_argument('--h_dim', type=int, default=128)
    parser.add_argument('--n_hid', type=int, default=4)
    parser.add_argument('--x_dim', type=int, default=152)
    parser.add_argument('--t_dim', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--m_dim', type=int, default=8)
    parser.add_argument('--l2wd', type=float, default=0.00001)
    parser.add_argument('--l1wd', type=float, default=0.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--atype', type=str, default='leakyrelu')
    parser.add_argument('--visionOnly', type=int, default=0)
    parser.add_argument('--vision', type=int, default=1)
    parser.add_argument('--visionHist', type=int, default=0)
    parser.add_argument('--gender', type=int, default=1)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--nntype', type=str, default='conv4')
    parser.add_argument('--vtype', type=str, default='full')
    parser.add_argument('--dtype', type=str, default='pdb')
    parser.add_argument('--btype', type=str, default='perc')

    return check_args(parser.parse_args())


parentpath='/groups/branson/home/imd/Documents/janelia/research/FlyTrajPred/pytorch/models/'

trainF=1
use_cuda=1
if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    args.y_dim = args.num_bin*args.m_dim


    matfile = '/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Code/eyrun_simulate_data.mat'
    (trx,motiondata,params,basesize) = load_eyrun_data(matfile)
    #bin_means = get_centre_bin(params['binedges'].T, tr_config['num_bin']) 


    if trainF:
        model = trainIters(args, print_every=100)
    else:
        pass

    print(model)
    #print(tr_config['savepath'])
    #python -u main_conv.py --save_dir ./runs/conv4_cat50_relu/ --epoch 35000 --gender 0 --vtype full --visionOnly 0 --vision 1 --lr 0.01 --h_dim 128 --t_dim 50 --atype 'relu'

