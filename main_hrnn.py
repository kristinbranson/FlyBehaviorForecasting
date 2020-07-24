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

teacher_forcing_ratio = 1.0
from flyNetwork_RNN import FlyNetworkSKIP6

from util import *
from gen_dataset import sample_batch_videos, test_batch_videos, \
            load_eyrun_data, load_videos, quick_sample_batch_videos


SOURCE=0
TARGET=1
MOTION=2
MALE=0
FEMALE=1

TRAIN=0
VALID=1
TEST=2
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
    T, batch_sz, D = input_variable.size()
    num_feat = args.num_mfeat
    hidden = model.initHidden(N, use_cuda=use_cuda) 

    if hiddenF:
        output, hidden  = model.forward(input_variable, hidden)
    else:
        output, hidden_init  = model.forward(input_variable, hidden)

    output          = output.view([T*batch_sz, num_feat, num_bins])
    prediction      = F.softmax(output, dim=2)
    target_variable = target_variable.contiguous()
    target_variable = target_variable.view([T*batch_sz, num_feat, -1])
    
    loss, lossNM = multiclass_cross_entropy(prediction, \
                                            target_variable, \
                                            NB=args.num_mfeat)

    if mode=='train':
        loss.backward()
        optimizer.step()

    amax_pred = torch.max(prediction, dim=2)[1]
    amax_targ = torch.max(target_variable, dim=2)[1]
    accuracy_matrix = amax_pred == amax_targ
    acc_rate = torch.sum(accuracy_matrix).float() / (T*batch_sz*num_feat) 
    score = log_likelihood(prediction, target_variable, NB=args.num_mfeat, T=T)

    if accuracyMetricF:

        targetGauss_np = convTarget(target_variable.data.cpu().numpy(), sigma=0.25)
        score3, scoreFeat3 = metric3( prediction.data.cpu().numpy(), \
                            targetGauss_np)

        return  loss.item(), lossNM.data.cpu().numpy(), \
                acc_rate.item(), accuracy_matrix.data.cpu().numpy(),\
                score.item(), score3, scoreFeat3
    return  loss.item(), lossNM.data.cpu().numpy(), \
                        acc_rate.item(), score.item()


def trainIters(model, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):

    start = time.time()
    plot_losses_tr, plot_losses_vl = [], []
    tr_accs, vl_accs, tr_nms, vl_nms, tr_scores, vl_scores \
                                        = [], [], [], [], [], []
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    video_data_list = load_videos(onehotF=args.onehot, \
            vtype=args.vtype, dtype=args.dtype, \
            bin_type=args.bin_type, num_bin=args.num_bin)
    load_start = time.time()
    valid_batch = sample_batch_videos(video_data_list[VALID], K=10000, \
                            tau=50, etype='vl', genderF=args.gender)
    load_end = time.time()
    print ('Validation Data Loading time %f' % ((load_end-load_start)/60.0))

    Ntr =  2500
    batch_sz = args.batch_sz
    start_epoch = 1
    for iter in range(start_epoch, n_iters + 1):

        ## Sample Data
        train_batch = sample_batch_videos(video_data_list[TRAIN], \
                        K=Ntr, tau=50, etype='tr', \
                        genderF=args.gender)
        if iter == 1 : print ('Loading time %f' % ((load_end-load_start)/60.0))

        num_batches = int(Ntr/batch_sz)

        iter_start = time.time()
        loss_total, acc_total, score_total = 0, 0, 0  # Reset every print_every
        for i in range(num_batches):
            if use_cuda:
                input_variable  = Variable(torch.FloatTensor(train_batch[SOURCE][:,i*batch_sz:(i+1)*batch_sz]).cuda(), requires_grad=True)
                target_variable = Variable(torch.FloatTensor(train_batch[TARGET][:,i*batch_sz:(i+1)*batch_sz]).cuda(), requires_grad=True)
            else:
                input_variable  = Variable(torch.FloatTensor(train_batch[SOURCE][:,i*batch_sz:(i+1)*batch_sz]), requires_grad=True)
                target_variable = Variable(torch.FloatTensor(train_batch[TARGET][:,i*batch_sz:(i+1)*batch_sz]), requires_grad=True)

            load_start = time.time()
            tr_loss, tr_nm, tr_acc, tr_score = feval(input_variable, \
                                target_variable, model,\
                                optimizer, 
                                num_bins=args.num_bin,\
                                max_length=args.num_steps, \
                                mode='train', N=args.batch_sz)
            load_end = time.time()
            if i == 1 and iter == 1 : print ('Training time %f' % ((load_end-load_start)/60.0))

            score_total += tr_score
            loss_total += tr_loss
            acc_total += tr_acc

        score_total /= num_batches
        loss_total /= num_batches
        acc_total /= num_batches
        iter_end = time.time()
        if iter == 1 or iter % print_every * 100 == 0 : print ('Train Epoch time %f' % ((iter_end-iter_start)/60.0))



        if iter % print_every == 0 or iter == 1:
    
            ## Sample Data
            Nvl = valid_batch[0].shape[1]

            vl_nm = [] 
            vl_loss, vl_acc, vl_score = 0, 0, 0
            vl_batch_sz = 200
            num_batch_vl = int(float(Nvl) / vl_batch_sz)
            for jj in range(num_batch_vl):

                if use_cuda:
                    input_variable  = Variable(torch.FloatTensor(valid_batch[SOURCE][:,jj*vl_batch_sz:(jj+1)*vl_batch_sz,:]).cuda(), requires_grad=False)
                    target_variable = Variable(torch.FloatTensor(valid_batch[TARGET][:,jj*vl_batch_sz:(jj+1)*vl_batch_sz,:]).cuda(), requires_grad=False)
                else:
                    input_variable  = Variable(torch.FloatTensor(valid_batch[SOURCE][:,jj*vl_batch_sz:(jj+1)*vl_batch_sz,:]), requires_grad=False)
                    target_variable = Variable(torch.FloatTensor(valid_batch[TARGET][:,jj*vl_batch_sz:(jj+1)*vl_batch_sz,:]), requires_grad=False)


                vl_loss_jj, vl_nm_jj, vl_acc_jj, vl_score_jj \
                        = feval(input_variable, target_variable,\
                                model,\
                                optimizer, 
                                num_bins=args.num_bin,\
                                max_length=args.num_steps, \
                                mode='valid',\
                                N=vl_batch_sz)

                vl_score += vl_score_jj
                vl_loss += vl_loss_jj
                vl_acc += vl_acc_jj
                vl_nm.append(vl_nm_jj)

            vl_acc = vl_acc / num_batch_vl
            vl_loss = vl_loss / num_batch_vl
            vl_score = vl_score / num_batch_vl

            tr_accs.append(acc_total) 
            vl_accs.append(vl_acc)

            tr_nms.append(tr_nm) 
            tr_scores.append(tr_score)
            vl_scores.append(vl_score)

            vl_nms.append(np.mean(vl_nm, axis=0)) 
            plot_losses_tr.append(loss_total)
            plot_losses_vl.append(vl_loss)

            print('%s (%d %d%%) TR Loss : %.4f, Tr Score %.4f, TR Acc : %.6f, VL Loss : %.4f, VL Score %.4f, VL Acc : %.6f' \
                                    % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, \
                                         loss_total, score_total,\
                                         acc_total, vl_loss , vl_score,\
                                         vl_acc))

            
        if iter % (print_every*5) == 0:
            save(model, args.parentpath + args.savepath+'_'+str(iter))        

    print(args.savepath)
    if not os.path.exists('./results/'+args.dtype): os.makedirs('./results/'+args.dtype)
    np.save('./results/'+args.dtype+'/'+args.savepath+'_loss_motion_features_tr', np.asarray(tr_nms))
    np.save('./results/'+args.dtype+'/'+args.savepath+'_loss_motion_features_vl', np.asarray(vl_nms))
    np.save('./results/'+args.dtype+'/'+args.savepath+'_loss_tr', np.asarray(plot_losses_tr))
    np.save('./results/'+args.dtype+'/'+args.savepath+'_loss_vl', np.asarray(plot_losses_vl))
    np.save('./results/'+args.dtype+'/'+args.savepath+'_scores_tr', np.asarray(tr_scores))
    np.save('./results/'+args.dtype+'/'+args.savepath+'_scores_vl', np.asarray(vl_scores))
    np.save('./results/'+args.dtype+'/'+args.savepath+'_acc_tr', np.asarray(tr_accs))
    np.save('./results/'+args.dtype+'/'+args.savepath+'_acc_vl', np.asarray(vl_accs))


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of FlyNetwork collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--print_every', type=int, default=100, help='print every')#, required=True)
    parser.add_argument('--save_every' , type=int, default=50,help= 'save every')#, required=True)
    parser.add_argument('--epoch', type=int, default=200000, help='The number of epochs to run')
    parser.add_argument('--batch_sz', type=int, default=32, help='The size of batch')
    parser.add_argument('--num_steps', type=int, default=50, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrsch_type', type=str, default='step')
    parser.add_argument('--onehot', type=int, default=0)
    parser.add_argument('--num_bin', type=int, default=101)
    parser.add_argument('--num_mfeat', type=int, default=8)
    parser.add_argument('--visionF', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=100)
    parser.add_argument('--x_dim', type=int, default=152)
    parser.add_argument('--gender', type=int, default=0)
    parser.add_argument('--l2wd', type=float, default=0.0001)
    parser.add_argument('--l1wd', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--dtype', type=str, default='gmr')
    parser.add_argument('--rnn_type', type=str, default='gru')
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--vtype', type=str, default='full')
    parser.add_argument('--bin_type', type=str, default='perc')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--file_dir', type=str, \
            default='./fout/', help='Directory name to save the model')


    return check_args(parser.parse_args())


trainF=1
use_cuda=1
if __name__ == '__main__':

    args = parse_args()
    args.y_dim = args.num_bin*args.num_mfeat
    args.savepath = 'flyHRNN_'+str(args.num_steps)+'steps_'\
                             +str(args.batch_sz)+'batch_sz_'\
                             +str(args.epoch)+'epochs_'\
                             +str(args.lr)+'lr_'\
                             +str(args.num_bin)+'bins_'\
                             +str(args.h_dim)+'hids_'\
                             +args.rnn_type\
                             +'_onehot'+str(args.onehot)\
                             +'_visionF'+str(args.visionF)\
                             +'_vtype:'+str(args.vtype)\
                             +'_dtype:'+str(args.dtype)\
                             +'_btype:'+str(args.bin_type)
    if args.gender == MALE:
        args.savepath += '_maleflies'
    elif args.gender == FEMALE:
        args.savepath += '_femaleflies'
    
    args.parentpath='/groups/branson/home/imd/Documents/janelia/research/fly_behaviour_sim/71g01/models/%s/' % args.dtype
    if not os.path.exists(args.parentpath): os.makedirs(args.parentpath)
    print(args.parentpath)

    model = FlyNetworkSKIP6(args)
    model = model.cuda()

    if trainF:
        trainIters(model, args.epoch, print_every=50)
    else:
        pass

    print(model)
    print(args.savepath)


