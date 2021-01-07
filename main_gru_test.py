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

from flyNetwork_RNN import FlyNetworkGRU, FlyNetworkSKIP6
from util import *
from gen_dataset import sample_batch_videos, test_batch_videos, \
                        load_videos, quick_sample_batch_videos, gender_classify


SOURCE=0
TARGET=1
MOTION=2
MALE=0
FEMALE=1
TRAIN=0
VALID=1
TEST=2

def videos_numpy_to_tensor(videos, device):
    return [[torch.tensor(f).float().to(device) if type(f) == np.ndarray else f for f in video] \
            for video in videos]

def sample_batch_videos_tensor(videos, genderF=2, onehotF=0, batch_size=10000, concatF=1, \
                        num_frames=50, etype='tr', itype='full', device="cuda"):
    num_video = len(videos)

    # Give equal number of examples to each video.  If the batch size isn't
    # divisible by the number of videos, assign the remainder randomly
    num, remainder = batch_size // num_video, batch_size % num_video
    num_per_video = [num + int(i < remainder) for i in range(num_video)]
    num_per_video = np.array(num_per_video)[np.random.permutation(num_video)]

    vsources, msources = [], []
    
    for j in range(num_video):
      num = num_per_video[j]
      while num > 0:  # this loop is only necessary if some target sequences are invalid
        vision_data, motion_data, binned_motion_data, _, basesize = videos[j]
        male_ind, female_ind = gender_classify(basesize['majax'])
        ind = male_ind if genderF==MALE else female_ind

        # Randomly select sequences within the video
        num_flies = len(ind)
        num_samples = int(math.ceil(num / float(num_flies)))
        start_times = torch.randint(2, vision_data.shape[0] - num_frames, size=[num_samples])
        
        # For each starting frame and fly pair, extract target data
        # TODO: do this without a loop as a single tensor
        target = torch.stack([binned_motion_data[t : t + num_frames, ind, :] for t in start_times], axis=0)
        target = target.transpose(1, 2)
        target = target.reshape((target.shape[0] * target.shape[1], target.shape[2], target.shape[3]))
        valid = target.sum(axis=(1, 2)) > 0  # to match Daniel's code, unsure why this is necessary
        num_valid = valid.sum().item()
        if num_valid < target.shape[0]:
            target = target[valid, :]
        target = target[:min(num_valid, num), :]

        # For each starting frame and fly pair, extract source feature data that concatenates
        # vision and motion data
        vdata = torch.stack([vision_data[t : t + num_frames, ind, :] for t in start_times], axis=0)
        mdata = torch.stack([motion_data[t : t + num_frames, ind, :] for t in start_times], axis=0)
        vdata[torch.isinf(vdata)] = 0.
        source = torch.cat([vdata, mdata], axis=3)
        source[torch.isnan(source)] = 0.
        source = source.transpose(1, 2)
        source = source.reshape(source.shape[0] * source.shape[1], source.shape[2], source.shape[3])
        if num_valid < source.shape[0]:
            source = source[valid, :]
        source = source[:min(num_valid, num), :]
        
        vsources.append(source)
        msources.append(target)

        #print('j=%d, num=%d, num_valid=%d' % (j, num, num_valid))
        num = num - min(num_valid, num)

    vsources = torch.cat(vsources, 0).transpose(0, 1)
    msources = torch.cat(msources, 0).transpose(0, 1)
    perm = torch.randperm(vsources.shape[1])
    return vsources[:, perm], msources[:, perm]

def sample_batch_videos(videos, genderF=2, onehotF=0, batch_size=10000, concatF=1, \
                        num_frames=50, etype='tr', itype='full', device="cuda"):
    if(type(videos[0][0]) == torch.Tensor):
        return sample_batch_videos_tensor(videos, genderF=genderF, onehotF=onehotF,
                                          batch_size=batch_size, concatF=concatF, 
                                          num_frames=num_frames, etype=etype, itype=itype,
                                          device=device)
    num_video = len(videos)

    # Give equal number of examples to each video.  If the batch size isn't
    # divisible by the number of videos, assign the remainder randomly
    num, remainder = batch_size // num_video, batch_size % num_video
    num_per_video = [num + int(i < remainder) for i in range(num_video)]
    num_per_video = np.array(num_per_video)[np.random.permutation(num_video)]

    vsources, msources = [], []
    
    for j in range(num_video):
      num = num_per_video[j]
      while num > 0:  # this loop is only necessary if some target sequences are invalid
        vision_data, motion_data, binned_motion_data, _, basesize = videos[j]
        male_ind, female_ind = gender_classify(basesize['majax'])
        ind = male_ind if genderF==MALE else female_ind

        # Randomly select sequences within the video
        num_flies = len(ind)
        num_samples = int(math.ceil(num / float(num_flies)))
        start_times = np.random.randint(2, vision_data.shape[0] - num_frames, size=num_samples)
        
        # For each starting frame and fly pair, extract target data
        target = np.array([binned_motion_data[t : t + num_frames, ind, :] for t in start_times])
        target = target.transpose(0, 2, 1, 3)
        target = target.reshape(target.shape[0] * target.shape[1], target.shape[2], target.shape[3])
        valid = target.sum(axis=(1, 2)) > 0  # to match Daniel's code, unsure why this is necessary
        num_valid = valid.sum()
        if num_valid < target.shape[0]:
            target = target[valid, :]
        target = target[:min(num_valid, num), :]

        # For each starting frame and fly pair, extract source feature data that concatenates
        # vision and motion data
        vdata = np.array([vision_data[t : t + num_frames, ind, :] for t in start_times])
        mdata = np.array([motion_data[t : t + num_frames, ind, :] for t in start_times])
        vdata[np.isinf(vdata)] = 0.
        source = np.concatenate([vdata, mdata], axis=3)
        source[np.isnan(source)] = 0.
        source = source.transpose(0, 2, 1, 3)
        source = source.reshape(source.shape[0] * source.shape[1], source.shape[2], source.shape[3])
        if num_valid < source.shape[0]:
            source = source[valid, :]
        source = source[:min(num_valid, num), :]
        
        vsources.append(source)
        msources.append(target)

        #print('j=%d, num=%d, num_valid=%d' % (j, num, num_valid))
        num = num - min(num_valid, num)

    vsources = np.concatenate(vsources, 0).transpose(1, 0, 2)
    msources = np.concatenate(msources, 0).transpose(1, 0, 2)
    perm = RNG.permutation(vsources.shape[1])
    vsources = torch.from_numpy(vsources[:,perm]).float().to(device)
    msources = torch.from_numpy(msources[:,perm]).float().to(device)
    return vsources, msources



def feval(input_variable, target_variable, model, optimizer, 
                                            num_bins=51,\
                                            max_length=100, \
                                            mode='train', \
                                            N=100, \
                                            accuracyMetricF=0, \
                                            use_cuda=1,
                                            hiddenF=1):

    if mode=='train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    loss = 0
    T, batch_sz, D = input_variable.size()
    num_feat = args.num_mfeat
    hidden = model.initHidden(N, use_cuda=use_cuda) 

    
    output, hidden_init, hidden \
                = model.forward(input_variable, hidden, hiddenF=hiddenF)
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
    score = log_likelihood(prediction, target_variable, \
                                           NB=args.num_mfeat, T=T)

    if accuracyMetricF:

        targetGauss_np = convTarget(target_variable.data.cpu().numpy(), sigma=0.25)
        score3, scoreFeat3 = metric3( prediction.data.cpu().numpy(), \
                            targetGauss_np)

        if hiddenF:
            return  loss.item(), lossNM.data.cpu().numpy(), \
                acc_rate.item(), accuracy_matrix.data.cpu().numpy(),\
                score.item(), score3, scoreFeat3, hidden

        return  loss.item(), lossNM.data.cpu().numpy(), \
                acc_rate.item(), accuracy_matrix.data.cpu().numpy(),\
                score.item(), score3, scoreFeat3

    #return loss.item(), lossNM.data.cpu().numpy(), acc_rate.item(), score.item()
    return loss, lossNM.data, acc_rate, score

def feval_batches(batches, num_batches, batch_sz, requires_grad, model, optimizer, args, mode, print_times=1):
    iter_start = time.time()
    nms = []
    for i in range(num_batches):
        input_variable  = Variable(batches[SOURCE][:,i*batch_sz:(i+1)*batch_sz], requires_grad=requires_grad)
        target_variable = Variable(batches[TARGET][:,i*batch_sz:(i+1)*batch_sz], requires_grad=requires_grad)

        load_start = time.time()
        tr_loss, tr_nm, tr_acc, tr_score = feval(input_variable, \
                            target_variable, model,\
                            optimizer, 
                            num_bins=args.num_bin,\
                            max_length=args.num_steps, \
                            mode='train', N=batch_sz)
        load_end = time.time()
        if i == 1 and print_times > 1:
            print ('%s time %f' % (mode, (load_end-load_start)/60.0))

        if i == 0:
            score_total = tr_score
            loss_total = tr_loss
            acc_total = tr_acc
        else:
            score_total += tr_score
            loss_total += tr_loss
            acc_total += tr_acc
        nms.append(tr_nm)

    score_total /= num_batches
    loss_total /= num_batches
    acc_total /= num_batches
    iter_end = time.time()
    if print_times >= 1:
        print ('%s Epoch time %f, tr_loss=%f' % (mode, (iter_end-iter_start)/60.0, tr_loss))
    
    return score_total.item(), loss_total.item(), acc_total.item(), torch.stack(nms, axis=0).cpu().numpy()

trainF=1
use_cuda=1

def lr_scheduler_init(optimizer, args):

    if args.lrsch_type == 'lambda':
        # Assuming optimizer has two groups.
        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    elif args.lrsch_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.gamma)

    elif args.lrsch_type == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    elif args.lrsch_type == 'multstep':
         scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch // 4 for i in range(1, 5)], gamma=args.gamma)

    elif args.lrsch_type == 'cyclic':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer)
    else:
        return None
    return scheduler


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of FlyNetwork collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--print_every', type=int, default=100, help='print every')#, required=True)
    parser.add_argument('--save_every' , type=int, default=50,help= 'save every')#, required=True)
    parser.add_argument('--epoch', type=int, default=10000, help='The number of epochs to run')
    parser.add_argument('--batch_sz', type=int, default=512, help='The size of batch')
    parser.add_argument('--num_steps', type=int, default=50, help='NUmber of steps')
    parser.add_argument('--lr', type=float, default=0.01)
    
    parser.add_argument('--lrsch_type', type=str, default='multstep')
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--onehot', type=int, default=0)
    parser.add_argument('--num_bin', type=int, default=101)
    parser.add_argument('--num_mfeat', type=int, default=8)
    parser.add_argument('--visionF', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=100)
    parser.add_argument('--x_dim', type=int, default=152) 
    parser.add_argument('--resume_training', type=bool, default=False) 
    parser.add_argument('--gender', type=int, default=0, help='0 = male, 1 = female')
    parser.add_argument('--l2wd', type=float, default=0.0001)
    parser.add_argument('--l1wd', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--dtype', type=str, default='gmr')
    parser.add_argument('--rnn_type', type=str, default='gru', choices=['gru', 'hrnn'])
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--vtype', type=str, default='full')
    parser.add_argument('--bin_type', type=str, default='perc', choices=['linear', 'perc'])
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--file_dir', type=str, \
            default='./fout/', help='Directory name to save the model')
    parser.add_argument('--mypath', type=str, default="./bins/")

    return check_args(parser.parse_args())

args = parse_args()
if args.save_dir is None:
    args.save_dir = './models/%s/' % args.dtype
args.y_dim = args.num_bin*args.num_mfeat
args.savepath = 'flyNet_'+args.rnn_type\
                         +str(args.num_steps)+'steps_'\
                         +str(args.batch_sz)+'batch_sz_'\
                         +str(args.epoch)+'epochs_'\
                         +str(args.lr)+'lr_'\
                         +str(args.num_bin)+'bins_'\
                         +str(args.h_dim)+'hids_'\
                         +'_onehot'+str(args.onehot)\
                         +'_visionF'+str(args.visionF)\
                         +'_vtype:'+str(args.vtype)\
                         +'_dtype:'+str(args.dtype)\
                         +'_btype:'+str(args.bin_type)
if args.gender == MALE:
    args.savepath += '_maleflies'
elif args.gender == FEMALE:
    args.savepath += '_femaleflies'

args.parentpath='./models/%s/' % args.dtype
if not os.path.exists(args.parentpath): os.makedirs(args.parentpath)
print(args.parentpath)


## LOAD MODEL
if args.rnn_type == 'hrnn':
    model = FlyNetworkSKIP6(args)
else:
    model = FlyNetworkGRU(args)

model = model.cuda()


print_every=100
plot_every=100
n_iters = args.epoch

iter = 1
if args.resume_training:
    iter = print_every*5
    while os.path.exists(args.save_dir+ args.savepath+'_'+str(iter)):
        iter += print_every*5
    iter -= print_every*5
    iter = max(iter, 1)
    if os.path.exists(args.save_dir+ args.savepath+'_'+str(iter)):
        model = load(model, args.save_dir+ args.savepath+'_'+str(iter)) 

start = time.time()
plot_losses_tr, plot_losses_vl = [], []
tr_accs, vl_accs, tr_nms, vl_nms, tr_scores, vl_scores = [], [], [], [], [], []
#optimizer = optim.SGD(model.parameters(), lr=args.lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler_init(optimizer, args)

## Load Videos
video_data_list = load_videos(onehotF=args.onehot, vtype=args.vtype,\
                            dtype=args.dtype, bin_type=args.bin_type,\
                              num_bin=args.num_bin, include_test=False, mypath=args.mypath)


Ntr = 5000  # Number of (sampled) training examples per "epoch"
videos = video_data_list[TRAIN]
batch_size=K=Ntr
num_frames=tau=50
etype='tr'
genderF=args.gender
device=torch.device("cuda")
STORE_DATASETS_ON_GPU = [True, False, False]

video_data_list = [videos_numpy_to_tensor(v, device) if gpu else v for gpu, v in zip(STORE_DATASETS_ON_GPU, video_data_list)]


## Sample Valid Data
load_start = time.time()
valid_batch = sample_batch_videos(video_data_list[VALID], batch_size=10000, \
                    num_frames=50, etype='vl', \
                                  genderF=args.gender, device=device)
load_end = time.time()
print ('Validation Data Loading time %f' % ((load_end-load_start)/60.0))





batch_sz = args.batch_sz
Ntr = batch_sz
for iter in range(iter, n_iters + 1):

    ## Sample Train Data
    if True: #iter % 20 == 1:
        load_start = time.time()
        train_batch = sample_batch_videos(video_data_list[TRAIN], \
                        batch_size=Ntr, num_frames=50, etype='tr', \
                        genderF=args.gender, device=device)
        load_end = time.time()
        if iter == 1 or iter % (print_every/10) == 0:
            print ('Loading time %f' % ((load_end-load_start)/60.0))


    num_batches = int(Ntr/batch_sz)
    score_total, loss_total, acc_total, tr_nm = feval_batches(train_batch, num_batches, batch_sz, True, model, optimizer, args, "train", print_times=2 if iter==1 else iter % (print_every/10) == 0)


    if iter % print_every == 0 or iter == 1: 
        Nvl = valid_batch[0].shape[1]

        vl_nm = [] 
        vl_batch_sz = 200
        num_batch_vl = int(float(Nvl) / vl_batch_sz)
        vl_score, vl_loss, vl_acc, vl_nm = feval_batches(valid_batch, num_batch_vl, vl_batch_sz, False, model, optimizer, args, "valid", int(iter == 1  or iter % print_every * 100 == 0))

        tr_accs.append(acc_total) 
        vl_accs.append(vl_acc)

        tr_nms.append(tr_nm) 
        tr_scores.append(score_total)
        vl_scores.append(vl_score)

        vl_nms.append(vl_nm) 
        plot_losses_tr.append(loss_total)
        plot_losses_vl.append(vl_loss)

        lr_ = optimizer.param_groups[0]['lr']
        print('%s (%d %d%%) TR Loss : %.4f, Tr Score %.4f, TR Acc : %.6f, VL Loss : %.4f, VL Score %.4f, VL Acc : %.6f, LR %f' \
                                % (timeSince(start, iter / n_iters),
                                     iter, iter / n_iters * 100, \
                                     loss_total, score_total,\
                                     acc_total, vl_loss , vl_score,\
                                     vl_acc, lr_))


    if iter % (print_every*5) == 0:
        if args.save_dir is not None:
            save(model, args.save_dir+ args.savepath+'_'+str(iter)) 
        else:
            save(model, args.parentpath + args.savepath+'_'+str(iter))

    scheduler.step()

print('Model saved to %s' % args.savepath)
if not os.path.exists('./results/'+args.dtype): os.makedirs('./results/'+args.dtype)
np.save('./results/'+args.dtype+'/'+args.savepath+'_loss_motion_features_tr', np.asarray(tr_nms))
np.save('./results/'+args.dtype+'/'+args.savepath+'_loss_motion_features_vl', np.asarray(vl_nms))
np.save('./results/'+args.dtype+'/'+args.savepath+'_loss_tr', np.asarray(plot_losses_tr))
np.save('./results/'+args.dtype+'/'+args.savepath+'_loss_vl', np.asarray(plot_losses_vl))
np.save('./results/'+args.dtype+'/'+args.savepath+'_scores_tr', np.asarray(tr_scores))
np.save('./results/'+args.dtype+'/'+args.savepath+'_scores_vl', np.asarray(vl_scores))
np.save('./results/'+args.dtype+'/'+args.savepath+'_acc_tr', np.asarray(tr_accs))
np.save('./results/'+args.dtype+'/'+args.savepath+'_acc_vl', np.asarray(vl_accs))
