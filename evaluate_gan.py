import os, h5py, argparse, time, argparse
import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})

import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from evaluate_nstep import real_flies_simulatePlan_RNNs
from gen_dataset import load_eyrun_data, combine_vision_data, \
                        video16_path, gender_classify, load_vision,\
                        normalize_pos, gen_dataset
#from util_vis import * 
from util import *
from util_fly import get_default_fly_colors_single,\
                        draw_flies, compute_vision, motion2binidx, \
                        compute_motion, update_flies
from simulate_rnn import get_nstep_comparison_rnn, init_simulation, \
                           get_real_fly, get_simulate_fly, init_canvas
import simulate_rnn as srnn
import simulate_autoreg as sauto

from tqdm import tqdm
from binaryClassifier import ConvNet, LogisticRegression


RNG = np.random.RandomState(0)

TRAIN=0
VALID=1
TEST=2
MALE=0
FEMALE=1
NUM_FLY=20
NUM_MFEAT=8
FPS=30
PPM=7.790785

def get_trx_data(simtrx, flyvisions, t, tsim, sim_fly_ind):

    xs = simtrx['x'][t-tsim:t, sim_fly_ind]
    ys = simtrx['y'][t-tsim:t, sim_fly_ind]
    thetas = simtrx['theta'][t-tsim:t, sim_fly_ind]
    aa = simtrx['a'][t-tsim:t, sim_fly_ind]
    bb = simtrx['b'][t-tsim:t, sim_fly_ind]
    l_wing_angs = simtrx['l_wing_ang'][t-tsim:t, sim_fly_ind]
    r_wing_angs = simtrx['r_wing_ang'][t-tsim:t, sim_fly_ind]
    l_wing_lens = simtrx['l_wing_len'][t-tsim:t, sim_fly_ind]
    r_wing_lens = simtrx['r_wing_len'][t-tsim:t, sim_fly_ind]

    flyvisions  = np.asarray(flyvisions).transpose(2,0,1)
    point = np.asarray([xs, ys, thetas, aa, bb,\
                                l_wing_angs, r_wing_angs, \
                                l_wing_lens, r_wing_lens])
    data = np.concatenate([point,flyvisions])
    data = data.transpose(2,1,0)
    return data 


def get_precision_recall(dtype='gmr', ltype='sq', gender='male'):

    if dtype == 'gmr':
        model_list = ['gmr91', 'pdb']
        label_list = ['R91B01', 'CONTROL']
    elif  dtype == 'gmr91':
        model_list = ['gmr', 'pdb']
        label_list = ['R71G01', 'CONTROL']
    elif  dtype == 'pdb':
        model_list = ['gmr', 'gmr91']
        label_list = ['R71G01', 'R91B01']

    model_list += [dtype, 'MF', 'lr50', 'conv4_cat50', 'rnn50', 'skip50']
    label_list += ['TRAIN DATA', "MALE-FEMALE", 'LINEAR', 'CNN', 'RNN', 'HRNN']
    color_list = ['navy', 'silver', 'gray', 'black', 'red', 'green', 'deepskyblue', 'mediumpurple']
       
    from load_gan import get_rscr
    data = get_rscr(ltype, dtype, gender, model_list)

    plt.figure()
    ax = plt.axes([0,0,1,1])
    tptns = [] 
    for i, mtype in enumerate(model_list):

        bookkeep = []
        for thrd in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            data_i = data[mtype]
            preds = data_i[0]
            labels = data_i[1]
            
            tne, tpe, tp, tn = get_tne_tpe(labels, preds, thrd=thrd)
            bookkeep.append([tn, tp])
            #bookkeep.append([tne, tpe])
        tptns.append(bookkeep)

        xy = np.asarray(bookkeep)
        plt.plot(xy[:,0], xy[:,1], ls='-', marker='x', color=color_list[i], label=label_list[i])

    #plt.axvline(x=0.6875, linewidth=2, color='black')
    plt.xlabel('Real Accuracy')
    plt.ylabel('Fake Accuracy')

    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.xlim([0,1])
    #plt.ylim([0.4,1])

    ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5,1.32), ncol=3)
    ax.get_xaxis().set_visible(True)

    os.makedirs('./figs/ganeval/', exist_ok=True)
    #plt.savefig('./figs/ganeval/%s/%s_%s_gan_prec_recall.pdf' % (dtype, gender, ltype), format='pdf', bbox_inches='tight') 
    plt.savefig('./figs/ganeval/%s/%s_%s_gan_tp_tn.pdf' % (dtype, gender, ltype), format='pdf', bbox_inches='tight') 
    plt.close()
    return tptns



def simulation_with_gan_score(args, data_stats, model, mtype, vpath, t0, t1, tsim,\
        burning=100, model_epoch=200000, DEBUG=1, plottrxlen=100):

    sim_type = 'SMSF'
    real_vs_sim = 'sim'
    fname = 'eyrun_simulate_data.mat'
    matfile = args.datapath+vpath+fname
    (trx,motiondata,params,basesize) = load_eyrun_data(matfile)
    params['mtype'] = mtype

    if real_vs_sim == 'sim':
        real_male_flies = np.arange(1,10,1)
        real_female_flies = np.arange(10,20,1)
        simulated_male_flies = np.arange(0,1,1)
        simulated_female_flies = []
    else:
        real_male_flies = np.arange(0,10,1)
        real_female_flies = np.arange(10,20,1)
        simulated_male_flies = []
        simulated_female_flies = []

    if 'rnn' in mtype or 'skip' in mtype:
        num_hid=100
        male_model, female_model, \
                male_hiddens, female_hiddens \
                = srnn.model_selection(None, None, 'full', \
                            mtype, 200000, num_hid,\
                simulated_male_flies, simulated_female_flies, dtype)
    else: 
        num_hid=1000
        male_model, female_model\
                = sauto.model_selection(None, None, params,  \
                    mtype=mtype, model_epoch=35000, num_hid=num_hid)



    n_flies= trx['x'].shape[1]
    simtrx, x, y, theta, a, b, l_wing_ang, r_wing_ang, l_wing_len, \
            r_wing_len, male_state, female_state, feat_motion = \
                    init_simulation(trx, params, motiondata, basesize, \
                    n_flies, t0, t1, real_male_flies, real_female_flies, \
                    simulated_male_flies, simulated_female_flies, sim_type)

    htrx, hbodies, hflies, htexts, hbg, counter_plt, colors0 = \
            init_canvas(params, x,y,a,b,theta,l_wing_ang,r_wing_ang,\
                        l_wing_len,r_wing_len, n_flies, DEBUG=DEBUG,\
                        sim_type='Single')

    xprev = x.copy()
    yprev = y.copy()
    thetaprev = theta.copy()
    flyvisions=[]

    model.eval()

    print('Simulation Start...\n')
    for counter, t in tqdm(enumerate(range(t0+1,t1))):

        xprev[:] = x
        yprev[:] = y
        thetaprev[:] = theta

        ## Simulate Male Model
        x, y, theta, a, l_wing_ang, r_wing_ang, \
            l_wing_len, r_wing_len, \
            male_hiddens, male_sim_vision_chamber, feat_motion = \
            get_simulate_fly(male_model, male_state, male_hiddens, t, trx,\
                             simulated_male_flies, feat_motion,\
                                x, y, theta, a, b, \
                                l_wing_ang, r_wing_ang,\
                                l_wing_len, r_wing_len,\
                                xprev, yprev, thetaprev, 
                                basesize, params, mtype)

        ## Simulate Female Model
        #x, y, theta, a, l_wing_ang, r_wing_ang, \
        #    l_wing_len, r_wing_len, \
        #    female_hiddens, female_sim_vision_chamber, feat_motion =\
        #    get_simulate_fly(female_model, female_state, female_hiddens, t, trx,\
        #                     simulated_female_flies, feat_motion,\
        #                        x, y, theta, a, b, \
        #                        l_wing_ang, r_wing_ang,\
        #                        l_wing_len, r_wing_len,\
        #                        xprev, yprev, thetaprev, 
        #                        basesize, params, mtype)

        ## Real male Model
        x, y, theta, a, \
            l_wing_ang, r_wing_ang, \
            l_wing_len, r_wing_len, male_real_vision_chamber\
            = get_real_fly(real_male_flies, \
                                        motiondata, feat_motion,\
                                        t, trx, x, y, theta, 
                                        l_wing_ang, r_wing_ang,\
                                        l_wing_len, r_wing_len,\
                                        a, b, params)

        ## Real female Model
        x, y, theta, a, \
            l_wing_ang, r_wing_ang, l_wing_len, r_wing_len, \
            female_real_vision_chamber\
            = get_real_fly(real_female_flies, \
                                        motiondata, feat_motion,\
                                        t, trx, x, y, theta, 
                                        l_wing_ang, r_wing_ang,\
                                        l_wing_len, r_wing_len,\
                                        a, b, params)
       
        #flyvisions.append(np.vstack([male_sim_vision_chamber,\
        #                            female_sim_vision_chamber]))
        if real_vs_sim == 'real':
            flyvisions.append([male_real_vision_chamber[0]]) if 1 else [female_real_vision_chamber[0]]
        else:
            flyvisions.append(male_sim_vision_chamber) if 1 else female_sim_vision_chamber
        flyvisions = flyvisions[-tsim:]
       
        if (t-t0-1) < burning: 
            for fly in range(n_flies):
                x[fly] = trx['x'][t,fly]
                y[fly] = trx['y'][t,fly]
                theta[fly] = trx['theta'][t,fly]
                a[fly] = trx['a'][t,fly]
                l_wing_ang[fly] = trx['l_wing_ang'][t,fly]
                r_wing_ang[fly] = trx['r_wing_ang'][t,fly]
                l_wing_len[fly] = trx['l_wing_len'][t,fly]
                r_wing_len[fly] = trx['r_wing_len'][t,fly]
        else:
            if real_vs_sim == 'real':
                target_fly_ind = np.asarray([0]) \
                    if 1 else  np.asarray([10])
            else:
                target_fly_ind = simulated_male_flies \
                    if 1 else  simulated_female_flies

            data = get_trx_data(simtrx, flyvisions, t, tsim, target_fly_ind)
            data[np.isnan(data)] = 0.
            data = np.concatenate([data[:,:,:2], data[:,:,8:]], axis=2)
            data = normalize_pos(data)
            data = data[:,data.shape[1]-(tsim-2):]
            data = (data - data_stats[0])/data_stats[1]
            data[np.isinf(data)] = 0.

            X = torch.FloatTensor(data.transpose([0,2,1]))
            X = Variable(X.cuda())
            Y = torch.sigmoid(model(X)).flatten().data.cpu().numpy()

            pred = (Y > 0.5) * 1.
            #if np.sum(pred) > 0:
            #    import pdb; pdb.set_trace()


        simtrx['x'][t,:] = x
        simtrx['y'][t,:]=y
        simtrx['theta'][t,:]=theta
        simtrx['a'][t,:]=a
        simtrx['b'][t,:]=b
        simtrx['l_wing_ang'][t,:]=l_wing_ang
        simtrx['r_wing_ang'][t,:]=r_wing_ang
        simtrx['l_wing_len'][t,:]=l_wing_len
        simtrx['r_wing_len'][t,:]=r_wing_len

        if DEBUG==1 and (t-t0-1) > burning:
            tprev = np.maximum(t0+1,t-plottrxlen)
            for fly in range(n_flies):
                htrx[fly].set_data(simtrx['x'][tprev:t+1,fly],simtrx['y'][tprev:t+1,fly])

            colors = colors0.copy()
            if pred[0] > 0: colors[0] = (0,1,0,1) 
            update_flies(hbodies,hflies,htexts,x,y,a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len, colors=colors)
            plt.axis('off')
            plt.pause(.001)
            counter_plt.set_text('{:.2f}sec'.format(counter / FPS))

            plt.annotate('{:.2f}ppm'.format(PPM*10),
                            xy=[55,params['bg'].shape[0]-45],
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', size=14, color='black')
            plt.plot([20,20+PPM*10],[params['bg'].shape[0]-40,params['bg'].shape[0]-40],'-',color='black',linewidth=2.)
 
            if t % 1 == 0 and t < t1 : #/10.0:
                if 'rnn' in params['mtype'] or 'skip' in params['mtype']:
                    os.makedirs('./figs/ganeval/sim/%s/' % vpath, exist_ok=True)   
                    plt.savefig('./figs/ganeval/sim/%s/' % vpath +params['mtype']\
                            +'_Single' +'_%s_full_%shid' % (real_vs_sim, num_hid)\
                            +'_epoch%d_32bs_v3_%5d.png' % (model_epoch,t), format='png')
                else:
                    plt.savefig('./figs/all/data_1000frames_%s_%5d.png' % (videotype, t), format='png', bbox_inches='tight')


def visualize_negative_examples(args, dtype, fakedata, model, simulated_male_flies,\
                                vpath, t0, t1, tsim, mtype):

    fname = 'eyrun_simulate_data.mat'
    matfile = args.datapath+vpath+fname
    (trx,motiondata,params,basesize) = load_eyrun_data(matfile)
    
    frame_ind = np.load('./fakedata/%s/frame_index_%dt0_%dt1_60tsim.npy' % (vpath, t0, t1))

    malefakedata = fakedata[:,simulated_male_flies]
    malefakedata = malefakedata[:,:,malefakedata.shape[2]-(tsim-2):,:]
    A,B,C,D = malefakedata.shape

    malefakedata = malefakedata.reshape([A*B,C, 153])
    malefakedata = normalize_pos(malefakedata)

    malefakedata[np.isnan(malefakedata)] = 0.
    malefakedata = np.concatenate([malefakedata[:,:,:2], malefakedata[:,:,8:]], axis=2)

    X = torch.FloatTensor(malefakedata.transpose([0,2,1])).cuda()
    pred = F.sigmoid(model(X).flatten())
    y = pred < 0.5 
   
    y = y.reshape([A,B]).data.cpu().numpy()
    neg_ind = np.argwhere(y) 

    N, M = neg_ind.shape
    NN = min(10,N)
    for k in range(NN):
        print(k)
        i,fly_j = neg_ind[k]
        data = fakedata[i, fly_j][-tsim+1:]

        frame_i = frame_ind[i] 

        #if data.shape[0] >= 58:
        # even flies are simulated, odd are real
        for ii in range(tsim-2):

            x = trx['x'][frame_i+ii,:].copy()
            y = trx['y'][frame_i+ii,:].copy()
            theta = trx['theta'][frame_i+ii,:].copy()


            x[fly_j] = data[ii,0]
            y[fly_j] = data[ii,1]
            theta[fly_j] = data[ii,2]

            b = basesize['minax'].copy()
            a = trx['a'][frame_i+ii,:].copy()
            l_wing_ang = trx['l_wing_ang'][frame_i+ii,:].copy()
            r_wing_ang = trx['r_wing_ang'][frame_i+ii,:].copy()
            l_wing_len = trx['l_wing_len'][frame_i+ii,:].copy()
            r_wing_len = trx['r_wing_len'][frame_i+ii,:].copy()

            n_flies = len(x)
            if ii == 0:
                htrx, hbodies, hflies, htexts, counter_plt = visualize_init_frame(params,fly_j, n_flies, \
                        x,y,a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len) 
            else:
                htrx, hbodies, hflies, htexts, counter_plt = visualize_update_frame(params, n_flies, htrx, hbodies, hflies, htexts, counter_plt, ii, \
                        x, y, a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len)

            os.makedirs('./figs/ganeval/neg_example/%s/' % dtype, exist_ok=True)
            plt.savefig('./figs/ganeval/neg_example/%s/%s_frame%d_fly%d_%d_%2d.png' \
                % (dtype, mtype, i, fly_j, tsim, ii), format='png', bbox_inches='tight')

    plt.close()


def visualize_update_frame(params, n_flies, htrx, hbodies, hflies, htexts, counter_plt, counter, \
        x, y, a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len, colors=None):
    
    for fly in range(n_flies):
        htrx[fly].set_data(x[fly],y[fly])

    update_flies(hbodies,hflies,htexts,x,y,a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len, colors=colors)
    counter_plt.set_text('{:.2f}sec'.format(counter / FPS))

    plt.annotate('{:.2f}ppm'.format(PPM*10),
                     xy=[55,params['bg'].shape[0]-45],
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', size=14, color='black')
    plt.plot([20,20+PPM*10],[params['bg'].shape[0]-40,params['bg'].shape[0]-40],'-',color='black',linewidth=2.) 
    plt.axis('off')
    
    return htrx, hbodies, hflies, htexts, counter_plt


def visualize_init_frame(params, fly_j, n_flies, x, y, a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len, colors=None):

    fig,ax = plt.subplots(figsize=(15,15))
    ax = plt.axes([0,0,1,1])

    if colors is None:
        colors = get_default_fly_colors_single(fly_j, n_flies)

    hbg = plt.imshow(params['bg'],cmap=cm.gray,vmin=0.,vmax=1.)
    htrx = []
    for fly in range(n_flies):
        htrxcurr, = ax.plot(x[fly],y[fly],'-',color=np.append(colors[fly,:-1],.5),linewidth=3)
        htrx.append(htrxcurr)

    hbodies,hflies,htexts = draw_flies(x,y,a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len,ax=ax,colors=colors, textOff=0,linewidth=5)
    counter_plt = plt.annotate('{:.2f}sec'.format(0. / FPS),
                        xy=[1024-55,params['bg'].shape[0]-45],
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', size=18, color='black')
    plt.axis('off')
    return htrx, hbodies, hflies, htexts, counter_plt



def feval(input_variable, target_variable, model, optimizer,
                                            ltype='sq',\
                                            mode='train', \
                                            N=100, \
                                            use_cuda=1):

    if mode=='train':
        model.train()
    else:
        model.eval()

    if mode == 'train': optimizer.zero_grad()
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
  
    T, batch_sz, D = input_variable.size()
    loss, acc, pred = model.loss(input_variable, target_variable, ltype=ltype)

    if mode=='train':
        loss.backward()
        optimizer.step()

    return loss.item(), acc.item(), pred



def main(args, train_set, valid_set, test_set):

    model = ConvNet(args.x_dim, args.c_dim, args)
    if args.use_cuda: model.cuda()

    start = time.time()
    plot_losses_tr, plot_losses_vl = [], []
    epoch_list, tr_acc_list, vl_acc_list = [], [], []
    tr_accs, vl_accs, tr_nms, vl_nms, tr_scores, vl_scores \
                                        = [], [], [], [], [], []
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    Ntr = train_set[0].shape[0]
    from tqdm import tqdm
    for iter in tqdm(range(1, args.epoch+ 1)):

        ## Sample Data
        perm_tr = RNG.permutation(Ntr)[:args.batch_sz]
        X = train_set[0][perm_tr]#.view([args.batch_sz,-1])
        X = torch.FloatTensor(X.transpose([0,2,1]))
        Y = torch.FloatTensor(train_set[1][perm_tr])
        if args.use_cuda: X, Y = X.cuda(), Y.cuda()
        X = Variable(X)
        Y = Variable(Y)

        batch_sz = args.batch_sz
        num_batches = int(Ntr/batch_sz)

        tr_loss, tr_acc, tr_pred = feval(X, Y, model,\
                                optimizer, ltype=args.ltype,\
                                mode='train', N=args.batch_sz)

        if iter % 10 == 1:
            Nvl = valid_set[0].shape[0]
            batch_sz = (args.batch_sz)
            num_batch = Nvl // batch_sz
            vl_losses, vl_accs = [], []
            for ii in range(num_batch):

                X = valid_set[0][ii*batch_sz:(ii+1)*batch_sz]
                X = torch.FloatTensor(X.transpose([0,2,1]))
                Y = torch.FloatTensor(valid_set[1][ii*batch_sz:(ii+1)*batch_sz])
                if args.use_cuda: X, Y = X.cuda(), Y.cuda()
                X = Variable(X)
                Y = Variable(Y)

                vl_loss, vl_acc, vl_pred = feval(X, Y, model,\
                                        optimizer, \
                                        mode='eval', ltype=args.ltype,\
                                        N=batch_sz)
                vl_losses.append(vl_loss)
                vl_accs.append(vl_acc)

            #if scheduler: scheduler.batch_step()
            print('tr loss %f tr acc %f | vl loss %f vl acc %f' \
                        % (tr_loss, tr_acc, \
                            np.mean(vl_losses), \
                            np.mean(vl_accs)))
            epoch_list.append(iter)
            tr_acc_list.append(np.mean(tr_accs))
            vl_acc_list.append(np.mean(vl_accs))

    Nte = test_set[0].shape[0]
    batch_sz = (args.batch_sz)
    num_batch = Nte // batch_sz #+ int(Nte % batch_sz)
    te_losses, te_accs, te_preds = [], [], []
    precisions, recalls = [], []

    for ii in range(num_batch):
        
        X = test_set[0][ii*batch_sz:(ii+1)*batch_sz]
        X = X[:,:,:args.x_dim]
        X = torch.FloatTensor(X.transpose([0,2,1]))
        Y_=test_set[1][ii*batch_sz:(ii+1)*batch_sz]
        Y = torch.FloatTensor(Y_)
        if args.use_cuda: X, Y = X.cuda(), Y.cuda()
        X = Variable(X)
        Y = Variable(Y)

        te_loss, te_acc, te_pred = feval(X, Y, model,\
                                optimizer, \
                                mode='eval', ltype=args.ltype,\
                                N=batch_sz)

        #precision, recall = get_tne_tpe(Y_, te_pred)
        te_preds.append(te_pred)
        te_losses.append(te_loss)
        te_accs.append(te_acc)
        #precisions.append(precision)
        #recalls.append(recall)
        #if precision

    te_preds = np.hstack(te_preds)
    O = te_preds.shape[0]
    precision, recall, _, _ = get_tne_tpe(test_set[1][:O], te_preds)

    #if scheduler: scheduler.batch_step()
    print('tr loss %f tr acc %f | te loss %f te acc %f' \
                % (tr_loss, tr_acc, \
                    np.mean(te_losses), \
                    np.mean(te_accs)))
                
    N = te_preds.shape[0]
    te_target = test_set[1][:N]
    te_errors = ((te_preds >0.5) != te_target)
    return model, np.asarray(epoch_list), np.asarray(tr_acc_list), \
            np.asarray(vl_acc_list), np.mean(te_accs), np.mean(te_losses), te_errors,\
            precision, recall, te_preds



def get_tne_tpe(label, pred, thrd=0.5):

    fake_mask = (pred < thrd) * 1.
    real_mask = (pred >= thrd)* 1.

    neg_labels = (label == 0)* 1.
    pos_labels = (label == 1)* 1.

    NP = pos_labels.sum()
    NN = neg_labels.sum()
    tn = fake_mask * neg_labels
    tp = real_mask * pos_labels
    fn = pos_labels * fake_mask
    fp = neg_labels * real_mask

    #TODO true negative error rate & true positive error rate
    precision = tp.sum() / (fp.sum() + tp.sum())
    recall = tp.sum() / (fn.sum() + tp.sum()) 
   
    return recall, precision, tp.sum() / NP, tn.sum() / NN


def real_traj(t0, t1, vpath, tsim=15, gender=0):

    #(trx,motiondata,params,basesize) = load_eyrun_data('/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Code/eyrun_simulate_data.mat') 
    #real_vdata = np.load('/groups/branson/home/imd/Documents/data/fly_tracking/oct8/vdata.npy')

    vision_matfile = args.datapath+vpath+'movie-vision.mat'
    vc_data = load_vision(vision_matfile)[1:]

    matfile = args.datapath+vpath+fname
    trx,motiondata,params,basesize = load_eyrun_data(matfile)
    binedges = params['binedges']

    male_ind, female_ind = gender_classify(basesize['majax'])        

    xs = trx['x'][t0,:].copy()
    ys = trx['y'][t0,:].copy()
    thetas = trx['theta'][t0,:].copy()
 
    b = basesize['minax'].copy()
    a = trx['a'][t0,:].copy()
    l_wing_angs = trx['l_wing_ang'][t0,:].copy()
    r_wing_angs = trx['r_wing_ang'][t0,:].copy()
    l_wing_lens = trx['l_wing_len'][t0,:].copy()
    r_wing_lens = trx['r_wing_len'][t0,:].copy()

    iterator = male_ind if gender ==0 else female_ind
 
    frame_ind = [] 
    data = []
    N = trx['x'].shape[0]
    for ii in range(t0,t1,tsim):
        tmp = [] 
        points = []
        for fly in iterator:

            # vision features for frame t-1
            x = trx['x'][ii:ii+tsim, fly]
            y = trx['y'][ii:ii+tsim, fly]
            theta = trx['theta'][ii:ii+tsim,fly]
            aa = trx['a'][ii:ii+tsim,fly]
            bb = trx['b'][ii:ii+tsim,fly]
            l_wing_ang = trx['l_wing_ang'][ii:ii+tsim,fly]
            r_wing_ang = trx['r_wing_ang'][ii:ii+tsim,fly]
            l_wing_len = trx['l_wing_len'][ii:ii+tsim,fly]
            r_wing_len = trx['r_wing_len'][ii:ii+tsim,fly]


            #(flyvision,chambervision) = compute_vision(xs,ys,thetas,a,b,fly,params)
            fly_data = np.asarray([x, y, theta, aa, bb,\
                                    l_wing_ang, r_wing_ang, \
                                    l_wing_len, r_wing_len])
            point = np.hstack([fly_data.T[1:], vc_data[ii+1:ii+tsim,fly,:]])
            points.append(point)
            tmp.append([ii, fly])

        if np.isnan(np.asarray(points)).sum() == 0:
            data.append(points)
            frame_ind += tmp

    return np.asarray(data), np.asarray(frame_ind)


def gen_real_traj():

    (trx,motiondata,params,basesize) = load_eyrun_data(matfile)
    binedges = params['binedges']
    params['mtype'] = mtype

    x = trx['x'][t0,:].copy()
    y = trx['y'][t0,:].copy()
    theta = trx['theta'][t0,:].copy()
    
    # even flies are simulated, odd are real
    n_flies = len(x)
    real_flies = np.arange(0,n_flies,1)
    simulated_flies = [] 

    b = basesize['minax'].copy()
    a = trx['a'][t0,:].copy()
    l_wing_ang = trx['l_wing_ang'][t0,:].copy()
    r_wing_ang = trx['r_wing_ang'][t0,:].copy()
    l_wing_len = trx['l_wing_len'][t0,:].copy()
    r_wing_len = trx['r_wing_len'][t0,:].copy()

    xprev = x.copy()
    yprev = y.copy()
    thetaprev = theta.copy()

    state = [None]*max(len(simulated_flies), len(real_flies))

    feat_motion  = motiondata[:,t0,:].copy()
    mymotiondata = np.zeros(motiondata.shape)
    dataset = []
    pos_errors, theta_errors, wing_ang_errors, wing_len_errors \
                                                = [], [], [], []
    acc_rates, loss_rates = [], []
    print('Simulation Start...\n')
    progress = tqdm(enumerate(range(t0+1,t1)))
    for ii, t in progress:

        xprev[:] = x
        yprev[:] = y
        thetaprev[:] = theta

        for flyi in range(len(real_flies)):

            fly = real_flies[flyi]
            fly = real_flies[flyi]
            # vision features for frame t-1
            (flyvision,chambervision) = compute_vision(x,y,theta,a,b,fly,params)

            x[fly] = trx['x'][t,fly]
            y[fly] = trx['y'][t,fly]
            theta[fly] = trx['theta'][t,fly]
            a[fly] = trx['a'][t,fly]
            l_wing_ang[fly] = trx['l_wing_ang'][t,fly]
            r_wing_ang[fly] = trx['r_wing_ang'][t,fly]
            l_wing_len[fly] = trx['l_wing_len'][t,fly]
            r_wing_len[fly] = trx['r_wing_len'][t,fly]

            data = combine_vision_data(simtrx_curr, flyvisions, num_burn=5)



def get_data(dtype, ltype):

    if ltype == 'sq':

        test_acc_male_gmr71 = [0.9153658355154642, 0.7826829116518904, 0.5114634015211245, 0.8334285531725202, 0.7889654903576292, 0.6220689559804982, 0.6513792934088871, 0.7117241168844288]
        test_acc_fale_gmr71 = [0.8717142684119088, 0.6674285599163601, 0.5079999889646257, 0.8408571192196437, 0.781333311398824, 0.618999985853831, 0.7186666488647461, 0.6753333250681559]
        test_prec_male_gmr71 = [0.11748373118678733, 0.21544051612942486, 0.5141117644479272, 0.19260407503594465, 0.2213888686955301, 0.39667541099367903, 0.37454329847343826, 0.27298665920500226] 
        test_prec_fale_gmr71 = [0.16161027473754705, 0.351209051258668, 0.3560165542938318, 0.16781018608437326, 0.2705239399409491, 0.4575901186428286, 0.31390774664846877, 0.3443044299860936]
        test_rec_male_gmr71 = [0.05555743910959822, 0.23027619859988777, 0.4834215370026032, 0.14662832687038738, 0.20977835171711348, 0.37745173068784343, 0.34600796763567293, 0.3152840544518348] 
        test_rec_fale_gmr71 = [0.0972451629033187, 0.3238082475210353, 0.6504928447345157, 0.15667519944103012, 0.17787792651662768, 0.3232419158583402, 0.2612624842473847, 0.31452125010148263]

        #test_acc_male_gmr71 = [0.5265853535838243, 0.831714265687125, 0.7941379074392647, 0.6213792973551256, 0.6165517157521742, 0.6893103246031136] 
        #test_acc_fale_gmr71 = [0.5259999845709119, 0.8165714110646929, 0.8186666468779246, 0.6293333232402801, 0.732999986410141, 0.6843333184719086]

        #test_prec_male_gmr71 = [0.5302749148432627, 0.11944775753131182, 0.2138259002994689, 0.3759808272223094, 0.23251053959169193, 0.33744502258899917] 
        #test_prec_fale_gmr71 = [0.5753178724248296, 0.26096038293229323, 0.13391206135230105, 0.4206971619374475, 0.2991296805704998, 0.38758185507802806]

        #test_rec_male_gmr71 = [0.4410717233867697, 0.2227274916678184, 0.20515110295423467, 0.39589306267970914, 0.5532481478415349, 0.29775536134027536] 
        #test_rec_fale_gmr71 = [0.3860023265800577, 0.11740476290592071, 0.2345901832722845, 0.3354852009786134, 0.24764061251271172, 0.2533339904386432]

        test_acc_male_gmr91 = [0.9045454372059215, 0.6336363499814813, 0.5409090735695579, 0.8345454281026666, 0.7499999891627919, 0.5581818087534471, 0.5727272575551813, 0.5645454498854551] 
        test_acc_fale_gmr91 = [0.8704761749222165, 0.7271428392046974, 0.5609523682367235, 0.8427272493189032, 0.7799999793370564, 0.5666666547457377, 0.5879999796549479, 0.5739999930063884]
        test_prec_male_gmr91 = [0.06465200643954755, 0.32436737003850846, 0.5129868190469217, 0.1875689040936734, 0.25531294366302154, 0.47850392363041705, 0.4018764647711494, 0.4231388167325227] 
        test_prec_fale_gmr91 = [0.1329970665743684, 0.24021715327889723, 0.46968729410703375, 0.14207265755656298, 0.25795043130689604, 0.47654238879257177, 0.4296480609107436, 0.40009074479928824]
        test_rec_male_gmr91 = [0.1286914097418299, 0.41942870978840896, 0.42629836112918384, 0.15070092517241213, 0.26751006323002335, 0.41658008207380814, 0.46506671693091006, 0.47207987304409377] 
        test_rec_fale_gmr91 = [0.12976183028873362, 0.3146838928518827, 0.4235086544108654, 0.17898541397549203, 0.1894841999997812, 0.4067971392656512, 0.4103883015969117, 0.46906839485343615]

        #test_acc_male_gmr91 = [0.5009090764956041, 0.825454527681524, 0.7499999837441877, 0.5645454390482469, 0.5609090815890919, 0.578181805935773] 
        #test_acc_fale_gmr91 = [0.5404761760007768, 0.8190908919681202, 0.7919999798138936, 0.5926666537920634, 0.6353333234786988, 0.5746666590372721]

        #test_prec_male_gmr91 = [0.5727682052768643, 0.174298031034493, 0.26072199384499734, 0.41317106720210645, 0.4282407651145452, 0.48368604002610205] 
        #test_prec_fale_gmr91 = [0.4287071149077201, 0.1948227782895975, 0.24367816293948188, 0.42738224441440154, 0.2316846396792747, 0.4931594990682056]

        #test_rec_male_gmr91 = [0.4425312826358766, 0.18322434020964132, 0.2531200357683354, 0.467295111041582, 0.4674550849071858, 0.3767375920592259] 
        #test_rec_fale_gmr91 = [0.5119394620604584, 0.17666031165338592, 0.1768490921146602, 0.4120582524332427, 0.502200486646392, 0.3716462821189634]


        test_acc_male_pdb = [0.7963636111129414, 0.8222727111794732, 0.5509090789339759, 0.9627906763276388, 0.8103448049775486, 0.6137930956380121, 0.5713792998215248, 0.5886206852978674] 
        test_acc_fale_pdb = [0.6695348648137824, 0.8372092801471089, 0.5630232414533926, 0.9544185843578604, 0.7806666394074758, 0.5853333244721095, 0.5629999925692876, 0.5809999932845433]
        test_prec_male_pdb = [0.11802574323228225, 0.23411040135934913, 0.36674498544703854, 0.04473509372181823, 0.1760641892687866, 0.36351596913211337, 0.45519814393136815, 0.33729507766486355] 
        test_prec_fale_pdb = [0.3863193356633044, 0.1709984524239141, 0.31082114312902653, 0.0626898758704497, 0.2540778156397402, 0.39861281149015354, 0.411797930634267, 0.5143921910206403]
        test_rec_male_pdb = [0.29531298694268265, 0.12596073774398264, 0.5519967850968266, 0.03183441299772054, 0.20796249652475032, 0.4207017347567347, 0.42247468373422215, 0.49655420139726486] 
        test_rec_fale_pdb = [0.28876699688454177, 0.15983913366410132, 0.5761816946785377, 0.03036280023798553, 0.19391475178903778, 0.4585001693839349, 0.47808865966029157, 0.3336425348705326]

        #test_acc_male_pdb = [0.5490908947857943, 0.9586046296496724, 0.8099999736095297, 0.6013792999859514, 0.567241370677948, 0.603448261474741] 
        #test_acc_fale_pdb = [0.5665116136850312, 0.9637209052263305, 0.7936666508515676, 0.5996666550636292, 0.5733333170413971, 0.5846666485071182]

        #test_prec_male_pdb = [0.6162789481384323, 0.0264946259895358, 0.26863772648082884, 0.39537354163155275, 0.2884922263629246, 0.38954598325824397] 
        #test_prec_fale_pdb = [0.5095660922806122, 0.032422744602615364, 0.2384055278678526, 0.4131302655556465, 0.573166387687978, 0.51869123269179]

        #test_rec_male_pdb = [0.30603298135441115, 0.05737856742276026, 0.11972484558392633, 0.4180560613840844, 0.5939640159554465, 0.42860271833993974] 
        #test_rec_fale_pdb = [0.37320439355284524, 0.04237615206399457, 0.18154800230767434, 0.4030002108352454, 0.3006654066367929, 0.3326981098195913]


        if dtype=='gmr':

            test_acc_male = test_acc_male_gmr71 
            test_acc_fale = test_acc_fale_gmr71

            test_prec_male = test_prec_male_gmr71 
            test_prec_fale = test_prec_fale_gmr71

            test_rec_male = test_rec_male_gmr71 
            test_rec_fale = test_rec_fale_gmr71

        elif dtype=='gmr91':
            test_acc_male = test_acc_male_gmr91
            test_acc_fale = test_acc_fale_gmr91

            test_prec_male = test_prec_male_gmr91 
            test_prec_fale = test_prec_fale_gmr91

            test_rec_male = test_rec_male_gmr91 
            test_rec_fale = test_rec_fale_gmr91

        elif dtype == 'pdb':

            test_acc_male = test_acc_male_pdb
            test_acc_fale = test_acc_fale_pdb

            test_prec_male = test_prec_male_pdb 
            test_prec_fale = test_prec_fale_pdb

            test_rec_male = test_rec_male_pdb 
            test_rec_fale = test_rec_fale_pdb

    else:
    
        test_acc_male_gmr71 = [0.9190243729730931, 0.77170729637146, 0.5075609633108464, 0.7979999780654907, 0.8358620446303795, 0.6841379136874758, 0.6586206719793123, 0.6993103232877008] 
        test_acc_fale_gmr71 = [0.8757142663002014, 0.6577142715454102, 0.5079999923706054, 0.8477142674582345, 0.8446666558583578, 0.6923333088556926, 0.7246666530768077, 0.6736666480700175]
        test_prec_male_gmr71 = [0.11289564320917331, 0.20336364255594125, 0.6332317570849825, 0.3188621125014896, 0.1605916493866096, 0.3837710863904099, 0.36291821977946614, 0.260112990655651] 
        test_prec_fale_gmr71 = [0.18803427747984433, 0.3341823555987191, 0.5457021582123619, 0.1406636029254147, 0.17462378870412743, 0.34067677373802013, 0.2263764250284011, 0.3502346873700493]
        test_rec_male_gmr71 = [0.0532313572429562, 0.2651129938656135, 0.3711763295981283, 0.09085860006321227, 0.17434513686357625, 0.26151075563354625, 0.34064907660997223, 0.354016693514943] 
        test_rec_fale_gmr71 = [0.06331393076363827, 0.36082249514515496, 0.4609073348024484, 0.16821294927270616, 0.14296167588337058, 0.28979647057853664, 0.3341976129698627, 0.3097418509785506]

        #test_acc_male_gmr71 = [0.5217073050940909, 0.8339999777930124, 0.8441379049728657, 0.693103434710667, 0.6306896435803381, 0.7120689523631129] 
        #test_acc_fale_gmr71 = [0.5002857020923069, 0.8294285501752581, 0.8566666464010875, 0.7533333202203115, 0.718666652838389, 0.6659999867280324]

        #test_prec_male_gmr71 = [0.5198564962344625, 0.17337975898420988, 0.1906197547838807, 0.28308717878752887, 0.3656461948684756, 0.34914836561207524] 
        #test_prec_fale_gmr71 = [0.48052445823928736, 0.1796295657411311, 0.15600638254459742, 0.3410533402307743, 0.2900281583330838, 0.40692778039804495]

        #test_rec_male_gmr71 = [0.4599395976837538, 0.16404587544163715, 0.12586916662209838, 0.34350805270255413, 0.3852163059201615, 0.2379779669364294] 
        #test_rec_fale_gmr71 = [0.5328038038051741, 0.1716668748237192, 0.13548631986263676, 0.16074431527075203, 0.2881519638417213, 0.27166651252636725]


        test_acc_male_gmr91 = [0.9136363484642722, 0.6554545272480358, 0.5327272577719255, 0.8454545302824541, 0.7499999837441877, 0.5609090788797899, 0.5654545372182672, 0.5554545413364064] 
        test_acc_fale_gmr91 = [0.8880952199300131, 0.7457142727715629, 0.5414285617215293, 0.8418181538581848, 0.7779999852180481, 0.5839999794960022, 0.5879999736944834, 0.571999986966451]
        test_prec_male_gmr91 = [0.07120076823675654, 0.3157849828343254, 0.5486700035257873, 0.1710843137229032, 0.26543830796112994, 0.4679146432135932, 0.44909170738022336, 0.4425383716173434] 
        test_prec_fale_gmr91 = [0.10054631837658332, 0.2394481422819425, 0.4972597722690046, 0.1450513374722889, 0.24135260317464224, 0.4439156092522506, 0.449551026366953, 0.4655241322466866]
        test_rec_male_gmr91 = [0.10341213874827321, 0.3815694117084545, 0.40740577060208966, 0.14514278291279661, 0.2561296956446359, 0.4215853356996021, 0.4330432168684096, 0.4699615962311449] 
        test_rec_fale_gmr91 = [0.12533617485376142, 0.2763935647662968, 0.43958950100616195, 0.176328309640883, 0.20936043606870677, 0.4082676081960239, 0.38973994558621805, 0.41219411073216305]
        #test_acc_male_gmr91 = [0.5145454406738281, 0.8436363447796215, 0.7554545239968733, 0.5609090788797899, 0.5709090774709528, 0.5636363544247367] 
        #test_acc_fale_gmr91 = [0.5223809409709204, 0.8309090625156056, 0.7986666480700175, 0.5873333175977071, 0.6286666552225749, 0.5646666487058004]

        #test_prec_male_gmr91 = [0.5246997879841987, 0.15638119603008185, 0.2640080562105909, 0.405873071747287, 0.43074590997330103, 0.45410685170201326] 
        #test_prec_fale_gmr91 = [0.449323424932652, 0.18006180560996454, 0.24651437311847296, 0.45293904311118094, 0.3284414753381066, 0.5086434838664496]

        #test_rec_male_gmr91 = [0.46416880587204135, 0.16473558941307517, 0.23912959056879057, 0.4835761943892942, 0.4438101537781823, 0.43669876814302955] 
        #test_rec_fale_gmr91 = [0.5263821626554865, 0.16917055171775594, 0.16202782018575682, 0.3982740373783215, 0.41681438159659173, 0.3761243370193513]

        test_acc_male_pdb = [0.7984090731902556, 0.8363636176694523, 0.54954544454813, 0.9688371808029884, 0.8531034239407244, 0.7106896371677004, 0.5734482643933132, 0.5875861994151411] 
        test_acc_fale_pdb = [0.6462790605633758, 0.8025581185207811, 0.5430232390414836, 0.9660464857899865, 0.8159999827543895, 0.6853333155314127, 0.5566666563351949, 0.5853333195050557]
        test_prec_male_pdb = [0.14413915450956474, 0.2194941815279905, 0.30727713818241686, 0.03935796632130305, 0.17519414322595186, 0.29613250232871596, 0.4354154763074371, 0.3558391023839553] 
        test_prec_fale_pdb = [0.24583425133206915, 0.2654538486817778, 0.4671387968267059, 0.040476384953219015, 0.23314055253842098, 0.2860732857988017, 0.3827186491610701, 0.4518273662096029]
        test_rec_male_pdb = [0.2641782480614622, 0.11182154014043257, 0.6118623368883394, 0.024521639179322725, 0.12098988046561394, 0.29206954084369763, 0.4365396955682441, 0.48026607522436504] 
        test_rec_fale_pdb = [0.47809859373274616, 0.135571924827218, 0.46079116820077676, 0.0286212204370933, 0.14279908926229198, 0.3637864845125623, 0.5237870998352636, 0.390193295817767]
        #test_acc_male_pdb = [0.5529545349153605, 0.9665116016254869, 0.8513792938199537, 0.6579310174646049, 0.5675861897139713, 0.5996551616438504] 
        #test_acc_fale_pdb = [0.5648837061815484, 0.9672092739925828, 0.8356666485468547, 0.7583333214124044, 0.5649999896685283, 0.5646666526794434]

        #test_prec_male_pdb = [0.5472119801200681, 0.028170113043995644, 0.18912592523238422, 0.2947611298819793, 0.34808023268019633, 0.36513109589287707] 
        #test_prec_fale_pdb = [0.39348929283608586, 0.021785924011888756, 0.18404857582144293, 0.2722385905169525, 0.4923397191642303, 0.47880566212538994]

        #test_rec_male_pdb = [0.36888173559857723, 0.04021497679320716, 0.11665333079674121, 0.40471890106150116, 0.5348616564430059, 0.46182263108836935] 
        #test_rec_fale_pdb = [0.4900584293490853, 0.04559940442615477, 0.14962257809818716, 0.22184944664340592, 0.39690840684978246, 0.4178259843567701]


        if dtype=='gmr':

            test_acc_male = test_acc_male_gmr71 
            test_acc_fale = test_acc_fale_gmr71

            test_prec_male = test_prec_male_gmr71 
            test_prec_fale = test_prec_fale_gmr71

            test_rec_male = test_rec_male_gmr71 
            test_rec_fale = test_rec_fale_gmr71

        if dtype=='gmr91':

            test_acc_male = test_acc_male_gmr91 
            test_acc_fale = test_acc_fale_gmr91

            test_prec_male = test_prec_male_gmr91 
            test_prec_fale = test_prec_fale_gmr91

            test_rec_male = test_rec_male_gmr91 
            test_rec_fale = test_rec_fale_gmr91

        if dtype == 'pdb':

            test_acc_male = test_acc_male_pdb 
            test_acc_fale = test_acc_fale_pdb

            test_prec_male = test_prec_male_pdb 
            test_prec_fale = test_prec_fale_pdb

            test_rec_male = test_rec_male_pdb 
            test_rec_fale = test_rec_fale_pdb

    return test_acc_male, test_acc_fale, \
            test_prec_male, test_prec_fale,\
            test_rec_male, test_rec_fale



"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of AVB collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--print_every', type=int, default=1000, help='print every')#, required=True)
    parser.add_argument('--save_every' , type=int, default=25,help= 'save every')#, required=True)
    parser.add_argument('--model_epoch', type=int, default=200000, help='The number of epochs to run')
    parser.add_argument('--batch_sz', type=int, default=100, help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory name to save the model')
    parser.add_argument('--ch_dim', type=list, default=[147, 32, 64, 128])
    parser.add_argument('--c_dim', type=float, default=1)
    parser.add_argument('--h_dim', type=int, default=100)
    parser.add_argument('--x_dim', type=int, default=152)
    parser.add_argument('--t_dim', type=int, default=14)
    parser.add_argument('--num_mfeat', type=int, default=8)
    parser.add_argument('--num_bin', type=int, default=51)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2wd', type=float, default=0.0001)
    parser.add_argument('--l1wd', type=float, default=0.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--operation', type=str, default='faketrx', choices=['faketrx', 'eval_disc', 'plot'])
    parser.add_argument('--gender', type=int, default=0)
    parser.add_argument('--tsim', type=int, default=60)
    parser.add_argument('--t0', type=int, default=0)
    parser.add_argument('--t1', type=int, default=10000)
    parser.add_argument('--visionF', type=int, default=1)
    parser.add_argument('--dtype', type=str, default='gmr')
    parser.add_argument('--btype', type=str, default='linear')
    parser.add_argument('--ltype', type=str, default='sq')
    parser.add_argument('--mtype', type=str, default='rnn50', help='binary classifier type')
    parser.add_argument('--atype', type=str, default='relu', choices=['relu', 'sigmoid'], help='activation function')
    parser.add_argument('--videotype', type=str, default='full')
    parser.add_argument('--datapath', type=str, default='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/')
    parser.add_argument('--basepath', type=str, default='/groups/branson/home/imd/Documents/janelia/research/fly_behaviour_sim/71g01/')
    return check_args(parser.parse_args())




fname = 'eyrun_simulate_data.mat'
#basepath='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'
if __name__ == '__main__':

    args = parse_args()
    args.y_dim = args.num_bin*args.num_mfeat


    if args.operation == 'faketrx':
        if args.mtype=='lr50':

            ### LR ###
            t_dim=50
            save_path='./runs/linear_reg_'+str(t_dim) +'tau/model/weight_gender0'
            if not args.visionF: save_path = save_path +'_visionF0'
            male_model = np.load(save_path+'.npy')

            save_path='./runs/linear_reg_'+str(t_dim) +'tau/model/weight_gender1'
            if not args.visionF: save_path = save_path +'_visionF0'
            female_model = np.load(save_path+'.npy')

            for testvideo_num in range(0,len(video16_path[args.dtype][TEST])):
                if  testvideo_num == 1 or testvideo_num == 2 :
                    simulated_male_flies = np.arange(0,9,1)
                    simulated_female_flies = np.arange(9,19,1)
                else:
                    simulated_male_flies = np.arange(0,10,1)
                    simulated_female_flies = np.arange(10,20,1)

                print ('testvideo %d %s' % (testvideo_num, video16_path[args.dtype][TEST][testvideo_num]))
                real_flies_simulatePlan_RNNs(video16_path[args.dtype][TEST][testvideo_num],\
                        male_model, female_model, \
                        simulated_male_flies, simulated_female_flies,\
                        monlyF=abs(1-args.visionF), genDataset=True,\
                        tsim=args.tsim, mtype=args.mtype, t_dim=t_dim, \
                        t0=args.t0, t1=t1, btype=args.btype)

        elif args.mtype=='nn4_cat50' or args.mtype=='conv4_cat50':

            t_dim=50
            vtype='full'
            model_epoch=25000 if args.mtype == 'nn4_cat50' else 35000 
            batch_sz=100 if args.mtype == 'nn4_cat50' else 32 
            from simulate_autoreg import model_selection
            for testvideo_num in range(0,len(video16_path[args.dtype][TEST])):

                vpath = video16_path[args.dtype][TEST][testvideo_num]
                print ('testvideo %d %s' % (testvideo_num, vpath))
                matfile = args.datapath+vpath+fname
                _,_,params,basesize = load_eyrun_data(matfile)

                if  testvideo_num == 1 or testvideo_num == 2 :
                    simulated_male_flies = np.arange(0,9,1)
                    simulated_female_flies = np.arange(9,19,1)
                else:
                    simulated_male_flies = np.arange(0,10,1)
                    simulated_female_flies = np.arange(10,20,1)


                params['mtype'] = args.mtype 
                male_model, female_model  = \
                    model_selection(None, None, params, visionF=args.visionF,\
                        model_epoch=model_epoch, t_dim=t_dim, vtype=vtype,\
                        batch_sz=batch_sz, mtype=args.mtype, dtype=args.dtype, num_bin=args.num_bin)
                
                real_flies_simulatePlan_RNNs(video16_path[args.dtype][TEST][testvideo_num],\
                        male_model, female_model, \
                        simulated_male_flies, simulated_female_flies,\
                        monlyF=abs(1-args.visionF), genDataset=True, \
                        tsim=args.tsim, mtype=args.mtype, t_dim=t_dim, \
                        t0=args.t0, t1=args.t1, btype=args.btype, num_bin=args.num_bin)


        elif 'rnn' in args.mtype or 'skip' in args.mtype:
    
            ### RNN & SKIP ###
            model_epoch=200000
            from simulate_rnn import model_selection
            for testvideo_num in range(0,len(video16_path[args.dtype][TEST])):
    
                vpath = video16_path[args.dtype][TEST][testvideo_num]
                print ('testvideo %d %s' % (testvideo_num, vpath))
                matfile = args.datapath+vpath+fname
    
                if  testvideo_num == 1 or testvideo_num == 2 :
                    simulated_male_flies = np.arange(0,9,1)
                    simulated_female_flies = np.arange(9,19,1)
                else:
                    simulated_male_flies = np.arange(0,10,1)
                    simulated_female_flies = np.arange(10,20,1)
        
                male_model, female_model, male_hiddens, female_hiddens = \
                    model_selection(args, None, None, args.videotype, args.mtype, \
                        model_epoch, args.h_dim, simulated_male_flies, \
                        simulated_female_flies, dtype=args.dtype, btype=args.btype, num_bin=args.num_bin)
    
                real_flies_simulatePlan_RNNs(video16_path[args.dtype][TEST][testvideo_num],\
                        male_model, female_model, \
                        simulated_male_flies, simulated_female_flies,\
                        male_hiddens, female_hiddens,\
                        monlyF=abs(1-args.visionF), genDataset=True, \
                        tsim=args.tsim, mtype=args.mtype, t_dim=args.t_dim,\
                        t0=args.t0, t1=args.t1, btype=args.btype, num_bin=args.num_bin)
    
    elif args.operation == 'plot':

        from evaluate_chase import error_bar_plot
        label_list = ['R91B01', 'CONTROL', 'TRAIN DATA', "MALE-FEMALE", 'LINEAR', 'CNN', 'RNN', 'HRNN']
        color_list = ['slategray', 'silver', 'gray', 'black', 'red', 'green', 'deepskyblue', 'mediumpurple']

        test_acc_male_ce, test_acc_fale_ce,  test_prec_male_ce, \
                test_prec_fale_ce,  test_rec_male_ce, test_rec_fale_ce \
                                                = get_data(args.dtype, 'ce')

        test_acc_male_sq, test_acc_fale_sq,  test_prec_male_sq, \
                test_prec_fale_sq,  test_rec_male_sq, test_rec_fale_sq \
                                                = get_data(args.dtype, 'sq')

        gan_male = np.asarray([test_acc_male_ce, test_acc_male_sq])
        gan_fale = np.asarray([test_acc_fale_ce, test_acc_fale_sq])

        error_bar_plot(gan_male, label_list, color_list, 'ganeval', '_%s_male' % args.dtype, vmin=0.5, vmax=1.0,\
                            N=8, text_labels=['Jensen-Shannon', 'Least-Squares'], ylabel='Disc. Accuracy')
        error_bar_plot(gan_fale, label_list, color_list, 'ganeval', '_%s_fale' % args.dtype, vmin=0.5, vmax=1.0,\
                            N=8, text_labels=['Jensen-Shannon', 'Least-Squares'], ylabel='Disc. Accuracy')


        test_acc_male_gmr71, test_acc_fale_gmr71,  test_prec_male, \
                test_prec_fale,  test_rec_male, test_rec_fale \
                                        = get_data('gmr', args.ltype)

        test_acc_male_gmr91, test_acc_fale_gmr91,  test_prec_male, \
                test_prec_fale,  test_rec_male, test_rec_fale \
                                        = get_data('gmr91', args.ltype)

        test_acc_male_pdb, test_acc_fale_pdb,  test_prec_male, \
                test_prec_fale,  test_rec_male, test_rec_fale \
                                        = get_data('pdb', args.ltype)

        gan_male = np.asarray([test_acc_male_gmr71, test_acc_male_gmr91, test_acc_male_pdb])
        gan_fale = np.asarray([test_acc_fale_gmr71, test_acc_fale_gmr91, test_acc_fale_pdb])
        error_bar_plot(gan_male, label_list, color_list, 'ganeval', args.ltype+'_male', ylabel='Disc. Accuracy', N=8, vmin=0.5, vmax=1.0)
        error_bar_plot(gan_fale, label_list, color_list, 'ganeval', args.ltype+'_female', ylabel='Disc. Accuracy', N=8, vmin=0.5, vmax=1.0)


        plt.figure()
        ax = plt.axes([0,0,1,1])
        model_list = ['train', 'MF', 'lr50', 'conv4_cat50', 'rnn50', 'skip50']
        label_list = ['TRAIN DATA', "MALE-FEMALE", 'LINEAR', 'CNN', 'RNN', 'HRNN']
        color_list = ['gray', 'black', 'red', 'green', 'deepskyblue', 'mediumpurple']
        for mtype, label, color in zip(model_list, label_list, color_list):

            male_accs = np.load('./ganeval/%s/%s_%s_male_accs.npy' % (args.dtype,mtype,args.ltype))
            #fale_accs = np.load('./ganeval/%s_female_accs' % mtype)            
            ax.plot(male_accs[0], male_accs[-1], color=color, lw=5, ls='-', label=label)

        plt.ylabel('Disc. Accuracy')
        plt.xlabel('Iteration')
        ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5,1.32), ncol=3)
        ax.get_xaxis().set_visible(True)

        os.makedirs('./figs/ganeval/', exist_ok=True)
        plt.savefig('./figs/ganeval/%s/male_%s_gan_eval.png' % (args.dtype, args.ltype), format='png', \
                        bbox_inches='tight') 
        plt.close()


        plt.figure()
        ax = plt.axes([0,0,1,1])
        for mtype, label, color in zip(model_list, label_list, color_list):

            fale_accs = np.load('./ganeval/%s/%s_%s_female_accs.npy' % (args.dtype, mtype, args.ltype))
            #fale_accs = np.load('./ganeval/%s_female_accs' % mtype)            
            ax.plot(fale_accs[0], fale_accs[-1], color=color, lw=5, ls='-', label=label)

        plt.ylabel('Disc. Accuracy')
        plt.xlabel('Iteration')
        ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5,1.32), ncol=3)
        ax.get_xaxis().set_visible(True)

        os.makedirs('./figs/ganeval/', exist_ok=True)
        plt.savefig('./figs/ganeval/%s/fale_%s_gan_eval.png' % (args.dtype, args.ltype), format='png', \
                        bbox_inches='tight') 
        plt.close()

        width = 1
        test_acc_male = np.asarray(test_acc_male)
        ylabel = 'Model Error'
        xlabel = 'Model'
        ftag = 'male_%s_test_accuracy_%s' % (args.ltype, args.dtype)
        fdir = './ganeval/%s/' % args.dtype
        plot_bar(width, test_acc_male, ylabel, xlabel, fdir, ftag, label_list, color_list)


        width = 1
        test_acc_fale = np.asarray(test_acc_fale)
        ylabel = 'Model Error'
        xlabel = 'Model'
        ftag = 'female_%s_test_accuracy_%s' % (args.ltype, args.dtype)
        fdir = './ganeval/%s/' % args.dtype
        plot_bar(width, test_acc_fale, ylabel, xlabel, fdir, ftag, label_list, color_list)


        plt.figure()
        ax = plt.axes([0,0,1,1])
        for prec, rec, label, color in zip(test_prec_male, test_rec_male, label_list, color_list):
            plt.plot(prec, rec, marker='x', ls='',  color=color, label=label, markersize=10)
        plt.ylabel('True Negative Error Rate')
        plt.xlabel('True Positive Error Rate')
        plt.xlim(0,1.0)
        plt.ylim(0,1.0)
        ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5,1.25), ncol=3)
        plt.savefig('./figs/ganeval/%s/male_%s_prec_recall.png' % (args.dtype, args.ltype), format='png', \
                        bbox_inches='tight') 
        plt.close()


        plt.figure()
        ax = plt.axes([0,0,1,1])
        for prec, rec, label, color in zip(test_prec_fale, test_rec_fale, label_list, color_list):
            plt.plot(prec, rec, marker='x', ls='',  color=color, label=label, markersize=10)
        plt.ylabel('True Negative Error Rate')
        plt.xlabel('True Positive Error Rate')
        plt.xlim(0,1.0)
        plt.ylim(0,1.0)
        ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5,1.25), ncol=3)

        plt.savefig('./figs/ganeval/%s/fale_%s_prec_recall.png' % (args.dtype, args.ltype), format='png', \
                        bbox_inches='tight') 
        plt.close()

    elif args.operation == 'eval_disc':

        male_real_frame_ind_list, fale_real_frame_ind_list = [], []
        malerealdata_list, falerealdata_list = [], []
        for testvideo_num in range(0,len(video16_path[args.dtype][TEST])):

            vpath = video16_path[args.dtype][TEST][testvideo_num]
            print ('testvideo %d %s' % (testvideo_num, vpath))
            malerealdata, male_real_frame_ind = real_traj(args.t0, args.t1, vpath, tsim=args.tsim, gender=0)
            falerealdata, fale_real_frame_ind = real_traj(args.t0, args.t1, vpath, tsim=args.tsim, gender=1)
            malerealdata = malerealdata[:,:,1:,:]
            falerealdata = falerealdata[:,:,1:,:]
            malerealdata_list.append(malerealdata.reshape([-1, args.tsim-2, 153]))
            falerealdata_list.append(falerealdata.reshape([-1, args.tsim-2, 153]))

            P = male_real_frame_ind.shape[0] 
            video_num = np.ones((P,1)) * testvideo_num
            male_real_frame_ind_list.append(np.hstack([male_real_frame_ind, video_num]))
            P = fale_real_frame_ind.shape[0] 
            video_num = np.ones((P,1)) * testvideo_num
            fale_real_frame_ind_list.append(np.hstack([fale_real_frame_ind, video_num]))

            #t0, t1, tsim, t_dim = 0, 10000, 60, 50
            #for ii, t in enumerate(range(t0+t_dim,t1,args.tsim)):
            #    frame_index.append(t)
            #frame_index = np.asarray(frame_index[50:])


        malerealdata = np.vstack(malerealdata_list)
        falerealdata = np.vstack(falerealdata_list)
        malerealdata = normalize_pos(malerealdata)
        falerealdata = normalize_pos(falerealdata)

        model_list = []
        test_accs_m, test_accs_f = [], []
        test_loss_m, test_loss_f = [], []
        test_prec_m, test_prec_f = [], []
        test_rec_m, test_rec_f = [], []
        if args.dtype == 'gmr':
            model_list = ['gmr91', 'pdb']
        elif  args.dtype == 'gmr91':
            model_list = ['gmr', 'pdb']
        elif  args.dtype == 'pdb':
            model_list = ['gmr', 'gmr91']

        #model_list += ['lr50', 'conv4_cat50', 'rnn50', 'skip50']
        #model_list += ['rnn50', 'skip50']
        model_list += [args.dtype, 'MF', 'lr50', 'conv4_cat50', 'rnn50', 'skip50']
        for mtype in model_list:

            male_fake_frame_ind_list, fale_fake_frame_ind_list = [], []
            if 'gmr' in mtype or 'pdb' in mtype:
                malefakedata_list, falefakedata_list = [], []
                for trainvideo_num in range(0,len(video16_path[mtype][TRAIN])):

                    vpath = video16_path[mtype][TRAIN][trainvideo_num]
                    print ('testvideo %d %s' % (trainvideo_num, vpath))
                    malefakedata, male_fake_frame_ind = real_traj(args.t0, args.t1, vpath, tsim=args.tsim, gender=0)
                    falefakedata, fale_fake_frame_ind = real_traj(args.t0, args.t1, vpath, tsim=args.tsim, gender=1)
                    malefakedata = malefakedata[:,:,1:,:]
                    falefakedata = falefakedata[:,:,1:,:]
                    malefakedata_list.append(malefakedata.reshape([-1, args.tsim-2, 153]))
                    falefakedata_list.append(falefakedata.reshape([-1, args.tsim-2, 153]))

                    P = male_fake_frame_ind.shape[0] 
                    video_num = np.ones((P,1)) * testvideo_num
                    male_fake_frame_ind_list.append(np.hstack([male_fake_frame_ind, video_num]))
                    P = fale_fake_frame_ind.shape[0] 
                    video_num = np.ones((P,1)) * testvideo_num
                    fale_fake_frame_ind_list.append(np.hstack([fale_fake_frame_ind, video_num]))

                A,B,C,D = malefakedata.shape

            elif 'MF' == mtype:
            
                malefakedata_list, falefakedata_list = [], []
                for testvideo_num in range(0,len(video16_path[args.dtype][TEST])):

                    vpath = video16_path[args.dtype][TEST][testvideo_num]
                    print ('testvideo %d %s' % (testvideo_num, vpath))
                    malefakedata, male_fake_frame_ind = real_traj(args.t0, args.t1, vpath, tsim=args.tsim, gender=0)
                    falefakedata, fale_fake_frame_ind = real_traj(args.t0, args.t1, vpath, tsim=args.tsim, gender=1)
                    malefakedata = malefakedata[:,:,1:,:]
                    falefakedata = falefakedata[:,:,1:,:]
                    malefakedata_list.append(falefakedata.reshape([-1, args.tsim-2, 153]))
                    falefakedata_list.append(malefakedata.reshape([-1, args.tsim-2, 153]))

                    P = fale_fake_frame_ind.shape[0] 
                    video_num = np.ones((P,1)) * testvideo_num
                    male_fake_frame_ind_list.append(np.hstack([fale_fake_frame_ind, video_num]))
                    P = male_fake_frame_ind.shape[0] 
                    video_num = np.ones((P,1)) * testvideo_num
                    fale_fake_frame_ind_list.append(np.hstack([male_fake_frame_ind, video_num]))

                A,B,C,D = malefakedata.shape

            else:

                frames_ind_list = []
                fakedata_list, malefakedata_list, falefakedata_list = [], [], []
                for testvideo_num in range(len(video16_path[args.dtype][TEST])):
                    vpath = video16_path[args.dtype][TEST][testvideo_num]
                    print ('testvideo %d %s' % (testvideo_num, vpath))

                    matfile = args.datapath+vpath+'eyrun_simulate_data.mat'
                    (trx,motiondata,params,basesize) = load_eyrun_data(matfile)
                    male_ind, female_ind = gender_classify(basesize['majax'])
                    simulated_male_flies = male_ind
                    simulated_female_flies = female_ind

                    frame_index = []
                    t_dim = 50
                    for ii, t in enumerate(range(args.t0+t_dim,args.t1,args.args.tsim)):
                        frame_index.append(t)
                    frame_index = np.asarray(frame_index[50:])

                    #if  testvideo_num == 1 or testvideo_num == 2 :
                    #    simulated_male_flies = np.arange(0,9,1)
                    #    simulated_female_flies = np.arange(9,19,1)
                    #else:
                    #    simulated_male_flies = np.arange(0,10,1)
                    #    simulated_female_flies = np.arange(10,20,1)

                    if 'lr50' == mtype:
                        fakepath = args.basepath+'/fakedata/%s/%s/%s_gender%d_%dt0_%dt1_%dtsim.npy' \
                                % (vpath, mtype, mtype, gender, args.t0, args.t1, args.tsim)
                    elif 'skip50' == mtype: #elif 'conv' in mtype:
                        fakepath = args.basepath+'/fakedata/%s/%s/%s_gender%d_100hid_%dt0_%dt1_%dtsim_200000epoch.npy' \
                                % (vpath, mtype, mtype, gender, args.t0, args.t1, args.tsim)
                    elif 'rnn50' == mtype: #elif 'conv' in mtype:
                        if 'perc' in btype:
                            fakepath = args.basepath+'/fakedata/%s/%s/%s_gender%d_100hid_%dt0_%dt1_%dtsim_%s_200000epoch.npy' \
                                % (vpath, mtype, mtype, gender, args.t0, args.t1, args.tsim, btype)
                        else:
                            fakepath = args.basepath+'/fakedata/%s/%s/%s_gender%d_100hid_%dt0_%dt1_%dtsim_200000epoch.npy' \
                                % (vpath, mtype, mtype, gender, args.t0, args.t1, args.tsim)
                    else:
                        fakepath = args.basepath+'/fakedata/%s/%s/%s_gender%d_100hid_%dt0_%dt1_%dtsim_200000epoch.npy' \
                                % (vpath, mtype, mtype, gender, args.t0, args.t1, args.tsim)

                    print(fakepath)

                    fakedata = np.load(fakepath) 
                    malefakedata = fakedata[:,simulated_male_flies]
                    malefakedata = malefakedata[:,:,malefakedata.shape[2]-(args.tsim-2):,:]
                    A,B,C,D = malefakedata.shape
                    malefakedata_list.append(malefakedata.reshape([A*B,C, 153]))

                    male_fake_frame_ind = []
                    for frame in frame_index:
                        for fly_ind in simulated_male_flies:
                            male_fake_frame_ind.append([frame, fly_ind, testvideo_num])
                    male_fake_frame_ind = np.asarray(male_fake_frame_ind)

                    falefakedata = fakedata[:,simulated_female_flies]
                    falefakedata = falefakedata[:,:,falefakedata.shape[2]-(args.tsim-2):,:]
                    A,B,C,D = falefakedata.shape
                    falefakedata_list.append(falefakedata.reshape([A*B,C, 153]))

                    fale_fake_frame_ind = []
                    for frame in frame_index:
                        for fly_ind in simulated_female_flies:
                            fale_fake_frame_ind.append([frame, fly_ind, testvideo_num])
                    fale_fake_frame_ind = np.asarray(fale_fake_frame_ind)

                    male_fake_frame_ind_list.append(male_fake_frame_ind)
                    fale_fake_frame_ind_list.append(fale_fake_frame_ind)

            malefakedata = np.vstack(malefakedata_list)
            falefakedata = np.vstack(falefakedata_list)
            malefakedata = normalize_pos(malefakedata)
            falefakedata = normalize_pos(falefakedata)

            male_real_frame_ind_list = np.vstack(male_real_frame_ind_list)
            male_fake_frame_ind_list = np.vstack(male_fake_frame_ind_list)
            fale_real_frame_ind_list = np.vstack(fale_real_frame_ind_list)
            fale_fake_frame_ind_list = np.vstack(fale_fake_frame_ind_list)

            
            train_set, valid_set, test_set, data_stats, real_ind_te, fake_ind_te \
                    = gen_dataset(malerealdata, malefakedata, male_real_frame_ind_list, male_fake_frame_ind_list)

            args = parse_args()
            args.epoch=500
            args.t_dim = C
            model, epochs, tr_accs, vl_accs, te_acc, te_loss, te_error, te_precision, te_recall, te_preds\
                                = main(args, train_set, valid_set, test_set)
            
            #simulation_with_gan_score(data_stats, model, mtype, video16_path[args.dtype][TEST][0], t0, t1, args.tsim, burning=100)
            #import pdb; pdb.set_trace()

            if 'lr50' == mtype:
                fakepath = './fakedata/%s/%s/%s_gender%d_%dt0_%dt1_%dtsim.npy' \
                        % (video16_path[args.dtype][TEST][0], args.mtype, args.mtype, args.gender, args.t0, args.t1, args.tsim)
            elif 'skip' in mtype:
                fakepath = './fakedata/%s/%s/%s_gender%d_100hid_%dt0_%dt1_%dtsim_200000epoch.npy' \
                        % (video16_path[args.dtype][TEST][0], args.mtype, args.mtype, args.gender, args.t0, args.t1, args.tsim)
            else: #elif 'conv' in mtype:
                fakepath = './fakedata/%s/%s/%s_gender%d_100hid_%dt0_%dt1_%dtsim_200000epoch.npy' \
                        % (video16_path[args.dtype][TEST][0], args.mtype, args.mtype, args.gender, args.t0, args.t1, args.tsim)
 
            #fakedata0 = np.load(fakepath) 
            #visualize_negative_examples(args.dtype, fakedata0, model, \
            #        simulated_male_flies, video16_path[args.dtype][TEST][0], args.t0, args.t1, args.tsim, args.mtype)


            test_loss_m.append(te_loss)
            test_accs_m.append(te_acc)
            test_prec_m.append(te_precision)
            test_rec_m.append(te_recall)
            os.makedirs('./ganeval/', exist_ok=True)
            os.makedirs('./ganeval/%s' % args.dtype, exist_ok=True)
            np.save('./ganeval/%s/%s_%s_male_accs' % (args.dtype, mtype, args.ltype), \
                    np.vstack([epochs, tr_accs, vl_accs]))
            reminder = fake_ind_te.shape[0] - (te_preds.shape[0] - real_ind_te.shape[0])
            np.save('./ganeval/%s/%s_%s_real_male_ind' % (args.dtype, mtype, args.ltype), real_ind_te)
            np.save('./ganeval/%s/%s_%s_fake_male_ind' % (args.dtype, mtype, args.ltype), fake_ind_te[:-reminder])
            np.save('./ganeval/%s/%s_%s_male_pred' % (args.dtype, mtype, args.ltype), te_preds)

            train_set, valid_set, test_set, data_stats, real_ind_te, fake_ind_te \
                    = gen_dataset(falerealdata, falefakedata, fale_real_frame_ind_list, fale_fake_frame_ind_list)

            args = parse_args()
            args.epoch=500
            args.t_dim = C
            model, epochs, tr_accs, vl_accs, te_acc, te_loss, te_errors, te_precision, te_recall, te_preds\
                    = main(args, train_set, valid_set, test_set)

            test_accs_f.append(te_acc)
            test_loss_f.append(te_loss)
            test_prec_f.append(te_precision)
            test_rec_f.append(te_recall)

            os.makedirs('./ganeval/', exist_ok=True)
            os.makedirs('./ganeval/%s' % args.dtype, exist_ok=True)
            np.save('./ganeval/%s/%s_%s_female_accs' % (args.dtype, mtype, args.ltype), \
                    np.vstack([epochs, tr_accs, vl_accs]))
            reminder = fake_ind_te.shape[0] - (te_preds.shape[0] - real_ind_te.shape[0])
            np.save('./ganeval/%s/%s_%s_real_female_ind' % (args.dtype, mtype, args.ltype), real_ind_te)
            np.save('./ganeval/%s/%s_%s_fake_female_ind' % (args.dtype, mtype, args.ltype), fake_ind_te[:-reminder])
            np.save('./ganeval/%s/%s_%s_female_pred' % (args.dtype, mtype, args.ltype), te_preds)

        print(test_accs_m, test_accs_f)
        print(test_loss_m, test_loss_f)
        print(test_prec_m, test_prec_f)
        print(test_rec_m, test_rec_f)


