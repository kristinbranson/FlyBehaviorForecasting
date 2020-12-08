import numpy as np
import h5py, os, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import scipy.io as sio

import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from gen_dataset import load_eyrun_data, video16_path, gender_classify
from util_fly import *
from util import copy_hiddens, check_args

from tqdm import tqdm
from apply_model_sjb import *

MALE=0
FEMALE=1
NUM_FLY=20
NUM_BIN=51
NUM_MFEAT=8
TRAIN=0
VALID=1
TEST=2

default_params = {}
default_params['n_motions'] = 8
default_params['binprob_exp'] = 1.2
default_params['mindist'] = 3.885505
default_params['n_oma'] = 72
default_params['I'] = None
default_params['J'] = None
default_params['PPM'] = 7.790785
default_params['chamber_outline_matfile'] = \
    '/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Code/chamber_outline.mat'
default_params['ranges'] = \
    np.array([24.449421,13.040234,9.382661,0.143254,0.464102,0.506068,0.216591,0.220717])
default_params['FPS'] = 30.
default_params['arena_center_x'] = 512.5059
default_params['arena_center_y'] = 516.4722
default_params['arena_radius'] = 476.3236



def get_real_fly(real_flies, motiondata, feat_motion, t, trx, x, y, theta, \
       l_wing_ang, r_wing_ang, l_wing_len, r_wing_len, a, b, params):


    vision_chamber_data = []
    for flyi in range(len(real_flies)):

        fly = real_flies[flyi]
        (flyvision,chambervision) = compute_vision(x,y,theta,a,b,fly,params)

        x[fly] = trx['x'][t,fly]
        y[fly] = trx['y'][t,fly]
        theta[fly] = trx['theta'][t,fly]
        a[fly] = trx['a'][t,fly]
        l_wing_ang[fly] = trx['l_wing_ang'][t,fly]
        r_wing_ang[fly] = trx['r_wing_ang'][t,fly]
        l_wing_len[fly] = trx['l_wing_len'][t,fly]
        r_wing_len[fly] = trx['r_wing_len'][t,fly]

        # motiondata[:,t,fly] = corresponds to movement from t-1 to t
        feat_motion[:,fly] = motiondata[:,t,fly]

        vision_chamber = np.hstack([flyvision,chambervision])
        vision_chamber_data.append(vision_chamber)

    return x, y, theta, a, \
            l_wing_ang, r_wing_ang, \
            l_wing_len, r_wing_len, \
            vision_chamber_data, feat_motion


def get_simulate_fly(model, state, t, trx, \
                     simulated_flies, feat_motion, \
                     x, y, theta, a, b, \
                     l_wing_ang, r_wing_ang,\
                     l_wing_len, r_wing_len,\
                     xprev, yprev, thetaprev, \
                     basesize, params, mtype, \
                     thrd=10, num_bin=51, hiddens=None, t_dim=50,\
                     visionF=1, visionOnly=0):

    from util_fly_sjb import compute_vision, binscores2motion, update_position
    vision_chamber_data = []
    device = simulated_flies.device
    for flyi in range(len(simulated_flies)):
        fly = simulated_flies[flyi]
        # vision features for frame t-1
        (flyvision,chambervision) = compute_vision(x,y,theta,a,b,fly,params)       

        if 'rnn' in mtype or 'skip' in mtype:
            # predicted motion from frame t-1 to t
            (binscores,state[flyi],hiddens[flyi],preds) \
                    = apply_bin_rnn((flyvision,chambervision,\
                            feat_motion[:,fly]), state[flyi], model,\
                            hiddens[flyi], params,fly, num_bin=num_bin)

            feat_motion[:,fly] = binscores2motion(binscores,params)

        elif 'lr' in mtype:
            # predicted motion from frame t-1 to t
            (binscores,state,pred) = apply_lr_fly_np(\
                                    state, \
                                    flyvision,chambervision,\
                                    feat_motion,\
                                    model, params,\
                                    visionF=visionF, t=t, \
                                    t_dim=t_dim, fly=fly,\
                                    visionOnly=visionOnly)
            feat_motion[:,fly] = binscores
       
        elif 'conv' in mtype and 'cat' in mtype:
            # predicted motion from frame t-1 to t
            (binscores,state,preds) \
                    = apply_conv_fly(state, flyvision, chambervision,\
                                    feat_motion, model, \
                                    params, t=t, visionF=visionF, \
                                    t_dim=t_dim, fly=fly, \
                                    visionOnlyF=visionOnly)

            feat_motion[:,fly]=binscores2motion(binscores,params)

        # predicted position at frame t
        (x[fly],y[fly],theta[fly],a[fly],
         l_wing_ang[fly],r_wing_ang[fly],
         l_wing_len[fly],r_wing_len[fly]) = \
            update_position(xprev[fly],yprev[fly],thetaprev[fly],
                            basesize['majax'][fly],
                            basesize['awing1'][fly],basesize['awing2'][fly],
                            basesize['lwing1'][fly],basesize['lwing2'][fly],
                            feat_motion[:,fly],params)


        vision_chamber = torch.cat([flyvision,chambervision])
        vision_chamber_data.append(vision_chamber)


    return x, y, theta, a, \
            l_wing_ang, r_wing_ang, \
            l_wing_len, r_wing_len, hiddens,\
            vision_chamber_data, feat_motion, state




def simulate_next_t_steps_v2(t0,curpos,male_state,female_state,\
                                feat_motion,male_model,\
                                female_model,params,\
                                basesize,motiondata,\
                                simulated_flies,\
                                tsim=30, monlyF=0, \
                                mtype='rnn', \
                                male_hiddens=None, \
                                female_hiddens=None,\
                                male_ind=None, female_ind=None, \
                                t_dim=7, num_bin=51):
    # simulate_next_t_steps(t0,curpos,hiddens,state,feat_motion,model,params,basesize,tsim=30)
    # inputs:
    # t0: start time point
    # curpos: initial position of flies
    # state, hiddens: initial hidden states of flies -- maybe we should do this without knowing the hidden state of other flies at some point?
    # feat_motion: motion features from previous time point
    # model: predictor
    # params: parameters
    # basesize: base size of flies
    # tsim: number of steps to simulate
    


    from util_fly_sjb import compute_vision, motion2binidx, \
                            binscores2motion, update_position
    # whether we are going to do the simulation within this function, or if it is already done
    #n_flies      = len(curpos['x'])
    n_flies      =len(male_ind) + len(female_ind)
    init_hiddens_male = copy_hiddens(male_hiddens, params)
    init_hiddens_fale = copy_hiddens(female_hiddens, params)

    device = curpos['x'].device
    simtrx = {}
    simtrx['x']=torch.zeros((tsim,n_flies), device=device)
    simtrx['y']=torch.zeros((tsim,n_flies), device=device)
    simtrx['theta']=torch.zeros((tsim,n_flies), device=device)
    simtrx['a']=torch.zeros((tsim,n_flies), device=device)
    simtrx['b']=torch.zeros((tsim,n_flies), device=device)
    simtrx['l_wing_ang']=torch.zeros((tsim,n_flies), device=device)
    simtrx['r_wing_ang']=torch.zeros((tsim,n_flies), device=device)
    simtrx['l_wing_len']=torch.zeros((tsim,n_flies), device=device)
    simtrx['r_wing_len']=torch.zeros((tsim,n_flies), device=device)

    simtrx['x'][0,:]=curpos['x']
    simtrx['y'][0,:]=curpos['y']
    simtrx['theta'][0,:]=curpos['theta']
    simtrx['a'][0,:]=curpos['a']
    simtrx['b'][0,:]=curpos['b']
    simtrx['l_wing_ang'][0,:]=curpos['l_wing_ang']
    simtrx['r_wing_ang'][0,:]=curpos['r_wing_ang']
    simtrx['l_wing_len'][0,:]=curpos['l_wing_len']
    simtrx['r_wing_len'][0,:]=curpos['r_wing_len']
 
    print("simulated_flies=%s" % str(simulated_flies))
    
    visionF = abs(1-monlyF)
    male_hiddens0, female_hiddens0 = [], []
    predictions_flies, flyvisions_flies = [], []
    #for fly2 in range(n_flies):
    for fly2 in simulated_flies:

        x=curpos['x'].clone()
        y=curpos['y'].clone()
        a=curpos['a'].clone()
        b=curpos['b'].clone()
        theta=curpos['theta'].clone()
        l_wing_ang=curpos['l_wing_ang']
        r_wing_ang=curpos['r_wing_ang']
        l_wing_len=curpos['l_wing_len']
        r_wing_len=curpos['r_wing_len']

        xprev = x.clone()
        yprev = y.clone()
        thetaprev = theta.clone()
        
        predictions, flyvisions = [], []
        hiddens_male = copy_hiddens(init_hiddens_male, params)
        hiddens_fale = copy_hiddens(init_hiddens_fale, params)
        for t in range(1,tsim):

            xprev[:]=x
            yprev[:]=y
            thetaprev[:]=theta
            for model, ind, hiddens, state \
                            in zip([male_model, female_model],\
                                    [male_ind, female_ind], \
                                    [hiddens_male, hiddens_fale],\
                                    [male_state, female_state]):
                            
                for flyi in range(len(ind)):
                
                    fly = ind[flyi]
                    if fly == fly2:
                        # vision features for frame t-1
                        (flyvision,chambervision)\
                            = compute_vision(x,y,theta,a,b,flyi,params)

                        # predicted motion from frame t-1 to t
                        if 'reg' in mtype:
                            # predicted motion from frame t-1 to t
                            (binscores,state[flyi],hiddens[flyi],preds) \
                            = apply_reg_model((flyvision, chambervision,\
                                                    feat_motion[:,fly]),\
                                                    state[flyi], \
                                                    model, hiddens[flyi], \
                                                    params,fly)
                            feat_motion[:,fly] = binscores #+ np.random.randn(1)*np.sqrt(binwidth)
                        elif 'rnn' in mtype or 'skip' in mtype:

                            (binscores,state[flyi],hiddens[flyi], preds)=\
                                    apply_bin_rnn((flyvision,chambervision,\
                                                    feat_motion[:,fly]),
                                                    state[flyi], model,\
                                                    hiddens[flyi],params,\
                                                    fly, num_bin=num_bin)
                            feat_motion[:,fly]=binscores2motion(binscores,params)
                        else:
                            (binscores,state,pred) = apply_lr_fly_np(\
                                                state, \
                                                flyvision,chambervision,\
                                                feat_motion,\
                                                model, params,\
                                                visionF, t, t_dim, fly)
                            feat_motion[:,fly] = binscores #binscores2motion(binscores,params)
                        predictions.append(binscores)
                        flyvisions.append(torch.cat([flyvision,chambervision]))
                    else:
                        feat_motion[:,fly] = motiondata[:,t0+t,fly]
                    

                    (x[fly],y[fly],theta[fly],a[fly],
                     l_wing_ang[fly],r_wing_ang[fly],
                     l_wing_len[fly],r_wing_len[fly])= \
                        update_position(xprev[fly],yprev[fly],thetaprev[fly],
                                        basesize['majax'][fly],
                                        basesize['awing1'][fly],basesize['awing2'][fly],
                                        basesize['lwing1'][fly],basesize['lwing2'][fly],
                                        feat_motion[:,fly],params)

                    if fly==fly2:
                        simtrx['x'][t,fly]=x[fly]
                        simtrx['y'][t,fly]=y[fly]
                        simtrx['theta'][t,fly]=theta[fly]
                        simtrx['a'][t,fly]=a[fly]
                        simtrx['b'][t,fly]=b[fly]
                        simtrx['l_wing_ang'][t,fly]=l_wing_ang[fly]
                        simtrx['r_wing_ang'][t,fly]=r_wing_ang[fly]
                        simtrx['l_wing_len'][t,fly]=l_wing_len[fly]
                        simtrx['r_wing_len'][t,fly]=r_wing_len[fly]

            if t==1:  
                if fly2 in male_ind:
                    ind2 = (male_ind == fly2).nonzero().flatten()[0]
                    male_hiddens0.append(copy_hiddens(hiddens_male, params)[ind2])
                else:
                    ind2 = (female_ind == fly2).nonzero().flatten()[0]
                    female_hiddens0.append(copy_hiddens(hiddens_fale, params)[ind2])
                #print(ind2, fly2)
        predictions_flies.append(torch.stack(predictions, 0))
        flyvisions_flies.append(torch.stack(flyvisions, 0))

    predictions_flies = torch.stack(predictions_flies, 0).transpose(0,1)
    return simtrx, feat_motion, predictions_flies, \
            male_hiddens0, female_hiddens0, flyvisions_flies


def get_nstep_comparison_rnn(x, y, theta, a, b, \
                    l_wing_ang, r_wing_ang, \
                    l_wing_len, r_wing_len,\
                    n_flies, trx,\
                    male_model, female_model, \
                    male_state, female_state,\
                    feat_motion, \
                    params, basesize, motiondata, tsim, tplot, \
                    simulated_flies,\
                    male_hiddens=None, \
                    female_hiddens=None,\
                    male_ind=None, female_ind=None,\
                    monlyF=0, mtype='rnn',t_dim=7, num_bin=51):

    # initial position
    curpos = {}
    curpos['x']=x.clone()
    curpos['y']=y.clone()
    curpos['theta']=theta.clone()
    curpos['a']=a.clone()
    curpos['b'] = basesize['minax'].clone()
    curpos['l_wing_ang']=l_wing_ang.clone()
    curpos['r_wing_ang']=r_wing_ang.clone()
    curpos['l_wing_len']=l_wing_len.clone()
    curpos['r_wing_len']=r_wing_len.clone()
    
    # simulate the next tsim time points
    print("simulated_flies=%s" % str(simulated_flies))
    simtrx, feat_motion, predictions, \
            male_hiddens, female_hiddens, flyvisions\
                            = simulate_next_t_steps_v2(tplot,curpos,\
                                        male_state, female_state,\
                                        feat_motion, \
                                        male_model, female_model,\
                                        params,basesize,motiondata, \
                                        simulated_flies,\
                                        male_hiddens=male_hiddens, \
                                        female_hiddens=female_hiddens, \
                                        male_ind=male_ind,\
                                        female_ind=female_ind,\
                                        tsim=tsim, monlyF=monlyF,\
                                        mtype=mtype, t_dim=t_dim,\
                                        num_bin=num_bin)

    return simtrx, feat_motion, predictions, \
            male_hiddens, female_hiddens, flyvisions




def model_selection(args, male_model, female_model, videotype, mtype, \
                    model_epoch, num_hid, simulated_male_flies, \
                    simulated_female_flies, dtype, btype='linear', num_bin=101,
                    use_cuda=0):

    print('Loading Model...\n')
    if 'rnn' in mtype or 'skip' in mtype:
        onehotF=0
        from util import load_rnn
        if male_model is None:
            print(mtype)
            if ('save_path_male' in args) and \
               (args.save_path_male is not None):
                load_path = args.save_path_male 
            elif mtype == 'rnn50':
                if 'perc' in btype and 'gmr' == args.dtype:
                    load_path=args.basepath+'/models/%s/flyNet12_50steps_32batch_sz_200000epochs_0.01lr_%dbins_%shids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_btype:%s_maleflies_%s' % (dtype, num_bin, num_hid, 0, videotype, dtype, btype, model_epoch)
                else:
                    load_path=args.basepath+'/models/%s/flyNet11_50steps_32batch_sz_200000epochs_0.01lr_%shids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_maleflies_%s' % (dtype, num_hid, 1, videotype, dtype, model_epoch)
            elif mtype == 'rnn25':
                load_path=args.basepath+'/models/%s/flyNet9_25steps_32batch_sz_200000epochs_0.01lr_%shids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_maleflies_%s' % (dtype, num_hid, onehotF, videotype, dtype, model_epoch)
            elif mtype == 'rnn100':
                load_path=args.basepath+'/models/%s/flyNet9_100steps_32batch_sz_200000epochs_0.01lr_%shids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_maleflies_%s' % (dtype, num_hid, onehotF, videotype, dtype, model_epoch)
            elif mtype == 'rnn150':
                load_path=argrs.basepath+'/models/%s/flyNet9_150steps_32batch_sz_200000epochs_0.01lr_%shids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_maleflies_%s' % (dtype, num_hid, onehotF, videotype, dtype, model_epoch)
            elif mtype == 'skip25':
                load_path = args.basepath+'/models/flyNetSKIP6_25steps_32batch_sz_0.01lr_%dhids_gru_onehot%d_visionF1_vtype:%s_maleflies_%s'% (num_hid, onehotF, videotype, model_epoch)
            elif mtype == 'skip50':
                if dtype == 'pdb':
                    load_path = args.basepath+'/models/%s/flyNetSKIP10_50steps_32batch_sz_0.01lr_%dhids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_maleflies_%s'% (dtype, num_hid, onehotF, videotype, dtype, model_epoch)
                else:
                    load_path = args.basepath+'/models/%s/flyNetSKIP10_50steps_32batch_sz_0.01lr_%dhids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_maleflies_%s'% (dtype, num_hid, onehotF, videotype, dtype, model_epoch)
            elif mtype == 'skip100':
                load_path = args.basepath+'/models/%s/flyNetSKIP9_100steps_32batch_sz_0.01lr_%dhids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_maleflies_%s'% (dtype, num_hid, onehotF, videotype, dtype, model_epoch)
            elif mtype == 'skip150':
                load_path = args.basepath+'/models/flyNetSKIP6_150steps_32batch_sz_0.01lr_%dhids_gru_onehot%d_visionF1_vtype:%s_maleflies_%s'% (num_hid, onehotF, videotype, model_epoch)
            male_model = load_rnn(args, mtype=mtype, gender=MALE, \
                             num_hid=num_hid, cudaF=use_cuda, num_bin=num_bin,\
                             load_path=load_path)
            print('Male Model Load Path %s' % load_path)
        if female_model is None:
            if ('save_path_female' in args) and \
               (args.save_path_female is not None):
                load_path = args.save_path_female 
            elif mtype == 'rnn50':
                if 'perc' in btype:
                    load_path=args.basepath+'/models/%s/flyNet12_50steps_32batch_sz_200000epochs_0.01lr_%dbins_%shids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_btype:%s_femaleflies_%s' % (dtype, num_bin, num_hid, 0, videotype, dtype, btype, model_epoch)
                else:
                    load_path=args.basepath+'/models/%s/flyNet11_50steps_32batch_sz_200000epochs_0.01lr_%shids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_femaleflies_%s' % (dtype, num_hid, 1, videotype, dtype, model_epoch)
            elif mtype == 'vrnn50':
                load_path=args.basepath+'/models/%s/vrnn50/flyVRNN_50steps_64batch_sz_200000epochs_0.01lr_%dbins_%shids_gru_onehot%d_visionF1_vtype:%s_btype:%s_femaleflies_%s' % (dtype, num_bin, num_hid, 0, videotype, btype, model_epoch)
            elif mtype == 'rnn25':
                load_path=args.basepath+'/models/%s/flyNet9_25steps_32batch_sz_200000epochs_0.01lr_%shids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_femaleflies_%s' % (dtype, num_hid, onehotF, videotype, dtype, model_epoch)
            elif mtype == 'rnn100':
                load_path=args.basepath+'/models/%s/flyNet9_100steps_32batch_sz_200000epochs_0.01lr_%shids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_femaleflies_%s' % (dtype, num_hid, onehotF, videotype, dtype, model_epoch)
            elif mtype == 'rnn150':
                load_path=args.basepath+'/models/%s/flyNet9_150steps_32batch_sz_200000epochs_0.01lr_%shids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_femaleflies_%s' % (dtype, num_hid, onehotF, videotype, dtype, model_epoch)
            elif mtype == 'skip25':
                load_path = args.basepath+'/models/flyNetSKIP6_25steps_32batch_sz_0.01lr_%dhids_gru_onehot%d_visionF1_vtype:%s_femaleflies_%s'% (num_hid, onehotF, videotype, model_epoch)
            elif mtype == 'skip50':
                if dtype == 'pdb':
                    load_path = args.basepath+'/models/%s/flyNetSKIP10_50steps_32batch_sz_0.01lr_%dhids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_femaleflies_%s'% (dtype, num_hid, onehotF, videotype, dtype, model_epoch)
                else:
                    load_path = args.basepath+'/models/%s/flyNetSKIP10_50steps_32batch_sz_0.01lr_%dhids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_femaleflies_%s'% (dtype, num_hid, onehotF, videotype, dtype, model_epoch)
            elif mtype == 'skip100':
                load_path = args.basepath+'/models/%s/flyNetSKIP9_100steps_32batch_sz_0.01lr_%dhids_gru_onehot%d_visionF1_vtype:%s_dtype:%s_femaleflies_%s'% (dtype, num_hid, onehotF, videotype, dtype, model_epoch)
            elif mtype == 'skip150':
                load_path = args.basepath+'/models/flyNetSKIP6_150steps_32batch_sz_0.01lr_%dhids_gru_onehot%d_visionF1_vtype:%s_femaleflies_%s'% (num_hid, onehotF, videotype, model_epoch)
            female_model = load_rnn(args, mtype=mtype, gender=FEMALE, \
                             num_hid=num_hid, cudaF=use_cuda, num_bin=num_bin,\
                             load_path=load_path)
            print('Female Model Load Path %s' % load_path)

        male_hiddens   = [male_model.initHidden(1, use_cuda=use_cuda) \
                            for i in range(len(simulated_male_flies))]
        female_hiddens = [female_model.initHidden(1, use_cuda=use_cuda)\
                            for i in range(len(simulated_female_flies))]
    else:
        model = motiondata
        male_hiddens = [None for i in range(len(real_male_flies))]
        female_hiddens = [None for i in range(len(real_female_flies))]

    return male_model, female_model, male_hiddens, female_hiddens


def init_simulation(trx, params, motiondata, basesize, n_flies, t0, t1,\
                        real_male_flies, real_female_flies, \
                        simulated_male_flies, simulated_female_flies, sim_type):


    # initial pose
    x = trx['x'][t0,:].copy()
    y = trx['y'][t0,:].copy()
    theta = trx['theta'][t0,:].copy()
  
    b = basesize['minax'].copy()
    a = trx['a'][t0,:].copy()
    l_wing_ang = trx['l_wing_ang'][t0,:].copy()
    r_wing_ang = trx['r_wing_ang'][t0,:].copy()
    l_wing_len = trx['l_wing_len'][t0,:].copy()
    r_wing_len = trx['r_wing_len'][t0,:].copy()
    male_ind, female_ind = gender_classify(basesize['majax'])        
    mymotiondata = np.zeros(motiondata.shape)

    if sim_type == 'SMSF':
        simulated_male_flies = np.arange(len(male_ind))
        simulated_female_flies = np.arange(len(male_ind), len(male_ind)+len(female_ind))

    male_state = [None]*(len(real_male_flies)+len(simulated_male_flies))
    female_state = [None]*(len(real_female_flies)+len(simulated_female_flies))
    feat_motion = motiondata[:,t0,:].copy()


    simtrx = {}
    tsim = T if t1 is None else t1
    simtrx['x'] = np.concatenate((trx['x'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['y'] = np.concatenate((trx['y'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['theta'] = np.concatenate((trx['theta'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['a'] = np.concatenate((trx['a'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['b'] = np.concatenate((trx['b'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['l_wing_ang'] = np.concatenate((trx['l_wing_ang'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['r_wing_ang'] = np.concatenate((trx['r_wing_ang'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['l_wing_len'] = np.concatenate((trx['l_wing_len'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['r_wing_len'] = np.concatenate((trx['r_wing_len'][:t0+1,:],np.zeros((tsim,n_flies))))

    return simtrx, x, y, theta, a, b, l_wing_ang, r_wing_ang, \
        l_wing_len, r_wing_len, male_state, female_state, feat_motion

def init_canvas(params, x,y,a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len,\
        n_flies,  sim_type='SMSF', DEBUG=1, fly_j=0):

    vdata, mdata, bdata = [], [], []
    if DEBUG >= 1:
        fig,ax = plt.subplots(figsize=(15,15))
        ax = plt.axes([0,0,1,1])
        if sim_type=='Single':
            colors = get_default_fly_colors_single(fly_j, n_flies)
            print(colors)
        elif sim_type=='LOO':
            colors = get_default_fly_colors_black(n_flies)
        else:
            colors = get_default_fly_colors_rb(n_flies+1)
    

        hbg = plt.imshow(params['bg'],cmap=cm.gray,vmin=0.,vmax=1.)
        htrx = []
        for fly in range(n_flies):
            htrxcurr, = ax.plot(x[fly],y[fly],'-',color=np.append(colors[fly,:-1],.5),linewidth=3)
            htrx.append(htrxcurr)

        hbodies,hflies,htexts = draw_flies(x,y,a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len,ax=ax,colors=colors, textOff=1,linewidth=5)
        counter_plt = plt.annotate('{:.2f}sec'.format(0. / default_params['FPS']),
                            xy=[1024-55,params['bg'].shape[0]-45],
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', size=18, color='black')

        plt.axis('off')

    return htrx, hbodies, hflies, htexts, hbg, counter_plt, colors, ax


def simulate_flies( args, real_male_flies, real_female_flies, \
                    simulated_male_flies, simulated_female_flies,\
                    male_model=None, female_model=None, plottrxlen=1,\
                    t0=0, t1=None, vision_save=False, histoF=False,\
                    visionF=0, bookkeepingF=True, fname='small', \
                    videotype='full', testvideo_num=0, vpath='',\
                    mtype='rnn50', DEBUG=0, burning=100, sim_type='SMSF',\
                    fly_single_ind=0, btype='linear', num_bin=101, dtype='rnn50',
                    use_cuda=0):

    from util_fly_sjb import compute_vision, motion2binidx, \
                    binscores2motion, update_position,\
                    get_default_fly_colors, draw_flies, update_flies

    
    from gen_dataset import video16_path
    fname = 'eyrun_simulate_data.mat'

    matfile = args.datapath+vpath+fname
    trx,motiondata,params,basesize = load_eyrun_data(matfile)
    n_flies= trx['x'].shape[1]
    T = trx['x'].shape[0]

    print(matfile)
    print(trx['x'].shape)

    if 'perc' in btype:
        binedges = np.load('./bins/percentile_%dbins.npy' % num_bin)
        params['binedges'] = binedges
    else:
        binedges = params['binedges']

    if sim_type == 'RMRF': mtype = 'data'
    params['mtype'] = mtype

    ## Initial pose
    simtrx, x, y, theta, a, b, l_wing_ang, r_wing_ang, l_wing_len, \
            r_wing_len, male_state, female_state, feat_motion = \
                    init_simulation(trx, params, motiondata, basesize, \
                    n_flies, t0, t1, real_male_flies, real_female_flies, \
                    simulated_male_flies, simulated_female_flies, sim_type)

    tsim = min([t1, T])
    male_ind, female_ind = gender_classify(basesize['majax'])                           

    xprev = x.copy()
    yprev = y.copy()
    thetaprev = theta.copy()


    ##Fly Visualization Initialization
    if DEBUG:
        if sim_type == 'Single':
            htrx, hbodies, hflies, htexts, hbg, counter_plt, colors, ax = \
                    init_canvas(params, x,y,a,b,theta,l_wing_ang,r_wing_ang,\
                    l_wing_len,r_wing_len, n_flies, DEBUG=DEBUG, \
                    sim_type=sim_type, fly_j=simulated_male_flies[0])
        else:
            htrx, hbodies, hflies, htexts, hbg, counter_plt, colors, ax = \
                init_canvas(params, x,y,a,b,theta,l_wing_ang,r_wing_ang,\
                l_wing_len,r_wing_len, n_flies, DEBUG=DEBUG, sim_type=sim_type)


    if sim_type == 'SMSF' or sim_type == 'LONG':
        simulated_male_flies = np.arange(len(male_ind))
        simulated_female_flies = np.arange(len(male_ind), len(female_ind)+len(male_ind))



    male_model, female_model, male_hiddens, female_hiddens \
            = model_selection(args, male_model, female_model, videotype, \
                        params['mtype'], args.model_epoch, args.h_dim,\
                    simulated_male_flies, simulated_female_flies, dtype, btype=btype, num_bin=num_bin)

    if sim_type == 'SMSF' or sim_type == 'LONG':
        male_hiddens   = [male_model.initHidden(1, use_cuda=use_cuda) \
                                for i in range(len(simulated_male_flies))]
        female_hiddens = [female_model.initHidden(1, use_cuda=use_cuda)\
                                for i in range(len(simulated_female_flies))]


  
       
    #histogram
    bins=100
    male_bucket = np.zeros([bins,bins])
    fale_bucket = np.zeros([bins,bins])
    male_dist_centre, fale_dist_centre = [], []
    male_velocity, fale_velocity = [], []
    male_pos = [np.hstack([trx['x'][0,male_ind], trx['y'][0,male_ind]])]
    fale_pos = [np.hstack([trx['x'][0,female_ind], trx['y'][0,female_ind]])]
    male_body_pos  = [[trx['theta'][0,male_ind], \
                       trx['l_wing_ang'][0,male_ind],\
                       trx['r_wing_ang'][0,male_ind],\
                       trx['l_wing_len'][0,male_ind],\
                       trx['r_wing_len'][0,male_ind]]]
    fale_body_pos  = [[trx['theta'][0,female_ind], \
                       trx['l_wing_ang'][0,female_ind],\
                       trx['r_wing_ang'][0,female_ind],\
                       trx['l_wing_len'][0,female_ind],\
                       trx['r_wing_len'][0,female_ind]]]
    male_motion, fale_motion = [], []
        
    hiddens_over_time = []
    print('Simulation Start %d %d...\n' % (t0, tsim))
    for counter, t in tqdm(enumerate(range(t0+1,tsim))):

        xprev[:] = x
        yprev[:] = y
        thetaprev[:] = theta
        male_dist_centre.append([x[male_ind]-default_params['arena_center_x'],\
                                 y[male_ind]-default_params['arena_center_y']])
        fale_dist_centre.append([x[female_ind]-default_params['arena_center_x'],\
                                 y[female_ind]-default_params['arena_center_y']])

        ## Simulate Male Model
        x, y, theta, a, l_wing_ang, r_wing_ang, \
            l_wing_len, r_wing_len, \
            male_hiddens, male_vision_chamber, feat_motion, _ = \
            get_simulate_fly(male_model, male_state, t, trx,\
                             simulated_male_flies, feat_motion,\
                                x, y, theta, a, b, \
                                l_wing_ang, r_wing_ang,\
                                l_wing_len, r_wing_len,\
                                xprev, yprev, thetaprev, 
                                basesize, params, mtype,\
                                num_bin=num_bin, hiddens=male_hiddens)

        ## Simulate Female Model
        x, y, theta, a, l_wing_ang, r_wing_ang, \
            l_wing_len, r_wing_len, \
            female_hiddens, female_vision_chamber, feat_motion, _ =\
            get_simulate_fly(female_model, female_state, t, trx,\
                             simulated_female_flies, feat_motion,\
                                x, y, theta, a, b, \
                                l_wing_ang, r_wing_ang,\
                                l_wing_len, r_wing_len,\
                                xprev, yprev, thetaprev, 
                                basesize, params, mtype, \
                                num_bin=num_bin, hiddens=female_hiddens)

        ## Real male Model
        x, y, theta, a, \
            l_wing_ang, r_wing_ang, \
            l_wing_len, r_wing_len, male_vision_chamber, _\
            = get_real_fly(real_male_flies, \
                                        motiondata, feat_motion,\
                                        t, trx, x, y, theta, 
                                        l_wing_ang, r_wing_ang,\
                                        l_wing_len, r_wing_len,\
                                        a, b, params)

        ## Real female Model
        x, y, theta, a, \
            l_wing_ang, r_wing_ang, l_wing_len, r_wing_len, \
            female_vision_chamber, _\
            = get_real_fly(real_female_flies, \
                                        motiondata, feat_motion,\
                                        t, trx, x, y, theta, 
                                        l_wing_ang, r_wing_ang,\
                                        l_wing_len, r_wing_len,\
                                        a, b, params)

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


        if DEBUG == 2:
            mymotiondata[:,t,fly] = \
                compute_motion(xprev[fly],yprev[fly],thetaprev[fly],x[fly],
                               y[fly],theta[fly],a[fly],l_wing_ang[fly],
                               r_wing_ang[fly],l_wing_len[fly],r_wing_len[fly],basesize,t,
                               fly,params)

        male_velocity.append([x[male_ind]-xprev[male_ind], y[male_ind]-yprev[male_ind]])
        fale_velocity.append([x[female_ind]-xprev[female_ind], y[female_ind]-yprev[female_ind]])

        male_pos.append(np.hstack([x[male_ind].copy(), y[male_ind].copy()]))
        male_motion.append(feat_motion[:,male_ind].copy())
        male_body_pos.append(np.asarray([theta[male_ind], \
                                l_wing_ang[male_ind], r_wing_ang[male_ind], \
                                l_wing_len[male_ind], r_wing_len[male_ind]]))

 
        fale_pos.append(np.hstack([x[female_ind].copy(), y[female_ind].copy()]))
        fale_motion.append(feat_motion[:,female_ind].copy())
        fale_body_pos.append(np.asarray([theta[female_ind], l_wing_ang[female_ind], r_wing_ang[female_ind], \
                              l_wing_len[female_ind], r_wing_len[female_ind]])) 

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
            tprev0 = np.maximum(t0+1,t-plottrxlen)
            tprev1 = t0+plottrxlen
            tprev2 = np.maximum(t0+burning,t-plottrxlen)
            tprev = np.maximum(t0+1,t-plottrxlen)
            for fly in range(n_flies):
                #htrx[fly].set_data(simtrx['x'][tprev0_real:tprev1,fly],simtrx['y'][tprev0_real:tprev1,fly], color='purple')

                if 'Single' in sim_type and fly_single_ind == fly:
                    htrx[fly].set_data(simtrx['x'][tprev1:t+1,fly],simtrx['y'][tprev1:t+1,fly])
                elif 'LONG' in sim_type:
                    if (t-t0-1) > burning: 
                        htrx[fly].set_data(simtrx['x'][tprev2:t+1,fly],simtrx['y'][tprev2:t+1,fly])
                else:
                    htrx[fly].set_data(simtrx['x'][tprev0:t+1,fly],simtrx['y'][tprev0:t+1,fly])

            if 'Single' in sim_type:
                ax.plot(simtrx['x'][tprev0:tprev1,fly_single_ind],\
                        simtrx['y'][tprev0:tprev1,fly_single_ind],\
                        '-',color='thistle',linewidth=3)


            update_flies(hbodies,hflies,htexts,x,y,a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len)
            plt.axis('off')
            plt.pause(.001)
            counter_plt.set_text('{:.2f}sec'.format(counter / default_params['FPS']))

            plt.annotate('{:.2f}ppm'.format(default_params['PPM']*10),
                            xy=[55,params['bg'].shape[0]-45],
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', size=14, color='black')
            plt.plot([20,20+default_params['PPM']*10],[params['bg'].shape[0]-40,params['bg'].shape[0]-40],'-',color='black',linewidth=2.)
 
            if t % 1 == 0 and t < t1 : #/10.0:
                if 'rnn' in params['mtype'] or 'skip' in params['mtype']:
                    if not os.path.exists('./figs/sim/%s/' % vpath): os.makedirs('./figs/sim/%s/' % vpath)   
                    sname = './figs/sim/%s/' % vpath +params['mtype']\
                            +'_'+sim_type\
                            +'_%s_%shid_%s_%dbins' % (videotype, args.h_dim, btype, num_bin)\
                            +'_epoch%d_32bs_%05d.png' % (args.model_epoch,t)
                    plt.savefig(sname, format='png')
                else:
                    plt.savefig('./figs/all/data_1000frames_%s_%5d.png' % (videotype, t), format='png', bbox_inches='tight')

    mtype = params['mtype']
    if mtype == 'data':
        ftag = str(t0)+'t0_'+str(t1)+'t1_'+sim_type+'_testvideo%d_%s' % (testvideo_num, dtype)
    else:
        ftag = str(t0)+'t0_'+str(t1)+'t1_epoch'+str(args.model_epoch)+'_'+sim_type+'_%s_%s_%dhid_lr%f_testvideo%d_%s' % (videotype, btype, args.h_dim, 0.01, testvideo_num, dtype)

    print ('ftag %s' % ftag)
    arena_radius = default_params['arena_radius']

    male_motion = np.asarray(male_motion) 
    fale_motion = np.asarray(fale_motion)

    male_pos = np.asarray(male_pos)
    fale_pos = np.asarray(fale_pos)
    male_body_pos = np.asarray(male_body_pos)
    fale_body_pos = np.asarray(fale_body_pos)

    #np.save('./trx/'+mtype+'_motion_male_'+ftag, simtrx)
    if not os.path.exists('./trx/%s/' % (vpath)): os.makedirs('./trx/%s/' % (vpath))   
    sio.savemat('./trx/%s/'%vpath+mtype+'_trx_'+ftag, simtrx)

    if bookkeepingF:
        if not os.path.exists('./motion/%s/' % (vpath)): os.makedirs('./motion/%s/' % (vpath))   
        np.save('./motion/'+vpath+'/'+mtype+'_motion_male_'+ftag, male_motion)
        np.save('./motion/'+vpath+'/'+mtype+'_motion_fale_'+ftag, fale_motion)
        np.save('./motion/'+vpath+'/'+mtype+'_position_male_'+ftag, male_pos)
        np.save('./motion/'+vpath+'/'+mtype+'_position_fale_'+ftag, fale_pos)
        np.save('./motion/'+vpath+'/'+mtype+'_body_position_male_'+ftag, male_body_pos)
        np.save('./motion/'+vpath+'/'+mtype+'_body_position_fale_'+ftag, fale_body_pos)


    male_velocity    = np.asarray(male_velocity) 
    fale_velocity    = np.asarray(fale_velocity)
    male_velocity    = np.sqrt(np.sum( male_velocity**2, axis=1)).flatten() 
    fale_velocity    = np.sqrt(np.sum( fale_velocity**2, axis=1)).flatten() 
    moving_male_ind = (male_velocity > 1.0)
    moving_fale_ind = (fale_velocity > 1.0)
    male_velocity_ns = male_velocity[moving_male_ind]
    fale_velocity_ns = fale_velocity[moving_fale_ind]
    if bookkeepingF:
        if not os.path.exists('./velocity/%s/' % (vpath)): os.makedirs('./velocity/%s/' % (vpath))   
        np.save('./velocity/'+vpath+'/'+mtype+'_velocity_male_'+ftag, male_velocity)
        np.save('./velocity/'+vpath+'/'+mtype+'_velocity_fale_'+ftag, fale_velocity)
        np.save('./velocity/'+vpath+'/'+mtype+'_velocity_woStationary_male_ind_'+ftag, moving_male_ind)
        np.save('./velocity/'+vpath+'/'+mtype+'_velocity_woStationary_fale_ind_'+ftag, moving_fale_ind)
        np.save('./velocity/'+vpath+'/'+mtype+'_velocity_woStationary_male_'+ftag, male_velocity_ns)
        np.save('./velocity/'+vpath+'/'+mtype+'_velocity_woStationary_fale_'+ftag, fale_velocity_ns)

    if histoF:
        male_histo = histogram(male_velocity/105, fname=mtype+'_velocity_male_histo_'+ftag, title='Velocity (Male)')
        fale_histo = histogram(fale_velocity/105, fname=mtype+'_velocity_fale_histo_'+ftag, title='Velocity (Female)')
        np.save('./hist/'+vpath+'/'+mtype+'_velocity_male_histo_'+ftag, male_histo)
        np.save('./hist/'+vpath+'/'+mtype+'_velocity_fale_histo_'+ftag, fale_histo)

        male_histo_ns = histogram(male_velocity_ns/105, fname=mtype+'_velocity_woStationary_male_histo_'+ftag, title='Velocity (Male)')
        fale_histo_ns = histogram(fale_velocity_ns/105, fname=mtype+'_velocity_woStationary_fale_histo_'+ftag, title='Velocity (Female)')
        np.save('./hist/'+vpath+'/'+mtype+'_velocity_woStationary_male_histo_'+ftag, male_histo_ns)
        np.save('./hist/'+vpath+'/'+mtype+'_velocity_woStationary_fale_histo_'+ftag, fale_histo_ns)


    male_dist_centre = np.asarray(male_dist_centre)
    fale_dist_centre = np.asarray(fale_dist_centre)
    male_dist_centre = np.sqrt(np.sum(male_dist_centre**2, axis=1)).flatten() 
    fale_dist_centre = np.sqrt(np.sum(fale_dist_centre**2, axis=1)).flatten() 

    male_dist_centre_ = male_dist_centre/arena_radius
    fale_dist_centre_ = fale_dist_centre/arena_radius
    if bookkeepingF:
        if not os.path.exists('./centredist/%s/' % (vpath)): os.makedirs('./centredist/%s/' % (vpath))   
        np.save('./centredist/'+vpath+'/'+mtype+'_centredist_male_'+ftag, male_dist_centre)
        np.save('./centredist/'+vpath+'/'+mtype+'_centredist_fale_'+ftag, fale_dist_centre)

    male_dist_centre_ns = male_dist_centre[moving_male_ind]
    fale_dist_centre_ns = fale_dist_centre[moving_fale_ind]
    male_dist_centre_ns_= male_dist_centre_ns/arena_radius
    fale_dist_centre_ns_= fale_dist_centre_ns/arena_radius
    if bookkeepingF:
        np.save('./centredist/'+vpath+'/'+mtype+'_centredist_woStationary_male_'+ftag, male_dist_centre_ns)
        np.save('./centredist/'+vpath+'/'+mtype+'_centredist_woStationary_fale_'+ftag, fale_dist_centre_ns)

    if histoF:
        male_dist_histo = histogram(male_dist_centre_, fname=mtype+'_dist2centre_male_histo_'+ftag, title='Distance to Centre (Male)')   
        fale_dist_histo = histogram(fale_dist_centre_, fname=mtype+'_dist2centre_fale_histo_'+ftag, title='Distance to Centre (Female)') 
        np.save('./hist/'+vpath+'/'+mtype+'_dist_male_histo_'+ftag, male_dist_histo)
        np.save('./hist/'+vpath+'/'+mtype+'_dist_fale_histo_'+ftag, fale_dist_histo)

        male_dist_histo_ns =histogram(male_dist_centre_ns_, fname=mtype+'_dist2centre_woStationary_male_histo_'+ftag, title='Distance to Centre (Male) excluding stationary flies')   
        fale_dist_histo_ns =histogram(fale_dist_centre_ns_, fname=mtype+'_dist2centre_woStationary_fale_histo_'+ftag, title='Distance to Centre (Female) excluding stationary flies') 
        np.save('./hist/'+vpath+'/'+mtype+'_dist_woStationary_male_histo_'+ftag, male_dist_histo_ns)
        np.save('./hist/'+vpath+'/'+mtype+'_dist_woStationary_fale_histo_'+ftag, fale_dist_histo_ns)


    male_bucket = male_bucket/(tsim-t0)
    fale_bucket = fale_bucket/(tsim-t0)
    print(max(male_bucket.max(), fale_bucket.max()))

    return male_pos, fale_pos



"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of FlyNetwork collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--btype', type=str, default='perc', choices=['linear', 'perc'])
    parser.add_argument('--mtype', type=str, default='rnn50', choices=['rnn50', 'hrnn50', 'skip50'])
    parser.add_argument('--model_epoch', type=int, default=200000)
    parser.add_argument('--sim_type', type=str, default='SMSF', choices=['LOO','SMSF', 'RMSF', 'SMRF'])
    parser.add_argument('--dtype', type=str, default='gmr', choices=['gmr', 'gmr91', 'hrnn', 'pdb'])
    parser.add_argument('--videotype', type=str, default='full', choices=['v1', 'full'])
    parser.add_argument('--num_bin', type=int, default=101)
    parser.add_argument('--num_mfeat', type=int, default=8)
    parser.add_argument('--visionF', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=100)
    parser.add_argument('--x_dim', type=int, default=152)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--fly_k', type=int, default=0)
    parser.add_argument('--savepath_male', type=str, default=None, help='Specify Male Model path')
    parser.add_argument('--savepath_female', type=str, default=None, help='Specify Female Model path')
    parser.add_argument('--basepath', type=str, default='/groups/branson/home/imd/Documents/janelia/research/fly_behaviour_sim/71g01/')
    parser.add_argument('--datapath', type=str, default='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/')
    parser.add_argument('--save_path_male', type=str, default=None)
    parser.add_argument('--save_path_female', type=str, default=None)

    return check_args(parser.parse_args())


if __name__ == '__main__':

    args = parse_args()
    args.y_dim = args.num_bin*args.num_mfeat


    if args.sim_type == 'LOO':
        t0,t1=0,30000
        fname='leave_one_out'
        dtype = args.dtype if args.dtype in video16_path or args.dtype != 'hrnn' else 'gmr'
        video_list = video16_path[dtype]
        burning=100


        for testvideo_num in range(0,len(video_list[TEST])):
            vpath = video_list[TEST][testvideo_num]
            print ('testvideo %d %s' % (testvideo_num, video_list[TEST][testvideo_num]))

            if  testvideo_num == 1 or testvideo_num == 2 :

                real_male_flies   = np.arange(1,9,1)
                real_female_flies = np.hstack([np.arange(9,10,1), np.arange(11,19,1)])

                simulated_male_flies = np.arange(0,1,1)
                simulated_female_flies = np.arange(10,11,1)

            else:
                real_male_flies = np.arange(1,10,1)
                real_female_flies = np.arange(11,20,1)

                simulated_male_flies = np.arange(0,1,1)
                simulated_female_flies = np.arange(10,11,1)

            print('Real Male and Female Flies')
            print(real_male_flies)
            print(real_female_flies)

            print('Simulated Male and Female Flies')
            print(simulated_male_flies)
            print(simulated_female_flies)

            simulate_flies(args, real_male_flies, real_female_flies, \
                        simulated_male_flies, simulated_female_flies,\
                        male_model=None, female_model=None, mtype=args.mtype,\
                        plottrxlen=100, t0=t0, t1=t1, bookkeepingF=1,\
                        vision_save=False, fname=fname, sim_type=args.sim_type,\
                        DEBUG=args.debug, burning=burning, videotype=args.videotype,\
                        testvideo_num=testvideo_num, vpath=vpath, btype=args.btype,\
                        num_bin=args.num_bin, dtype=args.dtype)

    elif args.sim_type == 'SMSF':
        t0,t1=0,30320
        burning=100
        fname='allsim'
        real_male_flies = []#np.arange(1,10,1)
        real_female_flies = []#np.arange(11,20,1)
        dtype = args.dtype if args.dtype in video16_path or args.dtype != 'hrnn' else 'gmr'
        video_list = video16_path[dtype]

        for testvideo_num in range(0,len(video_list[TEST])):
            vpath = video_list[TEST][testvideo_num]

            print ('testvideo %d %s' % (testvideo_num, video_list[TEST][testvideo_num]))

            if  testvideo_num == 1 or testvideo_num == 2 :
                simulated_male_flies = np.arange(0,9,1)
                simulated_female_flies = np.arange(9,19,1)
            else:
                simulated_male_flies = np.arange(0,10,1)
                simulated_female_flies = np.arange(10,20,1)

            print('Real Male and Female Flies')
            print(real_male_flies)
            print(real_female_flies)

            print('Simulated Male and Female Flies')
            print(simulated_male_flies)
            print(simulated_female_flies)

            simulate_flies( args, real_male_flies, real_female_flies, \
                            simulated_male_flies, simulated_female_flies,\
                            male_model=None, female_model=None, DEBUG=args.debug,\
                            plottrxlen=100, t0=t0, t1=t1, bookkeepingF=0,\
                            vision_save=False, fname=fname, mtype=args.mtype, \
                            videotype=args.videotype, burning=burning, vpath=vpath,\
                            testvideo_num=testvideo_num, btype=args.btype,\
                            num_bin=args.num_bin, dtype=args.dtype)

    elif args.sim_type == 'RMRF':
        t0,t1=0,10000
        burning=100
        fname='allreal'
        simulated_male_flies = []
        simulated_female_flies = []

        for testvideo_num in range(4,10):
            print ('testvideo %d' % testvideo_num)

            if testvideo_num == 2:
                real_male_flies = np.arange(0,10,1)
                real_female_flies = np.arange(10,19,1)
            elif  testvideo_num == 7 or testvideo_num == 6 or testvideo_num == 3:
                real_male_flies = np.arange(0,9,1)
                real_female_flies = np.arange(9,19,1)
            else:
                real_male_flies = np.arange(0,10,1)
                real_female_flies = np.arange(10,20,1)

            print('Real Male and Female Flies')
            print(real_male_flies)
            print(real_female_flies)

            print('Simulated Male and Female Flies')
            print(simulated_male_flies)
            print(simulated_female_flies)


            simulate_flies( args, real_male_flies, real_female_flies, \
                        simulated_male_flies, simulated_female_flies,\
                        male_model=None, female_model=None, DEBUG=args.debug, \
                        plottrxlen=100, t0=t0, t1=t1, bookkeepingF=True,\
                        vision_save=False, fname=fname, mtype=args.mtype, \
                        burning=burning, videotype=args.videotype,\
                        sim_type=args.sim_type, testvideo_num=testvideo_num,\
                        dtype=args.dtype)



    elif args.sim_type == 'RMSF':
        burning=100
        t0,t1=0,30320
        fname='realmale_simfemale'
        dtype = args.dtype if args.dtype in video16_path or args.dtype != 'hrnn' else 'gmr'
        video_list = video16_path[dtype]


        for testvideo_num in range(0,len(video_list[TEST])):
            vpath = video_list[TEST][testvideo_num]
            print ('testvideo %d %s' % (testvideo_num, video_list[TEST][testvideo_num]))
            simulated_female_flies = np.arange(10,20,1)
            real_male_flies = np.arange(0,10,1)
            if  testvideo_num == 1 or testvideo_num == 2 :
                simulated_female_flies = np.arange(9,19,1)
                real_male_flies = np.arange(0,9,1)


            simulated_male_flies = []#np.arange(0,10,1)
            real_female_flies = []#np.arange(11,20,1)


            print('Real Male and Female Flies')
            print(real_male_flies)
            print(real_female_flies)

            print('Simulated Male and Female Flies')
            print(simulated_male_flies)
            print(simulated_female_flies)

            simulate_flies( args, real_male_flies, real_female_flies, \
                        simulated_male_flies, simulated_female_flies,\
                        male_model=None, female_model=None, vpath=vpath, \
                        plottrxlen=100, t0=t0, t1=t1, bookkeepingF=1,\
                        vision_save=False, fname=fname, mtype=args.mtype,\
                        sim_type=args.sim_type, burning=burning, DEBUG=args.debug,\
                        videotype=args.videotype, testvideo_num=testvideo_num,\
                        dtype=args.dtype)


    elif args.sim_type == 'SMRF':
        burning=100
        t0,t1=0,30320
        fname='simmale_realfemale'
        dtype = args.dtype if args.dtype in video16_path or args.dtype != 'hrnn' else 'gmr'
        video_list = video16_path[dtype]

        for testvideo_num in range(0,len(video_list[TEST])):
            vpath = video_list[TEST][testvideo_num]
            print ('testvideo %d %s' % (testvideo_num, video_list[TEST][testvideo_num]))

            if  testvideo_num == 1 or testvideo_num == 2 :
                simulated_male_flies = np.arange(0,9,1)
                real_female_flies = np.arange(9,19,1)
            else:
                real_female_flies = np.arange(10,20,1)
                simulated_male_flies = np.arange(0,10,1)
                #simulated_female_flies = np.arange(10,20,1)

            simulated_female_flies = []
            real_male_flies = []#np.arange(0,10,1)

            print('Real Male and Female Flies')
            print(real_male_flies)
            print(real_female_flies)

            print('Simulated Male and Female Flies')
            print(simulated_male_flies)
            print(simulated_female_flies)


            simulate_flies( args, real_male_flies, real_female_flies, \
                        simulated_male_flies, simulated_female_flies,\
                        male_model=None, female_model=None, vpath=vpath,\
                        plottrxlen=100, t0=t0, t1=t1, bookkeepingF=1,\
                        vision_save=False, fname=fname, mtype=args.mtype,\
                        videotype=args.videotype, testvideo_num=testvideo_num,\
                        sim_type=args.sim_type, burning=burning, DEBUG=args.debug,\
                        dtype=args.dtype)


    elif args.sim_type == 'custom':
        t0,t1=0,1000
        fname='custom'
        real_male_flies = []#np.arange(0,10,1)
        real_female_flies = np.arange(10,20,1)

        simulated_male_flies = np.arange(0,10,1)
        simulated_female_flies = []#np.arange(10,20,1)

        print('Real Male and Female Flies')
        print(real_male_flies)
        print(real_female_flies)

        print('Simulated Male and Female Flies')
        print(simulated_male_flies)
        print(simulated_female_flies)

        simulate_flies( args, real_male_flies, real_female_flies, \
                        simulated_male_flies, simulated_female_flies,\
                        male_model=None, female_model=None, \
                        plottrxlen=100, t0=t0, t1=t1, bookkeepingF=False,\
                        vision_save=False, fname=fname, sim_type=argssim_type,\
                        DEBUG=args.debug, burning=burning, videotype=args.videotype,\
                        testvideo_num=testvideo_num, vpath=vpath,\
                        fly_single_ind=fly_k, btype=args.btype, \
                        num_bin=args.num_bin, dtype=args.dtype)
    elif args.sim_type == 'Single':
        t0,t1=0,10000
        #t0,t1=1000,1070
        #t0, t1 = 0, None#int(30320 * 0.9), 30320
        fname='leave_one_out'
        dtype = args.dtype if args.dtype in video16_path or args.dtype != 'hrnn' else 'gmr'
        video_list = video16_path[dtype]
        burning=100
        fly_k = args.fly_k
        for testvideo_num in range(0,len(video_list[TEST])):
            vpath = video_list[TEST][testvideo_num]
            print ('testvideo %d %s' % (testvideo_num, video_list[TEST][testvideo_num]))

            if  testvideo_num == 1 or testvideo_num == 2 :

                real_male_flies   = np.hstack([np.arange(0,fly_k+1,1), np.arange(fly_k+1,19,1)])
                real_female_flies = np.arange(9,19,1)

                simulated_male_flies = np.asarray([fly_k])
                simulated_female_flies = np.asarray([])#np.arange(10,11,1)

            else:
                real_male_flies = np.hstack([np.arange(0,fly_k), np.arange(fly_k+1,10,1)])
                real_female_flies = np.arange(10,20,1)

                simulated_male_flies = np.asarray([fly_k])
                simulated_female_flies = np.asarray([])# np.arange(10,11,1)

            print('Real Male and Female Flies')
            print(real_male_flies)
            print(real_female_flies)

            print('Simulated Male and Female Flies')
            print(simulated_male_flies)
            print(simulated_female_flies)

            simulate_flies(args, real_male_flies, real_female_flies, \
                        simulated_male_flies, simulated_female_flies,\
                        male_model=None, female_model=None, mtype=args.mtype,\
                        plottrxlen=100, t0=t0, t1=t1, bookkeepingF=0,\
                        vision_save=False, fname=fname, sim_type=argssim_type,\
                        DEBUG=args.debug, burning=burning, videotype=args.videotype,\
                        testvideo_num=testvideo_num, vpath=vpath,\
                        fly_single_ind=fly_k, btype=args.btype, \
                        num_bin=args.num_bin, dtype=args.dtype)


    if args.sim_type == 'LONG':
        t0,t1=0,30320
        #t0, t1 = int(30320 * 0.9), 30320
        #t0, t1 = int(30320 * 0.7), int(30320 * 0.9)
        #t1=t0+1000
        burning=100
        fname='allsim'
        real_male_flies = []#np.arange(1,10,1)
        real_female_flies = []#np.arange(11,20,1)
        dtype = args.dtype if args.dtype in video16_path or args.dtype != 'hrnn' else 'gmr'
        video_list = video16_path[dtype]

        #for testvideo_num, vpath in enumerate(video_list[TEST]):
        for testvideo_num in range(3,len(video_list[TEST])):
            #for t0, t1 in [(9900,10080), (21900,22130), (27900,28300), (12900,13500)]:
            for t0, t1 in [(24900,25080), (11900,12130), (20900,21300), (15900,16500)]:

                vpath = video_list[TEST][testvideo_num]

                print ('testvideo %d %s %s' % (testvideo_num, video_list[TEST][testvideo_num], args.sim_type))

                if  testvideo_num == 1 or testvideo_num == 2 :
                    simulated_male_flies = np.arange(0,9,1)
                    simulated_female_flies = np.arange(9,19,1)
                else:
                    simulated_male_flies = np.arange(0,10,1)
                    simulated_female_flies = np.arange(10,20,1)

                print('Real Male and Female Flies')
                print(real_male_flies)
                print(real_female_flies)

                print('Simulated Male and Female Flies')
                print(simulated_male_flies)
                print(simulated_female_flies)

                simulate_flies( args, real_male_flies, real_female_flies, \
                                simulated_male_flies, simulated_female_flies,\
                                male_model=None, female_model=None, DEBUG=args.debug,\
                                plottrxlen=100, t0=t0, t1=t1, bookkeepingF=0,\
                                vision_save=False, fname=fname, mtype=args.mtype, \
                                videotype=args.videotype, burning=burning, vpath=vpath,\
                                testvideo_num=testvideo_num, sim_type=args.sim_type,\
                                btype=args.btype, num_bin=args.num_bin,\
                                dtype=args.dtype)





