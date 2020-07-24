import numpy as np
import h5py, os, time, argparse
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import scipy.io as sio

import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from gen_dataset import load_eyrun_data, video16_path, gender_classify, load_vision
#from simulate_linear_reg import apply_lr_fly_np
from util_fly import *
from util import * 

from simulate_rnn import get_simulate_fly, get_real_fly
from apply_model import *

MALE=0
FEMALE=1
NUM_FLY=20
NUM_BIN=51
NUM_MFEAT=8
NUM_VFEAT=144
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


# simulate_next_t_steps(t0,curpos,state,feat_motion,model,params,basesize,tsim=30)
# inputs:
# t0: start time point
# curpos: initial position of flies
# feat_motion: motion features from previous time point
# model: predictor
# params: parameters
# basesize: base size of flies
# tsim: number of steps to simulate
#
# output:
# simtrx: dictionary with trajectories for each fly
def simulate_next_t_steps(t0,curpos, state, \
        feat_motion,male_model, female_model,\
        male_ind, female_ind, params,basesize, motiondata, simulated_flies,\
            tsim=30, monlyF=0, mtype='rnn', t_dim=50, visionOnly=0,\
            vc_data=None):


    from util_fly import compute_vision, motion2binidx, \
                            binscores2motion, update_position

    visionF=abs(1-monlyF)
    x=curpos['x'].copy()
    y=curpos['y'].copy()
    a=curpos['a'].copy()
    b=curpos['b'].copy()
    theta=curpos['theta'].copy()
    l_wing_ang=curpos['l_wing_ang']
    r_wing_ang=curpos['r_wing_ang']
    l_wing_len=curpos['l_wing_len']
    r_wing_len=curpos['r_wing_len']

    # whether we are going to do the simulation within this function, or if it is already done
    n_flies = len(x)

    simtrx = {}
    simtrx['x']=np.zeros((tsim+1,n_flies))
    simtrx['y']=np.zeros((tsim+1,n_flies))
    simtrx['theta']=np.zeros((tsim+1,n_flies))
    simtrx['a']=np.zeros((tsim+1,n_flies))
    simtrx['b']=np.zeros((tsim+1,n_flies))
    simtrx['l_wing_ang']=np.zeros((tsim+1,n_flies))
    simtrx['r_wing_ang']=np.zeros((tsim+1,n_flies))
    simtrx['l_wing_len']=np.zeros((tsim+1,n_flies))
    simtrx['r_wing_len']=np.zeros((tsim+1,n_flies))


    simtrx['x'][0,:]=curpos['x']
    simtrx['y'][0,:]=curpos['y']
    simtrx['theta'][0,:]=curpos['theta']
    simtrx['a'][0,:]=curpos['a']
    simtrx['b'][0,:]=curpos['b']
    simtrx['l_wing_ang'][0,:]=curpos['l_wing_ang']
    simtrx['r_wing_ang'][0,:]=curpos['r_wing_ang']
    simtrx['l_wing_len'][0,:]=curpos['l_wing_len']
    simtrx['r_wing_len'][0,:]=curpos['r_wing_len']

    predictions_flies, flyvisions_flies = [], []
    #for fly2 in range(n_flies):
    for fly2 in simulated_flies:

        x=curpos['x'].copy()
        y=curpos['y'].copy()
        a=curpos['a'].copy()
        b=curpos['b'].copy()
        theta=curpos['theta'].copy()
        l_wing_ang=curpos['l_wing_ang'].copy()
        r_wing_ang=curpos['r_wing_ang'].copy()
        l_wing_len=curpos['l_wing_len'].copy()
        r_wing_len=curpos['r_wing_len'].copy()

        xprev = x.copy()
        yprev = y.copy()
        thetaprev = theta.copy()

        if 'conv' in mtype:
            state[:,:8,:] = motiondata[:,t0-t_dim:t0,:].transpose(2,0,1)
            #state[fly,8:,-1] = vc_data[t0-t_dim:t0,:,:]
        else:
            state = motiondata[:,t0-t_dim:t0,:].transpose(2,0,1)
         
        predictions, flyvisions = [], []
        for t in range(1,tsim):

            xprev[:]=x
            yprev[:]=y
            thetaprev[:]=theta

            for model, ind  in zip([male_model, female_model],\
                                    [male_ind, female_ind]):

                for flyi in range(len(ind)):
                            
                    fly = ind[flyi]
                    if fly == fly2:

                        #Vision features for frame t-1
                        (flyvision,chambervision)\
                                =compute_vision(x,y,theta,a,b,fly,params)

                        if 'lr' in mtype:
                            # predicted motion from frame t-1 to t
                            (binscores,state,pred) = apply_lr_fly_np(\
                                                    state, \
                                                    flyvision,chambervision,\
                                                    feat_motion,\
                                                    model, params,\
                                                    visionF=args.visionF, t=t0+t, \
                                                    t_dim=t_dim, fly=fly,\
                                                    visionOnly=visionOnly)
                            feat_motion[:,fly] = binscores #binscores2motion(binscores,params)
                        elif 'nn' in mtype:
                            # predicted motion from frame t-1 to t
                            (binscores,state,preds) \
                                    = apply_nn_fly(state, flyvision, \
                                                    chambervision,\
                                                    feat_motion, model, \
                                                    params, t_dim=t_dim,\
                                                    fly=fly)
                            feat_motion[:,fly]=binscores2motion(\
                                                    binscores,params)
                        elif 'conv' in mtype and 'cat' in mtype:
                            # predicted motion from frame t-1 to t
                            (binscores,state,preds) \
                                    = apply_conv_fly(state, flyvision, \
                                                    chambervision,\
                                                    feat_motion, model, \
                                                    params, t_dim=t_dim,\
                                                    fly=fly)
                            feat_motion[:,fly]=binscores2motion(\
                                                    binscores,params)
                        elif 'conv' in mtype and 'reg' in mtype:
                            # predicted motion from frame t-1 to t
                            (binscores,state,preds) \
                                    = apply_conv_reg_fly(state, flyvision, \
                                                    chambervision,\
                                                    feat_motion, model, \
                                                    params, t_dim=t_dim,\
                                                    fly=fly)
                            feat_motion[:,fly]=binscores2motion(\
                                                    binscores,params)

                        else:
                            # predicted motion from frame t-1 to t
                            (binscores,state,pred) = apply_reg_fly(\
                                                    state, \
                                                    flyvision,chambervision,\
                                                    feat_motion,\
                                                    model, params,\
                                                    args.visionF, t, t_dim, fly)
                            feat_motion[:,fly] = binscores #binscores2motion(binscores,params)
                        predictions.append(binscores)
                        flyvisions.append(np.hstack([flyvision,chambervision]))

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
                    if fly == fly2: 
                        simtrx['x'][t,fly]=x[fly]
                        simtrx['y'][t,fly]=y[fly]
                        simtrx['theta'][t,fly]=theta[fly]
                        simtrx['a'][t,fly]=a[fly]
                        simtrx['b'][t,fly]=b[fly]
                        simtrx['l_wing_ang'][t,fly]=l_wing_ang[fly]
                        simtrx['r_wing_ang'][t,fly]=r_wing_ang[fly]
                        simtrx['l_wing_len'][t,fly]=l_wing_len[fly]
                        simtrx['r_wing_len'][t,fly]=r_wing_len[fly]
        flyvisions_flies.append(flyvisions)
        predictions_flies.append(predictions)
    predictions_flies = np.asarray(predictions_flies).transpose(1,0,2)

    return simtrx, feat_motion, predictions_flies, flyvisions_flies



def model_selection( args, male_model, female_model, params, \
                                        model_epoch=5000,\
                                        visionOnlyF=0):
  

    if 'nn' in args.mtype or 'conv' in args.mtype:
        
        vis_histo = 0
        if 1:
            basepath = '/groups/branson/home/imd/Documents/janelia/research/FlyTrajPred_v2/pytorch/'
            basepath4 = '/groups/branson/home/imd/Documents/janelia/research/FlyTrajPred_v4/pytorch/'

            ftag_m = 'gender%d_%s_bs%d_'% (0, args.vtype, args.batch_sz) 
            ftag_f = 'gender%d_%s_bs%d_'% (1, args.vtype, args.batch_sz) 
            if visionOnlyF:
                ftag_m= ftag_m+'visionOnly1'
                ftag_f= ftag_f+'visionOnly1'
            elif not args.visionF: 
                ftag_m = ftag_m+'visionF0'
                ftag_f = ftag_f+'visionF0'

            from flyNetwork_MLP import NeuralNetwork4, ConvNet
            t_dim = int(args.mtype.split('_')[1][3:])

            if male_model is None:
                load_path='./runs/%s/%s/model/model_%sepoch%d' \
                            % (args.mtype, args.dtype, ftag_m, model_epoch)

                #args = parse_args()
                args.gender=0
                args.visionOnly = visionOnlyF
                args.visionHist = vis_histo
                if 'nn' in args.mtype:
                    male_model = NeuralNetwork4(args)
                elif 'conv' in args.mtype and 'relu' in args.mtype:
                    args.atype = 'relu'
                    male_model = ConvNet(args, args.x_dim, args.y_dim)
                elif 'conv' in args.mtype:

                    male_model = ConvNet(args, args.x_dim, args.y_dim)
                print(load_path)
                male_model = load(male_model, load_path)

            if female_model is None:
                load_path='./runs/%s/%s/model/model_%sepoch%d' \
                            % (args.mtype, args.dtype, ftag_f, model_epoch)

                #args = parse_args()
                args.gender=1
                args.visionOnly = visionOnlyF
                args.visionHist = vis_histo
                if 'nn' in args.mtype:
                    female_model = NeuralNetwork4(args)
                elif 'conv' in args.mtype:
                    female_model = ConvNet(args, args.x_dim, args.y_dim)
                print(load_path)
                female_model = load(female_model, load_path)

    elif 'lr' in args.mtype:
 
        if male_model is None:
            save_path='./runs/linear_reg_'+str(t_dim) +'tau/%s/model/weight_gender0' % args.dtype
            if visionOnlyF:
                save_path = save_path+'_visionOnly1'
            elif not args.visionF: 
                save_path = save_path +'_visionF0'
            print(save_path+'.npy')
            male_model = np.load(save_path+'.npy')

        if female_model is None:
            save_path='./runs/linear_reg_'+str(t_dim) +'tau/%s/model/weight_gender1' % args.dtype
            if visionOnlyF:
                save_path = save_path+'_visionOnly1'
            elif not args.visionF: 
                save_path = save_path +'_visionF0'
            print(save_path+'.npy')
            female_model = np.load(save_path+'.npy')
    else:
        model = motiondata

    return male_model, female_model


def simulate_flies( args, real_male_flies, real_female_flies, \
                    simulated_male_flies, simulated_female_flies,\
                    male_model=None, female_model=None, plottrxlen=1,\
                    t0=0, t1=30320, vision_save=False, histoF=False, t_dim=50,\
                    bookkeepingF=True, fname='small', \
                    burning=100,DEBUG=1, testvideo_num=0,\
                    sim_type='SMSF', batch_sz=32,\
                    visionOnlyF=0, vpath='', ftag='',\
                    fly_single_ind=0 ):

    from util_fly import compute_vision, motion2binidx, \
                            binscores2motion, update_position

    from gen_dataset import video16_path
    if vpath == '': vpath = video16_path[args.dtype][TEST][testvideo_num]
    fname = 'eyrun_simulate_data.mat'
    basepath='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'


    vision_matfile = basepath+vpath+'movie-vision.mat'
    vc_data = load_vision(vision_matfile)[1:]

    matfile = basepath+vpath+fname
    trx,motiondata,params,basesize = load_eyrun_data(matfile)
    binedges = params['binedges']
    params['mtype'] = args.mtype
    n_flies= trx['x'].shape[1]
    T = trx['x'].shape[0]

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

    xprev = x.copy()
    yprev = y.copy()
    thetaprev = theta.copy()


    #NUM_FLY = 
    if 'conv' in args.mtype:
        state = np.zeros(( n_flies, NUM_VFEAT+NUM_MFEAT, t_dim ))
        if t0 > 50:
            state[:,:8,:] = motiondata[:,t0-50:t0,:].transpose(2,0,1)
            state[:,8:,:] = vc_data[t0-50:t0,:,:].transpose(1,2,0)

        ### MANIPULATION PURPOSE 
        #fly_j=6
        #zeroout_ind = [39, 31, 29, 38, 33, 32, 12, 36,  9, 42,\
        #    35, 44, 49, 47, 34,  6,  8,  7, 27, 46, 45, 26, 48, 28, 24, 25]
        #vision_feat = state[fly_j:,8:8+72,:]
        #state[fly_j:,8:8+72,zeroout_ind] = 0.15

    elif visionOnlyF:
        state = np.zeros(( n_flies, NUM_VFEAT, t_dim ))
    else:
        state = np.zeros(( n_flies, NUM_MFEAT, t_dim ))
    feat_motion = motiondata[:,t0,:].copy()

    male_ind, female_ind = gender_classify(basesize['majax'])        
    #male_ind, female_ind =simulated_male_flies, simulated_female_flies
    mymotiondata = np.zeros(motiondata.shape)

    simtrx = {}
    #tsim = T#min(t1,T)-t0-1
    tsim = T if t1 is None else t1
    #tsim = t1-t0-1
    simtrx['x'] = np.concatenate((trx['x'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['y'] = np.concatenate((trx['y'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['theta'] = np.concatenate((trx['theta'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['a'] = np.concatenate((trx['a'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['b'] = np.concatenate((trx['b'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['l_wing_ang'] = np.concatenate((trx['l_wing_ang'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['r_wing_ang'] = np.concatenate((trx['r_wing_ang'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['l_wing_len'] = np.concatenate((trx['l_wing_len'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['r_wing_len'] = np.concatenate((trx['r_wing_len'][:t0+1,:],np.zeros((tsim,n_flies))))


    ##Fly Visualization Initialization
    vdata, mdata, bdata = [], [], []
    if DEBUG >= 1:
        fig = plt.figure(figsize=(15,15))
        ax = plt.axes([0,0,1,1])

        #colors = get_default_fly_colors(n_flies)
        if sim_type=='Single':
            fly_j = simulated_male_flies[0]
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

        hbodies,hflies,htexts = draw_flies(x,y,a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len,ax=ax,colors=colors, textOff=True)
        #plt.axis('image')

        counter_plt = plt.annotate('{:.2f}sec'.format(0. / default_params['FPS']),
                            xy=[1024-55,params['bg'].shape[0]-45],
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', size=18, color='black')


        plt.axis('off')


    load_start = time.time()
    male_model, female_model = model_selection( args, male_model, \
                                                female_model, params, \
                                                model_epoch=args.model_epoch,\
                                                visionOnlyF=visionOnlyF)
    load_end = time.time()
    print ('Model Loading time %f' % ((load_end-load_start)/60.0))

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
    print('Simulation Start...\n')
    from tqdm import tqdm
    for counter, t in tqdm(enumerate(range(t0+1,t1))):

        xprev[:] = x
        yprev[:] = y
        thetaprev[:] = theta
        male_dist_centre.append([x[male_ind]-default_params['arena_center_x'],\
                                 y[male_ind]-default_params['arena_center_y']])
        fale_dist_centre.append([x[female_ind]-default_params['arena_center_x'],\
                                 y[female_ind]-default_params['arena_center_y']])

        if (t-t0-1) > burning: 
            ## Simulate Male Model
            x, y, theta, a, l_wing_ang, r_wing_ang, \
                l_wing_len, r_wing_len, _, \
                male_vision_chamber, feat_motion, state = \
                get_simulate_fly(male_model, state, t, trx,\
                                 simulated_male_flies, feat_motion,\
                                    x, y, theta, a, b, \
                                    l_wing_ang, r_wing_ang,\
                                    l_wing_len, r_wing_len,\
                                    xprev, yprev, thetaprev, 
                                    basesize, params, args.mtype,\
                                    visionF=args.visionF, \
                                    visionOnly=visionOnlyF,\
                                    t_dim=t_dim)

            ## Simulate Female Model
            x, y, theta, a, l_wing_ang, r_wing_ang, \
                l_wing_len, r_wing_len, _,\
                female_vision_chamber, feat_motion, state =\
                get_simulate_fly(female_model, state, t, trx, \
                                 simulated_female_flies, feat_motion,\
                                    x, y, theta, a, b, \
                                    l_wing_ang, r_wing_ang,\
                                    l_wing_len, r_wing_len,\
                                    xprev, yprev, thetaprev, 
                                    basesize, params, args.mtype,\
                                    visionF=args.visionF, \
                                    visionOnly=visionOnlyF, \
                                    t_dim=t_dim)

            ## Real male Model
            x, y, theta, a, \
                l_wing_ang, r_wing_ang, \
                l_wing_len, r_wing_len, male_vision_chamber, feat_motion\
                = get_real_fly(real_male_flies, \
                                            motiondata, feat_motion,\
                                            t, trx, x, y, theta, 
                                            l_wing_ang, r_wing_ang,\
                                            l_wing_len, r_wing_len,\
                                            a, b, params)

            ## Real female Model
            x, y, theta, a, \
                l_wing_ang, r_wing_ang, \
                l_wing_len, r_wing_len, female_vision_chamber, feat_motion\
                = get_real_fly(real_female_flies, \
                                            motiondata, feat_motion,\
                                            t, trx, x, y, theta, 
                                            l_wing_ang, r_wing_ang,\
                                            l_wing_len, r_wing_len,\
                                            a, b, params)

        else:
            for fly in range(n_flies):
                x[fly] = trx['x'][t,fly]
                y[fly] = trx['y'][t,fly]
                theta[fly] = trx['theta'][t,fly]
                a[fly] = trx['a'][t,fly]
                l_wing_ang[fly] = trx['l_wing_ang'][t,fly]
                r_wing_ang[fly] = trx['r_wing_ang'][t,fly]
                l_wing_len[fly] = trx['l_wing_len'][t,fly]
                r_wing_len[fly] = trx['r_wing_len'][t,fly]

                if 'conv' in args.mtype:
                    state[fly,:,:(t_dim-1)] = state[fly,:,1:] 
                    state[fly,:8,-1] = feat_motion[:,fly]
                    state[fly,8:,-1] = vc_data[t,fly,:]
                if visionOnlyF:
                    state[fly,:,:(t_dim-1)] = state[fly,:,1:] 
                    state[fly,:,-1] = vc_data[t,fly,:]
                else:
                    state[fly,:,:(t_dim-1)] = state[fly,:,1:] 
                    state[fly,:8,-1] = feat_motion[:,fly]

                feat_motion[:,fly] = motiondata[:,t,fly]


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
                if 'Single' in sim_type and fly_single_ind == fly:
                    htrx[fly].set_data(simtrx['x'][tprev1:t+1,fly],simtrx['y'][tprev1:t+1,fly])
                elif 'LONG' in sim_type:
                    if (t-t0-1) > burning: 
                        htrx[fly].set_data(simtrx['x'][tprev2:t+1,fly],simtrx['y'][tprev2:t+1,fly])
                else:
                    htrx[fly].set_data(simtrx['x'][tprev:t+1,fly],simtrx['y'][tprev:t+1,fly])

            if 'Single' in sim_type:
                ax.plot(simtrx['x'][tprev0:tprev1,fly_single_ind],\
                        simtrx['y'][tprev0:tprev1,fly_single_ind],\
                        '-',color='thistle',linewidth=3)

            update_flies(hbodies,hflies,htexts,x,y,a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len)
            plt.pause(.001)
            counter_plt.set_text('{:.2f}sec'.format(counter / default_params['FPS']))
            #counter_plt = plt.annotate('{:.2f}sec'.format(counter / default_params['FPS']),
            #                xy=[1024-55,params['bg'].shape[0]-45],
            #                xytext=(0, 3),  # 3 points vertical offset
            #                textcoords="offset points",
            #                ha='center', va='bottom', size=18, color='w')


            # plot scale bar
            plt.annotate('{:.2f}ppm'.format(default_params['PPM']*10),
                            xy=[55,params['bg'].shape[0]-45],
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', size=14, color='black')
            plt.plot([20,20+default_params['PPM']*10],[params['bg'].shape[0]-40,params['bg'].shape[0]-40],'-',color='black',linewidth=2.)

            if t % 1 == 0 and t < t1 : #/10.0:
                os.makedirs('./figs/sim/%s/' % vpath, exist_ok=True)   
                if 'nn' in params['mtype'] or 'conv' in params['mtype']:
                    plt.savefig('./figs/sim/%s/%s_%s_' % (vpath, params['mtype'], sim_type)\
                            +str(args.h_dim)+'hid_tau%d_' % (50) \
                            +'vision%d_visionOnly%d' % (visionF,visionOnlyF) \
                            +'_epoch'+str(args.model_epoch)\
                            +'_%dbs_%s_%05d.png' % (args.batch_sz,args.dtype,t), format='png')

                elif 'lr' in params['mtype']:
                    plt.savefig('./figs/sim/%s/lr_%s_' % (vpath, sim_type) +str(t_dim)\
                            +'tau_vision%d_visionOnly%d_%05d.png' % (visionF,visionOnlyF,t), \
                            format='png')
                            #+str(t0)+'t0_'+str(t1)+'t1_%05d.png' % t,\

                else:
                    plt.savefig('./figs/all/data_1000frames_%5d.png' % t, format='png', bbox_inches='tight')

    mtype = args.mtype 
    #ftag = str(t0)+'t0_'+str(t1)+'t1'
    if visionOnlyF:
        ftag += 'visionOnly_'+str(t0)+'t0_'+str(t1)+'t1_'+sim_type +'_testvideo%d_%s' % (testvideo_num, args.dtype)
    elif visionF:
        ftag += str(t0)+'t0_'+str(t1)+'t1_'+sim_type +'_testvideo%d_%s' % (testvideo_num, args.dtype)
    else:
        ftag += 'visionF0_'+str(t0)+'t0_'+str(t1)+'t1_'+sim_type +'_testvideo%d_%s' % (testvideo_num, args.dtype)

    print('ftag %s' % ftag)
    arena_radius = default_params['arena_radius']

    male_motion = np.asarray(male_motion) 
    fale_motion = np.asarray(fale_motion)
    male_body_pos = np.asarray(male_body_pos)
    fale_body_pos = np.asarray(fale_body_pos)

    #np.save('./trx/'+mtype+'_motion_male_'+ftag, simtrx)
    print('./trx/'+mtype+'_trx_'+ftag)
    sio.savemat('./trx/'+mtype+'_trx_'+ftag, simtrx)

    male_pos = np.asarray(male_pos)
    fale_pos = np.asarray(fale_pos)
    if bookkeepingF:
        np.save('./motion/'+vpath+'/'+mtype+'_motion_male_'+ftag, male_motion)
        np.save('./motion/'+vpath+'/'+mtype+'_motion_fale_'+ftag, fale_motion)
        np.save('./motion/'+vpath+'/'+mtype+'_position_male_'+ftag, male_pos)
        np.save('./motion/'+vpath+'/'+mtype+'_position_fale_'+ftag, fale_pos)
        np.save('./motion/'+vpath+'/'+mtype+'_body_position_male_'+ftag, male_body_pos)
        np.save('./motion/'+vpath+'/'+mtype+'_body_position_fale_'+ftag, fale_body_pos)


    male_velocity    = np.asarray(male_velocity) 
    fale_velocity    = np.asarray(fale_velocity)
    male_velocity    = np.sqrt(np.sum(male_velocity**2, axis=1)).flatten() 
    fale_velocity    = np.sqrt(np.sum(fale_velocity**2, axis=1)).flatten() 
    moving_male_ind = (male_velocity > 1.0)
    moving_fale_ind = (fale_velocity > 1.0)
    male_velocity_ns = male_velocity[moving_male_ind]
    fale_velocity_ns = fale_velocity[moving_fale_ind]

    if bookkeepingF:
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


    male_bucket = male_bucket/(t1-t0)
    fale_bucket = fale_bucket/(t1-t0)
    print(max(male_bucket.max(), fale_bucket.max()))

    return male_pos, fale_pos



"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of FlyNetwork collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--btype', type=str, default='perc')
    parser.add_argument('--mtype', type=str, default='conv4_cat50', choices=['lr50', 'conv4_cat50'])
    parser.add_argument('--sim_type', type=str, default='SMSF')
    parser.add_argument('--dtype', type=str, default='gmr')
    parser.add_argument('--videotype', type=str, default='full')
    parser.add_argument('--num_bin', type=int, default=51)
    parser.add_argument('--m_dim', type=int, default=8)
    parser.add_argument('--visionF', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=128)
    parser.add_argument('--n_hid', type=int, default=4)
    parser.add_argument('--t_dim', type=int, default=50)
    parser.add_argument('--x_dim', type=int, default=152)
    parser.add_argument('--batch_sz', type=int, default=32)
    parser.add_argument('--model_epoch', type=int, default=30000)
    parser.add_argument('--vtype', type=str, default='full')
    parser.add_argument('--atype', type=str, default='leakyrelu')
    return check_args(parser.parse_args())


basepath = '/groups/branson/home/imd/Documents/janelia/research/FlyTrajPred/pytorch/'
if __name__ == '__main__':

    args = parse_args()
    args.y_dim = args.num_bin*args.m_dim

    if args.sim_type == 'Single':
        t0,t1=0,1000
        fname='leave_one_out'
        video_list = video16_path[args.dtype]

        visionOnlyF=0

        #for t0, t1 in [(109,271), (409,551), (649, 811), (879,1051), (1179, 1351)]:
        #for t0, t1 in [(269,431), (569,711), (809, 971), (1049,1211), (1349, 1511)]:
        #for t0, t1, fly_k in [(8700,8870,9), (1900,2070,3), (18900, 19070,8), (11900,12070,2), (10000, 10170,0)]:
        for t0, t1, fly_k in [(6400,6570,6), (8900,9070,9), (17900, 18070,3), (29900,30070,1), (18900, 19070,7)]:
            #for testvideo_num, vpath in enumerate(video_list[TEST]):
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

                    #simulated_male_flies = np.arange(0,1,1)
                    simulated_male_flies = np.asarray([fly_k])
                    simulated_female_flies = np.asarray([])# np.arange(10,11,1)

                print('Real Male and Female Flies')
                print(real_male_flies)
                print(real_female_flies)

                print('Simulated Male and Female Flies')
                print(simulated_male_flies)
                print(simulated_female_flies)

                simulate_flies( args, real_male_flies, real_female_flies, \
                            simulated_male_flies, simulated_female_flies,\
                            male_model=None, female_model=None, DEBUG=1,\
                            plottrxlen=100, t0=t0, t1=t1, bookkeepingF=0,\
                            vision_save=False, fname=fname, vpath=vpath,\
                            testvideo_num=testvideo_num, sim_type=args.sim_type,\
                            batch_sz=args.batch_sz, t_dim=args.t_dim, visionOnlyF=visionOnlyF,\
                            fly_single_ind=fly_k)

    elif args.sim_type == 'LONG':
        #t0,t1=0,1000
        t0,t1=0,30320
        fname='allsim'
        visionOnlyF=0
        real_male_flies = []#np.arange(1,10,1)
        real_female_flies = []#np.arange(11,20,1)

        print('Real Male and Female Flies')
        print(real_male_flies)
        print(real_female_flies)

        video_list = video16_path[args.dtype]

        for testvideo_num in range(0,len(video_list[TEST])):
        #for testvideo_num in [len(video_list[TEST])-1]:
        #for testvideo_num in range(0,10):
            #for t0, t1 in [(14900,15080), (16900,17130), (25900, 26300), (23900,24500)]:
            for t0, t1 in [(19900,20080), (28900,29130), (18900, 19300), (22900,23500)]:
                #if testvideo_num == 2:
                #    simulated_male_flies = np.arange(0,10,1)
                #    simulated_female_flies = np.arange(10,19,1)
                print (video_list[TEST][testvideo_num])
                if  testvideo_num == 1 or testvideo_num == 2 :
                    simulated_male_flies = np.arange(0,9,1)
                    simulated_female_flies = np.arange(9,19,1)
                else:
                    simulated_male_flies = np.arange(0,10,1)
                    simulated_female_flies = np.arange(10,20,1)

                print('Simulated Male and Female Flies')
                print(simulated_male_flies)
                print(simulated_female_flies)

                simulate_flies( args, real_male_flies, real_female_flies, \
                                simulated_male_flies, simulated_female_flies,\
                                male_model=None, female_model=None, \
                                plottrxlen=100, t0=t0, t1=t1, bookkeepingF=0,\
                                vision_save=False, fname=fname, \
                                DEBUG=1, testvideo_num=testvideo_num,\
                                visionOnlyF=visionOnlyF, t_dim=args.t_dim, \
                                batch_sz=args.batch_sz, sim_type=args.sim_type)



    elif args.sim_type == 'LOO':
        #t0,t1=0,1000
        t0,t1=0,30000
        fname='leave_one_out'
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
        video_list = video16_path[args.dtype]

        visionOnlyF=0

        for testvideo_num, vpath in enumerate(video_list[TEST]):
        #for testvideo_num in range(0,10):
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

            simulate_flies( args, real_male_flies, real_female_flies, \
                        simulated_male_flies, simulated_female_flies,\
                        male_model=None, female_model=None, DEBUG=0,\
                        plottrxlen=100, t0=t0, t1=t1, bookkeepingF=1,\
                        vision_save=False, fname=fname, vpath=vpath,\
                        testvideo_num=testvideo_num, sim_type=args.sim_type,\
                        batch_sz=args.batch_sz, t_dim=args.t_dim, visionOnlyF=visionOnlyF)

    elif args.sim_type == 'SMSF':
        #t0,t1=0,1000
        t0,t1=0,30320
        fname='allsim'
        visionOnlyF=0
        real_male_flies = []#np.arange(1,10,1)
        real_female_flies = []#np.arange(11,20,1)

        print('Real Male and Female Flies')
        print(real_male_flies)
        print(real_female_flies)

        video_list = video16_path[args.dtype]

        for testvideo_num in range(0,len(video_list[TEST])):
        #for testvideo_num in [len(video_list[TEST])-1]:
        #for testvideo_num in range(0,10):

            #if testvideo_num == 2:
            #    simulated_male_flies = np.arange(0,10,1)
            #    simulated_female_flies = np.arange(10,19,1)
            print (video_list[TEST][testvideo_num])
            if  testvideo_num == 1 or testvideo_num == 2 :
                simulated_male_flies = np.arange(0,9,1)
                simulated_female_flies = np.arange(9,19,1)
            else:
                simulated_male_flies = np.arange(0,10,1)
                simulated_female_flies = np.arange(10,20,1)

            print('Simulated Male and Female Flies')
            print(simulated_male_flies)
            print(simulated_female_flies)

            simulate_flies( args, real_male_flies, real_female_flies, \
                            simulated_male_flies, simulated_female_flies,\
                            male_model=None, female_model=None, \
                            plottrxlen=100, t0=t0, t1=t1, bookkeepingF=1,\
                            vision_save=False, fname=fname, \
                            DEBUG=0, testvideo_num=testvideo_num,\
                            visionOnlyF=visionOnlyF, t_dim=args.t_dim)

    elif args.sim_type == 'RMSF':
        t0,t1=0,1000
        fname='realmale_simfemale'
        real_male_flies = np.arange(0,10,1)
        real_female_flies = []#np.arange(11,20,1)

        simulated_male_flies = []#np.arange(0,10,1)
        simulated_female_flies = np.arange(10,20,1)

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
                        vision_save=False, fname=fname)


    elif args.sim_type == 'SMRF':
        t0,t1=0,1000
        fname='simmale_realfemale'
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
                        vision_save=False, fname=fname)


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
                        vision_save=False, fname=fname)




    #render_flies( model=None, hiddenF=1, plottrxlen=1,\
    #                t0=0, t1=30320, vision_save=False, gender=1)



