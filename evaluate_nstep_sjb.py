import numpy as np
import h5py, copy, os, sys, argparse
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import scipy.io as sio

import torch
from tqdm import tqdm
from util import *
from util_fly import *
from gen_dataset import load_eyrun_data_sjb, combine_vision_data_sjb as combine_vision_data, \
                        video16_path, gender_classify, gender_classify_sjb, \
                        load_vision_sjb
from simulate_rnn_sjb import get_nstep_comparison_rnn, get_simulate_fly
from simulate_autoreg import simulate_next_t_steps

VDATA=0
MDATA=1
BDATA=2
MALE=0
FEMALE=1
SOURCE=0
TARGET=1
NUM_FLY=20
NUM_BIN=51
NUM_VFEAT=144
NUM_MFEAT=8
TRAIN=0
VALID=1
TEST=2


'''
# For temporary debugging, TODO remove
# For real_flies_simulatePlan_RNNs()
vpath=video_list[TEST][testvideo_num]
hiddens_male=None
hiddens_female=None
mtype=args.mtype
monlyF=abs(1-args.visionF)
plottrxlen=100
tsim=args.tsim
t0=0
t1=None
t_dim=args.t_dim
genDataset=False
binwidth=2.0
num_hid=100
model_epoch=200000
btype='perc'
num_bin=args.num_bin
gender=0
use_cuda=args.use_cuda

# For get_simulate_fly()
model=male_model
state=male_state
t=t_j
simulated_flies=simulated_male_flies
thrd=10
hiddens=hiddens_male
t_dim=50
visionF=1
visionOnly=0

fly = simulated_flies[0]
xs,ys,thetas,a_s,bs = x,y,theta,a,b
distF=0
'''

def real_flies_simulatePlan_RNNs(vpath, male_model, female_model,\
                simulated_male_flies, simulated_female_flies,\
                hiddens_male=None, hiddens_female=None, mtype='rnn', \
                monlyF=0, plottrxlen=100, tsim=1, t0=0, t1=None,\
                t_dim=7, genDataset=False, ifold=0, binwidth=2.0,\
                num_hid=100, model_epoch=200000, btype='linear',\
                num_bin=51,gender=0, use_cuda=1):

    print(mtype, monlyF, tsim)
    device = torch.device("cuda" if use_cuda else "cpu")

    DEBUG = 0
    fname = 'eyrun_simulate_data.mat'
    basepath='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'
    matfile = basepath+vpath+fname
    (trx,motiondata,params,basesize) = load_eyrun_data_sjb(matfile, device=device)

    vision_matfile = basepath+vpath+'movie-vision.mat'
    vc_data = load_vision_sjb(vision_matfile, device=device)[1:]


    if 'perc' in btype:
        binedges = np.load('./bins/percentile_%dbins.npy' % num_bin)
        params['binedges'] = torch.tensor(binedges).to(device)
    else:
        binedges = params['binedges']

    male_ind, female_ind = gender_classify_sjb(basesize['majax'])
    params['mtype'] = mtype


    #initial pose 
    print("TSIM: %d" % tsim)
    
    if t1 is None: t1= trx['x'].shape[0] - tsim
    x = trx['x'][t0+t_dim,:].clone()
    y = trx['y'][t0+t_dim,:].clone()
    theta = trx['theta'][t0+t_dim,:].clone()
    
    # even flies are simulated, odd are real
    n_flies = x.shape[0]
    real_flies = torch.arange(0,n_flies,1, device=device)
    simulated_flies = [] 

    b = basesize['minax'].clone()
    a = trx['a'][t0+t_dim,:].clone()
    l_wing_ang = trx['l_wing_ang'][t0+t_dim,:].clone()
    r_wing_ang = trx['r_wing_ang'][t0+t_dim,:].clone()
    l_wing_len = trx['l_wing_len'][t0+t_dim,:].clone()
    r_wing_len = trx['r_wing_len'][t0+t_dim,:].clone()

    xprev = x.clone()
    yprev = y.clone()
    thetaprev = theta.clone()

    # simulated_male_flies = simulated_male_flies[:len(male_ind)]
    #simulated_female_flies = simulated_female_flies[:len(female_ind)]
    simulated_male_flies = torch.arange(len(male_ind), device=device)
    simulated_female_flies = torch.arange(len(male_ind),len(male_ind)+len(female_ind), device=device)


    if 'rnn' in mtype or 'skip' in mtype:
        hiddens_male   = [male_model.initHidden(1, use_cuda=use_cuda) \
                                for i in range(len(simulated_male_flies))]
        hiddens_female = [female_model.initHidden(1, use_cuda=use_cuda)\
                                for i in range(len(simulated_female_flies))]


    simulated_flies = torch.cat([simulated_male_flies, simulated_female_flies])
    NUM_FLY = len(simulated_male_flies) + len(simulated_female_flies)
    print('Number of flies : %d' % NUM_FLY)
    if 'rnn' in mtype or 'skip' in mtype:
        male_state = [None]*(len(male_ind))
        female_state = [None]*(len(female_ind))

    elif 'conv' in mtype:
        state = torch.zeros(( NUM_FLY, NUM_VFEAT+NUM_MFEAT, t_dim ), device=device)

    elif 'lr' in mtype or 'nn' in mtype:
        state = torch.zeros(( NUM_FLY, NUM_MFEAT, t_dim ), device=device)

    feat_motion  = motiondata[:,t0+t_dim,:].clone()
    mymotiondata = torch.zeros(motiondata.shape, device=device)
    predictions_flies, flyvisions = [], []
    vel_errors, pos_errors, theta_errors, wing_ang_errors, wing_len_errors \
                                                = [], [], [], [], []
    acc_rates, loss_rates = [], []
    simtrx_numpys, dataset, dataset_frames = [], [], []

        
    print('Simulation Start %d %d %d...\n' % (t0+t_dim,t1,tsim))


    progress = tqdm(enumerate(range(t0+t_dim,t1,tsim)))
    for ii, t in progress:

        print(ii, t)
        if 'rnn' in mtype or 'skip' in mtype:
            for t_j in range(t-tsim, t):

                for fly_j in range(n_flies):
                    x[fly_j] = trx['x'][t_j,fly_j]
                    y[fly_j] = trx['y'][t_j,fly_j]
                    theta[fly_j] = trx['theta'][t_j,fly_j]
                    a[fly_j] = trx['a'][t_j,fly_j]
                    l_wing_ang[fly_j] = trx['l_wing_ang'][t_j,fly_j]
                    r_wing_ang[fly_j] = trx['r_wing_ang'][t_j,fly_j]
                    l_wing_len[fly_j] = trx['l_wing_len'][t_j,fly_j]
                    r_wing_len[fly_j] = trx['r_wing_len'][t_j,fly_j]

                    feat_motion[:,fly_j] = motiondata[:,t_j,fly_j]

                xprev[:] = x
                yprev[:] = y
                thetaprev[:] = theta

                x, y, theta, a, l_wing_ang, r_wing_ang, \
                    l_wing_len, r_wing_len, \
                    hiddens_male, male_vision_chamber, feat_motion, _ = \
                    get_simulate_fly(male_model, male_state, t_j, trx,\
                                     simulated_male_flies, feat_motion,\
                                        x, y, theta, a, b, \
                                        l_wing_ang, r_wing_ang,\
                                        l_wing_len, r_wing_len,\
                                        xprev, yprev, thetaprev, 
                                        basesize, params, mtype,\
                                        num_bin=num_bin, hiddens=hiddens_male)

                ## Simulate Female Model
                x, y, theta, a, l_wing_ang, r_wing_ang, \
                    l_wing_len, r_wing_len, \
                    hiddens_female, female_vision_chamber, feat_motion, _ =\
                    get_simulate_fly(female_model, female_state, t_j, trx,\
                                     simulated_female_flies, feat_motion,\
                                        x, y, theta, a, b, \
                                        l_wing_ang, r_wing_ang,\
                                        l_wing_len, r_wing_len,\
                                        xprev, yprev, thetaprev,\
                                        basesize, params, mtype,\
                                        num_bin=num_bin, hiddens=hiddens_female)

        for flyi in range(len(real_flies)):

            fly = real_flies[flyi]
            x[fly] = trx['x'][t,fly].clone()
            y[fly] = trx['y'][t,fly].clone()
            theta[fly] = trx['theta'][t,fly].clone()
            a[fly] = trx['a'][t,fly].clone()
            l_wing_ang[fly] = trx['l_wing_ang'][t,fly].clone()
            r_wing_ang[fly] = trx['r_wing_ang'][t,fly].clone()
            l_wing_len[fly] = trx['l_wing_len'][t,fly].clone()
            r_wing_len[fly] = trx['r_wing_len'][t,fly].clone()

            # motiondata[:,t,fly] = corresponds to movement from t-1 to t
            feat_motion[:,fly] = motiondata[:,t,fly]


            if 'conv' in mtype:
                state = torch.zeros(( NUM_FLY, NUM_VFEAT+NUM_MFEAT, t_dim ), device=device)
                state[:,:8,:] = motiondata[:,t-50:t,:].transpose(2,0,1)
                state[:,8:,:] = vc_data[t-50:t,:,:].transpose(1,2,0)

        #Start doing nstep (tsim) predictions at each time step t
        if 'rnn' in mtype or 'skip' in mtype:
            simtrx_curr, feat_motion, predictions, \
                    hiddens_male, hiddens_female, flyvisions\
                                = get_nstep_comparison_rnn(\
                                x, y, theta, a, b, \
                                l_wing_ang, r_wing_ang, \
                                l_wing_len, r_wing_len,\
                                NUM_FLY, trx, \
                                male_model, female_model,\
                                male_state, female_state, \
                                feat_motion, \
                                params, basesize, motiondata,\
                                tsim, t, simulated_flies,\
                                monlyF=monlyF,\
                                mtype=mtype, \
                                male_hiddens=hiddens_male,\
                                female_hiddens=hiddens_female,\
                                male_ind=male_ind,\
                                female_ind=female_ind,
                                num_bin=num_bin)
        else:
            simtrx_curr, feat_motion, predictions, flyvisions\
                                = get_nstep_comparison(\
                                x, y, theta, a, b, \
                                l_wing_ang, r_wing_ang, \
                                l_wing_len, r_wing_len, \
                                NUM_FLY, trx, \
                                male_model, female_model, \
                                male_ind, female_ind,\
                                state, feat_motion, \
                                params, basesize, motiondata,\
                                tsim, t, simulated_flies, \
                                monlyF=monlyF, mtype=mtype, \
                                t_dim=t_dim, num_bin=num_bin)
        
        if genDataset:
            flyvisions = torch.stack(flyvisions, 0)
            data = combine_vision_data(simtrx_curr, flyvisions, num_fly=NUM_FLY, num_burn=2)
            dataset.append(data)
            dataset_frames.append(t)
            
        simtrx_numpy = simtrx2numpy(simtrx_curr)
        simtrx_numpys.append(simtrx_numpy)
        if 1:
            vel_error, pos_error, theta_error, wing_ang_error, wing_len_error = [], [], [], [], []
            for tt in range(1,tsim):#[1,3,5,10,15]:
                results = get_error_sjb(simtrx_curr, trx, t, tt)
                vel_error.append(results[2])
                pos_error.append(results[3])
                theta_error.append(results[4])
                wing_ang_error.append(results[5])
                wing_len_error.append(results[6])

            if 0:
                loss, acc_rate = get_loss_change_motion(predictions, \
                                                        motiondata, t,\
                                                        gender)
                acc_rates.append(acc_rate)
                loss_rates.append(loss)
                progress.set_description('Accuracy : %f, Loss %f' % (acc_rate, loss))

            vel_error = np.asarray(vel_error)
            pos_error = np.asarray(pos_error)
            theta_error = np.asarray(theta_error)
            wing_ang_error = np.asarray(wing_ang_error)
            wing_len_error = np.asarray(wing_len_error)

            vel_errors.append(vel_error)
            pos_errors.append(pos_error)
            theta_errors.append(theta_error)
            wing_ang_errors.append(wing_ang_error)
            wing_len_errors.append(wing_len_error)

            progress.set_description(('%d VEL MSE: %f POSITION MSE : %f THETA MSE %f' \
                    + 'WING ANG MSE %f WING LEN MSE %f')
                    % (t, np.nanmean(vel_error[-1]), np.nanmean(pos_error[-1]),\
                    np.nanmean(theta_error[-1]), \
                    np.nanmean(wing_ang_error[-1]), \
                    np.nanmean(wing_len_error[-1])))




    if 'rnn' in mtype or 'skip' in mtype:
        os.makedirs('./simtrx/%s/' % (vpath), exist_ok=True)   
        os.makedirs('./simtrx/%s/%s' % (vpath, mtype), exist_ok=True)   
        np.save('./simtrx/'+vpath+'/'+mtype+'/'+mtype+'_gender'\
            +str(gender)+'_'+str(num_hid)+'hid_'+str(t0)+'t0_'\
            +str(t1)+'t1_%dtsim_%s_%depoch' % (tsim, btype, model_epoch) + str(ifold), \
            np.asarray(simtrx_numpys))

    elif 'lr' in mtype:
        os.makedirs('./simtrx/%s/' % (vpath), exist_ok=True)   
        os.makedirs('./simtrx/%s/%s' % (vpath, mtype), exist_ok=True)   
        np.save('./simtrx/'+vpath+'/'+mtype+'/'+mtype+'_gender'\
            +str(gender)+'_'+str(t0)+'t0_'+str(t1)\
            +'t1_%dtsim' % tsim + str(ifold), \
            np.asarray(simtrx_numpys))



    if genDataset:
        os.makedirs('./fakedata/%s/' % (vpath), exist_ok=True)   
        os.makedirs('./fakedata/%s/%s' % (vpath, mtype), exist_ok=True)   
        if 'lr' in mtype:
            ffname = './fakedata/'+vpath+'/'+mtype+'/'+mtype+'_gender'\
                +str(gender)+'_'+str(t0)\
                +'t0_'+str(t1)+'t1_%dtsim' % tsim
            np.save(ffname, np.asarray(dataset))
            print('Data Generated Path: %s' % ffname)
        else:
            np.save('./fakedata/'+vpath+'/'+mtype+'/'+mtype+'_gender'\
                +str(gender)+'_'+str(num_hid)+'hid_'+str(t0)\
                +'t0_'+str(t1)+'t1_%dtsim_%s_%depoch' % (tsim, btype, model_epoch), \
                np.asarray(dataset))
        np.save('./fakedata/'+vpath+'/frame_index_'\
                +str(t0) +'t0_'+str(t1)+'t1_%dtsim' % (tsim), \
                np.asarray(dataset_frames))


    visionF = 1-int(monlyF)
    results = np.stack([vel_errors, pos_errors, theta_errors, wing_ang_errors, wing_len_errors])
    os.makedirs('%s/metrics/%s/' % (args.basepath, vpath), exist_ok=True)   
    os.makedirs('%s/metrics/%s/%s' % (args.basepath, vpath, mtype), exist_ok=True)   
    if 'rnn' in mtype or 'skip' in mtype:
        fname=args.basepath+'/metrics/'+vpath+'/'+mtype+'/'+mtype+'_'+str(t0)+'t0_'+str(t1)+'t1_%dtsim_%s_%depoch_%dfold' % (tsim, btype, model_epoch, ifold)
    else:
        fname=args.basepath+'/metrics/'+vpath+'/'+mtype+'/'+mtype+'_visionF'+str(visionF)+'_'+str(t0)+'t0_'+str(t1)+'t1_%dtsim_%depoch_%dfold' % (tsim, model_epoch, ifold)
    print(fname)
    np.save(fname, np.asarray(results))

    print('Final Velocity Error %f' % (np.nanmean(vel_errors)))
    print('Final Position Error %f' % (np.nanmean(pos_errors)))
    print('Final Theta Error %f' % (np.nanmean(theta_errors)))
    print('Final Wing Ang Error %f' % (np.nanmean(wing_ang_errors)))
    print('Final Wing Len Error %f' % (np.nanmean(wing_len_errors)))

    return simtrx_curr


def get_nstep_comparison(x, y, theta, a, b, \
                    l_wing_ang, r_wing_ang, l_wing_len, r_wing_len,\
                    n_flies, trx, male_model, female_model,\
                    male_ind, female_ind, state, \
                    feat_motion, params, basesize, motiondata,\
                    tsim, tplot, simulated_flies, \
                    monlyF=0, mtype='rnn', t_dim=7, num_bin=51):

    # initial position
    curpos = {}
    curpos['x']=x.copy()
    curpos['y']=y.copy()
    curpos['theta']=theta.copy()
    curpos['a']=a.copy()
    curpos['b'] = basesize['minax'].copy()
    curpos['l_wing_ang']=l_wing_ang.copy()
    curpos['r_wing_ang']=r_wing_ang.copy()
    curpos['l_wing_len']=l_wing_len.copy()
    curpos['r_wing_len']=r_wing_len.copy()
    
    simtrx_curr, feat_motion, predictions, flyvisions\
                            = simulate_next_t_steps(tplot, curpos,\
                                state, feat_motion,\
                                male_model, female_model,\
                                male_ind, female_ind,\
                                params,basesize, motiondata,\
                                simulated_flies,\
                                tsim=tsim, monlyF=monlyF,\
                                mtype=mtype, t_dim=t_dim,\
                                num_bin=num_bin)

    return simtrx_curr, feat_motion, predictions, flyvisions


def simtrx2numpy(simtrx):

    numpy_trx = []
    x_data = simtrx['x']
    y_data = simtrx['y']
    theta_data = simtrx['theta']
    a_data = simtrx['a']
    l_wing_ang_data= simtrx['l_wing_ang']
    r_wing_ang_data= simtrx['r_wing_ang']
    l_wing_len_data = simtrx['l_wing_len']
    r_wing_len_data = simtrx['r_wing_len']

    numpy_trx = [x_data, y_data, theta_data, a_data, \
                    l_wing_ang_data, r_wing_ang_data,\
                    l_wing_len_data, r_wing_len_data]

    return np.asarray(numpy_trx)

def get_error(model_trx, data_trx, time, nstep):

    assert nstep>0, 'Nstep should be greater than 0'
    x_data = data_trx['x'][time+nstep,:]
    y_data = data_trx['y'][time+nstep,:]
    theta_data = data_trx['theta'][time+nstep,:]
    l_wing_ang_data= data_trx['l_wing_ang'][time+nstep,:]
    r_wing_ang_data= data_trx['r_wing_ang'][time+nstep,:]
    l_wing_len_data = data_trx['l_wing_len'][time+nstep,:]
    r_wing_len_data = data_trx['r_wing_len'][time+nstep,:]

    x_model = model_trx['x'][nstep,:]
    y_model = model_trx['y'][nstep,:]
    theta_model = model_trx['theta'][nstep,:]
    l_wing_ang_model = model_trx['l_wing_ang'][nstep,:]
    r_wing_ang_model = model_trx['r_wing_ang'][nstep,:]
    l_wing_len_model = model_trx['l_wing_len'][nstep,:]
    r_wing_len_model = model_trx['r_wing_len'][nstep,:]

    x_error = (x_data - x_model)**2
    y_error = (y_data - y_model)**2
    position_error = np.sqrt(x_error + y_error)

    vel_x_data  = data_trx['x'][time+nstep,:] - data_trx['x'][time+nstep-1,:]
    vel_y_data  = data_trx['y'][time+nstep,:] - data_trx['y'][time+nstep-1,:]
    vel_x_model = model_trx['x'][nstep,:] - model_trx['x'][nstep-1,:]
    vel_y_model = model_trx['y'][nstep,:] - model_trx['y'][nstep-1,:]
    vel_x_error = (vel_x_data - vel_x_model)**2
    vel_y_error = (vel_y_data - vel_y_model)**2
    velocity_error = np.sqrt(vel_x_error + vel_y_error)

    theta_diff = abs(theta_data - theta_model)
    theta_error = (theta_diff > np.pi) * (2*np.pi-theta_diff)\
                        + (theta_diff <= np.pi) * theta_diff

    l_wing_len_error = abs(l_wing_len_data - l_wing_len_model)
    r_wing_len_error = abs(r_wing_len_data - r_wing_len_model)


    l_wing_ang_diff = abs(l_wing_ang_data - l_wing_ang_model)
    l_wing_ang_error = (l_wing_ang_diff > np.pi) * (2*np.pi-l_wing_ang_diff)\
                        + (l_wing_ang_diff <= np.pi) * l_wing_ang_diff

    r_wing_ang_diff = abs(r_wing_ang_data - r_wing_ang_model)
    r_wing_ang_error = (r_wing_ang_diff > np.pi) * (2*np.pi-r_wing_ang_diff)\
                        + (r_wing_ang_diff <= np.pi) * r_wing_ang_diff

    wing_ang_error = (l_wing_ang_error + r_wing_ang_error) * 0.5
    wing_len_error = (l_wing_len_error + r_wing_len_error) * 0.5

    return np.sqrt(x_error), np.sqrt(y_error), \
            velocity_error, position_error, \
                theta_error, wing_ang_error, wing_len_error 


def get_error_sjb(model_trx, data_trx, time, nstep):

    assert nstep>0, 'Nstep should be greater than 0'
    x_data = data_trx['x'][time+nstep,:]
    y_data = data_trx['y'][time+nstep,:]
    theta_data = data_trx['theta'][time+nstep,:]
    l_wing_ang_data= data_trx['l_wing_ang'][time+nstep,:]
    r_wing_ang_data= data_trx['r_wing_ang'][time+nstep,:]
    l_wing_len_data = data_trx['l_wing_len'][time+nstep,:]
    r_wing_len_data = data_trx['r_wing_len'][time+nstep,:]

    x_model = model_trx['x'][nstep,:]
    y_model = model_trx['y'][nstep,:]
    theta_model = model_trx['theta'][nstep,:]
    l_wing_ang_model = model_trx['l_wing_ang'][nstep,:]
    r_wing_ang_model = model_trx['r_wing_ang'][nstep,:]
    l_wing_len_model = model_trx['l_wing_len'][nstep,:]
    r_wing_len_model = model_trx['r_wing_len'][nstep,:]

    x_error = (x_data - x_model)**2
    y_error = (y_data - y_model)**2
    position_error = torch.sqrt(x_error + y_error)

    vel_x_data  = data_trx['x'][time+nstep,:] - data_trx['x'][time+nstep-1,:]
    vel_y_data  = data_trx['y'][time+nstep,:] - data_trx['y'][time+nstep-1,:]
    vel_x_model = model_trx['x'][nstep,:] - model_trx['x'][nstep-1,:]
    vel_y_model = model_trx['y'][nstep,:] - model_trx['y'][nstep-1,:]
    vel_x_error = (vel_x_data - vel_x_model)**2
    vel_y_error = (vel_y_data - vel_y_model)**2
    velocity_error = torch.sqrt(vel_x_error + vel_y_error)

    theta_diff = torch.abs(theta_data - theta_model)
    theta_error = (theta_diff > np.pi) * (2*np.pi-theta_diff)\
                        + (theta_diff <= np.pi) * theta_diff

    l_wing_len_error = torch.abs(l_wing_len_data - l_wing_len_model)
    r_wing_len_error = torch.abs(r_wing_len_data - r_wing_len_model)


    l_wing_ang_diff = torch.abs(l_wing_ang_data - l_wing_ang_model)
    l_wing_ang_error = (l_wing_ang_diff > np.pi) * (2*np.pi-l_wing_ang_diff)\
                        + (l_wing_ang_diff <= np.pi) * l_wing_ang_diff

    r_wing_ang_diff = torch.abs(r_wing_ang_data - r_wing_ang_model)
    r_wing_ang_error = (r_wing_ang_diff > np.pi) * (2*np.pi-r_wing_ang_diff)\
                        + (r_wing_ang_diff <= np.pi) * r_wing_ang_diff

    wing_ang_error = (l_wing_ang_error + r_wing_ang_error) * 0.5
    wing_len_error = (l_wing_len_error + r_wing_len_error) * 0.5

    return torch.sqrt(x_error).cpu().numpy(), torch.sqrt(y_error).cpu().numpy(), \
            velocity_error.cpu().numpy(), position_error.cpu().numpy(), \
                theta_error.cpu().numpy(), wing_ang_error.cpu().numpy(), wing_len_error.cpu().numpy()



def baseline0_zero_nstep_prediction(vpath, gender=1, t_dim=50,\
                                        t0=0, t1=None, tsim=15):

    mtype='zero'
    DEBUG = 0
    plottrxlen = 100

    fname = 'eyrun_simulate_data.mat'
    basepath='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'
    matfile = basepath+vpath+fname
    (trx,motiondata,params,basesize) = load_eyrun_data(matfile)    
    binedges = params['binedges']
    params['mtype'] = 'rnn'

    # initial pose 
    print("TSIM: %d" % tsim)

    if t1 is None: t1= trx['x'].shape[0] - tsim
    x = trx['x'][t0+t_dim,:].copy()
    y = trx['y'][t0+t_dim,:].copy()
    theta = trx['theta'][t0+t_dim,:].copy()
    
    # even flies are simulated, odd are real
    n_flies = len(x)
    real_flies = np.arange(0,n_flies,1)
    simulated_flies = [] 

    b = basesize['minax'].copy()
    a = trx['a'][t0,:].copy()
    l_wing_ang = trx['l_wing_ang'][t0+t_dim,:].copy()
    r_wing_ang = trx['r_wing_ang'][t0+t_dim,:].copy()
    l_wing_len = trx['l_wing_len'][t0+t_dim,:].copy()
    r_wing_len = trx['r_wing_len'][t0+t_dim,:].copy()

    xprev = x.copy()
    yprev = y.copy()
    thetaprev = theta.copy()

    state = [None]*max(len(simulated_flies), len(real_flies))

    feat_motion  = motiondata[:,t0+t_dim,:].copy()
    mymotiondata = np.zeros(motiondata.shape)

    vel_errors, pos_errors, theta_errors, wing_ang_errors, wing_len_errors \
                                                = [], [], [], [], []
    errors, acc_rates, loss_rates = [], [], []
    print('Simulation Start...\n')
    progress = tqdm(enumerate(range(t0+t_dim,t1,tsim)))
    for ii, t in progress:

        if ii >= 50:
            simtrx = {}
            simtrx['x']=np.tile(trx['x'][t-1,:], [tsim,1])
            simtrx['y']=np.tile(trx['y'][t-1,:], [tsim,1])
            simtrx['theta']=np.tile(trx['theta'][t-1,:], [tsim,1])
            simtrx['l_wing_ang']=np.tile(trx['l_wing_ang'][t-1,:], [tsim,1])
            simtrx['r_wing_ang']=np.tile(trx['r_wing_ang'][t-1,:], [tsim,1])
            simtrx['l_wing_len']=np.tile(trx['l_wing_len'][t-1,:], [tsim,1])
            simtrx['r_wing_len']=np.tile(trx['r_wing_len'][t-1,:], [tsim,1])


            vel_error, pos_error, theta_error, \
                    wing_ang_error, wing_len_error = [], [], [], [], []
            for tt in range(1,tsim):#[1,3,5,10,15]:
                results = get_error(simtrx, trx, t, tt)
                vel_error.append(results[2])
                pos_error.append(results[3])
                theta_error.append(results[4])
                wing_ang_error.append(results[5])
                wing_len_error.append(results[6])

            vel_error = np.asarray(vel_error)
            pos_error = np.asarray(pos_error)
            theta_error = np.asarray(theta_error)
            wing_ang_error = np.asarray(wing_ang_error)
            wing_len_error = np.asarray(wing_len_error)

            vel_errors.append(vel_error)
            pos_errors.append(pos_error)
            theta_errors.append(theta_error)
            wing_ang_errors.append(wing_ang_error)
            wing_len_errors.append(wing_len_error)

            progress.set_description(('VEL MSE: %f POSITION MSE : %f THETA MSE %f' \
                    + 'WING ANG MSE %f WING LEN MSE %f')
                    % (np.mean(vel_error[0]), \
                       np.mean(pos_error[0]), np.mean(theta_error[0]),\
                       np.nanmean(wing_ang_error[0]), \
                       np.nanmean(wing_len_error[0])))

    results = np.stack([vel_errors, pos_errors, theta_errors, wing_ang_errors, wing_len_errors])
    os.makedirs('./metrics/%s/' % (vpath), exist_ok=True)   
    os.makedirs('./metrics/%s/%s' % (vpath, mtype), exist_ok=True)   
    fname='./metrics/'+vpath+'/'+mtype+'/'+mtype+'_visionF1_'+str(t0)+'t0_'+str(t1)+'t1_%dtsim' % tsim
    print(fname)
    np.save(fname, np.asarray(results))
    print('Final Velocity Error %f' % (np.nanmean(vel_errors)))
    print('Final Position Error %f' % (np.nanmean(pos_errors)))
    print('Final Theta Error %f' % (np.nanmean(theta_errors)))
    print('Final Wing Ang Error %f' % (np.nanmean(wing_ang_errors)))
    print('Final Wing Len Error %f' % (np.nanmean(wing_len_errors)))


def baseline0_constant_nstep_prediction(vpath, gender=1, t_dim=50,\
                                        t0=0, t1=None, tsim=15):

    mtype='const'
    DEBUG = 0
    plottrxlen = 100

    fname = 'eyrun_simulate_data.mat'
    basepath='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'
    matfile = basepath+vpath+fname
    (trx,motiondata,params,basesize) = load_eyrun_data(matfile)    
    binedges = params['binedges']
    params['mtype'] = 'rnn'

    # initial pose 
    print("TSIM: %d" % tsim)

    if t1 is None: t1= trx['x'].shape[0] - tsim
    x = trx['x'][t0+t_dim,:].copy()
    y = trx['y'][t0+t_dim,:].copy()
    theta = trx['theta'][t0+t_dim,:].copy()
    
    # even flies are simulated, odd are real
    n_flies = len(x)
    real_flies = np.arange(0,n_flies,1)
    simulated_flies = [] 

    b = basesize['minax'].copy()
    a = trx['a'][t0,:].copy()
    l_wing_ang = trx['l_wing_ang'][t0+t_dim,:].copy()
    r_wing_ang = trx['r_wing_ang'][t0+t_dim,:].copy()
    l_wing_len = trx['l_wing_len'][t0+t_dim,:].copy()
    r_wing_len = trx['r_wing_len'][t0+t_dim,:].copy()

    xprev = x.copy()
    yprev = y.copy()
    thetaprev = theta.copy()

    state = [None]*max(len(simulated_flies), len(real_flies))

    feat_motion  = motiondata[:,t0+t_dim,:].copy()
    mymotiondata = np.zeros(motiondata.shape)

    vel_errors, pos_errors, theta_errors, wing_ang_errors, wing_len_errors \
                                                = [], [], [], [], []
    errors, acc_rates, loss_rates = [], [], []
    print('Simulation Start...\n')
    progress = tqdm(enumerate(range(t0+t_dim,t1,tsim)))
    for ii, t in progress:

        if ii >= 50:
            simtrx = {}
            vel_x = (trx['x'][t-1,:] - trx['x'][t-2,:])
            vel_y = (trx['y'][t-1,:] - trx['y'][t-2,:])
            vel_t = (trx['theta'][t-1,:] - trx['theta'][t-2,:])
            simtrx['x']=np.tile(trx['x'][t,:], [tsim,1]) + vel_x * np.tile(np.arange(1,tsim+1), [n_flies,1]).T
            simtrx['y']=np.tile(trx['y'][t,:], [tsim,1]) + vel_y * np.tile(np.arange(1,tsim+1), [n_flies,1]).T
            simtrx['theta']=np.tile(trx['theta'][t,:]+vel_t, [tsim,1]) * np.tile(np.arange(1,tsim+1), [n_flies,1]).T
            simtrx['l_wing_ang']=np.tile(trx['l_wing_ang'][t,:], [tsim,1])
            simtrx['r_wing_ang']=np.tile(trx['r_wing_ang'][t,:], [tsim,1])
            simtrx['l_wing_len']=np.tile(trx['l_wing_len'][t,:], [tsim,1])
            simtrx['r_wing_len']=np.tile(trx['r_wing_len'][t,:], [tsim,1])

            vel_error, pos_error, theta_error, \
                    wing_ang_error, wing_len_error = [], [], [], [], []
            for tt in range(1,tsim):#[1,3,5,10,15]:
                results = get_error(simtrx, trx, t, tt)
                vel_error.append(results[2])
                pos_error.append(results[3])
                theta_error.append(results[4])
                wing_ang_error.append(results[5])
                wing_len_error.append(results[6])


            vel_error = np.asarray(vel_error)
            pos_error = np.asarray(pos_error)
            theta_error = np.asarray(theta_error)
            wing_ang_error = np.asarray(wing_ang_error)
            wing_len_error = np.asarray(wing_len_error)

            vel_errors.append(vel_error)
            pos_errors.append(pos_error)
            theta_errors.append(theta_error)
            wing_ang_errors.append(wing_ang_error)
            wing_len_errors.append(wing_len_error)

            progress.set_description(('VEL MSE: %f POSITION MSE : %f THETA MSE %f' \
                    + 'WING ANG MSE %f WING LEN MSE %f')
                    % (np.mean(vel_error[0]), \
                       np.mean(pos_error[0]), np.mean(theta_error[0]),\
                       np.nanmean(wing_ang_error[0]), \
                       np.nanmean(wing_len_error[0])))

    results = np.stack([vel_errors, pos_errors, theta_errors, wing_ang_errors, wing_len_errors])
    os.makedirs('%s/metrics/%s/' % (args.basepath, vpath), exist_ok=True)   
    os.makedirs('%s/metrics/%s/%s' % (args.basepath, vpath, mtype), exist_ok=True)   
    fname=args.basepath+'/metrics/'+vpath+'/'+mtype+'/'+mtype+'_visionF1_'+str(t0)+'t0_'+str(t1)+'t1_%dtsim' % tsim
    print(fname)
    np.save(fname, np.asarray(results))

    print('Final Velocity Error %f' % (np.nanmean(vel_errors)))
    print('Final Position Error %f' % (np.nanmean(pos_errors)))
    print('Final Theta Error %f' % (np.nanmean(theta_errors)))
    print('Final Wing Ang Error %f' % (np.nanmean(wing_ang_errors)))
    print('Final Wing Len Error %f' % (np.nanmean(wing_len_errors)))


def baseline1_constVel_nstep_prediction(vpath, gender=1, t0=0, t1=None, tsim=15):


    DEBUG = 0
    plottrxlen = 100

    fname = 'eyrun_simulate_data.mat'
    basepath='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'
    matfile = basepath+vpath+fname
    (trx,motiondata,params,basesize) = load_eyrun_data(matfile)    
    binedges = params['binedges']
    params['mtype'] = 'rnn'

    # initial pose 
    print("TSIM: %d" % tsim)

    if t1 is None: t1= trx['x'].shape[0] - tsim
    x = trx['x'][t0+t_dim,:].copy()
    y = trx['y'][t0+t_dim,:].copy()
    theta = trx['theta'][t0+t_dim,:].copy()
    
    # even flies are simulated, odd are real
    n_flies = len(x)
    real_flies = np.arange(0,n_flies,1)
    simulated_flies = [] 

    b = basesize['minax'].copy()
    a = trx['a'][t0,:].copy()
    l_wing_ang = trx['l_wing_ang'][t0+t_dim,:].copy()
    r_wing_ang = trx['r_wing_ang'][t0+t_dim,:].copy()
    l_wing_len = trx['l_wing_len'][t0+t_dim,:].copy()
    r_wing_len = trx['r_wing_len'][t0+t_dim,:].copy()

    xprev = x.copy()
    yprev = y.copy()
    thetaprev = theta.copy()

    state = [None]*max(len(simulated_flies), len(real_flies))

    feat_motion  = motiondata[:,t0+t_dim,:].copy()
    mymotiondata = np.zeros(motiondata.shape)
    vel_errors, pos_errors, theta_errors, wing_ang_errors, wing_len_errors \
                                                = [], [], [], [], []
    errors, acc_rates, loss_rates = [], [], []
    print('Simulation Start...\n')
    #progress = tqdm(enumerate(range(t0+t_dim,t1)))
    progress = tqdm(enumerate(range(t0+t_dim,t1,tsim)))
    for ii, t in progress:

        if ii>=50:
            simtrx = {}
            simtrx['x'] = ((trx['x'][t,:] - trx['x'][t-1,:])[:,None] \
                            * np.tile(np.arange(tsim), [n_flies,1])).T\
                            + np.tile(trx['x'][t,:], [tsim,1])
            simtrx['y'] = ((trx['y'][t,:] - trx['y'][t-1,:])[:,None] \
                            * np.tile(np.arange(tsim), [n_flies,1])).T\
                            + np.tile(trx['y'][t,:], [tsim,1])
            simtrx['theta'] = ((trx['theta'][t,:] - trx['theta'][t-1,:])[:,None] \
                            * np.tile(np.arange(tsim), [n_flies,1])).T\
                            + np.tile(trx['theta'][t,:], [tsim,1])
            simtrx['l_wing_ang'] = ((trx['l_wing_ang'][t,:] \
                            - trx['l_wing_ang'][t-1,:])[:,None] \
                            * np.tile(np.arange(tsim), [n_flies,1])).T\
                            + np.tile(trx['l_wing_ang'][t,:], [tsim,1])
            simtrx['r_wing_ang'] = ((trx['r_wing_ang'][t,:] \
                            - trx['r_wing_ang'][t-1,:])[:,None] \
                            * np.tile(np.arange(tsim), [n_flies,1])).T\
                            + np.tile(trx['r_wing_ang'][t,:], [tsim,1])
            simtrx['l_wing_len'] = ((trx['l_wing_len'][t,:] \
                            - trx['l_wing_len'][t-1,:])[:,None] \
                            * np.tile(np.arange(tsim), [n_flies,1])).T\
                            + np.tile(trx['l_wing_len'][t,:], [tsim,1])
            simtrx['r_wing_len'] = ((trx['r_wing_len'][t,:] \
                            - trx['r_wing_len'][t-1,:])[:,None] \
                            * np.tile(np.arange(tsim), [n_flies,1])).T\
                            + np.tile(trx['r_wing_len'][t,:], [tsim,1])

            vel_error, pos_error, theta_error, \
                    wing_ang_error, wing_len_error = [], [], [], [], []
            for tt in range(1,tsim):#[1,3,5,10,15]:
 
                results = get_error(simtrx, trx, t, tt)
                vel_error.append(results[2])
                pos_error.append(results[3])
                theta_error.append(results[4])
                wing_ang_error.append(results[5])
                wing_len_error.append(results[6])

            vel_error = np.asarray(vel_error)
            pos_error = np.asarray(pos_error)
            theta_error = np.asarray(theta_error) 
            wing_ang_error = np.asarray(wing_ang_error)
            wing_len_error = np.asarray(wing_len_error)

            vel_errors.append(vel_error)
            pos_errors.append(pos_error)
            theta_errors.append(theta_error)
            wing_ang_errors.append(wing_ang_error)
            wing_len_errors.append(wing_len_error)

            progress.set_description(('VEL MSE: %f  POSITION MSE : %f THETA MSE %f' \
                    + 'WING ANG MSE %f WING LEN MSE %f')
                    % (np.mean(vel_error[0]), np.mean(pos_error[0]), np.mean(theta_error[0]),\
                       np.nanmean(wing_ang_error[0]), np.nanmean(wing_len_error[0])))


    results = np.stack([vel_errors, pos_errors, theta_errors, wing_ang_errors, wing_len_errors])
    os.makedirs('./metrics/%s/' % (vpath), exist_ok=True)   
    os.makedirs('./metrics/%s/%s' % (vpath, mtype), exist_ok=True)   
    fname='./metrics/'+vpath+'/'+mtype+'/'+mtype+'_visionF1_'+str(t0)+'t0_'+str(t1)+'t1_%dtsim' % tsim
    print(fname)
    np.save(fname, np.asarray(results))

    print('Final Velocity Error %f' % (np.nanmean(vel_errors)))
    print('Final Position Error %f' % (np.nanmean(pos_errors)))
    print('Final Theta Error %f' % (np.nanmean(theta_errors)))
    print('Final Wing Ang Error %f' % (np.nanmean(wing_ang_errors)))
    print('Final Wing Len Error %f' % (np.nanmean(wing_len_errors)))
    #print('Accuracy %f Loss %f' % (np.mean(acc_rates), np.mean(loss_rates)))
    # end loop over frames
    pass



def plot_nstep_errors(models, dtype, video_list, t_dim=50, gender=0, t0=21224, tsim=15, \
        visionF=1, labels=None, vlowmax=None,
        colors=['blue','red','green', 'magenta', 'purple', 'black']):

    if labels is None: labels = models
    fname0 = 'eyrun_simulate_data.mat'
    basepath='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'
    parentpath='/groups/branson/home/imd/Documents/janelia/research/fly_behaviour_sim/71g01'

    
    ftag = ['velo', 'pos', 'bodyAng', 'wingang']
    for j in range(len(ftag)):

        pos_errs = []
        pos_stds = []

        #for testvideo_num, vpath in enumerate(video_list[TEST]):
        for testvideo_num in range(0,len(video_list[TEST])):
            vpath = video_list[TEST][testvideo_num]
            print ('testvideo %d %s' % (testvideo_num, video_list[TEST][testvideo_num]))
            matfile = basepath+vpath+fname0
            (trx,motiondata,params,basesize) = load_eyrun_data(matfile)
            t1= trx['x'].shape[0] - tsim
            male_ind, female_ind = gender_classify(basesize['majax'])        

            pos_err_models = []
            pos_std_models = []

            for mtype0 in models:
                if 'const' == mtype0 or 'copy' == mtype0 or 'zero' == mtype0:
                    fname = args.basepath+'/metrics/'+vpath+'/'+mtype0+'/'+mtype0+'_visionF'+str(visionF)+'_'+str(t0)+'t0_'+str(t1)+'t1_30tsim.npy'
                    err_test = np.load(fname)
                else:

                    err_tests = []
                    for kk in range(10):
                        if 'rnn' in mtype0:
                            fname = args.basepath+'/metrics/'+vpath+'/'+mtype0+'/'+mtype0+'_'+str(t0)+'t0_'+str(t1)+'t1_30tsim_150000epoch'
                        elif 'skip' in mtype0:
                            fname = args.basepath+'/metrics/'+vpath+'/'+mtype0+'/'+mtype0+'_'+str(t0)+'t0_'+str(t1)+'t1_30tsim_100000epoch'
                        else:
                            fname = args.basepath+'/metrics/'+vpath+'/'+mtype0+'/'+mtype0+'_visionF'+str(visionF)+'_'+str(t0)+'t0_'+str(t1)+'t1_30tsim'
                            if ('pdb' in dtype or 'gmr91' in dtype) and 'lr' in mtype0: fname += '_200000epoch'
 
                        err_test0 = np.load(fname + '_%dfold.npy' % kk)
                        #if dtype == 'gmr91': err_test0 = err_test0[:,:,:,1:]
                        err_tests.append(err_test0)

                    err_test0 = np.min(err_tests, axis=0)
                    err_test = err_test0[:]
         
                print(fname)
                pos_err_tests = []
                pos_std_tests = []
                for i in range(1, tsim-1):
                    
                    if gender == 0:
                        pos_err_test  = np.nanmean(err_test[j,:,i,male_ind])
                        pos_std_test  = np.nanstd( err_test[j,:,i,male_ind])
                    else:
                        pos_err_test  = np.nanmean(err_test[j,:,i,female_ind])
                        pos_std_test  = np.nanstd( err_test[j,:,i,female_ind])

                    pos_err_tests.append(pos_err_test)
                    pos_std_tests.append(pos_std_test)

                pos_err_models.append(pos_err_tests)
                pos_std_models.append(pos_std_tests)

            pos_errs.append(pos_err_models)
            pos_stds.append(pos_std_models)

        pos_err_models = np.nanmean(pos_errs, axis=0)
        pos_std_models = np.nanmean(pos_stds, axis=0)
        xx = np.arange(2,tsim) 
        plt.figure()
        ax = plt.axes([0,0,1,1])
        for i,pos_err_test in enumerate(pos_err_models):
            plt.errorbar(xx, pos_err_test, ls='-', color=colors[i], label=labels[i], lw=3, alpha=0.8)

        if 'body' in ftag[j]:
            plt.ylim([0, pos_err_models[1:].max()])

        plt.xlabel('N-steps')
        plt.ylabel('Error rate')
        SMALL_SIZE = 22
        matplotlib.rc('font', size=SMALL_SIZE)
        matplotlib.rc('axes', titlesize=SMALL_SIZE)

        ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5,1.25), ncol=3)

        os.makedirs('./figs/nstep/%s/' % dtype, exist_ok=True)   
        plt.savefig('./figs/nstep/%s/eval_%dsteps_%s_gender%d_%s.pdf' \
            % (dtype, tsim, ftag[j], gender, dtype), format='pdf', bbox_inches='tight') 
        
    return 


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of FlyNetwork collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--btype', type=str, default='perc')
    parser.add_argument('--mtype', type=str, default='rnn50')
    parser.add_argument('--sim_type', type=str, default='SMSF')
    parser.add_argument('--dtype', type=str, default='gmr')
    parser.add_argument('--videotype', type=str, default='full')
    parser.add_argument('--num_bin', type=int, default=101)
    parser.add_argument('--num_mfeat', type=int, default=8)
    parser.add_argument('--visionF', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=100)
    parser.add_argument('--t_dim', type=int, default=50)
    parser.add_argument('--tsim', type=int, default=30)
    parser.add_argument('--x_dim', type=int, default=152)
    parser.add_argument('--gender', type=int, default=1)
    parser.add_argument('--plotF', type=int, default=0)
    parser.add_argument('--basepath', type=str, default='/groups/branson/home/imd/Documents/janelia/research/fly_behaviour_sim/71g01/')
    parser.add_argument('--datapath', type=str, default='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/')
    parser.add_argument('--save_path_male', type=str, default=None)
    parser.add_argument('--save_path_female', type=str, default=None)
    parser.add_argument('--use_cuda', type=int, default=1)

    return check_args(parser.parse_args())


if True:
    fname = 'eyrun_simulate_data.mat'

    args = parse_args()
    args.dtype = 'gmr'
    args.mtype = 'rnn50'
    args.tsim = 30
    args.basepath = './'
    args.save_path_male = './models/gmr/flyNet_gru50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_maleflies_10000'
    args.save_path_female = './models/gmr/flyNet_gru50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_femaleflies_10000'
    args.y_dim = args.num_bin*args.num_mfeat
    video_list = video16_path[args.dtype]

    if args.plotF:
        colors = ['black', 'silver', 'red', 'green', 'deepskyblue', 'mediumpurple']
        models = [ 'const', 'zero', 'lr50', 'conv4_cat50', 'rnn50', 'skip50']
        labels = [ 'CONST', 'HALT', 'LINEAR'  , 'CNN', 'RNN', 'HRNN']
        vlowmax = [[2, 4.75], [5, 75], [0.1,0.9], [0.04,0.08], [0.04,0.08]]
        plot_nstep_errors(models, args.dtype, video_list, t_dim=args.t_dim, gender=args.gender, t0=0, tsim=args.tsim, visionF=args.visionF, labels=labels, colors=colors, vlowmax=vlowmax)

    else:
        if args.mtype=='lr50':

            ### LR ###
            save_path='./runs/linear_reg_'+str(args.t_dim) +'tau/%s/model/weight_gender0' % args.dtype
            if not args.visionF: save_path = save_path +'_visionF0'
            male_model = np.load(save_path+'.npy')

            save_path='./runs/linear_reg_'+str(args.t_dim) +'tau/%s/model/weight_gender1' % args.dtype
            if not args.visionF: save_path = save_path +'_visionF0'
            female_model = np.load(save_path+'.npy')

            for testvideo_num in range(0,len(video_list[TEST])):

                if  testvideo_num == 1 or testvideo_num == 2 :
                    simulated_male_flies = np.arange(0,9,1)
                    simulated_female_flies = np.arange(9,19,1)
                else:
                    simulated_male_flies = np.arange(0,10,1)
                    simulated_female_flies = np.arange(10,20,1)


                print ('testvideo %d %s' % (testvideo_num, video_list[TEST][testvideo_num]))
                for ifold in range(10):
                    real_flies_simulatePlan_RNNs(video_list[TEST][testvideo_num],\
                            male_model, female_model, \
                            simulated_male_flies, simulated_female_flies,\
                            monlyF=abs(1-args.visionF), ifold=ifold,\
                            tsim=args.tsim, mtype=args.mtype, t_dim=args.t_dim,\
                            num_bin=args.num_bin, use_cuda=args.use_cuda, btype=args.btype)


        elif args.mtype=='nn4_cat50' or args.mtype=='conv4_cat50':

            args.visionF=1
            vtype='full'
            model_epoch=25000 if args.mtype == 'nn4_cat50' else 25000 
            batch_sz=100 if args.mtype == 'nn4_cat50' else 32 
            from simulate_autoreg import model_selection
            for testvideo_num in range(0,len(video_list[TEST])):

                vpath = video_list[TEST][testvideo_num]
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
                        model_epoch=model_epoch, t_dim=args.t_dim, vtype=vtype,\
                        batch_sz=batch_sz, mtype=args.mtype, dtype=args.dtype)

                for ifold in range(10):
                    real_flies_simulatePlan_RNNs(video_list[TEST][testvideo_num],\
                            male_model, female_model, \
                            simulated_male_flies, simulated_female_flies,\
                            monlyF=abs(1-args.visionF), ifold=ifold, \
                            model_epoch=model_epoch, \
                            tsim=args.tsim, mtype=args.mtype,\
                            t_dim=args.t_dim, num_bins=args.num_bin, \
                            use_cuda=args.use_cuda)



        elif 'rnn' in args.mtype or 'skip' in args.mtype:

            ### LR MO ###
            model_epoch=200000
            from simulate_rnn_sjb import model_selection
            for testvideo_num in range(0,len(video_list[TEST])):

                vpath = video_list[TEST][testvideo_num]
                print ('testvideo %d of %d %s' % (testvideo_num, len(video_list[TEST]), vpath))
                matfile = args.datapath+vpath+fname

                if  testvideo_num == 1 or testvideo_num == 2 :
                    simulated_male_flies = np.arange(0,9,1)
                    simulated_female_flies = np.arange(9,19,1)
                else:
                    simulated_male_flies = np.arange(0,10,1)
                    simulated_female_flies = np.arange(10,20,1)


                male_model, female_model, male_hiddens, female_hiddens = \
                    model_selection(args, None, None, \
                        args.videotype, args.mtype, model_epoch, \
                        args.h_dim, simulated_male_flies, simulated_female_flies, \
                        dtype=args.dtype, btype=args.btype, use_cuda=args.use_cuda)

                for ifold in range(10):
                    print('ifold %d of %d' % (ifold, 10))
                    real_flies_simulatePlan_RNNs(video_list[TEST][testvideo_num],\
                            male_model, female_model, \
                            simulated_male_flies, simulated_female_flies,\
                            male_hiddens, female_hiddens,\
                            model_epoch=model_epoch, \
                            monlyF=abs(1-args.visionF), ifold=ifold,\
                            tsim=args.tsim, mtype=args.mtype, \
                            t_dim=args.t_dim, btype=args.btype,\
                            num_bin=args.num_bin,\
                            gender=args.gender)


        elif args.mtype == 'zero':
            for testvideo_num in range(0,len(video_list[TEST])):

                vpath = video_list[TEST][testvideo_num]
                print ('testvideo %d %s' % (testvideo_num, vpath))
                baseline0_zero_nstep_prediction(vpath, tsim=args.tsim)


        elif args.mtype == 'const':
            for testvideo_num in range(0,len(video_list[TEST])):

                vpath = video_list[TEST][testvideo_num]
                print ('testvideo %d %s' % (testvideo_num, vpath))
                baseline0_constant_nstep_prediction(vpath, tsim=args.tsim)


        elif args.mtype == 'copy':
            for testvideo_num in range(0,len(video_list[TEST])):

                vpath = video_list[TEST][testvideo_num]
                print ('testvideo %d %s' % (testvideo_num, vpath))
                baseline1_constVel_nstep_prediction(vpath, tsim=args.tsim)




