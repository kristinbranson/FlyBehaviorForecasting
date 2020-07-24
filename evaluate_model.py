import numpy as np
import h5py, copy, math, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
matplotlib.rcParams.update({'font.size': 22})
import scipy.io as sio

from util import * 
from gen_dataset import load_eyrun_data, video16_path, load_vision
from util_vis import * 
from util_fly import *

import torch
from torch.autograd import Variable
from scipy.stats import wasserstein_distance, energy_distance 

TRAIN=0
VALID=1
TEST=2
VDATA=0
MDATA=1
BDATA=2
MALE=0
FEMALE=1
SOURCE=0
TARGET=1
NUM_BIN=51
NUM_MFEAT=8
EPS=1e-6
RADIUS = 476.3236
PPM = 7.790785

def get_data_stats(t0, t1, gender='male'):

    #DATA TE
    dtype='gmr'
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):

        vpath = video16_path[dtype][TEST][testvideo_num]
        ftag = str(t0)+'t0_'+str(t1)+'t1_testvideo%d' % testvideo_num
        data_pos = np.load(basepath0 +'/motion/%s/data_position_%s_' % (vpath, gender) +ftag+'.npy')
        vel = np.load(basepath0 +'/velocity/%s/data_velocity_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        data_cd = np.load(basepath0 +'/centredist/%s/data_centredist_%s_' % (vpath, gender) +ftag+'.npy')
        wd = RADIUS -data_cd
        wds = np.hstack([wds, wd])

        data_body_pos = np.load(basepath0 +'/motion/%s/data_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = data_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        #data_wing_ang_mot = np.load(basepath +'/motion/data_body_position_'+gender+'_'+ftag+'.npy')
        wing_angle = np.hstack([abs(data_body_pos[:,1].flatten()),\
                                    data_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(data_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        data_mot = np.load(basepath0 +'/motion/%s/data_motion_%s_' % (vpath, gender) +ftag+'.npy')
        min_dtheta = 0#data_mot[:,2].flatten().min()
        dtheta = data_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    data_gmr_te = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
                'wing_angle':wing_angles, 'dtheta':dthetas}

    #DATA TR
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TRAIN])):
        vpath = video16_path[dtype][TRAIN][testvideo_num]

        ftag = str(t0)+'t0_'+str(t1)+'t1_testvideo%d' % testvideo_num
        data_pos = np.load(basepath0 +'/motion/%s/data_position_%s_' % (vpath, gender)+ftag+'.npy')
        vel = np.load(basepath0 +'/velocity/%s/data_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        vels = np.hstack([vels, vel])

        data_cd = np.load(basepath0 +'/centredist/%s/data_centredist_%s_' % (vpath, gender) +ftag+'.npy')
        wd = RADIUS -data_cd
        wds = np.hstack([wds, wd])

        data_body_pos = np.load(basepath0 +'/motion/%s/data_body_position_%s_' % (vpath, gender)+ftag+'.npy')
        theta = data_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        #data_wing_ang_mot = np.load(basepath +'/motion/data_body_position_'+gender+'_'+ftag+'.npy')
        wing_angle = np.hstack([abs(data_body_pos[:,1].flatten()),\
                                    data_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(data_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        data_mot = np.load(basepath0 +'/motion/%s/data_motion_%s_' % (vpath, gender) +ftag+'.npy')
        min_dtheta = 0
        dtheta = data_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    data_gmr_tr = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
                'wing_angle':wing_angles, 'dtheta':dthetas}


    dtype='gmr91'
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):

        vpath = video16_path[dtype][TEST][testvideo_num]
        ftag = str(t0)+'t0_'+str(t1)+'t1_testvideo%d' % testvideo_num
        data_pos = np.load(basepath0 +'/motion/%s/data_position_%s_' % (vpath, gender) +ftag+'.npy')
        vel = np.load(basepath0 +'/velocity/%s/data_velocity_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        data_cd = np.load(basepath0 +'/centredist/%s/data_centredist_%s_' % (vpath, gender) +ftag+'.npy')
        wd = RADIUS -data_cd
        wds = np.hstack([wds, wd])

        data_body_pos = np.load(basepath0 +'/motion/%s/data_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = data_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        #data_wing_ang_mot = np.load(basepath +'/motion/data_body_position_'+gender+'_'+ftag+'.npy')
        wing_angle = np.hstack([abs(data_body_pos[:,1].flatten()),\
                                    data_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(data_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        data_mot = np.load(basepath0 +'/motion/%s/data_motion_%s_' % (vpath, gender) +ftag+'.npy')
        min_dtheta = 0#data_mot[:,2].flatten().min()
        dtheta = data_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    data_gmr91_te = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
                'wing_angle':wing_angles, 'dtheta':dthetas}

    #DATA TR
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TRAIN])):
        vpath = video16_path[dtype][TRAIN][testvideo_num]

        ftag = str(t0)+'t0_'+str(t1)+'t1_testvideo%d' % testvideo_num
        data_pos = np.load(basepath0 +'/motion/%s/data_position_%s_' % (vpath, gender)+ftag+'.npy')
        vel = np.load(basepath0 +'/velocity/%s/data_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        vels = np.hstack([vels, vel])

        data_cd = np.load(basepath0 +'/centredist/%s/data_centredist_%s_' % (vpath, gender) +ftag+'.npy')
        wd = RADIUS -data_cd
        wds = np.hstack([wds, wd])

        data_body_pos = np.load(basepath0 +'/motion/%s/data_body_position_%s_' % (vpath, gender)+ftag+'.npy')
        theta = data_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        #data_wing_ang_mot = np.load(basepath +'/motion/data_body_position_'+gender+'_'+ftag+'.npy')
        wing_angle = np.hstack([abs(data_body_pos[:,1].flatten()),\
                                    data_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(data_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        data_mot = np.load(basepath0 +'/motion/%s/data_motion_%s_' % (vpath, gender) +ftag+'.npy')
        min_dtheta = 0
        dtheta = data_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    data_gmr91_tr = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
                'wing_angle':wing_angles, 'dtheta':dthetas}



    dtype='pdb'
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):

        vpath = video16_path[dtype][TEST][testvideo_num]
        ftag = str(t0)+'t0_'+str(t1)+'t1_testvideo%d' % testvideo_num
        data_pos = np.load(basepath0 +'/motion/%s/data_position_%s_' % (vpath, gender) +ftag+'.npy')
        vel = np.load(basepath0 +'/velocity/%s/data_velocity_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        data_cd = np.load(basepath0 +'/centredist/%s/data_centredist_%s_' % (vpath, gender) +ftag+'.npy')
        wd = RADIUS -data_cd
        wds = np.hstack([wds, wd])

        data_body_pos = np.load(basepath0 +'/motion/%s/data_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = data_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        #data_wing_ang_mot = np.load(basepath +'/motion/data_body_position_'+gender+'_'+ftag+'.npy')
        wing_angle = np.hstack([abs(data_body_pos[:,1].flatten()),\
                                    data_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(data_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        data_mot = np.load(basepath0 +'/motion/%s/data_motion_%s_' % (vpath, gender) +ftag+'.npy')
        min_dtheta = 0#data_mot[:,2].flatten().min()
        dtheta = data_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    data_pdb_te = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
                'wing_angle':wing_angles, 'dtheta':dthetas}

    #DATA TR
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TRAIN])):
        vpath = video16_path[dtype][TRAIN][testvideo_num]

        ftag = str(t0)+'t0_'+str(t1)+'t1_testvideo%d' % testvideo_num
        data_pos = np.load(basepath0 +'/motion/%s/data_position_%s_' % (vpath, gender)+ftag+'.npy')
        vel = np.load(basepath0 +'/velocity/%s/data_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        vels = np.hstack([vels, vel])

        data_cd = np.load(basepath0 +'/centredist/%s/data_centredist_%s_' % (vpath, gender) +ftag+'.npy')
        wd = RADIUS -data_cd
        wds = np.hstack([wds, wd])

        data_body_pos = np.load(basepath0 +'/motion/%s/data_body_position_%s_' % (vpath, gender)+ftag+'.npy')
        theta = data_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        #data_wing_ang_mot = np.load(basepath +'/motion/data_body_position_'+gender+'_'+ftag+'.npy')
        wing_angle = np.hstack([abs(data_body_pos[:,1].flatten()),\
                                    data_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(data_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        data_mot = np.load(basepath0 +'/motion/%s/data_motion_%s_' % (vpath, gender) +ftag+'.npy')
        min_dtheta = 0
        dtheta = data_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    data_pdb_tr = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
                'wing_angle':wing_angles, 'dtheta':dthetas}


    return data_gmr_tr, data_gmr_te, data_gmr91_tr, data_gmr91_te, data_pdb_tr, data_pdb_te



def get_stats_ind(t0, t1, dtype, testvideo_num=0, gender='male'):

    if not gender: fly_ind = 0
    min_dtheta=0

    #RNN LOO 50
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]
        fly_ind = 1 if gender and (testvideo_num == 1 or testvideo_num == 2) else 0

        t1 = 30000
        model_epoch=150000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_LOO_full_%dhid_lr0.010000_testvideo%d_%s' \
                % (t0, t1, model_epoch, num_hid, testvideo_num, dtype)
        vel  = np.load(basepath0 +'/velocity/%s/rnn50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn50_position_%s_' % (vpath, gender)+ftag+'.npy')
        T =  rnn_pos.shape[0]-1
        vel = vel.reshape([-1, T])[fly_ind]
        vels = np.hstack([vels, vel])

        rnn_cd  = np.load(basepath0 +'/centredist/%s/rnn50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -rnn_cd
        wd = wd.reshape([-1, T])[fly_ind]
        wds = np.hstack([wds, wd])

        rnn_body_pos = np.load(basepath0 +'/motion/%s/rnn50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = rnn_body_pos[:,0,fly_ind].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnn_body_pos[:,1, fly_ind]),\
                                rnn_body_pos[:,2, fly_ind]])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(rnn_pos)[:,fly_ind]
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnn_mot = np.load(basepath0 +'/motion/%s/rnn50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = rnn_mot[:,2,fly_ind] - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])
    
    rnn_loo50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}



    #RNN SMRF 50
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]
        fly_ind = 1 if gender and (testvideo_num == 1 or testvideo_num == 2) else 0

        model_epoch=100000 if 'pdb' in dtype else 150000 
        num_hid=100
        ftag = '%dt0_30320t1_epoch%d_SMRF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, model_epoch, num_hid, testvideo_num, dtype)
        vel  = np.load(basepath0 +'/velocity/%s/rnn50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn50_position_%s_' % (vpath, gender)+ftag+'.npy')
        T =  rnn_pos.shape[0]-1
        vel = vel.reshape([-1, T]).flatten() #[fly_ind]
        vels = np.hstack([vels, vel])

        rnn_cd  = np.load(basepath0 +'/centredist/%s/rnn50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -rnn_cd
        wd = wd.reshape([-1, T]).flatten() #[fly_ind]
        wds = np.hstack([wds, wd])

        rnn_body_pos = np.load(basepath0 +'/motion/%s/rnn50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = rnn_body_pos[:,0,:].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnn_body_pos[:,1, :].flatten()), rnn_body_pos[:,2, :].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(rnn_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnn_mot = np.load(basepath0 +'/motion/%s/rnn50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = rnn_mot[:,2,:].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])
    
    rnn_smrf50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}



    #RNN RMSF 50
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]
        fly_ind = 1 if gender and (testvideo_num == 1 or testvideo_num == 2) else 0

        model_epoch=100000 if 'pdb' in dtype else 150000 
        num_hid=100
        ftag = '%dt0_30320t1_epoch%d_RMSF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, model_epoch, num_hid, testvideo_num, dtype)
        vel      = np.load(basepath0 +'/velocity/%s/rnn50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn50_position_%s_' % (vpath, gender)+ftag+'.npy')
        T =  rnn_pos.shape[0]-1
        vel = vel.reshape([-1, T]).flatten()
        vels = np.hstack([vels, vel])
        rnn_cd  = np.load(basepath0 +'/centredist/%s/rnn50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        #rnn_cd = rnn_cd.reshape([10,-1])[:,0]
        wd = RADIUS -rnn_cd
        wd = wd.reshape([-1, T]).flatten()
        wds = np.hstack([wds, wd])

        rnn_body_pos = np.load(basepath0 +'/motion/%s/rnn50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = rnn_body_pos[:,0,:].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnn_body_pos[:,1,:].flatten()),\
                rnn_body_pos[:,2,:].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn50_position_%s_' % (vpath, gender)+ftag+'.npy')
        inter_dist  = compute_inter_animal_distance(rnn_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnn_mot = np.load(basepath0 +'/motion/%s/rnn50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = rnn_mot[:,2,:].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    rnn_rmsf50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    #RNN 50
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]
        fly_ind = 1 if gender and (testvideo_num == 1 or testvideo_num == 2) else 0

        t1 = 30320
        model_epoch=200000 if dtype == 'pdb' else 70000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, t1, model_epoch, num_hid, testvideo_num, dtype)

        vel  = np.load(basepath0 +'/velocity/%s/rnn50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn50_position_%s_' % (vpath, gender)+ftag+'.npy')
        T =  rnn_pos.shape[0]-1
        vel = vel.reshape([-1, T])[fly_ind]
        vels = np.hstack([vels, vel])

        rnn_cd  = np.load(basepath0 +'/centredist/%s/rnn50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -rnn_cd
        wd = wd.reshape([-1, T])[fly_ind]
        wds = np.hstack([wds, wd])

        rnn_body_pos = np.load(basepath0 +'/motion/%s/rnn50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = rnn_body_pos[:,0,fly_ind].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnn_body_pos[:,1,fly_ind]),\
                                rnn_body_pos[:,2,fly_ind]])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(rnn_pos)[:,fly_ind]
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnn_mot = np.load(basepath0 +'/motion/%s/rnn50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = rnn_mot[:,2,fly_ind] - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    rnn_smsf50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    #SKIP LOO 50
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]
        fly_ind = 1 if gender and (testvideo_num == 1 or testvideo_num == 2) else 0

        model_epoch=100000  if 'pdb' in dtype else 150000 
        num_hid=100
        ftag = '%dt0_30000t1_epoch%d_LOO_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, model_epoch, num_hid, testvideo_num, dtype)
        vel  = np.load(basepath0 +'/velocity/%s/skip50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        skip_pos  = np.load(basepath0 +'/motion/%s/skip50_position_%s_' % (vpath, gender)+ftag+'.npy')
        T =  skip_pos.shape[0]-1
        vel = vel.reshape([-1, T])[fly_ind]
        vels = np.hstack([vels, vel])

        skip_cd  = np.load(basepath0 +'/centredist/%s/skip50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -skip_cd
        wd = wd.reshape([-1, T])[fly_ind]
        wds = np.hstack([wds, wd])

        skip_body_pos = np.load(basepath0 +'/motion/%s/skip50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = skip_body_pos[:,0,fly_ind]
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(skip_body_pos[:,1,fly_ind]),\
                                skip_body_pos[:,2,fly_ind]])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(skip_pos)[:,fly_ind]
        inter_dists = np.hstack([inter_dists, inter_dist])

        skip_mot = np.load(basepath0 +'/motion/%s/skip50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = skip_mot[:,2,fly_ind] - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    skip_loo50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    #SKIP RMSF 50
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]
        fly_ind = 1 if gender and (testvideo_num == 1 or testvideo_num == 2) else 0

        model_epoch=100000  if 'pdb' in dtype else 150000 
        num_hid=100
        ftag = '%dt0_30320t1_epoch%d_RMSF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, model_epoch, num_hid, testvideo_num, dtype)
        vel  = np.load(basepath0 +'/velocity/%s/skip50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        skip_pos  = np.load(basepath0 +'/motion/%s/skip50_position_%s_' % (vpath, gender)+ftag+'.npy')
        T =  skip_pos.shape[0]-1
        vel = vel.reshape([-1, T]).flatten()
        vels = np.hstack([vels, vel])

        skip_cd  = np.load(basepath0 +'/centredist/%s/skip50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -skip_cd
        wd = wd.reshape([-1, T]).flatten()
        wds = np.hstack([wds, wd])

        skip_body_pos = np.load(basepath0 +'/motion/%s/skip50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = skip_body_pos[:,0,:].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(skip_body_pos[:,1,:].flatten()),\
                                    skip_body_pos[:,2,:].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(skip_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        skip_mot = np.load(basepath0 +'/motion/%s/skip50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = skip_mot[:,2,:].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    skip_rmsf50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    #SKIP SMRF 50
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]
        fly_ind = 1 if gender and (testvideo_num == 1 or testvideo_num == 2) else 0

        model_epoch=100000  if 'pdb' in dtype else 150000 
        num_hid=100
        ftag = '%dt0_30320t1_epoch%d_SMRF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, model_epoch, num_hid, testvideo_num, dtype)
        vel  = np.load(basepath0 +'/velocity/%s/skip50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        skip_pos  = np.load(basepath0 +'/motion/%s/skip50_position_%s_' % (vpath, gender)+ftag+'.npy')
        T =  skip_pos.shape[0]-1
        vel = vel.reshape([-1, T]).flatten()
        vels = np.hstack([vels, vel])

        skip_cd  = np.load(basepath0 +'/centredist/%s/skip50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -skip_cd
        wd = wd.reshape([-1, T]).flatten()
        wds = np.hstack([wds, wd])

        skip_body_pos = np.load(basepath0 +'/motion/%s/skip50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = skip_body_pos[:,0,:].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(skip_body_pos[:,1,:].flatten()),\
                                    skip_body_pos[:,2,:].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(skip_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        skip_mot = np.load(basepath0 +'/motion/%s/skip50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = skip_mot[:,2,:].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    skip_smrf50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}



    #SKIP 50
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]
        fly_ind = 1 if gender and (testvideo_num == 1 or testvideo_num == 2) else 0

        model_epoch=100000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, t1, model_epoch, num_hid, testvideo_num, dtype)
        vel  = np.load(basepath0 +'/velocity/%s/skip50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        skip_pos  = np.load(basepath0 +'/motion/%s/skip50_position_%s_' % (vpath, gender)+ftag+'.npy')
        T =  skip_pos.shape[0]-1
        vel = vel.reshape([-1, T])[fly_ind]
        vels = np.hstack([vels, vel])

        skip_cd  = np.load(basepath0 +'/centredist/%s/skip50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -skip_cd
        wd = wd.reshape([-1, T])[fly_ind]
        wds = np.hstack([wds, wd])

        skip_body_pos = np.load(basepath0 +'/motion/%s/skip50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = skip_body_pos[:,0,fly_ind]
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(skip_body_pos[:,1,fly_ind]),\
                                skip_body_pos[:,2,fly_ind]])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(skip_pos)[:,fly_ind]
        inter_dists = np.hstack([inter_dists, inter_dist])

        skip_mot = np.load(basepath0 +'/motion/%s/skip50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = skip_mot[:,2,fly_ind] - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    skip_smsf50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    return rnn_loo50, rnn_rmsf50, rnn_smrf50, rnn_smsf50,\
            skip_loo50, skip_rmsf50, skip_smrf50, skip_smsf50



def get_stats_mo_vo(t0, t1, testvideo_num=0, gender='male', sim_type='SMSF'):

    #lr50 MO
    print('LR50 MO')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):

        vpath = video16_path[dtype][TEST][testvideo_num]
        t_dim = 50
        ftag = 'visionF0_'+str(t0)+'t0_'+str(t1)+'t1_SMSF_testvideo%d' % testvideo_num
        vel = np.load(basepath0+'/velocity/%s/lr50_velocity_%s_' % (vpath, gender) + ftag+'.npy')
        vels = np.hstack([vels, vel])

        #ftag = str(t_dim)+'tdim_'+str(t0)+'t0_'+str(t1)+'t1'
        cd = np.load(basepath0+'/centredist/%s/lr50_centredist_%s_' % (vpath, gender) + ftag+'.npy')
        wd = RADIUS - cd
        wds = np.hstack([wds, wd])

        lr50mo_body_pos = np.load(basepath0+'/motion/%s/lr50_body_position_%s_' % (vpath, gender) + ftag+'.npy')
        theta = lr50mo_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(lr50mo_body_pos[:,1].flatten()),\
                                lr50mo_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        lr50vo_pos    = np.load(basepath0 +'/motion/%s/lr50_position_%s_' % (vpath, gender)+ftag+'.npy')
        inter_dist  = compute_inter_animal_distance(lr50vo_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        lr50vo_mot = np.load(basepath0+'/motion/%s/lr50_motion_%s_' % (vpath,gender)+ftag+'.npy')
        dtheta = lr50vo_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    lr50mo = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    #lr50 VO
    print('LR50 VO')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):

        vpath = video16_path[dtype][TEST][testvideo_num]
        t_dim = 50
        ftag = 'visionOnly_'+str(t0)+'t0_'+str(t1)+'t1_SMSF_testvideo%d' % testvideo_num
        vel = np.load(basepath0+'/velocity/%s/lr50_velocity_%s_' % (vpath, gender) + ftag+'.npy')
        vels = np.hstack([vels, vel])

        #ftag = str(t_dim)+'tdim_'+str(t0)+'t0_'+str(t1)+'t1'
        cd = np.load(basepath0+'/centredist/%s/lr50_centredist_%s_' % (vpath, gender) + ftag+'.npy')
        wd = RADIUS - cd
        wds = np.hstack([wds, wd])

        lr50vo_body_pos = np.load(basepath0+'/motion/%s/lr50_body_position_%s_' % (vpath, gender) + ftag+'.npy')
        theta = lr50vo_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(lr50mo_body_pos[:,1].flatten()),\
                                lr50mo_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        lr50mo_pos    = np.load(basepath0 +'/motion/%s/lr50_position_%s_' % (vpath, gender)+ftag+'.npy')
        inter_dist  = compute_inter_animal_distance(lr50mo_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        lr50vo_mot = np.load(basepath0+'/motion/%s/lr50_motion_%s_' % (vpath,gender)+ftag+'.npy')
        dtheta = lr50vo_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    lr50vo = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}

    #NN REG 50 Motion Only
    print('NN REG50 MO')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        ftag = 'visionF0_'+str(t0)+'t0_'+str(t1)+'t1_SMSF_testvideo%d' % testvideo_num
        vel = np.load(basepath0+'/velocity/%s/nn50_velocity_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        cd = np.load(basepath0+'/centredist/%s/nn50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS - cd
        wds = np.hstack([wds, wd])

        nn50_body_pos = np.load(basepath0+'/motion/%s/nn50_body_position_%s_' % (vpath, gender)+ftag+'.npy')
        theta = nn50_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(nn50_body_pos[:,1].flatten()),\
                                nn50_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        nn50_pos    = np.load(basepath0 +'/motion/%s/nn50_position_%s_' % (vpath, gender)+ftag+'.npy')
        inter_dist  = compute_inter_animal_distance(nn50_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        nn50_mot = np.load(basepath0+'/motion/%s/nn50_motion_%s_'%(vpath,gender)+ftag+'.npy')
        dtheta = nn50_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    nn_reg50mo = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    #NN REG 50 Vision Only
    print('NN REG50 VO')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        ftag = 'visionOnly_'+str(t0)+'t0_'+str(t1)+'t1_SMSF_testvideo%d' % testvideo_num
        vel = np.load(basepath0+'/velocity/%s/nn_velocity_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        cd = np.load(basepath0+'/centredist/%s/nn_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS - cd
        wds = np.hstack([wds, wd])

        nn50_body_pos = np.load(basepath0+'/motion/%s/nn_body_position_%s_' % (vpath, gender)+ftag+'.npy')
        theta = nn50_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(nn50_body_pos[:,1].flatten()),\
                                nn50_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        nn50_pos    = np.load(basepath0 +'/motion/%s/nn_position_%s_' % (vpath, gender)+ftag+'.npy')
        inter_dist  = compute_inter_animal_distance(nn50_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        nn50_mot = np.load(basepath0+'/motion/%s/nn_motion_%s_'%(vpath,gender)+ftag+'.npy')
        dtheta = nn50_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    nn_reg50vo = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    return lr50mo, lr50vo, nn_reg50mo, nn_reg50vo, \


def get_reg_stats(t0, t1, testvideo_num=0, gender='male', sim_type='SMSF'):

    #NN REG 50
    print('NN REG50')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]
        ftag = str(t0)+'t0_'+str(t1)+'t1_SMSF_testvideo%d' % testvideo_num
        vel = np.load(basepath0+'/velocity/%s/nn4_reg50_velocity_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        cd = np.load(basepath0+'/centredist/%s/nn4_reg50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS - cd
        wds = np.hstack([wds, wd])

        nn50_body_pos = np.load(basepath0+'/motion/%s/nn4_reg50_body_position_%s_' % (vpath, gender)+ftag+'.npy')
        theta = nn50_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(nn50_body_pos[:,1].flatten()),\
                                nn50_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        nn50_pos    = np.load(basepath0 +'/motion/%s/nn4_reg50_position_%s_' % (vpath, gender)+ftag+'.npy')
        inter_dist  = compute_inter_animal_distance(nn50_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        nn50_mot = np.load(basepath0+'/motion/%s/nn4_reg50_motion_%s_'%(vpath,gender)+ftag+'.npy')
        dtheta = nn50_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    nn_reg50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}



    #RNNREG
    print('RNN REG50')
    lr=0.01
    num_hid=100
    videotype='full'
    model_epoch=60000

    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        ftag = str(t0)+'t0_'+str(t1)+'t1_epoch'+str(model_epoch)+'_'\
                +sim_type+'_'+str(videotype)+'_%dhid_lr%f_testvideo%d' % (num_hid, lr, testvideo_num)
        #ftag = str(t0)+'t0_'+str(t1)+'t1_epoch80000_testvideo%d' % testvideo_num
        vel  = np.load(basepath0 +'/velocity/%s/rnn_reg50_velocity_%s_'%(vpath,gender)+ftag+'.npy')
        rnnreg_pos  = np.load(basepath0 +'/motion/%s/rnn_reg50_position_%s_'%(vpath,gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        rnnreg_cd  = np.load(basepath0 +'/centredist/%s/rnn_reg50_centredist_%s_'%(vpath,gender)+ftag+'.npy')
        wd = RADIUS -rnnreg_cd
        wds = np.hstack([wds, wd])

        rnnreg_body_pos = np.load(basepath0 +'/motion/%s/rnn_reg50_body_position_%s_'%(vpath,gender)+ftag+'.npy')
        theta = rnnreg_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnnreg_body_pos[:,1].flatten()),\
                                    rnnreg_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(rnnreg_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnnreg_mot = np.load(basepath0 +'/motion/%s/rnn_reg50_motion_%s_'%(vpath,gender)+ftag+'.npy')
        dtheta = rnnreg_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    rnn50_reg = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}

    return nn_reg50, rnn50_reg



def get_stats_nstep(t0, t1, dtype, testvideo_num=0, gender='male', sim_type='SMSF'):


    #RNN  25
    print('RNN CAT25')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        model_epoch=150000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, t1, model_epoch, num_hid, testvideo_num, dtype)
        vel  = np.load(basepath0 +'/velocity/%s/rnn25_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn25_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        rnn_cd  = np.load(basepath0 +'/centredist/%s/rnn25_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -rnn_cd
        wds = np.hstack([wds, wd])

        rnn_body_pos = np.load(basepath0 +'/motion/%s/rnn25_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = rnn_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnn_body_pos[:,1].flatten()),\
                                rnn_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(rnn_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnn_mot = np.load(basepath0 +'/motion/%s/rnn25_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = rnn_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    rnn25 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}

    #RNN 50
    print('RNN CAT50')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        model_epoch=100000
        model_epoch=200000 if dtype == 'pdb' else 70000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, t1, model_epoch, num_hid, testvideo_num, dtype)
        vel  = np.load(basepath0 +'/velocity/%s/rnn50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn50_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        rnn_cd  = np.load(basepath0 +'/centredist/%s/rnn50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -rnn_cd
        wds = np.hstack([wds, wd])

        rnn_body_pos = np.load(basepath0 +'/motion/%s/rnn50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = rnn_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnn_body_pos[:,1].flatten()),\
                                rnn_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(rnn_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnn_mot = np.load(basepath0 +'/motion/%s/rnn50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = rnn_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    rnn50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}

    #RNN  100
    print('RNN CAT100')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        model_epoch=150000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, t1, model_epoch, num_hid, testvideo_num, dtype)
        vel  = np.load(basepath0 +'/velocity/%s/rnn100_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn100_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        rnn_cd  = np.load(basepath0 +'/centredist/%s/rnn100_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -rnn_cd
        wds = np.hstack([wds, wd])

        rnn_body_pos = np.load(basepath0 +'/motion/%s/rnn100_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = rnn_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnn_body_pos[:,1].flatten()),\
                                rnn_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(rnn_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnn_mot = np.load(basepath0 +'/motion/%s/rnn100_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = rnn_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    rnn100 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}

    #RNN  150
    print('RNN CAT150')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        model_epoch=200000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, t1, model_epoch, num_hid, testvideo_num, dtype)
        vel  = np.load(basepath0 +'/velocity/%s/rnn150_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn150_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        rnn_cd  = np.load(basepath0 +'/centredist/%s/rnn150_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -rnn_cd
        wds = np.hstack([wds, wd])

        rnn_body_pos = np.load(basepath0 +'/motion/%s/rnn150_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = rnn_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnn_body_pos[:,1].flatten()),\
                                rnn_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(rnn_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnn_mot = np.load(basepath0 +'/motion/%s/rnn150_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = rnn_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    rnn150 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}



    #SKIP 25
    print('SKIP CAT25')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        model_epoch=100000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d'\
                % (t0, t1, model_epoch, num_hid, testvideo_num)
        vel  = np.load(basepath0 +'/velocity/%s/skip25_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        skip_pos  = np.load(basepath0 +'/motion/%s/skip25_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        skip_cd  = np.load(basepath0 +'/centredist/%s/skip25_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -skip_cd
        wds = np.hstack([wds, wd])

        skip_body_pos = np.load(basepath0 +'/motion/%s/skip25_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = skip_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(skip_body_pos[:,1].flatten()),\
                                skip_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(skip_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        skip_mot = np.load(basepath0 +'/motion/%s/skip25_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = skip_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    skip_cat25 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    #SKIP 50
    print('SKIP CAT50')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        model_epoch=100000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d'\
                % (t0, t1, model_epoch, num_hid, testvideo_num)
        vel  = np.load(basepath0 +'/velocity/%s/skip50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        skip_pos  = np.load(basepath0 +'/motion/%s/skip50_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        skip_cd  = np.load(basepath0 +'/centredist/%s/skip50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -skip_cd
        wds = np.hstack([wds, wd])

        skip_body_pos = np.load(basepath0 +'/motion/%s/skip50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = skip_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(skip_body_pos[:,1].flatten()),\
                                skip_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(skip_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        skip_mot = np.load(basepath0 +'/motion/%s/skip50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = skip_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    skip_cat50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}

    #SKIP 100
    print('SKIP CAT100')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        model_epoch=100000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d'\
                % (t0, t1, model_epoch, num_hid, testvideo_num)
        vel  = np.load(basepath0 +'/velocity/%s/skip100_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        skip_pos  = np.load(basepath0 +'/motion/%s/skip100_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        skip_cd  = np.load(basepath0 +'/centredist/%s/skip100_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -skip_cd
        wds = np.hstack([wds, wd])

        skip_body_pos = np.load(basepath0 +'/motion/%s/skip100_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = skip_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(skip_body_pos[:,1].flatten()),\
                                skip_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(skip_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        skip_mot = np.load(basepath0 +'/motion/%s/skip100_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = skip_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    skip_cat100 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    #SKIP 150
    print('SKIP CAT150')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        model_epoch=100000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d'\
                % (t0, t1, model_epoch, num_hid, testvideo_num)
        vel  = np.load(basepath0 +'/velocity/%s/skip150_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        skip_pos  = np.load(basepath0 +'/motion/%s/skip150_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        skip_cd  = np.load(basepath0 +'/centredist/%s/skip150_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -skip_cd
        wds = np.hstack([wds, wd])

        skip_body_pos = np.load(basepath0 +'/motion/%s/skip150_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = skip_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(skip_body_pos[:,1].flatten()),\
                                skip_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(skip_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        skip_mot = np.load(basepath0 +'/motion/%s/skip150_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = skip_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    skip_cat150 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    return rnn25, rnn50, rnn100, rnn150, \
            skip_cat25, skip_cat50, skip_cat100, skip_cat150


def get_stats_perc_linear(t0, t1, testvideo_num=0, gender='male', sim_type='SMSF'):

    #RNN 50
    print('RNN CAT50')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        #model_epoch=150000 if (gender == 'female' and dtype == 'gmr91') else 100000
        #model_epoch=70000  if (gender == 'male' and dtype == 'gmr') else 200000
        model_epoch=150000 if dtype == 'pdb' else 70000
        model_epoch=200000 if dtype == 'gmr' else 70000
        if 1: model_epoch=200000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_perc_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, t1, model_epoch, num_hid, testvideo_num, dtype)
        print(vpath, ftag)
        vel  = np.load(basepath0 +'/velocity/%s/rnn50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn50_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        rnn_cd  = np.load(basepath0 +'/centredist/%s/rnn50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -rnn_cd
        wds = np.hstack([wds, wd])

        rnn_body_pos = np.load(basepath0 +'/motion/%s/rnn50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = rnn_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnn_body_pos[:,1].flatten()),\
                                rnn_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(rnn_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnn_mot = np.load(basepath0 +'/motion/%s/rnn50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = rnn_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    rnn50_perc = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
                    'wing_angle':wing_angles, 'dtheta':dthetas}


    #RNN 50
    print('RNN CAT50')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        #model_epoch=150000 if (gender == 'female' and dtype == 'gmr91') else 100000
        #model_epoch=70000  if (gender == 'male' and dtype == 'gmr') else 200000
        model_epoch=150000 if dtype == 'pdb' else 70000
        model_epoch=200000 if dtype == 'gmr' else 70000
        if 1: model_epoch=200000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_linear_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, t1, model_epoch, num_hid, testvideo_num, dtype)
        print(vpath, ftag)
        vel  = np.load(basepath0 +'/velocity/%s/rnn50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn50_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        rnn_cd  = np.load(basepath0 +'/centredist/%s/rnn50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -rnn_cd
        wds = np.hstack([wds, wd])

        rnn_body_pos = np.load(basepath0 +'/motion/%s/rnn50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = rnn_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnn_body_pos[:,1].flatten()),\
                                rnn_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(rnn_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnn_mot = np.load(basepath0 +'/motion/%s/rnn50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = rnn_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    rnn50_linear = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}

    return rnn50_perc, rnn50_linear



def get_stats(t0, t1, dtype, testvideo_num=0, gender='male', sim_type='SMSF'):

    min_dtheta=0

    #lr50
    print('LR50')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        ftag = str(t0)+'t0_'+str(t1)+'t1_SMSF_testvideo%d_%s' % (testvideo_num, dtype)
        vel = np.load(basepath0+'/velocity/%s/lr50_velocity_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        cd = np.load(basepath0+'/centredist/%s/lr50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS - cd
        wds = np.hstack([wds, wd])

        lr50_body_pos = np.load(basepath0+'/motion/%s/lr50_body_position_%s_' % (vpath, gender)+ftag+'.npy')
        theta = lr50_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(lr50_body_pos[:,1].flatten()),\
                                lr50_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        lr50_pos    = np.load(basepath0 +'/motion/%s/lr50_position_%s_' % (vpath, gender)+ftag+'.npy')
        inter_dist  = compute_inter_animal_distance(lr50_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        lr50_mot = np.load(basepath0+'/motion/%s/lr50_motion_%s_'%(vpath,gender)+ftag+'.npy')
        dtheta = lr50_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    lr50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}



    ##NN CAT 50
    #print('NN CAT50')
    #inter_dists = np.asarray([])
    #vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    #wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    #for testvideo_num in range(len(video16_path[dtype][TEST])):
    #    vpath = video16_path[dtype][TEST][testvideo_num]
    #    ftag = str(t0)+'t0_'+str(t1)+'t1_SMSF_testvideo%d' % testvideo_num
    #    vel = np.load(basepath0+'/velocity/%s/nn4_cat50_velocity_%s_' % (vpath, gender)+ftag+'.npy')
    #    vels = np.hstack([vels, vel])

    #    cd = np.load(basepath0+'/centredist/%s/nn4_cat50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
    #    wd = RADIUS - cd
    #    wds = np.hstack([wds, wd])

    #    nn50_body_pos = np.load(basepath0+'/motion/%s/nn4_cat50_body_position_%s_' % (vpath, gender)+ftag+'.npy')
    #    theta = nn50_body_pos[:,0].flatten()
    #    thetas = np.hstack([thetas, theta])

    #    wing_angle = np.hstack([abs(nn50_body_pos[:,1].flatten()),\
    #                            nn50_body_pos[:,2].flatten()])
    #    wing_angles = np.hstack([wing_angles, wing_angle])

    #    nn50_pos    = np.load(basepath0 +'/motion/%s/nn4_cat50_position_%s_' % (vpath, gender)+ftag+'.npy')
    #    inter_dist  = compute_inter_animal_distance(nn50_pos).flatten()
    #    inter_dists = np.hstack([inter_dists, inter_dist])

    #    nn50_mot = np.load(basepath0+'/motion/%s/nn4_cat50_motion_%s_'%(vpath,gender)+ftag+'.npy')
    #    dtheta = nn50_mot[:,2].flatten() - min_dtheta
    #    dthetas = np.hstack([dthetas, dtheta])

    #nn_cat50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
    #        'wing_angle':wing_angles, 'dtheta':dthetas}


    #CONV CAT 50
    print('CONV CAT50')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]
        ftag = str(t0)+'t0_'+str(t1)+'t1_SMSF_testvideo%d_%s' % (testvideo_num, dtype)
        vel = np.load(basepath0+'/velocity/%s/conv4_cat50_velocity_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        cd = np.load(basepath0+'/centredist/%s/conv4_cat50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS - cd
        wds = np.hstack([wds, wd])

        nn50_body_pos = np.load(basepath0+'/motion/%s/conv4_cat50_body_position_%s_' % (vpath, gender)+ftag+'.npy')
        theta = nn50_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(nn50_body_pos[:,1].flatten()),\
                                nn50_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        nn50_pos    = np.load(basepath0 +'/motion/%s/conv4_cat50_position_%s_' % (vpath, gender)+ftag+'.npy')
        inter_dist  = compute_inter_animal_distance(nn50_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        nn50_mot = np.load(basepath0+'/motion/%s/conv4_cat50_motion_%s_'%(vpath,gender)+ftag+'.npy')
        dtheta = nn50_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    conv4_cat50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    ##OLD RNN 
    #print('OLDRNN CAT50')
    #inter_dists = np.asarray([])
    #vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    #wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    #for testvideo_num in range(len(video16_path[dtype][TEST])):
    #    vpath = video16_path[dtype][TEST][testvideo_num]

    #    model_epoch=60000
    #    num_hid=100
    #    ftag = '%dt0_%dt1_epoch%d_SMSF_v1_%dhid_lr0.010000_testvideo%d'\
    #            % (t0, t1, model_epoch, num_hid, testvideo_num)
    #    vel  = np.load(basepath0 +'/velocity/%s/old_rnn50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
    #    rnn_pos  = np.load(basepath0 +'/motion/%s/old_rnn50_position_%s_' % (vpath, gender)+ftag+'.npy')
    #    vels = np.hstack([vels, vel])

    #    rnn_cd  = np.load(basepath0 +'/centredist/%s/old_rnn50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
    #    wd = RADIUS -rnn_cd
    #    wds = np.hstack([wds, wd])

    #    rnn_body_pos = np.load(basepath0 +'/motion/%s/old_rnn50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
    #    theta = rnn_body_pos[:,0].flatten()
    #    thetas = np.hstack([thetas, theta])

    #    wing_angle = np.hstack([abs(rnn_body_pos[:,1].flatten()),\
    #                            rnn_body_pos[:,2].flatten()])
    #    wing_angles = np.hstack([wing_angles, wing_angle])

    #    inter_dist  = compute_inter_animal_distance(rnn_pos).flatten()
    #    inter_dists = np.hstack([inter_dists, inter_dist])

    #    rnn_mot = np.load(basepath0 +'/motion/%s/old_rnn50_motion_%s_' % (vpath, gender)+ftag+'.npy')
    #    min_dtheta = 0
    #    dtheta = rnn_mot[:,2].flatten() - min_dtheta
    #    dthetas = np.hstack([dthetas, dtheta])

    #old_rnn50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
    #        'wing_angle':wing_angles, 'dtheta':dthetas}


    #RNN 50
    print('RNN CAT50')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        #model_epoch=150000 if (gender == 'female' and dtype == 'gmr91') else 100000
        #model_epoch=70000  if (gender == 'male' and dtype == 'gmr') else 200000
        model_epoch=150000 if dtype == 'pdb' else 70000
        model_epoch=200000 if dtype == 'gmr' else 70000
        if 1: model_epoch=200000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, t1, model_epoch, num_hid, testvideo_num, dtype)
        print(vpath, ftag)
        vel  = np.load(basepath0 +'/velocity/%s/rnn50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn50_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        rnn_cd  = np.load(basepath0 +'/centredist/%s/rnn50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -rnn_cd
        wds = np.hstack([wds, wd])

        rnn_body_pos = np.load(basepath0 +'/motion/%s/rnn50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = rnn_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnn_body_pos[:,1].flatten()),\
                                rnn_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(rnn_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnn_mot = np.load(basepath0 +'/motion/%s/rnn50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = rnn_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    rnn50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}

    #RNN 50
    print('RNN CAT50')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        #model_epoch=150000 if (gender == 'female' and dtype == 'gmr91') else 100000
        #model_epoch=70000  if (gender == 'male' and dtype == 'gmr') else 200000
        model_epoch=150000 if dtype == 'pdb' else 70000
        model_epoch=200000 if dtype == 'gmr' else 70000
        if 1: model_epoch=200000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, t1, model_epoch, num_hid, testvideo_num, dtype)
        print(vpath, ftag)
        vel  = np.load(basepath0 +'/velocity/%s/rnn50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        rnn_pos  = np.load(basepath0 +'/motion/%s/rnn50_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        rnn_cd  = np.load(basepath0 +'/centredist/%s/rnn50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -rnn_cd
        wds = np.hstack([wds, wd])

        rnn_body_pos = np.load(basepath0 +'/motion/%s/rnn50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = rnn_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(rnn_body_pos[:,1].flatten()),\
                                rnn_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(rnn_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        rnn_mot = np.load(basepath0 +'/motion/%s/rnn50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = rnn_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    rnn50_perc = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    #SKIP 50
    print('SKIP CAT50')
    inter_dists = np.asarray([])
    vels, wds, wds = np.asarray([]), np.asarray([]), np.asarray([])
    wing_angles, thetas, dthetas = np.asarray([]), np.asarray([]), np.asarray([])
    for testvideo_num in range(len(video16_path[dtype][TEST])):
        vpath = video16_path[dtype][TEST][testvideo_num]

        if 'gmr' == dtype: model_epoch=100000 
        if 'pdb' in dtype: model_epoch=50000 
        if 'gmr91' == dtype: model_epoch=100000
        #model_epoch=40000 if 'pdb' in dtype else 100000
        num_hid=100
        ftag = '%dt0_%dt1_epoch%d_SMSF_full_%dhid_lr0.010000_testvideo%d_%s'\
                % (t0, t1, model_epoch, num_hid, testvideo_num, dtype)
        print(vpath, ftag)
        vel  = np.load(basepath0 +'/velocity/%s/skip50_velocity_%s_' % (vpath, gender) +ftag+'.npy')
        skip_pos  = np.load(basepath0 +'/motion/%s/skip50_position_%s_' % (vpath, gender)+ftag+'.npy')
        vels = np.hstack([vels, vel])

        skip_cd  = np.load(basepath0 +'/centredist/%s/skip50_centredist_%s_' % (vpath,gender)+ftag+'.npy')
        wd = RADIUS -skip_cd
        wds = np.hstack([wds, wd])

        skip_body_pos = np.load(basepath0 +'/motion/%s/skip50_body_position_%s_' % (vpath, gender) +ftag+'.npy')
        theta = skip_body_pos[:,0].flatten()
        thetas = np.hstack([thetas, theta])

        wing_angle = np.hstack([abs(skip_body_pos[:,1].flatten()),\
                                skip_body_pos[:,2].flatten()])
        wing_angles = np.hstack([wing_angles, wing_angle])

        inter_dist  = compute_inter_animal_distance(skip_pos).flatten()
        inter_dists = np.hstack([inter_dists, inter_dist])

        skip_mot = np.load(basepath0 +'/motion/%s/skip50_motion_%s_' % (vpath, gender)+ftag+'.npy')
        min_dtheta = 0
        dtheta = skip_mot[:,2].flatten() - min_dtheta
        dthetas = np.hstack([dthetas, dtheta])

    skip_cat50 = {'vel':vels, 'wd':wds, 'theta':thetas, 'inter_dist':inter_dists,\
            'wing_angle':wing_angles, 'dtheta':dthetas}


    return lr50, conv4_cat50, rnn50, rnn50_perc, skip_cat50




def histo_diff(data, labels, key, bin_width, keep_perc, \
                left_chop, right_chop, ftag, plotF, fdir, colors=None):

    #for data_dict in data: 
    #    data_dict['wd'] = data_dict['wd'] / PPM
    #    data_dict['inter_dist'] = data_dict['inter_dist'] / PPM
    #import pdb; pdb.set_trace()
    print(key)
    lists = [data_dict[key] for data_dict in data]
    min_result = list(map(lambda x: np.nanmin(x), lists))
    min_val = min([min(min_result), 0])

    max_result = list(map(lambda x: np.nanmax(x), lists))
    max_val = max(max_result)

    hists = []
    for data_dict, label in zip(data, labels):

        val = data_dict[key]

        num_bins = int(math.ceil((max_val - min_val) / bin_width))
        bin_index= np.arange(num_bins) * bin_width
        bin_loc  = bin_index + bin_width / 2 + min_val

        hist = manual_histogram(val, bin_width=bin_width, min_val=min_val)
        hists.append(hist)

    hists_ = zeropad_hists(hists)

    bookkeep = []
    for i, (label, hist) in enumerate(zip(labels[1:], hists_[1:])):
        cdist = correlation_distance(hists_[0], hists_[i+1])
        adist = np.sum(abs(hists_[0]-hists_[i+1]))
        print('%s    : %d Step - CS and Abs. distance %f %f' % \
                (label, args.t1, cdist, adist))
        bookkeep.append([cdist, adist])

    bookkeep = np.asarray(bookkeep)
    n = bookkeep.shape[0]
    width=1/n

    #width=1
    plot_histo_ind(bin_loc, hists_[0:2], keep_perc, min_val, max_val, \
                fdir, '_data_distribution_'+ftag, bin_width=bin_width, labels=labels,\
                left_chop=left_chop, right_chop=right_chop, \
                colors=colors)

    if colors is not None:
        plot_histo_ind(bin_loc, hists_, keep_perc, min_val, max_val, \
                    fdir, ftag, bin_width=bin_width, labels=labels,\
                    left_chop=left_chop, right_chop=right_chop, \
                    colors=colors)

        ylabel = 'Correlation Distance'
        xlabel = 'Distribution distance against test dataset'
        ftag = 'CorrelationDist_' + ftag
        plot_bar(width, bookkeep[:,0], ylabel, xlabel, fdir, ftag, labels[1:], colors[1:])

        ylabel = 'L1 Distance'
        ftag = 'L1Dist_' + ftag
        plot_bar(width, bookkeep[:,1], ylabel, xlabel, fdir, ftag, labels[1:], colors[1:])


    else:
        plot_histo_ind(bin_loc, hists_, keep_perc, min_val, max_val, \
                    fdir, ftag, bin_width=bin_width, labels=labels,\
                    left_chop=left_chop, right_chop=right_chop)


        ylabel = 'Correlation Distance'
        xlabel = 'Distribution distance against test dataset'
        ftag = 'CorrelationDist_' + ftag
        plot_bar(width, bookkeep[:,0], ylabel, xlabel, fdir, ftag, labels[1:])

        ylabel = 'L1 Distance'
        ftag = 'L1Dist_' + ftag
        plot_bar(width, bookkeep[:,1], ylabel, xlabel, fdir, ftag, labels[1:])


def plot_bar_anal():

    male_gmr_id = np.asarray([0.047116, 0.056290, 0.048523, 0.051359, 0.071427, 0.060737, 0.078263])
    fale_gmr_id = np.asarray([0.034572, 0.085895, 0.058700, 0.075325, 0.080112, 0.078667, 0.079917])
                                                                                          
    male_gmr_wd = np.asarray([0.044456, 0.211412, 0.198050, 0.373734, 0.204690, 0.329584, 0.449256])
    fale_gmr_wd = np.asarray([0.073834, 0.215440, 0.293732, 0.423417, 0.178636, 0.572546, 0.588914])
                                                                                          
    male_gmr_td = np.asarray([0.015093, 0.153972, 0.259265, 0.219477, 0.165595, 0.324927, 0.359722])
    fale_gmr_td = np.asarray([0.023222, 0.267458, 0.257004, 0.306299, 0.336984, 0.314887, 0.374290])
                                                                                          
    male_gmr_vl = np.asarray([0.028586, 0.130851, 0.443838, 0.471106, 0.124278, 0.633899, 0.672400])
    fale_gmr_vl = np.asarray([0.039923, 0.069532, 0.402091, 0.401284, 0.061512, 0.466728, 0.517803])


    male_gmr91_id = np.asarray([0.159202, 0.181054, 0.782285, 1.427747, 0.112546, 0.157116, 0.276987])
    fale_gmr91_id = np.asarray([0.165969, 0.126120, 0.074902, 0.776773, 0.096693, 0.071255, 0.100624])
                                                                                          
    male_gmr91_wd = np.asarray([0.132526, 0.271577, 0.450326, 0.334684, 0.266793, 0.300909, 0.415036])
    fale_gmr91_wd = np.asarray([0.364645, 0.376877, 0.487691, 0.670024, 0.384425, 0.504219, 0.360230])
                                                                                          
    male_gmr91_td = np.asarray([0.095609, 0.131015, 0.183015, 0.221102, 0.161387, 0.206444, 0.264850])
    fale_gmr91_td = np.asarray([0.126957, 0.107021, 0.175319, 0.178092, 0.078196, 0.156549, 0.162809])
                                                                                          
    male_gmr91_vl = np.asarray([0.108888, 0.225564, 0.251443, 0.241057, 0.222538, 0.253548, 0.282698])
    fale_gmr91_vl = np.asarray([0.165912, 0.374063, 0.331864, 0.264555, 0.366271, 0.348937, 0.330629])


    male_pdb_id = np.asarray([0.038032, 0.116126, 0.040599, 0.127216, 0.110027, 0.044722, 0.572404])
    fale_pdb_id = np.asarray([0.049771, 0.092979, 0.101701, 0.096957, 0.078241, 0.098920, 0.083070])
        
    male_pdb_wd = np.asarray([0.192638, 0.157932, 0.285125, 0.593328, 0.144632, 0.355875, 1.145780])
    fale_pdb_wd = np.asarray([0.139950, 0.225176, 0.444526, 0.303277, 0.238459, 0.344578, 0.217677])
    
    male_pdb_td = np.asarray([0.031382, 0.190727, 0.399982, 0.501522, 0.218960, 0.440331, 0.383586])
    fale_pdb_td = np.asarray([0.074512, 0.158589, 0.437725, 0.406574, 0.214679, 0.471250, 0.393466])

    male_pdb_vl = np.asarray([0.045448, 0.166080, 0.474586, 0.556328, 0.205569, 0.629141, 0.635638])
    fale_pdb_vl = np.asarray([0.098486, 0.223174, 0.530458, 0.570015, 0.199216, 0.536336, 0.561052])


    colors = ['blue', 'deepskyblue', 'dodgerblue', 'mediumslateblue', 'mediumpurple', 'mediumorchid']
    male_labels = ['RNN LOO', 'RNN SMRF', 'RNN SMSF', 'HRNN LOO', 'HRNN SMRF', 'HRNN SMSF']
    fale_labels = ['RNN LOO', 'RNN RMSF', 'RNN SMSF', 'HRNN LOO', 'HRNN RMSF', 'HRNN SMSF']
    male_models = np.asarray([male_gmr_vl[1:], male_gmr_wd[1:], male_gmr_id[1:], male_gmr_td[1:]])
    fale_models = np.asarray([fale_gmr_vl[1:], fale_gmr_wd[1:], fale_gmr_id[1:], fale_gmr_td[1:]])

    text_labels = ['Velocity', 'Wall Dist.', 'Inter. Dist.', 'Angular Motion']
    ftag = 'smsf_loo_rmsf_smrf_gmr71'


    from evaluate_chase import error_bar_plot
    error_bar_plot(male_models, male_labels, colors, 'histogram', ftag+'_male', N=4, text_labels=text_labels, vmax=1.2)
    error_bar_plot(fale_models, fale_labels, colors, 'histogram', ftag+'_female', N=4, text_labels=text_labels, vmax=1.2)


    from evaluate_chase import error_bar_plot
    ftag = 'smsf_loo_rmsf_smrf_gmr91'
    male_models = np.asarray([male_gmr91_vl[1:], male_gmr91_wd[1:], male_gmr91_id[1:], male_gmr91_td[1:]])
    fale_models = np.asarray([fale_gmr91_vl[1:], fale_gmr91_wd[1:], fale_gmr91_id[1:], fale_gmr91_td[1:]])

    error_bar_plot(male_models, male_labels, colors, 'histogram', ftag+'_male', N=4, text_labels=text_labels, vmax=1.2)
    error_bar_plot(fale_models, fale_labels, colors, 'histogram', ftag+'_female', N=4, text_labels=text_labels, vmax=1.2)


    ftag = 'smsf_loo_rmsf_smrf_pdb'
    male_models = np.asarray([male_pdb_vl[1:], male_pdb_wd[1:], male_pdb_id[1:], male_pdb_td[1:]])
    fale_models = np.asarray([fale_pdb_vl[1:], fale_pdb_wd[1:], fale_pdb_id[1:], fale_pdb_td[1:]])

    error_bar_plot(male_models, male_labels, colors, 'histogram', ftag+'_male', N=4, text_labels=text_labels, vmax=1.2)
    error_bar_plot(fale_models, fale_labels, colors, 'histogram', ftag+'_female', N=4, text_labels=text_labels, vmax=1.2)


    male_gmr_id = np.asarray([0.047116, 0.074685, 0.070098, 0.070615, 0.042751])
    fale_gmr_id = np.asarray([0.034572, 0.084840, 0.077861, 0.052340, 0.207916])
        
    male_gmr_wd = np.asarray([0.044638, 0.387907, 0.392004, 0.350534, 0.309177])
    fale_gmr_wd = np.asarray([0.074299, 0.424486, 0.433829, 0.283102, 0.252239])
    
    male_gmr_td = np.asarray([0.015093, 0.248053, 0.226069, 0.280222, 0.345459])
    fale_gmr_td = np.asarray([0.023222, 0.359300, 0.299847, 0.315408, 0.296952])

    male_gmr_vl = np.asarray([0.028586, 0.421533, 0.476608, 0.554322, 0.683169])
    fale_gmr_vl = np.asarray([0.039923, 0.469489, 0.432701, 0.423260, 0.454214])

    male_labels = ['RNN25', 'RNN50', 'RNN100', 'RNN150']
    fale_labels = ['RNN25', 'RNN50', 'RNN100', 'RNN150']

    ftag = 'rnn_nsteps_gmr'
    male_models = np.asarray([male_gmr_vl[1:], male_gmr_wd[1:], male_gmr_id[1:], male_gmr_td[1:]])
    fale_models = np.asarray([fale_gmr_vl[1:], fale_gmr_wd[1:], fale_gmr_id[1:], fale_gmr_td[1:]])

    error_bar_plot(male_models, male_labels, colors, 'histogram', ftag+'_male', N=4, text_labels=text_labels, vmax=1.2)
    error_bar_plot(fale_models, fale_labels, colors, 'histogram', ftag+'_female', N=4, text_labels=text_labels, vmax=1.2)


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
    parser.add_argument('--t0', type=int, default=0)
    parser.add_argument('--t1', type=int, default=30320) 
    parser.add_argument('--tsim', type=int, default=30)
    parser.add_argument('--x_dim', type=int, default=152)
    parser.add_argument('--atype', type=str, default='diff_models', choices=['diff_models', 'diff_nstep', 'diff_simtype'])
    parser.add_argument('--basepath', type=str, default='/groups/branson/home/imd/Documents/janelia/research/fly_behaviour_sim/71g01/')
    parser.add_argument('--datapath', type=str, default='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/')
    
    return check_args(parser.parse_args())


if __name__ == '__main__':

    args = parse_args()
    args.y_dim = args.num_bin*args.num_mfeat
    basepath0 = args.basepath 
    os.makedirs('./figs/histogram/', exist_ok=True)

    hyper_param  = [(0.05, 0.01, 0, 0), (0.175/10, 0.04, 0, 0), (0.0175, 0.1, 0, 0), (4.76, 0.4, 1, 0), (4.76, 0.03, 1, 0)]
    hyper_labels = ['Velocity', 'Delta Theta', 'Wing Angle', 'Wall Distance', 'Inter Dist']
    hyper_keys   = ['vel', 'dtheta', 'wing_angle', 'wd', 'inter_dist']

    #plot_bar_anal()
    #import pdb; pdb.set_trace()

    data_gmr_tr_male, data_gmr_te_male, data_gmr91_tr_male, data_gmr91_te_male, data_pdb_tr_male, data_pdb_te_male = get_data_stats(args.t0, args.t1, gender='male')
    data_gmr_tr_fale, data_gmr_te_fale, data_gmr91_tr_fale, data_gmr91_te_fale, data_pdb_tr_fale, data_pdb_te_fale = get_data_stats(args.t0, args.t1, gender='fale')

    if 'diff_models' == args.atype:
        lr50_male, conv4_cat50_male, \
                rnn50_male, rnn50_perc_male, skip_cat50_male,\
                = get_stats(args.t0, args.t1, args.dtype, gender='male')
        lr50_fale, conv4_cat50_fale, \
                rnn50_fale, rnn50_perc_fale, skip_cat50_fale,\
                = get_stats(args.t0, args.t1, args.dtype, gender='fale')
    elif 'diff_nstep' == args.atype:
        rnn25_male, rnn50_male, rnn100_male, rnn150_male, \
                skip_cat25_male, skip_cat50_male,\
                skip_cat100_male, skip_cat150_male\
                = get_stats_nstep(args.t0, args.t1, args.dtype, gender='male')
        rnn25_fale, rnn50_fale, rnn100_fale, rnn150_fale, \
                skip_cat25_fale, skip_cat50_fale,\
                skip_cat100_fale, skip_cat150_fale \
                = get_stats_nstep(args.t0, args.t1, args.dtype, gender='fale')
    elif 'diff_simtype' == args.atype:
        rnn_loo50_m, rnn_rmsf50_m, rnn_smrf50_m, rnn_smsf50_m, \
            skip_loo50_m, skip_rmsf50_m, skip_smrf50_m, skip_smsf50_m =\
            get_stats_ind(args.t0, args.t1, args.dtype, testvideo_num=0, gender='male')

        rnn_loo50_f, rnn_rmsf50_f, rnn_smrf50_f, rnn_smsf50_f, \
            skip_loo50_f, skip_rmsf50_f, skip_smrf50_f, skip_smsf50_f =\
            get_stats_ind(args.t0, args.t1, args.dtype, testvideo_num=0, gender='fale')


    for (bin_width, keep_perc, left_chop, right_chop), hyper_label, \
            hyper_key in zip(hyper_param, hyper_labels, hyper_keys):

        print(hyper_key)

        if 'diff_models' == args.atype:

            colors=['black', 'gray', 'navy', 'silver',\
                    'red',  'green', \
                    'deepskyblue', 'mediumpurple', \
                    'magenta', 'orange', 'cyan', \
                    'darkgreen', \
                    ]

            if 'gmr' == args.dtype:
                labels = ['TEST DATA', 'TRAIN DATA', 'R91B01', 'CONTROL',\
                                    'LINEAR', 'CNN50', 'RNN50', 'RNN50 PERC', 'HRNN50']
                data_male = [data_gmr_te_male, data_gmr_tr_male, \
                        data_gmr91_te_male, data_pdb_te_male, lr50_male, \
                        conv4_cat50_male, rnn50_male, rnn50_perc_male, skip_cat50_male]
                data_fale = [data_gmr_te_fale, data_gmr_tr_fale, 
                        data_gmr91_te_fale, data_pdb_te_fale, lr50_fale, \
                        conv4_cat50_fale, rnn50_fale, rnn50_perc_fale, skip_cat50_fale]
   
            elif 'gmr91' == args.dtype:
                labels = ['TEST DATA', 'TRAIN DATA', 'R71G01', 'CONTROL',\
                                    'LINEAR', 'CNN50', 'RNN50', 'HRNN50']
                data_male = [data_gmr91_te_male, data_gmr91_tr_male, \
                        data_gmr_te_male, data_pdb_te_male, lr50_male, \
                        conv4_cat50_male, rnn50_male, skip_cat50_male]
                data_fale = [data_gmr_te_fale, data_gmr_tr_fale, 
                        data_gmr91_te_fale, data_pdb_te_fale, lr50_fale, \
                        conv4_cat50_fale, rnn50_fale, skip_cat50_fale]
            
            elif 'pdb' in args.dtype:
                labels = ['TEST DATA', 'TRAIN DATA', 'R71G01', 'R91B01',\
                                    'LINEAR', 'CNN50', 'RNN50', 'HRNN50']
                data_male = [data_pdb_te_male, data_pdb_tr_male, \
                        data_gmr_te_male, data_gmr91_te_male, lr50_male, \
                        conv4_cat50_male, rnn50_male, skip_cat50_male]
                data_fale = [data_gmr_te_fale, data_gmr_tr_fale, 
                        data_gmr91_te_fale, data_pdb_te_fale, lr50_fale, \
                        conv4_cat50_fale, rnn50_fale, skip_cat50_fale]
            
            histo_diff(data_male, labels, hyper_key, bin_width=bin_width, \
                        keep_perc=keep_perc, left_chop=left_chop, \
                        right_chop=right_chop, plotF=1,\
                        ftag='%s_%s_male_allmodel' % (hyper_label,args.dtype), \
                        fdir='histogram/', colors=colors)

            histo_diff(data_fale, labels, hyper_key, bin_width=bin_width, \
                        keep_perc=keep_perc, left_chop=left_chop, \
                        right_chop=right_chop, plotF=1,\
                        ftag='%s_%s_fale_allmodel' % (hyper_label,args.dtype), \
                        fdir='histogram/', colors=colors)

        elif 'diff_nstep' == args.atype:
            colors=['black', 'gray', 'salmon', 'tomato', 'orangered',\
                    'deeppink', 'hotpink', 'palevioletred', \
                    ]


            labels = ['TEST DATA', 'TRAIN DATA', 'RNN_CAT25', 'RNN_CAT50', 'RNN_CAT100', 'RNN_CAT150']
            data_male = [data_gmr_te_male, data_gmr_tr_male, rnn25_male, rnn50_male, rnn100_male, rnn150_male]
            data_fale = [data_gmr_te_fale, data_gmr_tr_fale, rnn25_fale, rnn50_fale, rnn100_fale, rnn150_fale]
    
            histo_diff(data_male, labels, hyper_key, bin_width=bin_width, \
                        keep_perc=keep_perc, left_chop=left_chop, \
                        right_chop=right_chop, plotF=1,\
                        ftag='%s_%s_rnn_male' % (hyper_label, args.dtype), \
                        fdir='histogram/', colors=colors)
    
            histo_diff(data_fale, labels, hyper_key, bin_width=bin_width, \
                        keep_perc=keep_perc, left_chop=left_chop, \
                        right_chop=right_chop, plotF=1,\
                        ftag='%s_%s_rnn_fale' % (hyper_label, args.dtype), \
                        fdir='histogram/', colors=colors)


            labels = ['TEST DATA', 'TRAIN DATA', 'HRNN_CAT25', 'HRNN_CAT50', 'HRNN_CAT100', 'HRNN_CAT150']
            data_male = [data_gmr_te_male, data_gmr_tr_male, skip_cat25_male, skip_cat50_male, skip_cat100_male, skip_cat150_male]
            data_fale = [data_gmr_te_fale, data_gmr_tr_fale, skip_cat25_fale, skip_cat50_fale, skip_cat100_fale, skip_cat150_fale]
   
            colors=['black', 'gray', 'slateblue', 'mediumpurple', 
                    'mediumslateblue', 'blueviolet', 'darkorchid', 'mediumorchid']


            histo_diff(data_male, labels, hyper_key, bin_width=bin_width, \
                        keep_perc=keep_perc, left_chop=left_chop, \
                        right_chop=right_chop, plotF=1,\
                        ftag='%s_hrnn_male' % hyper_label, \
                        fdir='histogram/', colors=colors)
    
            histo_diff(data_fale, labels, hyper_key, bin_width=bin_width, \
                        keep_perc=keep_perc, left_chop=left_chop, \
                        right_chop=right_chop, plotF=1,\
                        ftag='%s_hrnn_fale' % hyper_label, \
                        fdir='histogram/', colors=colors)

 
        elif 'diff_simtype' == args.atype:
            colors=['black', 'gray', 'limegreen', 'green', 'darkgreen',\
                    'blue', 'deepskyblue', 'mediumpurple', \
                    'magenta', 'orange', 'cyan', \
                     \
                    ]

            labels = ['TEST DATA', 'TRAIN DATA',  
                        'RNN LOO', 'RNN SMRF', 'RNN SMSF',\
                        'SKIP LOO','SKIP SMRF', 'SKIP SMSF']
            data_male = [ data_gmr_te_male, data_gmr_tr_male, \
                    rnn_loo50_m, rnn_smrf50_m, rnn_smsf50_m, \
                    skip_loo50_m, skip_smrf50_m, skip_smsf50_m]

  
            histo_diff(data_male, labels, hyper_key, bin_width=bin_width, \
                        keep_perc=keep_perc, left_chop=left_chop, \
                        right_chop=right_chop, plotF=1,\
                        ftag='%s_%s_male_smrf' % (hyper_key, args.dtype), \
                        fdir='histogram/', colors=colors)


            labels = ['TEST DATA', 'TRAIN DATA',  
                        'RNN LOO', 'RNN RMSF', 'RNN SMSF',\
                        'SKIP LOO', 'SKIP RMSF', 'SKIP SMSF']
            data_fale = [data_gmr_te_fale, data_gmr_tr_fale, \
                    rnn_loo50_f, rnn_rmsf50_f, rnn_smsf50_f,\
                    skip_loo50_f, skip_rmsf50_f, skip_smsf50_f]
 
            histo_diff(data_fale, labels, hyper_key, bin_width=bin_width, \
                        keep_perc=keep_perc, left_chop=left_chop, \
                        right_chop=right_chop, plotF=1,\
                        ftag='%s_%s_fale_rmsf' % (hyper_key, args.dtype), \
                        fdir='histogram/', colors=colors)

 


