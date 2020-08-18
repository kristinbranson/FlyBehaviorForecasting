import numpy as np
import h5py, copy, os, sys, pickle
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import scipy.io as sio

import torch
from tqdm import tqdm
from util_fly import *
from gen_dataset import load_eyrun_data, combine_vision_data, \
                        video16_path, gender_classify, load_vision
from simulate_rnn import get_nstep_comparison_rnn, get_simulate_fly
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

def plot_nstep_errors(models, dtype, video_list, t_dim=50, gender=0, t0=21224, tsim=15, \
        visionF=1, labels=None, vlowmax=None,
        colors=['blue','red','green', 'magenta', 'purple', 'black']):

    if labels is None: labels = models
    fname0 = 'eyrun_simulate_data.mat'
    basepath='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'
    parentpath='/groups/branson/home/imd/Documents/janelia/research/fly_behaviour_sim/71g01'


    #ftag = ['velo', 'pos', 'bodyAng', 'wingang']
    ftag = ['velo', 'pos']
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
                    fname = './metrics/'+vpath+'/'+mtype0+'/'+mtype0+'_visionF'+str(visionF)+'_'+str(t0)+'t0_'+str(t1)+'t1_30tsim.npy'
                    err_test = np.load(fname)
                else:

                    err_tests = []
                    for kk in range(10):
                        if 'rnn' in mtype0:
                            fname = './metrics/'+vpath+'/'+mtype0+'/'+mtype0+'_'+str(t0)+'t0_'+str(t1)+'t1_30tsim_150000epoch'
                        elif 'skip' in mtype0:
                            fname = './metrics/'+vpath+'/'+mtype0+'/'+mtype0+'_'+str(t0)+'t0_'+str(t1)+'t1_30tsim_150000epoch'
                        else:
                            fname = './metrics/'+vpath+'/'+mtype0+'/'+mtype0+'_visionF'+str(visionF)+'_'+str(t0)+'t0_'+str(t1)+'t1_30tsim'
                            if ('pdb' in dtype or 'gmr91' in dtype) and 'lr' in mtype0: fname += '_200000epoch'
 
                        err_test0 = np.load(fname + '_%dfold.npy' % kk)
                        #if dtype == 'gmr91': err_test0 = err_test0[:,:,:,1:]
                        err_tests.append(err_test0)

                        import pdb; pdb.set_trace()
                        print(fname, mtype0, err_test0.shape)
                    #err_test0 = np.min(err_tests, axis=0)
                    #err_test = err_test0[:]

def data4Kristin(mtype0, vpath, t_dim=50, gender='male', t0=0, tsim=30, visionF=1):

    fname0 = 'eyrun_simulate_data.mat'
    basepath='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'
    parentpath='/groups/branson/home/imd/Documents/janelia/research/fly_behaviour_sim/71g01'


    #ftag = ['velo', 'pos', 'bodyAng', 'wingang']
    #ftag = ['velo', 'pos']
    #for j in range(len(ftag)):

    pos_errs = []
    pos_stds = []

    print ('%s' % (vpath))
    matfile = basepath+vpath+fname0
    (trx,motiondata,params,basesize) = load_eyrun_data(matfile)
    t1= trx['x'].shape[0] - tsim
    male_ind, female_ind = gender_classify(basesize['majax'])        

    pos_err_models = []
    pos_std_models = []

    err_tests = []

    if 'const' == mtype0 or 'copy' == mtype0 or 'zero' == mtype0:
        fname = './metrics/'+vpath+'/'+mtype0+'/'+mtype0+'_visionF'+str(visionF)+'_'+str(t0)+'t0_'+str(t1)+'t1_30tsim.npy'
        err_test = np.load(fname)
        err_tests.append(err_test)
    else:

        for kk in range(10):
            if 'rnn' in mtype0:
                fname = './metrics/'+vpath+'/'+mtype0+'/'+mtype0+'_'+str(t0)+'t0_'+str(t1)+'t1_30tsim_150000epoch'
            elif 'skip' in mtype0:
                fname = './metrics/'+vpath+'/'+mtype0+'/'+mtype0+'_'+str(t0)+'t0_'+str(t1)+'t1_30tsim_150000epoch'
            else:
                fname = './metrics/'+vpath+'/'+mtype0+'/'+mtype0+'_visionF'+str(visionF)+'_'+str(t0)+'t0_'+str(t1)+'t1_30tsim'
                if ('pdb' in dtype or 'gmr91' in dtype) and 'lr' in mtype0: fname += '_200000epoch'
        
            err_test0 = np.load(fname + '_%dfold.npy' % kk)
            err_tests.append(err_test0)

    if 'lr' in mtype:
        for fold_k in range(10):
            err_stds = err_tests[fold_k].mean(axis=-1).std(axis=1)
            for feat in range(err_stds.shape[0]):
                for step in range(err_stds.shape[1]):
                    eps = np.random.normal(0, err_stds[feat,step], \
                            size=(err_tests[0].shape[1], err_tests[0].shape[-1]))

                    err_tests[fold_k][feat,:,step,:] += eps
    
    return err_tests


def adjust_rnn_skip(err_tests):

    new_err_tests = []
    for err_test in err_tests:
        new_err_tests.append(err_test[:,50:])
    return new_err_tests

dtype='gmr91'
tsim=30
gender=1
video_list = video16_path[dtype]
mtype='lr50'

if __name__ == '__main__':

    #colors = ['black', 'silver', 'red', 'green', 'deepskyblue', 'mediumpurple']
    #models = [ 'const', 'zero', 'lr50', 'conv4_cat50', 'rnn50', 'skip50']
    #labels = [ 'CONST', 'HALT', 'LINEAR'  , 'CNN', 'RNN', 'HRNN']
    #vlowmax = [[2, 4.75], [5, 75], [0.1,0.9], [0.04,0.08], [0.04,0.08]]

    #plot_nstep_errors(models, dtype, video_list, t_dim=50, gender=gender, \
    #        t0=0, tsim=30, visionF=1, labels=labels, colors=colors, vlowmax=vlowmax)

    
    t0, tsim, t_dim  = 0, 30, 50

    DEBUG = 0
    basepath='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'
    #vpath = video_list[TEST][0]


    for testvideo_num in range(0,len(video_list[TEST])):

        frame_index = []
        vpath = video_list[TEST][testvideo_num]
        matfile = basepath+vpath+'eyrun_simulate_data.mat'
        (trx,motiondata,params,basesize) = load_eyrun_data(matfile)


        t1= trx['x'].shape[0] - tsim
        for ii, t in enumerate(range(t0+t_dim,t1,tsim)):
            frame_index.append(t)
        frame_index = frame_index[50:]

        mtype='lr50'
        os.makedirs('./forKristin/nstep/%s/' % (vpath), exist_ok=True)   
        fname = './forKristin/nstep/%s/%s_%dt0_%dt1_%dtsim' % (vpath, mtype, t0, t1, tsim)
        err_tests = data4Kristin(mtype, vpath, t_dim=t_dim, gender=gender, t0=t0, tsim=tsim, visionF=1)
        import pdb; pdb.set_trace()
        assert err_tests[0].shape[1] == len(frame_index), 'length must be the same'
        pickle.dump([vpath, frame_index, err_tests], open(fname+'.pkl', 'wb'))


        mtype='conv4_cat50'
        os.makedirs('./forKristin/nstep/%s/' % (vpath), exist_ok=True)   
        fname = './forKristin/nstep/%s/%s_%dt0_%dt1_%dtsim' % (vpath, mtype, t0, t1, tsim)
        err_tests = data4Kristin(mtype, vpath, t_dim=t_dim, gender=gender, t0=t0, tsim=tsim, visionF=1)
        assert err_tests[0].shape[1] == len(frame_index), 'length must be the same'
        pickle.dump([vpath, frame_index, err_tests], open(fname+'.pkl', 'wb'))


        mtype='rnn50'
        os.makedirs('./forKristin/nstep/%s/' % (vpath), exist_ok=True)   
        fname = './forKristin/nstep/%s/%s_%dt0_%dt1_%dtsim' % (vpath, mtype, t0, t1, tsim)
        err_tests = data4Kristin(mtype, vpath, t_dim=t_dim, gender=gender, t0=t0, tsim=tsim, visionF=1)
        err_tests = adjust_rnn_skip(err_tests)
        assert err_tests[0].shape[1] == len(frame_index), 'length must be the same'
        pickle.dump([vpath, frame_index, err_tests], open(fname+'.pkl', 'wb'))


        mtype='skip50'
        os.makedirs('./forKristin/nstep/%s/' % (vpath), exist_ok=True)   
        fname = './forKristin/nstep/%s/%s_%dt0_%dt1_%dtsim' % (vpath, mtype, t0, t1, tsim)
        err_tests = data4Kristin(mtype, vpath, t_dim=t_dim, gender=gender, t0=t0, tsim=tsim, visionF=1)
        err_tests = adjust_rnn_skip(err_tests)
        assert err_tests[0].shape[1] == len(frame_index), 'length must be the same'
        pickle.dump([vpath, frame_index, err_tests], open(fname+'.pkl', 'wb'))



        mtype='zero'
        os.makedirs('./forKristin/nstep/%s/' % (vpath), exist_ok=True)   
        fname = './forKristin/nstep/%s/%s_%dt0_%dt1_%dtsim' % (vpath, mtype, t0, t1, tsim)
        err_tests = data4Kristin(mtype, vpath, t_dim=t_dim, gender=gender, t0=t0, tsim=tsim, visionF=1)
        assert err_tests[0].shape[1] == len(frame_index), 'length must be the same'
        pickle.dump([vpath, frame_index, err_tests], open(fname+'.pkl', 'wb'))


        mtype='const'
        os.makedirs('./forKristin/nstep/%s/' % (vpath), exist_ok=True)   
        fname = './forKristin/nstep/%s/%s_%dt0_%dt1_%dtsim' % (vpath, mtype, t0, t1, tsim)
        err_tests = data4Kristin(mtype, vpath, t_dim=t_dim, gender=gender, t0=t0, tsim=tsim, visionF=1)
        assert err_tests[0].shape[1] == len(frame_index), 'length must be the same'
        pickle.dump([vpath, frame_index, err_tests], open(fname+'.pkl', 'wb'))







