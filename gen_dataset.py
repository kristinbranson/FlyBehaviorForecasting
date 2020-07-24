import os, sys, h5py, math, argparse

import numpy as np
import scipy.io as sio

from util_fly import motion2binidx 
from util import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RANDSEED=0
RNG = np.random.RandomState(RANDSEED)

MALE=0
FEMALE=1
NUM_BIN=51
NUM_MFEAT=8

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

basepath = '/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'

pbd_4u = [[
    'pBDPGAL4U_TrpA_Rig1Plate10BowlA_20110323T114748/',\
    'pBDPGAL4U_TrpA_Rig1Plate10BowlC_20110610T160613/',\
    #'pBDPGAL4U_TrpA_Rig1Plate15BowlA_20121220T161640',\ #21
    'pBDPGAL4U_TrpA_Rig1Plate15BowlB_20120203T150713/',\
    'pBDPGAL4U_TrpA_Rig1Plate15BowlB_20120713T083042/',\
    'pBDPGAL4U_TrpA_Rig1Plate15BowlC_20110831T100911/',\
    'pBDPGAL4U_TrpA_Rig1Plate15BowlD_20111117T092639/'],\
    [\
    'pBDPGAL4U_TrpA_Rig1Plate15BowlD_20120425T130405/',\
    #'pBDPGAL4U_TrpA_Rig2Plate14BowlB_20110504T102739',\ #19
    'pBDPGAL4U_TrpA_Rig2Plate14BowlB_20110720T140728/'],\
    [
    'pBDPGAL4U_TrpA_Rig2Plate17BowlA_20120315T142016/',\
    'pBDPGAL4U_TrpA_Rig2Plate17BowlA_20120823T144547/',\
    #'pBDPGAL4U_TrpA_Rig2Plate17BowlC_20111007T140348',\ #19
    #'pBDPGAL4U_TrpA_Rig2Plate17BowlC_20120601T150533',\ #19
    'pBDPGAL4U_TrpA_Rig2Plate17BowlD_20111209T134749/',\
    'pBDPGAL4U_TrpA_Rig2Plate17BowlD_20120104T103048/']
    ]

gmr_71 = [[\
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlB_20110707T154653/',
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlC_20110707T154934/', #19
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlA_20110916T155922/',
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlD_20110916T155353/',
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlA_20110921T085351/'],\
    [\
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110921T085346/',\
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/',
    ],
    [\
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/', #19
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110916T155917/',\
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110921T084823/']]
    #'GMR_71G01_AE_01_TrpA_Rig1Plate15BowlA_20120316T144027/',
    #'GMR_71G01_AE_01_TrpA_Rig1Plate15BowlB_20120316T144030/', #21
    #'GMR_71G01_AE_01_TrpA_Rig1Plate15BowlC_20120316T144000/', #30
    #'GMR_71G01_AE_01_TrpA_Rig1Plate15BowlD_20120316T144003/', #27

gmr_91 = [[\
    'GMR_91B01_AE_01_TrpA_Rig1Plate15BowlA_20120329T131415/',
    'GMR_91B01_AE_01_TrpA_Rig1Plate15BowlB_20120329T131418/', #19
    'GMR_91B01_AE_01_TrpA_Rig1Plate15BowlC_20120329T131338/',
    'GMR_91B01_AE_01_TrpA_Rig1Plate15BowlD_20120329T131343/',
    'GMR_91B01_AE_01_TrpA_Rig2Plate17BowlA_20120614T085804/'],\
    [\
    'GMR_91B01_AE_01_TrpA_Rig2Plate17BowlB_20120614T085806/',\
    'GMR_91B01_AE_01_TrpA_Rig2Plate17BowlD_20120614T090114/' #22
    ],
    [\
    'GMR_91B01_AE_01_TrpA_Rig2Plate17BowlA_20120614T085804/',\
    'GMR_91B01_AE_01_TrpA_Rig2Plate17BowlC_20120614T090112/']]

gmr_26 = [[\
    'GMR_26E01_AE_01_TrpA_Rig2Plate14BowlA_20110610T141315/',
    'GMR_26E01_AE_01_TrpA_Rig2Plate14BowlB_20110610T141310/', #19
    'GMR_26E01_AE_01_TrpA_Rig2Plate14BowlD_20110610T141503/',
    'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlA_20120531T140054/',
    'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlB_20120223T101810/'],\
    [\
    'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlB_20120531T140057/',\
    'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlC_20120531T140341/',\
    ],
    [\
    'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlC_20120223T101853/',
    #'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlD_20120223T101856/',\
    'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlD_20120531T140344/']]



video16_path = {'gmr':gmr_71, 'pdb':pbd_4u, 'gmr91':gmr_91, 'gmr26':gmr_26}

video16_july = [[\
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlB_20110707T154653/',
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlC_20110707T154934/'],  
    [
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',\
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/'],\
    [\
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/' #19
    ]]

video16_original = [['GMR_71G01_AE_01_TrpA_Rig2Plate14BowlB_20110707T154653/'],\
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110921T085346/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/',
                    ],
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/', #19
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110916T155917/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110921T084823/']]

video16_v2 = [['GMR_71G01_AE_01_TrpA_Rig2Plate17BowlA_20110916T155922/'],\
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110921T085346/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/',
                    ],
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/', #19
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110916T155917/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110921T084823/']]

video16_v3 = [['GMR_71G01_AE_01_TrpA_Rig2Plate17BowlA_20110921T085351/'],\
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110921T085346/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/',
                    ],
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/', #19
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110916T155917/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110921T084823/']]


video16_v4 = [['GMR_71G01_AE_01_TrpA_Rig2Plate17BowlD_20110916T155353/'],\
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110921T085346/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/',
                    ],
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/', #19
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110916T155917/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110921T084823/']]


video16_v5 = [['GMR_71G01_AE_01_TrpA_Rig2Plate17BowlA_20110921T085351/'],\
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110921T085346/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/',
                    ],
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/', #19
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110916T155917/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110921T084823/']]





TRAIN=0
VALID=1
TEST=2

def load_eyrun_data(matfile):

    f = h5py.File(matfile,'r')
    trx = {}
    trx['x'] = np.array(f['trx_x'])-1.
    trx['y'] = np.array(f['trx_y']) - 1.
    trx['theta'] = np.array(f['trx_theta'])
    trx['a'] = np.array(f['trx_a'])
    trx['b'] = np.array(f['trx_b'])
    trx['l_wing_ang'] = np.array(f['trx_l_wing_ang'])
    trx['l_wing_len'] = np.array(f['trx_l_wing_len'])
    trx['r_wing_ang'] = np.array(f['trx_r_wing_ang'])
    trx['r_wing_len'] = np.array(f['trx_r_wing_len'])

    motiondata = np.array(f['motiondata'])

    params = default_params.copy()
    params['mindist'] = f['mindist'][0]
    params['n_oma'] = int(f['n_oma'][0])
    params['I'] = np.array(f['I'])-1.
    params['J'] = np.array(f['J'])-1.
    params['PPM'] = f['PPM'][0]
    params['FPS'] = f['FPS'][0]
    params['binedges'] = np.array(f['binedges'])
    params['bincenters'] = np.array(f['bincenters'])
    params['ranges'] = np.array(f['ranges']).flatten()
    params['bg'] = np.array(f['bg']).T

    # arena is a circle, find the parameters
    params['arena_center_x'] = np.mean(params['J'])
    params['arena_center_y'] = np.mean(params['I'])
    
    # fly actually can't get all the way out to the arena boundary
    dctr =  np.sqrt( (trx['x']-params['arena_center_x'])**2. + \
                     (trx['y'] - params['arena_center_y'])**2. )
    params['arena_radius'] = np.nanmax(dctr)

    # median
    basesize = {}
    basesize['majax'] = np.nanmedian(trx['a'],axis=0)
    basesize['minax'] = np.nanmedian(trx['b'],axis=0)
    basesize['awing1'] = -np.nanmedian(trx['l_wing_ang'],axis=0)
    basesize['lwing1'] = np.nanmedian(trx['l_wing_len'],axis=0)
    basesize['awing2'] = np.nanmedian(trx['r_wing_ang'],axis=0)
    basesize['lwing2'] = np.nanmedian(trx['r_wing_len'],axis=0)

    return (trx,motiondata,params,basesize)




def load_vision(matfile):
    
    f = h5py.File(matfile,'r')
    v_data = np.asarray(f['vision']['ndata'])
    _, _, T, F = v_data.shape
    return v_data.reshape([-1, T, F]).transpose(1,2,0)


def blur_bins(bdata_idx, T,F,M, onehotF=1, num_bin=51):

    bdata_idx = bdata_idx.flatten()
    bdata = np.zeros((T*F*M,num_bin), dtype='float32')
    std2 = 1/4
    if not onehotF:
        for n in range(T*F*M):
            ind_percentile = bdata_idx[n]
            bdata[n, ind_percentile] = np.exp(0)/ np.sqrt(np.pi*2)
            bdata[n, np.minimum(num_bin-1, ind_percentile+1)] = np.exp(-std2*1/2)/ np.sqrt(np.pi*2)
            bdata[n, np.maximum(0,ind_percentile-1)] = np.exp(-std2*1/2)/ np.sqrt(np.pi*2)
            bdata[n, np.minimum(num_bin-1, ind_percentile+2)] = np.exp(-std2*1) / np.sqrt(np.pi*2) 
            bdata[n, np.maximum(0,ind_percentile-2)] = np.exp(-std2*1) / np.sqrt(np.pi*2) 
            bdata[n, np.minimum(num_bin-1, ind_percentile+3)] = np.exp(-std2*3/2) / np.sqrt(np.pi*2) 
            bdata[n, np.maximum(0,ind_percentile-3)] = np.exp(-std2*3/2) / np.sqrt(np.pi*2) 
            bdata[n, np.minimum(num_bin-1, ind_percentile+3)] = np.exp(-std2*2) / np.sqrt(np.pi*2) 
            bdata[n, np.maximum(0,ind_percentile-3)] = np.exp(-std2*2) / np.sqrt(np.pi*2) 
        bdata = bdata / np.sum(bdata, axis=1)[:,None]
    else:
        bdata[np.arange(T*F*M), bdata_idx.flatten()] = 1

    return bdata


def percentile_bin(mdata, bins, num_bin=51, onehotF=0):

    N, F, D = mdata.shape
    mdata2 = mdata.reshape([N*F, NUM_MFEAT])
    mbins = np.zeros((N*F, NUM_MFEAT, num_bin), dtype='float32')
    mu_bin, percentiles = [], []
    for d in range(NUM_MFEAT):
        #tmp = argpercentile(mdata2[i], np.linspace(0,100,50), axis=0)
        #percentile = np.percentile(mdata2[:,d], np.linspace(0,100,num_bin+1), axis=0)
        ind_percentile = np.searchsorted(bins, mdata2[:,d])
        if onehotF:
            mbins[np.arange(N*F), d, ind_percentile] = 1
        else:
            mbins[np.arange(N*F), d, ind_percentile] = 0.78
            mbins[np.arange(N*F), d, np.minimum(num_bin-1, ind_percentile+1)] = 0.10
            mbins[np.arange(N*F), d, np.maximum(0,ind_percentile-1)] = 0.10
            mbins[np.arange(N*F), d, np.minimum(num_bin-1, ind_percentile+2)] = 0.045
            mbins[np.arange(N*F), d, np.maximum(0,ind_percentile-2)] = 0.045 
        percentiles.append(percentile)
    mbins = mbins / np.sum(mbins, axis=2)[:,:,None]
    mbins = mbins.reshape([N,F,D,num_bin])
    return mbins, np.asarray(percentiles)


def create_binned_dataset(maxtime=30320, \
        genderF=2, concatF=1, onehotF=0):

    matfile = '/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Code/eyrun_simulate_data.mat'
    (trx,motiondata,params,basesize) = load_eyrun_data(matfile)
    motion_bin = params['binedges'].T

    #Vision Data
    vdata = np.load('/groups/branson/home/imd/Documents/data/fly_tracking/oct8/vdata.npy')
    bdata_idx = np.load('/groups/branson/home/imd/Documents/data/fly_tracking/oct8/bdata.npy')
    T, F, D = vdata.shape

    #Motion Data
    M = bdata_idx.shape[-1]
    mdata = motiondata.transpose([1,2,0])[:-1,:,:]
    mdata[np.isnan(mdata)] = 0.


    male_ind = np.arange(10)
    female_ind = np.arange(10,20)
    bdata = blur_bins(bdata_idx, T,F,M, onehotF=onehotF)


    T, F, M = mdata.shape
    bdata = bdata.reshape([T,F,M*NUM_BIN])
    male_ind, female_ind = gender_classify(basesize['majax'])
    if genderF==MALE:
        bdata = bdata[:, male_ind, :]
        vdata = vcdata[:, male_ind, :]
        mdata = mdata[:,male_ind,:]

    elif genderF==FEMALE:
        bdata = bdata[:, female_ind, :]
        vdata = vcdata[:, female_ind, :]
        mdata = mdata[:,female_ind,:]



    return preprocess_dataset(vdata,bdata, mdata, \
                        maxtime=maxtime, concatF=concatF)


def preprocess_dataset(vdata, bdata, mdata, \
                    maxtime, concatF=0):

    #Bin Data
    #import datahandler2 as dh
    #bdata, _, _, _, bins, _ = dh.load_fly_data_entire(dataset=4, feature_type=2, window_size=1, n_bins=tr_config['num_bin'])
    #bdata = bdata.reshape([NUM_FLY,(T+1),-1]).transpose([1,0,2])[1:,:,:]
    #import pdb; pdb.set_trace()

   
    T, F, D = vdata.shape
    vdata = vdata.reshape([T*F,-1])
    mdata = mdata.reshape([T*F,-1])
    vdata[np.isinf(vdata)] = 0.

    if concatF:
        xdata   = np.hstack([vdata, mdata]).reshape([T,F,-1])
        X_train = [xdata[:int(0.7*maxtime)], bdata[:int(0.7*maxtime)]]
        X_test  = [xdata[int(0.7*maxtime):int(0.9*maxtime)], \
                   bdata[int(0.7*maxtime):int(0.9*maxtime)]]
        X_valid = [xdata[int(0.9*maxtime):], bdata[int(0.9*maxtime):]]

        return X_train, X_valid, X_test
    
    else:
        X_train = [ vdata.reshape([T,F,-1])[:int(0.7*maxtime)], \
                    mdata.reshape([T,F,-1])[:int(0.7*maxtime)], \
                    bdata[:int(0.7*maxtime)]]
        X_test  = [ vdata.reshape([T,F,-1])[int(0.7*maxtime):], \
                    mdata.reshape([T,F,-1])[int(0.7*maxtime):], \
                    bdata[int(0.7*maxtime):]]
 
        return X_train, X_test


def batch_flytraj(data, tau=25, N=50000, itype='full'):

    xdata, ydata = data
    maxTime, F, D = xdata.shape

    vsources, vtargets = [], []
    msources, mtargets = [], []

    tt = (maxTime-1)/tau 
    i=0
    while i < N:
        for fly_i in range(F):

            starting_pt = (RNG.randint(tt, size=1) * tau)[0]
            if starting_pt+2*tau < maxTime:

                if itype == 'full':
                    vsamples = xdata[starting_pt:starting_pt+2*tau, fly_i, :]
                elif itype == 'mv':
                    vsamples = \
                        np.concatenate(
                            [xdata[starting_pt:starting_pt+2*tau, fly_i, :72],\
                             xdata[starting_pt:starting_pt+2*tau, fly_i, 144:]], axis=1)
                elif itype == 'mc':
                    vsamples = xdata[starting_pt:starting_pt+2*tau, fly_i, 72:]
                   
                msamples = ydata[starting_pt:starting_pt+2*tau, fly_i, :]
                msamples[np.isnan(msamples)] = 0.

                vsources.append(vsamples)
                msources.append(msamples)
                i += 1

    vsources = np.asarray(vsources, dtype='float32').transpose(1, 0, 2)
    msources = np.asarray(msources, dtype='float32').transpose(1, 0, 2)

    return [vsources, msources]


def batch_autoreg_flytraj(data, tau=7, N=50000, itype='full'):

    xdata, ydata    = data
    maxTime, F, D   = xdata.shape
    sources, targets= [], []

    i   =  0
    tt  = (maxTime-1)
    while i < N:
        for fly_i in range(F):

            starting_pt = (RNG.randint(tt, size=1))[0]
            if starting_pt+tau+1 < maxTime:

                mdata = xdata[starting_pt:starting_pt+tau, fly_i, 144:]
                vdata = xdata[starting_pt+tau, fly_i, :144]
                source = np.concatenate([mdata.flatten(), vdata])
                target = ydata[starting_pt+tau, fly_i, :]
                source[np.isnan(source)] = 0.

                sources.append(source)
                targets.append(target)
                i += 1

    sources = np.asarray(sources, dtype='float32')
    targets = np.asarray(targets, dtype='float32')
    return [sources, targets]


def gen_fly_regression(maxtime=30320, genderF=2, normF=False, nlogScaleAng=0):

    #from simulate import load_eyrun_data
    #vision_data = np.load('./data/real_fly_vision_data_0t0_30320t1.npy')
    #matfile = '/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Code/eyrun_simulate_data.mat'
    #(trx,motiondata,params,basesize) = load_eyrun_data(matfile)
    #motion_bin = params['binedges'].T

    vpath = 'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlB_20110707T154653/'
    fname = 'eyrun_simulate_data.mat'
    matfile = basepath+vpath+fname
    trx,motiondata,params,basesize = load_eyrun_data(matfile)
    motion_bin = params['binedges'].T
    NN = trx['x'].shape[0]
    vision_matfile = basepath+vpath+'movie-vision.mat'
    vision_data = load_vision(vision_matfile)
    #basepath4 = '/groups/branson/home/imd/Documents/janelia/research/FlyTrajPred_v4/pytorch/'
    #vision_data3 = np.load(basepath4+'data/'+vpath+'/real_fly_vision_data_0t0_%dt1.npy' % NN)
  
    if genderF==MALE:
        mr_data = motiondata[:,1:,:10]
        vr_data = vision_data[1:,:10,:]
    elif genderF==FEMALE: 
        mr_data = motiondata[:,1:,10:]   
        vr_data = vision_data[1:,10:,:]
    else:
        mr_data = motiondata[:,1:,:]
        vr_data = vision_data[1:,:,:]
    mr_data = mr_data.transpose(1,2,0)#[:-1,:,:]
    mr_data[np.isnan(mr_data)]=0.
    
    if nlogScaleAng:
        for i in [2,3,4]:
            a = np.sqrt(mr_data[:,:,i])[mr_data[:,:,i]>0]
            b = -np.sqrt(-mr_data[:,:,i])[mr_data[:,:,i]<0]
            mr_data[:,:,i][mr_data[:,:,i]>0] = a
            mr_data[:,:,i][mr_data[:,:,i]<0] = b


    mean = np.array([ 0.48268917,  0.01496615, 0.00149052, \
                     -0.00068649,  0.10472064, 0.11423372,  \
                      0.06161121,  0.0673553 ])
    std  = np.array([   0.51404585, 0.42001035, 0.3999898 , \
                        0.42712151, 0.40415932,
                        0.40614178, 0.30986684, 0.32055528])

    #Take fwd, side, yaw, wing angle only
    #indice = [0,1,2,4,5]
    #mr_data_ = np.take(mr_data, indice, axis=2)
    #mean_ = np.take(mean, indice)
    #std_  = np.take(std, indice)

    if normF: mr_data_ = (mr_data_ - mean_) /  std_
    ##Divide into Train, Valid, Test trajectories
    X_train = [mr_data[:int(0.7*maxtime)], vr_data[:int(0.7*maxtime)]]
    tmp1, tmp2 = mr_data[int(0.7*maxtime):], vr_data[int(0.7*maxtime):]

    N_tmp = tmp1.shape[0]
    X_valid = [tmp1[:int(0.33*N_tmp)], tmp2[:int(0.33*N_tmp)]]
    X_test  = [tmp1[int(0.33*N_tmp):], tmp2[int(0.33*N_tmp):]]
   
    return X_train, X_valid, X_test


def sample_batch_linear_reg(data, batch_sz, tau, visionF=1):

    mdata, vdata = data
    if visionF:
        fdata = np.concatenate([vdata, mdata], axis=2)
    else:
        fdata = mdata
    i = 0 
    X, Y = [], []
    T, F, D = fdata.shape
    while i < int(batch_sz/F):

        for fly_i in range(F):
    
            start_pt = RNG.randint(T-tau-1, size=1)[0]
            X.append(fdata[start_pt:start_pt+tau, fly_i])
            Y.append(mdata[start_pt+tau+1, fly_i])

        i+=1

    X = np.asarray(X,dtype='float32')
    Y = np.asarray(Y,dtype='float32')
    
    perm = RNG.permutation(X.shape[0])
    return X[perm], Y[perm]


def sample_batch_reg(data, batch_sz, tau, visionF=1):

    mdata, vdata = data
    if visionF:
        fdata = np.concatenate([mdata,vdata], axis=2)
    else:
        fdata = mdata
    i = 0 
    X, Y = [], []
    T, F, D = fdata.shape
    while i < int(batch_sz/F):

        for fly_i in range(F):

            start_pt = RNG.randint(T-tau-2, size=1)[0]
            X.append(fdata[start_pt:start_pt+tau, fly_i])
            Y.append(mdata[start_pt+1:start_pt+tau+1, fly_i])

        i+=1

    X = np.asarray(X)
    Y = np.asarray(Y)
            
    perm = RNG.permutation(X.shape[0])
    return X[perm], Y[perm]


def combine_vision_data(simtrx_curr, flyvisions, num_fly=20, num_burn=4):

    data = []
    N = simtrx_curr['x'].shape[0]
    for fly in range(num_fly):
        points = []
        for ii in range(num_burn, N):

            x = simtrx_curr['x'][ii,fly]
            y = simtrx_curr['y'][ii,fly]
            theta = simtrx_curr['theta'][ii,fly]
            aa = simtrx_curr['a'][ii,fly]
            bb = simtrx_curr['b'][ii,fly]
            l_wing_ang = simtrx_curr['l_wing_ang'][ii,fly]
            r_wing_ang = simtrx_curr['r_wing_ang'][ii,fly]
            l_wing_len = simtrx_curr['l_wing_len'][ii,fly]
            r_wing_len = simtrx_curr['r_wing_len'][ii,fly]
            point = np.hstack([np.asarray([x, y, theta, aa, bb,\
                                l_wing_ang, r_wing_ang, \
                                l_wing_len, r_wing_len]), \
                                flyvisions[fly, ii-num_burn, :]])
            points.append(point)
        data.append(points)
            
    return np.asarray(data)



def normalize_pos(data):
    ''' Noramlize the fly body to initial starting point of view
        x_t = x_t - x_0
        y_t = y_t - y_0
        pos = [cos theta_0, -sin theta_0; sin theta_0 cos theta_0] [x_t, y_t]'''

    new_data = data.copy()
    F, T, D = data.shape
    for ifly in range(F):
        data0  = data[ifly, 0, :2]
        angle0 = data[ifly, 0, 2] * 180 / np.pi
        rotate = [[np.cos(angle0), -np.sin(angle0)],\
                  [np.sin(angle0),  np.cos(angle0)]]

        for tt in range(T):
            data_t = data[ifly, tt, :2]
            new_data_t = data_t - data0
            new_data_t = np.dot(rotate, new_data_t)
            new_data[ifly, tt, :2] = new_data_t
    
    return new_data


def gen_dataset_v2(realdata, fakedata, real_ind, fake_ind):


    RANDSEED=0
    RNG = np.random.RandomState(RANDSEED)


    realdata[np.isnan(realdata)] = 0.
    fakedata[np.isnan(fakedata)] = 0.

    N1 = realdata.shape[0]
    perm_tr = RNG.permutation(N1)
    realdata = realdata[perm_tr]
    real_ind = real_ind[perm_tr]
    reallabel = np.ones(N1)

    Ntr = N1 // 3
    realdata_te, reallabel_te, real_ind_te = realdata[:Ntr], reallabel[:Ntr], real_ind[:Ntr]
    realdata_vl, reallabel_vl, real_ind_vl = realdata[Ntr:Ntr*2], reallabel[Ntr:Ntr*2], real_ind[Ntr:Ntr*2]
    realdata_tr, reallabel_tr, real_ind_tr = realdata[Ntr*2:], reallabel[Ntr*2:], real_ind[Ntr*2:]


    N2 = fakedata.shape[0]
    perm_tr = RNG.permutation(N2)
    fakedata = fakedata[perm_tr]
    fake_ind = fake_ind[perm_tr]
    fakelabel = np.zeros(N2)

    Ntr = N2 // 3
    fakedata_te, fakelabel_te, fake_ind_te = fakedata[:Ntr], fakelabel[:Ntr], fake_ind[:Ntr]
    fakedata_vl, fakelabel_vl, fake_ind_vl = fakedata[Ntr:Ntr*2], fakelabel[Ntr:Ntr*2], fake_ind[Ntr:Ntr*2]
    fakedata_tr, fakelabel_tr, fake_ind_tr = fakedata[Ntr*2:], fakelabel[Ntr*2:], fake_ind[Ntr*2:]

    N = min(realdata_tr.shape[0], fakedata_tr.shape[0]) 

    data_tr  = np.vstack([realdata_tr[:N], fakedata_tr[:N]])
    label_tr = np.hstack([reallabel_tr[:N], fakelabel_tr[:N]])
    perm     = RNG.permutation(data_tr.shape[0])
    data_tr  = data_tr[perm]
    label_tr = label_tr[perm]

    N = min(realdata_vl.shape[0], fakedata_vl.shape[0]) 
    data_vl  = np.vstack([realdata_vl[:N], fakedata_vl[:N]])
    label_vl = np.hstack([reallabel_vl[:N], fakelabel_vl[:N]])
    #perm     = RNG.permutation(data_vl.shape[0])
    #data_vl  = data_vl[perm]
    #label_vl = label_vl[perm]

    N = min(realdata_te.shape[0], fakedata_te.shape[0]) 
    data_te  = np.vstack([realdata_te[:N], fakedata_te[:N]])
    label_te = np.hstack([reallabel_te[:N], fakelabel_te[:N]])
    #perm     = RNG.permutation(data_te.shape[0])
    #data_te  = data_te[perm]
    #label_te = label_te[perm]


    maxv, minv = data_tr.max(0), data_tr.min(0)
    data_tr /=  (maxv-minv)
    data_vl /=  (maxv-minv)
    data_te /=  (maxv-minv)

    data_tr[np.isnan(data_tr)] = 0.
    data_vl[np.isnan(data_vl)] = 0.
    data_te[np.isnan(data_te)] = 0.

    mean = data_tr.mean(0)
    std  = data_tr.std(0)

    data_tr = (data_tr - mean) / std
    data_vl = (data_vl - mean) / std
    data_te = (data_te - mean) / std

    data_tr[np.isnan(data_tr)] = 0.
    data_vl[np.isnan(data_vl)] = 0.
    data_te[np.isnan(data_te)] = 0.

    train_set = [data_tr, label_tr]
    valid_set = [data_vl, label_vl]
    test_set  = [data_te, label_te]
    return train_set, valid_set, test_set, real_ind_te[:N], fake_ind_te[:N]






def gen_dataset(realdata, fakedata, real_ind, fake_ind):

    RANDSEED=0
    RNG = np.random.RandomState(RANDSEED)

    realdata[np.isnan(realdata)] = 0.
    fakedata[np.isnan(fakedata)] = 0.
    realdata = np.concatenate([realdata[:,:,:2], realdata[:,:,8:]], axis=2)
    fakedata = np.concatenate([fakedata[:,:,:2], fakedata[:,:,8:]], axis=2)

    N1 = realdata.shape[0]
    perm_tr = RNG.permutation(N1)
    realdata = realdata[perm_tr]
    real_ind = real_ind[perm_tr]
    reallabel = np.ones(N1)

    Ntr = N1 // 3
    realdata_te, reallabel_te, real_ind_te = realdata[:Ntr], reallabel[:Ntr], real_ind[:Ntr]
    realdata_vl, reallabel_vl, real_ind_vl = realdata[Ntr:Ntr*2], reallabel[Ntr:Ntr*2], real_ind[Ntr:Ntr*2]
    realdata_tr, reallabel_tr, real_ind_tr = realdata[Ntr*2:], reallabel[Ntr*2:], real_ind[Ntr*2:]

    N2 = fakedata.shape[0]
    perm_tr = RNG.permutation(N2)
    fakedata = fakedata[perm_tr]
    fake_ind = fake_ind[perm_tr]
    fakelabel = np.zeros(N2)

    Ntr = N2 // 3
    fakedata_te, fakelabel_te, fake_ind_te = fakedata[:Ntr], fakelabel[:Ntr], fake_ind[:Ntr]
    fakedata_vl, fakelabel_vl, fake_ind_vl = fakedata[Ntr:Ntr*2], fakelabel[Ntr:Ntr*2], fake_ind[Ntr:Ntr*2]
    fakedata_tr, fakelabel_tr, fake_ind_tr = fakedata[Ntr*2:], fakelabel[Ntr*2:], fake_ind[Ntr*2:]

    N = min(realdata_tr.shape[0], fakedata_tr.shape[0]) 
    data_tr  = np.vstack([realdata_tr[:N], fakedata_tr[:N]])
    label_tr = np.hstack([reallabel_tr[:N], fakelabel_tr[:N]])
    perm     = RNG.permutation(data_tr.shape[0])
    data_tr  = data_tr[perm]
    label_tr = label_tr[perm]

    N = min(realdata_vl.shape[0], fakedata_vl.shape[0]) 
    data_vl  = np.vstack([realdata_vl[:N], fakedata_vl[:N]])
    label_vl = np.hstack([reallabel_vl[:N], fakelabel_vl[:N]])
    #perm     = RNG.permutation(data_vl.shape[0])
    #data_vl  = data_vl[perm]
    #label_vl = label_vl[perm]

    N = min(realdata_te.shape[0], fakedata_te.shape[0]) 
    data_te  = np.vstack([realdata_te[:N], fakedata_te[:N]])
    label_te = np.hstack([reallabel_te[:N], fakelabel_te[:N]])
    #frame_ind_te = np.hstack([real_ind[:N], fake_ind[:N]])

    #perm     = RNG.permutation(data_te.shape[0])
    #data_te  = data_te[perm]
    #label_te = label_te[perm]


    #maxv, minv = np.nanmax(data_tr, axis=0), np.nanmax(data_tr, axis=0)
    #data_tr /=  (maxv-minv)
    #data_vl /=  (maxv-minv)
    #data_te /=  (maxv-minv)

    #mean = data_tr.mean(0)
    #std  = data_tr.std(0)

    #data_tr = (data_tr - mean) / std
    #data_vl = (data_vl - mean) / std
    #data_te = (data_te - mean) / std

    mean = data_tr.mean(0)
    std  = data_tr.std(0)

    data_tr = (data_tr - mean) / std
    data_vl = (data_vl - mean) / std
    data_te = (data_te - mean) / std

    data_tr[np.isnan(data_tr)] = 0.
    data_vl[np.isnan(data_vl)] = 0.
    data_te[np.isnan(data_te)] = 0.
    data_tr[np.isinf(data_tr)] = 0.
    data_vl[np.isinf(data_vl)] = 0.
    data_te[np.isinf(data_te)] = 0.

    train_set = [data_tr, label_tr]
    valid_set = [data_vl, label_vl]
    test_set  = [data_te, label_te]

    data_stats = [mean,std]
    return train_set, valid_set, test_set, data_stats, real_ind_te[:N], fake_ind_te[:N]



def feval(input_variable, target_variable, model, optimizer, 
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
    loss, acc = model.loss(input_variable, target_variable)

    if mode=='train':
        loss.backward()
        optimizer.step()

    return loss.item(), acc.item()


def gen_videos():

    fname = 'eyrun_simulate_data.mat'

    for vpath in video16_path[dtype]:
        t0, t1 = 0, 5
        fname='allreal'
        real_male_flies = np.arange(0,10,1)
        real_female_flies = np.arange(10,20,1)
        simulated_male_flies = []
        simulated_female_flies = []

        matfile = basepath+vpath+fname
        from simulate_rnn import simulate_flies
        simulate_flies( real_male_flies, real_female_flies, \
                        simulated_male_flies, simulated_female_flies,\
                        male_model=None, female_model=None, DEBUG=1, \
                        plottrxlen=100, t0=t0, t1=t1, bookkeepingF=False,\
                        vision_save=False, fname=fname, mtype=mtype, \
                        burning=burning, matfile=matfile)
    pass


def gender_classify(basesize):
    
    male_ind = np.argwhere(basesize < 19.5).flatten()
    female_ind = np.argwhere(basesize > 19.5).flatten()
    return male_ind, female_ind

def get_motiondata(dtype='gmr'):

    fname = 'eyrun_simulate_data.mat'
    motiondata_list = []
    for vpaths in video16_path[dtype]:
        for vpath in vpaths:
            print(vpath)
            matfile = basepath+vpath+fname
            trx,motiondata,params,basesize = load_eyrun_data(matfile)

            D, T, F = motiondata.shape
            motiondata[np.isnan(motiondata)] = 0.

            mdata = motiondata.transpose(1,2,0).reshape([-1,NUM_MFEAT])
            motiondata_list.append(mdata)

    motiondata_ = np.vstack(motiondata_list)
    return motiondata_, params


def get_percentile(dtype='gmr', num_bin=53):

    bins = []
    motiondata_, params = get_motiondata(dtype='gmr') 
    for d in range(NUM_MFEAT):
        percentile = np.percentile(motiondata_[:,d], \
                    np.linspace(0.5,99.5,num_bin+1), axis=0)
        bins.append(percentile[1:-1])
    
    ## The binning for left & right wing angle and length should be symmetric
    bins[4] = bins[5].copy()
    bins[7] = bins[6].copy()
    return np.asarray(bins).T


def mot2binidx(f_motion,bins):

    n_bins = bins.shape[0]-1
    binidx = np.zeros(NUM_MFEAT,dtype=int)
    for v in range(NUM_MFEAT):
        if f_motion[v] < bins[1,v]:
            binidx[v] = 0
        else:
            binidx[v] = np.nonzero(bins[1:-1,v]<=f_motion[v])[0][-1]+1

        if binidx[v] >= n_bins:
            raise
    return binidx



def gen_motion2bin_video(onehotF=0, dtype='gmr', bin_type='linear', num_bin=51):

    if bin_type != 'linear':
        bins = np.load('./bins/percentile_%sbins.npy' % num_bin)

    fname = 'eyrun_simulate_data.mat'
    from tqdm import tqdm
    for vpaths in tqdm(video16_path[dtype]):
        for vpath in vpaths:
            print(vpath)
            matfile = basepath+vpath+fname
            trx,motiondata,params,basesize = load_eyrun_data(matfile)
            if bin_type == 'linear':bins = params['binedges']

            D, T, F = motiondata.shape
            motiondata[np.isnan(motiondata)] = 0.

            b_motions = []
            for t in range(T):
                b_motions_fly = []
                for f in range(F):
                    
                    b_motion = mot2binidx(motiondata[:,t,f],bins)
                    b_motions_fly.append(b_motion) 
                b_motions.append(b_motions_fly) 

            b_motions = np.asarray(b_motions)
            bdata = blur_bins(b_motions, T,F,D, onehotF=onehotF, num_bin=num_bin)
            #bdata = percentile_bin(motiondata.transpose(1,2,0), bins, onehotF=onehotF)
            bdata = bdata.reshape([T,F,D*num_bin])

            path = './bins/' + vpath 
            if not os.path.exists(path): os.makedirs(path)
            np.save(path+'bin_motion_indx_onehotF%d_%s_%dbin' % (onehotF, bin_type, num_bin), b_motions)
            np.save(path+'motion_%dbin_onehotF%d_%s' % (num_bin, onehotF, bin_type), bdata)

    return b_motions


def number_nans():

    fname = 'eyrun_simulate_data.mat'
    for vpath in video16_path[dtype]:

        matfile = basepath+vpath+fname
        trx,motiondata,params,basesize = load_eyrun_data(matfile)
         
        first_motiondata = motiondata[0,:,:]
        nans = np.isnan(first_motiondata[1:])
        nan_flies = np.sum(nans, axis=0)

        D, F = nans.shape
        tmp = D % 10
        if tmp != 0:
            nans = nans[:-tmp,:]
        downsample_nans = np.sum(nans.reshape([10,D//10,F]), axis=0)


        if np.sum(downsample_nans):
            print(vpath)
            ftag = 'nanmap_motion_%s' % vpath[:-1]
            plt.figure(figsize=(14, 14))
            ind = np.arange(downsample_nans.shape[0]) 
            for f in range(F):
                ax1 = plt.subplot(math.ceil(F/2),2,f+1)
                ax1.bar(ind, downsample_nans[:,f], width=0.5)
            plt.savefig('./figs/video/'+ftag+'.png', format='png', bbox_inches='tight') 


        NN = trx['x'].shape[0]
        basepath4 = '/groups/branson/home/imd/Documents/janelia/research/FlyTrajPred_v4/pytorch/'
        vcdata = np.load(basepath4+'data/'+vpath+'/real_fly_vision_data_0t0_%dt1.npy' % NN)
        T, F, _ = vcdata.shape

        nans = np.isnan(vcdata)
        nan_vision = np.sum(nans, axis=2)

        D, F = nan_vision.shape
        tmp = D % 10
        if tmp != 0:
            nan_vision = nan_vision[:-tmp,:]
        downsample_nans = np.mean(nan_vision.reshape([10,D//10,F]), axis=0)

        if np.isinf(np.sum(downsample_nans)):
            ftag = 'nanmap_vision_%s' % vpath[:-1]
            plt.figure(figsize=(14, 14))
            for f in range(F):
                ax1 = plt.subplot(F//2, 2, f+1)
                ind = np.arange(downsample_nans.shape[0]) 
                p1 = ax1.bar(ind, downsample_nans[:,f], width=0.1)
            plt.savefig('./figs/video/'+ftag+'.png', format='png', bbox_inches='tight') 



def test_batch_videos(genderF=0, onehotF=0, K=10000, concatF=1, \
                                tau=50, etype='tr', itype='full'):

    #sample data from video
    fname = 'eyrun_simulate_data.mat'
    num_video = len(video16_path[dtype])


    vsources, vtargets = [], []
    msources, mtargets = [], []

    video_counter, video_ind = 0, 0
    vpath = video16_path[dtype][video_ind]
    matfile = basepath+vpath+fname
    trx,motiondata,params,basesize = load_eyrun_data(matfile)

    motion_bin = params['binedges'].T
    movie_vision = basepath+vpath+'movie-vision.mat'

    NN = trx['x'].shape[0]
    basepath4 = '/groups/branson/home/imd/Documents/janelia/research/FlyTrajPred_v4/pytorch/'
    vdata = np.load(basepath4+'data/'+vpath+'/real_fly_vision_data_0t0_%dt1.npy' % NN)
    T, F, _ = vdata.shape
    
    mypath = '/groups/branson/home/imd/Documents/data/fly_tracking/'
    bdata = np.load(mypath+vpath+'motion_bin_onehotF%d.npy' % onehotF)[1:]
    #bdata = blur_bins(bdata_idx, T,F,M, onehotF=onehotF)

    mdata = motiondata[:,:-1].transpose([1,2,0])

    T, F, M = mdata.shape
    #bdata = bdata.reshape([T,F,M*NUM_BIN])
    male_ind, female_ind = gender_classify(basesize['majax'])
    if genderF==MALE:
        bdata = bdata[1:, :10, :]
        vdata = vdata[1:, :10, :]
        mdata = mdata[1:, :10, :]

    elif genderF==FEMALE:
        bdata = bdata[1:, 10:, :]
        vdata = vdata[1:, 10:, :]
        mdata = mdata[1:, 10:, :]


    video_usecap = math.ceil(K / num_video)
    i = 0
    while i < K:

        #Preprocess Data
        train_set, valid_set, test_set = preprocess_dataset(vdata, bdata, mdata,\
                        T, concatF)

        if etype == 'tr':
            N = train_set[0].shape[0]
            xdata, ydata = train_set
        elif etype == 'vl':
            N = valid_set[0].shape[0]
            xdata, ydata = valid_set
        else:
            N = test_set[0].shape[0]
            xdata, ydata = test_set
        
        mdata_ = xdata[:,:,144:]
        T, F, M = mdata_.shape
        NN = T * F

        video_counter=0    
        while video_counter < video_usecap and i < K: 
        
            fly_sample_ind = RNG.randint(NN, size=1)[0]
            fly_i = fly_sample_ind // T 
            start_ind = (fly_sample_ind - T * fly_i) #% F      
            begin_ind = max([start_ind-tau//2, 0])
            end_ind   = min(int(start_ind+tau+1), T)

            issue = np.sum(np.isnan(mdata_[begin_ind:end_ind, fly_i, 0]))
            if not issue and (T - start_ind) > tau:
                vsamples, msamples = prep_binned_data(xdata, ydata, \
                            start_ind, start_ind+tau, fly_i, itype)
                #vsamples[np.isnan(vsamples)] = 0.
                assert np.isnan(vsamples).sum() == 0, \
                                'vsamples contains nan'
                i += 1
                video_counter += 1
                vsources.append(vsamples)
                msources.append(msamples)
            else:
                if (T - start_ind) > tau:
                    import pdb; pdb.set_trace()

        #train_set, valid_set, test_set = None, None, None

    vsources = np.asarray(vsources, dtype='float32').transpose(1, 0, 2)
    msources = np.asarray(msources, dtype='float32').transpose(1, 0, 2)
   
    return vsources, msources


def load_videos(onehotF=0, vtype='all', dtype='gmr', bin_type='linear', num_bin=51):

    if vtype == 'july':
        video_list = video16_july
    elif vtype == 'original' or vtype == 'v1':
        video_list = video16_original
    elif vtype == 'v2':
        video_list = video16_v2
    elif vtype == 'v3':
        video_list = video16_v3
    elif vtype == 'v4':
        video_list = video16_v4
    elif vtype == 'v5':
        video_list = video16_v5
    else: # vtype == 'full'
        video_list = video16_path[dtype]
    #sample data from video
    fname = 'eyrun_simulate_data.mat'

    video_tr_list, video_vl_list, video_te_list = [], [], []
    video_data_list = [video_tr_list, video_vl_list, video_te_list]
    for kk in range(3):
        num_video = len(video_list[kk])
        for i in range(num_video):
            print('%dth video loading' % i)
            vpath = video_list[kk][i]
            matfile = basepath+vpath+fname
            trx,motiondata,params,basesize = load_eyrun_data(matfile)
            print(vpath, trx['x'].shape[0])

            motion_bin = params['binedges'].T
            movie_vision = basepath+vpath+'movie-vision.mat'
            vision_matfile = basepath+vpath+'movie-vision.mat'
            vcdata = load_vision(vision_matfile)[1:]
            NN = trx['x'].shape[0]
         
            mypath = '/groups/branson/home/imd/Documents/data/fly_tracking/'
            bdata = np.load(mypath+vpath+'motion_%dbin_onehotF%d_%s.npy' % (num_bin, onehotF, bin_type))[1:]
            bdata = bdata.astype('float32')
            
            histo_motion(motiondata, vpath)
            motiondata[0:2][(motiondata[0:2] > 6)] = np.nan
            motiondata[0:2][(motiondata[0:2] <-6)] = np.nan

            motiondata[3:5][(motiondata[3:5] > 10)] = np.nan
            motiondata[3:5][(motiondata[3:5] <-10)] = np.nan

            motiondata[5:7][(motiondata[5:7] > 6)] = np.nan
            motiondata[5:7][(motiondata[5:7] <-6)] = np.nan

            motiondata[7:8][(motiondata[7:8] > 4)] = np.nan
            motiondata[7:8][(motiondata[7:8] <-4)] = np.nan

            motiondata = motiondata[:,:-1]
            mdata = motiondata.transpose([1,2,0])

            T, F, M = mdata.shape
            bdata = bdata.reshape([T,F,M*num_bin])

            video_data_list[kk].append([vcdata, mdata, bdata, motiondata, basesize])

    return video_data_list


def histo_motion(motiondata, vpath):

    bins = np.linspace(0,18,100)
    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0,0].hist(motiondata[0,:,:10].flatten(), bins)
    axs[0,1].hist(motiondata[1,:,:10].flatten(), bins)
    axs[1,0].hist(motiondata[2,:,:10].flatten(), bins)
    axs[1,1].hist(motiondata[3,:,:10].flatten(), bins)
    axs[0,0].set_yscale('log')
    #axs[0,0].set_ylim((0,100))
    os.makedirs('./figs/%s/' % vpath, exist_ok=True)
    plt.savefig('./figs/%s/motion_histo_male.png' % (vpath), format='png', bbox_inches='tight')
    #import pdb; pdb.set_trace()

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0,0].hist(motiondata[0,:,10:].flatten(), bins)
    axs[0,1].hist(motiondata[1,:,10:].flatten(), bins)
    axs[1,0].hist(motiondata[2,:,10:].flatten(), bins)
    axs[1,1].hist(motiondata[3,:,10:].flatten(), bins)
    axs[0,0].set_yscale('log')
    plt.savefig('./figs/%s/motion_histo_female.png' % (vpath), format='png', bbox_inches='tight')


def quick_sample_batch_videos(video_data_list, genderF=2, onehotF=0, K=10000, concatF=1, \
                        tau=50, etype='tr', itype='full'):

    num_video = len(video_data_list)

    vsources, vtargets = [], []
    msources, mtargets = [], []

    video_usecap = math.ceil(K / num_video)
    i, video_ind = 0, 0
    while i < K:

        video_counter = 0
        vcdata, mdata, bdata, motiondata, basesize = video_data_list[video_ind]

        T, F, M = mdata.shape
        male_ind, female_ind = gender_classify(basesize['majax'])
        if genderF==MALE:
            bdata = bdata[1:, male_ind, :]
            vdata = vcdata[1:, male_ind, :]
            mdata = mdata[1:,male_ind,:]

        elif genderF==FEMALE:
            bdata = bdata[1:, female_ind, :]
            vdata = vcdata[1:, female_ind, :]
            mdata = mdata[1:,female_ind,:]


        #Preprocess Data
        T, F, D = vdata.shape#
        vdata = vdata.reshape([T*F,-1])
        mdata = mdata.reshape([T*F,-1])
        vdata[np.isinf(vdata)] = 0.
        xdata = np.hstack([vdata, mdata]).reshape([T,F,-1])
        N = xdata.shape[0]

        mdata_ = xdata[:,:,144:]
        T, F, M = mdata_.shape
        NN = N * F

        
        while video_counter < video_usecap and i < K: 
        
            fly_sample_ind = RNG.randint(NN, size=1)[0]
            fly_i = fly_sample_ind // N 
            start_ind = (fly_sample_ind - N * fly_i) #% F
            begin_ind = max([start_ind-tau//5, 0])
            end_ind   = min(int(start_ind+tau+1), N)

            issue = np.sum(np.isnan(mdata_[begin_ind:end_ind, fly_i, 0]))
            if not issue and (N - start_ind) > tau:
                vsamples, msamples = prep_binned_data(xdata, bdata, \
                        start_ind, start_ind+tau, fly_i, itype)
                vsamples[np.isnan(vsamples)] = 0.
                assert np.isnan(vsamples).sum() == 0, \
                                'vsamples contains nan'

                i += 1
                video_counter += 1
                vsources.append(vsamples)
                msources.append(msamples)

        video_ind += 1
        train_set, valid_set, test_set = None, None, None

    vsources = np.asarray(vsources, dtype='float32').transpose(1, 0, 2)
    msources = np.asarray(msources, dtype='float32').transpose(1, 0, 2)
    return  vsources, msources

    #NNN = vsources.shape[1]
    #perm = RNG.permutation(NNN) 
    #return vsources[:,perm], msources[:,perm]



def sample_batch_videos(video_data_list, genderF=2, onehotF=0, K=10000, concatF=1, \
                        tau=50, etype='tr', itype='full'):

    num_video = len(video_data_list)

    vsources, vtargets = [], []
    msources, mtargets = [], []

    video_usecap = math.ceil(K / num_video)
    i, video_ind = 0, 0
    while i < K:

        video_counter = 0
        vcdata, mdata, bdata, motiondata, basesize = video_data_list[video_ind]

        T, F, M = mdata.shape
        male_ind, female_ind = gender_classify(basesize['majax'])
        if genderF==MALE:
            bdata = bdata[1:, male_ind, :]
            vdata = vcdata[1:, male_ind, :]
            mdata = mdata[1:,male_ind,:]

        elif genderF==FEMALE:
            bdata = bdata[1:, female_ind, :]
            vdata = vcdata[1:, female_ind, :]
            mdata = mdata[1:,female_ind,:]


        #Preprocess Data
        T, F, D = vdata.shape#
        vdata = vdata.reshape([T*F,-1])
        mdata = mdata.reshape([T*F,-1])
        vdata[np.isinf(vdata)] = 0.
        xdata = np.hstack([vdata, mdata]).reshape([T,F,-1])
        N = xdata.shape[0]

        mdata_ = xdata[:,:,144:]
        T, F, M = mdata_.shape
        NN = N * F
        counter =0 
        while video_counter < video_usecap and i < K: 

            vsamples, msamples = [], []
            for fly_i in range(F):

                starting_pt = (RNG.randint(N, size=1))[0]
                target = bdata[starting_pt+1:starting_pt+tau+1, fly_i, :]
                if starting_pt+tau+1 < N  and  target.sum() > 0: \
                        #np.sum(np.isnan(mdata_[starting_pt:starting_pt+tau+1, fly_i, :])) == 0:

                    #mdata = xdata[starting_pt:starting_pt+tau, fly_i, 144:]
                    #vdata = xdata[starting_pt+tau, fly_i, :144]
                    #source = np.concatenate([mdata.flatten(), vdata])
                    source = xdata[starting_pt:starting_pt+tau, fly_i,:]
                    target = bdata[starting_pt+1:starting_pt+tau+1, fly_i, :]
                    source[np.isnan(source)] = 0.
                    #target[np.isnan(target)] = 0.

                    #vsamples.append(source)
                    #msamples.append(target)
       
                    i += 1
                    video_counter += 1
                    vsources.append(source)
                    msources.append(target)
                else:
                    counter +=1
                    pass


        video_ind += 1
        train_set, valid_set, test_set = None, None, None
        #print(counter)

    vsources = np.asarray(vsources[:K], dtype='float32').transpose(1, 0, 2)
    msources = np.asarray(msources[:K], dtype='float32').transpose(1, 0, 2)
    NNN = vsources.shape[1]
    perm = RNG.permutation(NNN) 
    return vsources[:,perm], msources[:,perm]


def prep_binned_data(xdata, ydata, start_ind, end_ind, fly_i, itype):

    if itype == 'full':
        vsamples = xdata[start_ind:end_ind, fly_i, :]
    elif itype == 'mv':
        vsamples = \
            np.concatenate(
                [xdata[start_ind:end_ind, fly_i, :72],\
                 xdata[start_ind:end_ind, fly_i, 144:]], axis=1)
    elif itype == 'mc':
        vsamples = xdata[start_ind:end_ind, fly_i, 72:]

    msamples = ydata[start_ind:end_ind, fly_i, :]
    msamples[np.isnan(msamples)] = 0.
    
    return vsamples, msamples

    
def prep_regression_data(xdata, start_ind, end_ind, fly_i, itype):

    if itype == 'full':
        vsamples = xdata[start_ind:end_ind, fly_i, :]
    elif itype == 'mv':
        vsamples = \
            np.concatenate(
                [xdata[start_ind:end_ind, fly_i, :72],\
                 xdata[start_ind:end_ind, fly_i, 144:]], axis=1)
    elif itype == 'mc':
        vsamples = xdata[start_ind:end_ind, fly_i, 72:]
       
    msamples = xdata[start_ind+1:end_ind+1, fly_i, 144:]
    msamples[np.isnan(msamples)] = 0.
    
    return vsamples, msamples


def sample_regression_batch_videos(video_data_list, genderF=2, onehotF=0, K=10000, concatF=1, \
                        tau=50, etype='tr', itype='full'):

    #stds = array([0.4762483 , 0.36305289, 0.54262141, 0.4506655 , 0.39683208, 0.36324166, 0.31870881, 0.32580173])
    num_video = len(video_data_list)

    vsources, vtargets = [], []
    msources, mtargets = [], []

    video_usecap = math.ceil(K / num_video)
    i, video_ind = 0, 0
    while i < K:

        video_counter = 0
        vcdata, mdata, bdata, motiondata, basesize = video_data_list[video_ind]

        T, F, M = mdata.shape
        male_ind, female_ind = gender_classify(basesize['majax'])

        if genderF==MALE:
            bdata = bdata[1:, male_ind, :]
            vdata = vcdata[1:, male_ind, :]
            mdata = mdata[1:,male_ind,:]

        elif genderF==FEMALE:
            bdata = bdata[1:, female_ind, :]
            vdata = vcdata[1:, female_ind, :]
            mdata = mdata[1:,female_ind,:]


        #Preprocess Data
        T, F, D = vdata.shape#
        vdata = vdata.reshape([T*F,-1])
        mdata = mdata.reshape([T*F,-1])
        xdata = np.hstack([vdata, mdata]).reshape([T,F,-1])
        N = xdata.shape[0]

        NN = N * F

        counter = 0
        while video_counter < video_usecap and i < K: 
       
            vsamples, msamples = [], []
            for fly_i in range(F):

                starting_pt = (RNG.randint(N-tau-1, size=1))[0]
                if starting_pt+tau+1 < N and \
                        np.sum(np.isnan(xdata[starting_pt:starting_pt+tau+1, fly_i, 144:])) == 0:

                    source = xdata[starting_pt:starting_pt+tau, fly_i,:]
                    target = xdata[starting_pt+1:starting_pt+tau+1, fly_i,144:]

                    i += 1
                    video_counter += 1
                    source[np.isnan(source)] = 0.
                    target[np.isnan(target)] = 0.

                    vsources.append(source)
                    msources.append(target)

        video_ind += 1
        #print(counter)
    vsources = np.asarray(vsources[:K], dtype='float32').transpose(1, 0, 2)
    msources = np.asarray(msources[:K], dtype='float32').transpose(1, 0, 2)
 
    NNN = vsources.shape[1]
    perm = RNG.permutation(NNN) 
    return vsources[:,perm], msources[:,perm]


def quick_sample_regression_batch_videos(video_data_list, genderF=2, \
                        onehotF=0, K=10000, concatF=1, \
                        tau=50, etype='tr', itype='full'):

    #sample data from video
    fname = 'eyrun_simulate_data.mat'
    num_video = len(video_data_list)
    #video_ind = RNG.randint(num_video)

    vsources, vtargets = [], []
    msources, mtargets = [], []

    video_usecap = math.ceil(K / num_video)
    i, video_ind = 0, 0
    while i < K:

        video_counter = 0
        vcdata, mdata, bdata, motiondata, basesize = video_data_list[video_ind]

        T, F, M = mdata.shape
        male_ind, female_ind = gender_classify(basesize['majax'])
        if genderF==MALE:
            bdata = bdata[1:, male_ind, :]
            vdata = vcdata[1:, male_ind, :]
            mdata = mdata[1:,male_ind,:]

        elif genderF==FEMALE:
            bdata = bdata[1:, female_ind, :]
            vdata = vcdata[1:, female_ind, :]
            mdata = mdata[1:,female_ind,:]

        #Preprocess Data
        train_set, valid_set, test_set = preprocess_dataset(vdata,\
                                bdata, mdata, T, concatF)
        #f_cv, vc_data, bdata, mdata = None, None, None, None

        if etype == 'tr':
            N = train_set[0].shape[0]
            xdata, ydata = train_set
        elif etype == 'vl':
            N = valid_set[0].shape[0]
            xdata, ydata = valid_set
        else:
            N = test_set[0].shape[0]
            xdata, ydata = test_set
        mdata_ = xdata[:,:,144:]
        T, F, M = mdata_.shape
        NN = N * F

        while video_counter < video_usecap and i < K: 
        
            fly_sample_ind = RNG.randint(NN, size=1)[0]
            fly_i = fly_sample_ind // N 
            start_ind = (fly_sample_ind - N * fly_i) #% F      
            begin_ind = max([start_ind-tau//2, 0])
            end_ind   = min(int(start_ind+tau+1), N-tau)

            issue = np.sum(np.isnan(mdata_[begin_ind:end_ind, fly_i, 0]))
            if not issue and (N - start_ind) > tau:
                vsamples, msamples = prep_regression_data(xdata, \
                        start_ind, start_ind+tau, fly_i, itype=itype)
                vsamples[np.isnan(vsamples)] = 0.
                assert np.isnan(vsamples).sum() == 0, \
                                'vsamples contains nan'

                i += 1
                video_counter += 1
                vsources.append(vsamples)
                msources.append(msamples)
            else:
                if issue:
                    print('BAD %d' % i)

        video_ind += 1
        train_set, valid_set, test_set = None, None, None

    vsources = np.asarray(vsources, dtype='float32').transpose(1, 0, 2)
    msources = np.asarray(msources, dtype='float32').transpose(1, 0, 2)
  
    return  vsources, msources

 
"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of FlyNetwork collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--print_every', type=int, default=100, help='print every')#, required=True)
    parser.add_argument('--save_every' , type=int, default=50,help= 'save every')#, required=True)
    parser.add_argument('--onehotF', type=int, default=0)
    parser.add_argument('--num_bin', type=int, default=51)
    parser.add_argument('--dtype', type=str, default='gmr', choices=['gmr', 'gmr91', 'pdb'])
    # gmr = 71G01, gmr91 = 91B01, pdb = pBDP
    parser.add_argument('--bin_type', type=str, default='perc', choices=['linear', 'perc'])
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--file_dir', type=str, \
            default='./fout/', help='Directory name to save the model')

    return check_args(parser.parse_args())



if __name__ == '__main__':

    args = parse_args()
    if args.bin_type == 'perc':
        percentile_bin = get_percentile(dtype=args.dtype, num_bin=args.num_bin+2)
        np.save('./bins/percentile_%dbins' % args.num_bin, percentile_bin)
    gen_motion2bin_video(onehotF=args.onehotF, dtype=args.dtype, bin_type=args.bin_type,  num_bin=args.num_bin)
    #X_train, X_valid, X_test = gen_fly_regression()
    #sample_batch_reg(X_train, batch_sz=100)


    #gen_motion2bin_video(onehotF=0, dtype='gmr26')
    #gen_motion2bin_video(onehotF=0, dtype='gmr', bin_type='linear')

    #number_nans()
    #sample_batch_videos(tau=50, etype='tr', itype='full')
   
    #abc = load_videos(onehotF=0, vtype='all')
    pass



