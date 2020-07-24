import numpy as np
import h5py
import scipy.io as sio

import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def one_step_simulate_rnn(model, hidden, input_variable, \
                            batch_sz=32, num_feat=8, num_bin=51, \
                            teacherForce=0, cudaF=0, mtype='rnn'):

    model.eval()
    input_length = input_variable.size()[0]
    N = input_variable.size()[1]
    loss = 0
    T, batch_sz, D = input_variable.size()
    Dt = num_feat*num_bin
   
    if teacherForce:
        if mtype=='attn':
            output, hidden, _ = model.forward(XXX, hidden)
        else:
            output, hidden = model.forward(XXX, hidden)
        output = output.view([T*batch_sz, num_feat, num_bin])
        prediction = F.softmax(output, dim=2)

    else:
        predictions = []
        loss=0
        XXX = input_variable[0].view([1, batch_sz,D])
        output, hidden = model.forward(XXX, hidden)
        output = output.view([batch_sz, num_feat, num_bin])
        prediction = F.softmax(output, dim=2)
        prediction = prediction.view([1,-1,Dt])
        predictions.append(prediction)
        prediction = torch.cat(predictions, dim=0).view(\
                        [-1, num_feat, num_bin])

    if cudaF:
        predictions = prediction.data.cpu().numpy().reshape([T, batch_sz, num_feat, num_bin])
    else:
        predictions = prediction.data.numpy().reshape([T,batch_sz, num_feat, num_bin])
    return predictions, hidden




def apply_reg_rnn(feats, state, model, hidden, params,fly=None):
    
    num_bin = params['binedges'].shape[0]-1
    if 'rnn' in params['mtype'] or 'skip' in params['mtype']:
        feat_motion_tm1 = feats[2]

        if np.any(np.isnan(feat_motion_tm1)):
            print("Found nans in feat_motion, replacing with 0s")
            feat_motion_tm1[np.isnan(feat_motion_tm1)] = 0.
        vdata_t = np.hstack([feats[0], feats[1]]) #TODO double check the order of feats[0] and feats[1]
        fdata_t = np.hstack([vdata_t, feat_motion_tm1])[:,None, None]
        #fdata_t = np.hstack([feat_motion_tm1, vdata_t])[:,None, None]
        fdata_t = Variable(torch.FloatTensor(fdata_t.transpose([1,2,0])))

        model.eval()
        predictions = []
        loss=0
        preds, hidden = model.forward(fdata_t, hidden)
        binscores = preds.data.cpu().numpy().flatten()
    else:
        #print('REAL DATA')
        # this is a place-holder, returns groundtruth
        feat_motion = model[:,t,fly]
        if np.any(np.isnan(feat_motion)):
            print("Found nans in feat_motion, replacing with 0s")
            feat_motion[np.isnan(feat_motion)] = 0.
        binidx = motion2binidx(feat_motion,params)

        n_bins = params['binedges'].shape[0]-1
        binscores = np.zeros(n_bins*params['n_motions'])
        for v in range(params['n_motions']):
            # select bin according to output probability
            inds=np.arange(v*n_bins,(v+1)*n_bins)
            binscores[inds[binidx[v]]] = 1
        #print "Placeholder -- using groundtruth motion"
        preds = []

    return (binscores,state,hidden,preds)


def apply_bin_rnn(feats, state, model, hidden, params, fly=None, num_bin=51):
    
    #num_bin = params['binedges'].shape[0]-1
    if 'rnn' in params['mtype'] or 'skip' in params['mtype']:
        feat_motion_tm1 = feats[2]

        if np.any(np.isnan(feat_motion_tm1)):
            print("Found nans in feat_motion, replacing with 0s")
            feat_motion_tm1[np.isnan(feat_motion_tm1)] = 0.
        vdata_t = np.hstack([feats[0], feats[1]]) #TODO double check the order of feats[0] and feats[1]
        fdata_t = np.hstack([vdata_t, feat_motion_tm1])[:,None, None]
        fdata_t = Variable(torch.FloatTensor(fdata_t.transpose([1,2,0])))

        preds, hidden = one_step_simulate_rnn(model, hidden, fdata_t, \
                        num_bin=num_bin, batch_sz=1, teacherForce=0)
        preds = preds.reshape([params['n_motions'], params['binedges'].shape[0]-1])
        binscores = preds.flatten()
    else:
        #print('REAL DATA')
        # this is a place-holder, returns groundtruth
        feat_motion = model[:,t,fly]
        if np.any(np.isnan(feat_motion)):
            print("Found nans in feat_motion, replacing with 0s")
            feat_motion[np.isnan(feat_motion)] = 0.
        binidx = motion2binidx(feat_motion,params)

        n_bins = params['binedges'].shape[0]-1
        binscores = np.zeros(n_bins*params['n_motions'])
        for v in range(params['n_motions']):
            # select bin according to output probability
            inds=np.arange(v*n_bins,(v+1)*n_bins)
            binscores[inds[binidx[v]]] = 1
        #print "Placeholder -- using groundtruth motion"
        preds = []

    return (binscores,state,hidden,preds)


def apply_lr_fly_np(state, flyvision, chambervision, feat_motion, model, \
                    params, visionF=True, t=None, t_dim=None, fly=None, visionOnly=0):


    if visionOnly:
        vision_chamber = np.hstack([flyvision,chambervision])
        state[fly,:,:(t_dim-1)] = state[fly,:,1:] 
        state[fly,:,-1] = vision_chamber#motiondata[:,t-1,fly]
        state[np.isnan(state)] = 0.
        XX = state[fly].T.flatten()

    else:
        state[fly,:,:(t_dim-1)] = state[fly,:,1:] 
        state[fly,:,-1] = feat_motion[:,fly]#motiondata[:,t-1,fly]
        state[np.isnan(state)] = 0.

        XX = state[fly].T.flatten()
        if visionF:
            vision_chamber = np.hstack([flyvision,chambervision])
            XX = np.hstack([XX,vision_chamber])
    XX = np.hstack([XX, np.ones((1,))])
    preds = np.dot(XX, model.T)
    binscores = preds.flatten()

    return (binscores,state,preds)


def apply_nn_reg_fly(state, flyvision, chambervision, feat_motion, model, \
                    params,visionF=True, t=None, t_dim=None, fly=None,\
                    visionOnlyF=0):
   
    vision_chamber = np.hstack([flyvision,chambervision])
    if visionOnlyF:
        state[fly,:,:(t_dim-1)] = state[fly,:,1:] 
        state[fly,:,-1] = vision_chamber
        state[np.isnan(state)] = 0.
    else:
        state[fly,:,:(t_dim-1)] = state[fly,:,1:] 
        state[fly,:,-1] = feat_motion[:,fly]#motiondata[:,t-1,fly]
        state[np.isnan(state)] = 0.

    XX = state[fly].T.flatten()
    if visionF:
        XX = np.hstack([XX,vision_chamber])
    XXX = Variable(torch.FloatTensor(XX))

    model.eval()
    preds = model.forward(XXX)
    preds = preds.data.cpu().numpy()
    #preds = (preds * std + mean)
    binscores = preds.flatten()

    return (binscores,state,preds)


def apply_conv_reg_fly(state, flyvision, chambervision, feat_motion, model, \
                    params,visionF=True, t=None, t_dim=None, fly=None,\
                    visionOnlyF=0):
  
    vision_chamber = np.hstack([flyvision,chambervision])
    if visionOnlyF:
        state[fly,:,:(t_dim-1)] = state[fly,:,1:] 
        state[fly,:,-1] = vision_chamber
        state[np.isnan(state)] = 0.
    else:
        state[fly,:,:(t_dim-1)] = state[fly,:,1:] 
        state[fly,:8,-1] = feat_motion[:,fly]
        state[fly,8:,-1] = vision_chamber
        state[np.isnan(state)] = 0.

    XX = state[fly].T[None,:,:]#.flatten()
    XXX = Variable(torch.FloatTensor(XX))

    model.eval()
    preds, _ = model.forward(XXX)
    preds = preds.data.cpu().numpy()
    #preds = (preds * std + mean)
    binscores = preds.flatten()

    return (binscores,state,preds)


def apply_conv_fly(state, flyvision, chambervision, feat_motion, model, \
                    params,visionF=True, t=None, t_dim=None, fly=None,\
                    visionOnlyF=0):
 
    num_bin = params['binedges'].shape[0]-1
    vision_chamber = np.hstack([flyvision,chambervision])
    if visionOnlyF:
        state[fly,:,:(t_dim-1)] = state[fly,:,1:] 
        state[fly,:,-1] = vision_chamber
        state[np.isnan(state)] = 0.
    else:
        state[fly,:,:(t_dim-1)] = state[fly,:,1:] 
        state[fly,:8,-1] = feat_motion[:,fly]
        state[fly,8:,-1] = vision_chamber
        state[np.isnan(state)] = 0.

    XX = state[fly].T[None,:,:]#.flatten()
    XXX = Variable(torch.FloatTensor(XX))

    model.eval()
    logit, _ = model.forward(XXX)
    logit = logit.view([1,-1,num_bin])
    preds = F.softmax(logit, dim=-1)
    preds = preds.data.cpu().numpy()
    binscores = preds.flatten()

    return (binscores,state,preds)



def apply_nn_fly(state, flyvision, chambervision, feat_motion, \
                            model, params,t_dim=None,fly=None):
   
    num_bin = params['binedges'].shape[0]-1
    if 'nn' in params['mtype']:
        state[fly,:,:(t_dim-1)] = state[fly,:,1:] 
        state[fly,:,-1] = feat_motion[:,fly]
        state[np.isnan(state)] = 0.
    
        vision_chamber = np.hstack([flyvision,chambervision])
        XX = state[fly].T.flatten()
        XX = np.hstack([XX, vision_chamber])
        XXX = Variable(torch.FloatTensor(XX))
        XXX = XXX.view([1,-1])

        model.eval()
        logit = model.forward(XXX)
        logit = logit.view([1,-1,num_bin])
        preds = F.softmax(logit, dim=-1)
        preds = preds.data.cpu().numpy()
        binscores = preds.flatten()
    else:
        #print('REAL DATA')
        # this is a place-holder, returns groundtruth
        feat_motion = model[:,t,fly]
        if np.any(np.isnan(feat_motion)):
            print("Found nans in feat_motion, replacing with 0s")
            feat_motion[np.isnan(feat_motion)] = 0.
        binidx = motion2binidx(feat_motion,params)

        n_bins = params['binedges'].shape[0]-1
        binscores = np.zeros(n_bins*params['n_motions'])
        for v in range(params['n_motions']):
            # select bin according to output probability
            inds=np.arange(v*n_bins,(v+1)*n_bins)
            binscores[inds[binidx[v]]] = 1
        #print "Placeholder -- using groundtruth motion"
        preds = []

    return (binscores,state,preds)




def simulate_flies( male_model=None, female_model=None, plottrxlen=1,\
                    t0=0, t1=30320, vision_save=False, histoF=False,\
                    visionF=1, bookkeepingF=True, mtype='nn',\
                    testvideo_num=0):

    DEBUG = 0
    #matfile = '/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Code/eyrun_simulate_data.mat'
    #(trx,motiondata,params,basesize) = load_eyrun_data(matfile)
    #binedges = params['binedges']
   
    vpath = video16_path[testvideo_num]
    fname = 'eyrun_simulate_data.mat'
    basepath='/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'
    basepath4 = '/groups/branson/home/imd/Documents/janelia/research/FlyTrajPred_v4/pytorch/'

    matfile = basepath+vpath+fname
    trx,motiondata,params,basesize = load_eyrun_data(matfile)
    n_flies= trx['x'].shape[1]


    # initial pose
    x = trx['x'][t0,:].copy()
    y = trx['y'][t0,:].copy()
    theta = trx['theta'][t0,:].copy()
    
    # even flies are simulated, odd are real
    n_flies = trx['x'].shape[1]
    real_flies = []
    #male_flies = np.arange(0,10,1)
    #female_flies = np.arange(10,n_flies,1)

    if testvideo_num == 2:
        male_flies = np.arange(0,10,1)
        female_flies = np.arange(10,19,1)
    elif  testvideo_num == 7 or testvideo_num == 6 or testvideo_num == 3:
        male_flies = np.arange(0,9,1)
        female_flies = np.arange(9,19,1)
    else:
        male_flies = np.arange(0,10,1)
        female_flies = np.arange(10,20,1)


    b = basesize['minax'].copy()
    a = trx['a'][t0,:].copy()
    l_wing_ang = trx['l_wing_ang'][t0,:].copy()
    r_wing_ang = trx['r_wing_ang'][t0,:].copy()
    l_wing_len = trx['l_wing_len'][t0,:].copy()
    r_wing_len = trx['r_wing_len'][t0,:].copy()

    xprev = x.copy()
    yprev = y.copy()
    thetaprev = theta.copy()


    feat_motion = motiondata[:,t0,:].copy()
    mymotiondata = np.zeros(motiondata.shape)

    simtrx = {}
    tsim = t1-t0-1
    simtrx['x'] = np.concatenate((trx['x'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['y'] = np.concatenate((trx['y'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['theta'] = np.concatenate((trx['theta'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['a'] = np.concatenate((trx['a'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['b'] = np.concatenate((trx['b'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['l_wing_ang'] = np.concatenate((trx['l_wing_ang'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['r_wing_ang'] = np.concatenate((trx['r_wing_ang'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['l_wing_len'] = np.concatenate((trx['l_wing_len'][:t0+1,:],np.zeros((tsim,n_flies))))
    simtrx['r_wing_len'] = np.concatenate((trx['r_wing_len'][:t0+1,:],np.zeros((tsim,n_flies))))

    #histogram
    bins=100
    male_bucket = np.zeros([bins,bins])
    fale_bucket = np.zeros([bins,bins])
    male_dist_centre, fale_dist_centre = [], []
    male_velocity, fale_velocity = [], []
    male_pos = [np.hstack([trx['x'][0,:10], trx['y'][0,:10]])]
    fale_pos = [np.hstack([trx['x'][0,10:], trx['y'][0,10:]])]
    male_body_pos  = [[trx['theta'][0,:10], \
                       trx['l_wing_ang'][0,:10],\
                       trx['r_wing_ang'][0,:10],\
                       trx['l_wing_len'][0,:10],\
                       trx['r_wing_len'][0,:10]]]
    fale_body_pos  = [[trx['theta'][0,10:], \
                       trx['l_wing_ang'][0,10:],\
                       trx['r_wing_ang'][0,10:],\
                       trx['l_wing_len'][0,10:],\
                       trx['r_wing_len'][0,10:]]]
    male_motion, fale_motion = [], []


    ##Fly Visualization Initialization
    vdata, mdata, bdata = [], [], []
    if DEBUG == 1:
        fig,ax = plt.subplots(figsize=(15,15))
        colors = get_default_fly_colors(n_flies)

        hbg = plt.imshow(params['bg'],cmap=cm.gray,vmin=0.,vmax=1.)
        htrx = []
        for fly in range(n_flies):
            htrxcurr, = ax.plot(x[fly],y[fly],'-',color=np.append(colors[fly,:-1],.5),linewidth=1)
            htrx.append(htrxcurr)

        hbodies,hflies,htexts = draw_flies(x,y,a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len,ax=ax,colors=colors)
        plt.axis('image')

    print('Loading Model...\n')
    numpyF = 0
    #t_dim=1; epoch=120
    #t_dim=3; epoch=120
    #t_dim=5; epoch=400
    t_dim=7; epoch=600
    #t_dim=9; epoch=600
    if 'lr' in mtype:
        if male_model is None:
            save_path='./runs/linear_reg_'+str(t_dim) +'tau/model/weight_gender0'
            if not visionF: save_path = save_path +'_visionF0'
            male_model = np.load(save_path+'.npy')

        if female_model is None:
            save_path='./runs/linear_reg_'+str(t_dim) +'tau/model/weight_gender1'
            if not visionF: save_path = save_path +'_visionF0'
            female_model = np.load(save_path+'.npy')
    else:
        if male_model is None:
            save_path='./runs/nn4_7tau_reg/model/model_gender0_epoch5000'
            from main_nn4_reg import parse_args
            from fly_neural_network import NeuralNetwork4
            args = parse_args()
            args.gender=0
            model = NeuralNetwork4(args)
            male_model = load(model, save_path)

        if female_model is None:
            save_path='./runs/nn4_7tau_reg/model/model_gender1_epoch5000'
            from main_nn4_reg import parse_args
            from fly_neural_network import NeuralNetwork4
            args = parse_args()
            args.gender=1
            female_model = NeuralNetwork4(args)
            female_model = load(model, save_path)

    num_feat = 152 if visionF else 8
    fly_data_all = np.zeros(( NUM_FLY, num_feat, t_dim ))
    state = np.zeros(( NUM_FLY, NUM_MFEAT, t_dim ))

    print('Simulation Start...\n')
    from tqdm import tqdm
    for t in tqdm(range(t0+1,t1)):

        xprev[:] = x
        yprev[:] = y
        thetaprev[:] = theta

        male_dist_centre.append([x[:10]-default_params['arena_center_x'],\
                                 y[:10]-default_params['arena_center_y']])
        fale_dist_centre.append([x[10:]-default_params['arena_center_x'],\
                                 y[10:]-default_params['arena_center_y']])

        ###
        x, y, theta, a, \
            l_wing_ang, r_wing_ang, \
            l_wing_len, r_wing_len, feat_motion =\
            get_simulate_fly(male_model, state, t, t_dim, trx,\
                             motiondata, male_flies, feat_motion, \
                                x, y, theta, a, b, \
                                l_wing_ang, r_wing_ang,\
                                l_wing_len, r_wing_len,\
                                xprev, yprev, thetaprev, 
                                basesize, params, DEBUG, mtype=mtype,\
                                thrd=0, visionF=visionF)
        male_velocity.append([x[:10].copy()-xprev[:10].copy(), y[:10].copy()-yprev[:10].copy()])
        male_pos.append(np.hstack([x[:10].copy(), y[:10].copy()]))
        male_motion.append(feat_motion[:,:10].copy())
        male_body_pos.append(np.asarray([theta[:10], \
                                l_wing_ang[:10], r_wing_ang[:10], \
                                l_wing_len[:10], r_wing_len[:10]]))

        x, y, theta_, a, \
            l_wing_ang, r_wing_ang, \
            l_wing_len, r_wing_len, feat_motion =\
            get_simulate_fly(female_model, state, t, t_dim, trx,\
                             motiondata, female_flies, feat_motion, \
                                x, y, theta, a, b, \
                                l_wing_ang, r_wing_ang,\
                                l_wing_len, r_wing_len,\
                                xprev, yprev, thetaprev, 
                                basesize, params, DEBUG, mtype=mtype,\
                                thrd=0, visionF=visionF)

        fale_velocity.append([x[10:].copy()-xprev[10:].copy(), y[10:].copy()-yprev[10:].copy()])
        fale_pos.append(np.hstack([x[10:].copy(), y[10:].copy()]))
        fale_motion.append(feat_motion[:,10:].copy())
        fale_body_pos.append(np.asarray([theta[10:], \
                                l_wing_ang[10:], r_wing_ang[10:], \
                                l_wing_len[10:], r_wing_len[10:]])) 

        simtrx['x'][t,:]=x
        simtrx['y'][t,:]=y
        simtrx['theta'][t,:]=theta
        simtrx['a'][t,:]=a
        simtrx['b'][t,:]=b
        simtrx['l_wing_ang'][t,:]=l_wing_ang
        simtrx['r_wing_ang'][t,:]=r_wing_ang
        simtrx['l_wing_len'][t,:]=l_wing_len
        simtrx['r_wing_len'][t,:]=r_wing_len
        
        if DEBUG==1:
            tprev = np.maximum(t0+1,t-plottrxlen)
            for fly in range(n_flies):
                htrx[fly].set_data(simtrx['x'][tprev:t+1,fly],simtrx['y'][tprev:t+1,fly])

            update_flies(hbodies,hflies,htexts,x,y,a,b,theta,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len)
            plt.pause(.001)
            if t % 1 == 0 and t < t1 : #/10.0:
                if visionF:
                    if 'lr' in mtype:
                        plt.savefig('./figs/sim/lr/lr_'+str(t_dim)+'tau_'+str(t0)+'t0_'+str(t1)+'t1_'+str(epoch)+'epoch_%5d.png' % t, bbox_inches='tight', format='png')
                    else:
                        plt.savefig('./figs/sim/nn4/nn4_reg_'+str(t_dim)+'tau_'+str(t0)+'t0_'+str(t1)+'t1_'+str(epoch)+'epoch_%5d.png' % t, format='png')
                else:
                    if 'lr' in mtype:
                        #plt.savefig('./figs/sim/lr/lr_visionF0_'+str(t_dim)+'tau_'+str(t0)+'t0_'+str(t1)+'t1_'+str(epoch)+'epoch_%5d.png' % t, bbox_inches='tight', format='png')
                        plt.savefig('./figs/sim/lr/lr_visionF0_'+str(t_dim)+'tau_'+str(t0)+'t0_'+str(t1)+'t1_'+str(epoch)+'epoch_%5d.png' % t, format='png')
                    else:
                        plt.savefig('./figs/sim/nn4/nn4_reg_visionF0_'+str(t_dim)+'tau_'+str(t0)+'t0_'+str(t1)+'t1_'+str(epoch)+'epoch_%5d.png' % t, format='png')
    
    #mtype =  'nn4_reg'
    if not visionF: mtype = mtype + '_visionF0'
    ftag = str(t_dim)+'tdim_'+str(t0)+'t0_'+str(t1)+'t1_testvideo%d' % testvideo_num
    arena_radius = default_params['arena_radius']

    male_motion = np.asarray(male_motion) 
    fale_motion = np.asarray(fale_motion)


    male_pos = np.asarray(male_pos)
    fale_pos = np.asarray(fale_pos)

    male_body_pos = np.asarray(male_body_pos)
    fale_body_pos = np.asarray(fale_body_pos)
    if bookkeepingF:
        np.save(basepath4+'/motion/'+mtype+'_motion_male_'+ftag, male_motion)
        np.save(basepath4+'/motion/'+mtype+'_motion_fale_'+ftag, fale_motion)
        np.save(basepath4+'/motion/'+mtype+'_position_male_'+ftag, male_pos)
        np.save(basepath4+'/motion/'+mtype+'_position_fale_'+ftag, fale_pos)
        np.save(basepath4+'/motion/'+mtype+'_body_position_male_'+ftag, male_body_pos)
        np.save(basepath4+'/motion/'+mtype+'_body_position_fale_'+ftag, fale_body_pos)

    male_velocity    = np.asarray(male_velocity) 
    fale_velocity    = np.asarray(fale_velocity)
    male_velocity    = np.sqrt(np.sum( male_velocity**2, axis=1)).flatten() 
    fale_velocity    = np.sqrt(np.sum( fale_velocity**2, axis=1)).flatten() 
    moving_male_ind = (male_velocity > 1.0)
    moving_fale_ind = (fale_velocity > 1.0)
    male_velocity_ns = male_velocity[moving_male_ind]
    fale_velocity_ns = fale_velocity[moving_fale_ind]

    if bookkeepingF:
        np.save(basepath4+'/velocity/'+mtype+'_velocity_male_'+ftag, male_velocity)
        np.save(basepath4+'/velocity/'+mtype+'_velocity_fale_'+ftag, fale_velocity)
        np.save(basepath4+'/velocity/'+mtype+'_velocity_woStationary_male_ind_'+ftag, moving_male_ind)
        np.save(basepath4+'/velocity/'+mtype+'_velocity_woStationary_fale_ind_'+ftag, moving_fale_ind)
        np.save(basepath4+'/velocity/'+mtype+'_velocity_woStationary_male_'+ftag, male_velocity_ns)
        np.save(basepath4+'/velocity/'+mtype+'_velocity_woStationary_fale_'+ftag, fale_velocity_ns)

    if histoF:
        male_histo = histogram(male_velocity/105, fname=mtype+'_velocity_male_histo_'+ftag, title='Velocity (Male)')
        fale_histo = histogram(fale_velocity/105, fname=mtype+'_velocity_fale_histo_'+ftag, title='Velocity (Female)')
        np.save(basepath4+'/hist/'+mtype+'_velocity_male_histo_'+ftag, male_histo)
        np.save(basepath4+'/hist/'+mtype+'_velocity_fale_histo_'+ftag, fale_histo)

        male_histo_ns = histogram(male_velocity_ns/105, fname=mtype+'_velocity_woStationary_male_histo_'+ftag, title='Velocity (Male)')
        fale_histo_ns = histogram(fale_velocity_ns/105, fname=mtype+'_velocity_woStationary_fale_histo_'+ftag, title='Velocity (Female)')
        np.save(basepath4+'/hist/'+mtype+'_velocity_woStationary_male_histo_'+ftag, male_histo_ns)
        np.save(basepath4+'/hist/'+mtype+'_velocity_woStationary_fale_histo_'+ftag, fale_histo_ns)

    male_dist_centre = np.asarray(male_dist_centre)
    fale_dist_centre = np.asarray(fale_dist_centre)
    male_dist_centre = np.sqrt(np.sum(male_dist_centre**2, axis=1)).flatten() 
    fale_dist_centre = np.sqrt(np.sum(fale_dist_centre**2, axis=1)).flatten() 

    male_dist_centre_ = male_dist_centre/arena_radius
    fale_dist_centre_ = fale_dist_centre/arena_radius
    if bookkeepingF:
        np.save(basepath4+'/centredist/'+mtype+'_centredist_male_'+ftag, male_dist_centre)
        np.save(basepath4+'/centredist/'+mtype+'_centredist_fale_'+ftag, fale_dist_centre)

    male_dist_centre_ns = male_dist_centre[moving_male_ind]
    fale_dist_centre_ns = fale_dist_centre[moving_fale_ind]
    male_dist_centre_ns_= male_dist_centre_ns/arena_radius
    fale_dist_centre_ns_= fale_dist_centre_ns/arena_radius

    if bookkeepingF:
        np.save(basepath4+'/centredist/'+mtype+'_centredist_woStationary_male_'+ftag, male_dist_centre_ns)
        np.save(basepath4+'/centredist/'+mtype+'_centredist_woStationary_fale_'+ftag, fale_dist_centre_ns)

    if histoF:
        male_dist_histo = histogram(male_dist_centre_, fname=mtype+'_dist2centre_male_histo_'+ftag, title='Distance to Centre (Male)')   
        fale_dist_histo = histogram(fale_dist_centre_, fname=mtype+'_dist2centre_fale_histo_'+ftag, title='Distance to Centre (Female)') 
        np.save(basepath4+'/hist/'+mtype+'_dist_male_histo_'+ftag, male_dist_histo)
        np.save(basepath4+'/hist/'+mtype+'_dist_fale_histo_'+ftag, fale_dist_histo)

        male_dist_histo_ns =histogram(male_dist_centre_ns_, fname=mtype+'_dist2centre_woStationary_male_histo_'+ftag, title='Distance to Centre (Male) excluding stationary flies')   
        fale_dist_histo_ns =histogram(fale_dist_centre_ns_, fname=mtype+'_dist2centre_woStationary_fale_histo_'+ftag, title='Distance to Centre (Female) excluding stationary flies') 
        np.save(basepath4+'/hist/'+mtype+'_dist_woStationary_male_histo_'+ftag, male_dist_histo_ns)
        np.save(basepath4+'/hist/'+mtype+'_dist_woStationary_fale_histo_'+ftag, fale_dist_histo_ns)

    return male_pos, fale_pos


