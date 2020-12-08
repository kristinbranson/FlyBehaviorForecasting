

def get_real_positions(t, trx, motiondata, T, batch_sz):
    '''
    Extract a batch of fly trajectories (positions) and their respective features (feats)
    from the dataset trx
    Let positions be a a dictionary of fly positions (each key is something like
    x, y, l_wing_ang, etc.).  positions[k] will be a batch_sz X T X num_flies tensor 
    trajectories of T timesteps sampled starting at frame t
    and feats will be a batch_sz X num_feats X T X num_flies tensor extracted from 
    motiondata
    '''
    positions = {}
    for k,v in trx.items():
        p = [v[s : s + tsim, :] for s in range(t, t + T * batch_sz, batch_sz)]
        positions[k] = torch.stack(p, 0)
    motion_feats = [motiondata[:, s : s + T, :] for s in range(t, t + T * batch_sz, batch_sz)]
    return positions, motion_feats

def compute_features(positions, params, keys=['x', 'y', 'theta', 'a', 'b']):
    device = positions.values()[0].device

    x = positions['x']  # batch_sz X T X num_flies
    y = positions['y']  # batch_sz X T X num_flies
    theta = positions['theta']  # batch_sz X T X num_flies
    a = positions['a']  # batch_sz X T X num_flies
    b = positions['b']  # batch_sz X T X num_flies

    if(x.isnan().any() or y.isnan().any() or theta.isnan().any())
        flyvision = torch.zeros((params['n_oma']), device=device)
        chambervision = torch.zeros((params['n_oma']), device=device)
        return (flyvision, chambervision)
    
    # vision bin size
    step = 2.*np.pi/params['n_oma']
    
    # flip
    theta_r = -theta + np.pi  # batch_sz X T X num_flies
    
    # for rotating into fly's coordinate system
    cos_theta_r, sin_theta_r = torch.cos(theta_r), torch.sin(theta_r)
    cos_theta_pi2_r, sin_theta_pi2_r = torch.cos(theta_r), torch.sin(theta_r)
    xs = torch.cat([x + cos_theta_r * a, x - cos_theta_r * a,
                    x_ + cos_theta_pi2_r * b, x_ - cos_theta_pi2_r * b], 3)
    ys = torch.cat([y + sin_theta_r * a, y - sin_theta_r * a,
                    y_ + sin_theta_pi2_r * b, y_ - sin_theta_pi2_r * b], 3)

    # compute other flies view

    # initialize everything to be infinitely far away
    flyvision = torch.full([params['n_oma']], np.inf, device=device)

    # other flies positions ends of axes
    x_ = torch.tile(x[:,:,:,None], (1, 1, 1, n_flies)) # batch_sz X T X num_flies X num_flies
    y_ = torch.tile(y[:,:,:,None], (1, 1, 1, n_flies)) # batch_sz X T X num_flies X num_flies
    ax_a = torch.tile(a[:,:,:,None], (1, 1, 1, n_flies)) / 2 # batch_sz X T X num_flies X num_flies
    ax_b = torch.tile(b[:,:,:,None], (1, 1, 1, n_flies)) / 2 # batch_sz X T X num_flies X num_flies
    xs = torch.tile(xs[:,:,:,None,:], (1, 1, 1, n_flies, 1)) # batch_sz X T X num_flies X num_flies X 4
    ys = torch.tile(ys[:,:,:,None,:], (1, 1, 1, n_flies, 1)) # batch_sz X T X num_flies X num_flies X 4
    
    # convert to this fly's coord system
    dx = xs - x[:,:,:,None,None] # batch_sz X T X num_flies X num_flies X 4
    dy = ys - y[:,:,:,None,None] # batch_sz X T X num_flies X num_flies X 4
    dist = torch.sqrt(dx**2 + dy**2) # batch_sz X T X num_flies X num_flies X 4
    dx = dx / dist # batch_sz X T X num_flies X num_flies X 4
    dy = dy / dist # batch_sz X T X num_flies X num_flies X 4
    angle = torch.atan2(cos_theta_r * dy - sin_theta_r * dx,
                        cos_theta_r * dx + sin_theta_r * dy)
    angle = torch.remainder(angle,2.*np.pi) # batch_sz X T X num_flies X num_flies X 4

    # no -1 because of zero-indexing
    angle_bin = torch.floor(angle / step).int() # batch_sz X T X num_flies X num_flies X 4

    dists_bin_inf = torch.full([batch_sz, T, num_flies, num_flies, num_bins], np.inf)
    dists_bin = dist.min(4).repeat([1, 1, 1, 1, num_bins]) 
    min_bin, max_bin = angle_bin.min(4), angle_bin.max(4)
    R = torch.arange(params['n_oma']).repeat([batch_sz, T, num_flies, num_flies, num_bins])
    dists_bin = torch.where((max_bin - min_bin) * 2 < num_bins - 1,
                            torch.where(R >= min_bin & R < max_bin, dists_bin, dists_bin_inf),
                            torch.where(R <= min_bin | R >= max_bin, dists_bin, dists_bin_inf))
    dists_bin[torch.eye(num_flies).repeat([batch_sz, T, 1, 1, num_bins])] = np.inf
   
    for j in range(n_flies):
        if np.isnan(x_[j].item()):
            continue
            
        b = b_all[j,:]
        f_dist = dist[j,:]

        if torch.any(torch.isnan(b)).item() or torch.any(torch.isnan(f_dist)).item():
            raise Exception(ValueError,'nans found in flyvision')

        mi, ma = torch.min(b), torch.max(b)
        if ma - mi*2 < params['n_oma']-1-ma:
            bs = torch.arange(mi,ma+1, device=device)
        else:
            bs = torch.cat((torch.arange(0,mi+1, device=device),
                            torch.arange(ma,params['n_oma'], device=device)))

        #print "j = " + str(j) + ", bs = " + str(bs)
        flyvision[bs] = torch.min(flyvision[bs],torch.min(f_dist))

    # compute chamber view
    chambervision = torch.full([params['n_oma']], np.inf, device=device)
    dx = params['J'] - x
    dy = params['I'] - y
    dist = torch.sqrt(dx**2 + dy**2)
    dx = dx/dist
    dy = dy/dist
    angle = torch.atan2(rotvec[0]*dy - rotvec[1]*dx,
                       rotvec[0]*dx + rotvec[1]*dy)
    angle = torch.remainder(angle,2.*np.pi)
    if torch.any(torch.isnan(angle)).item():
        raise Exception(ValueError,'angle to arena wall is nan')
    #angle[np.isnan(angle)] = 0. ##THIS LINE ADDED BY DANIEL.
    b = torch.floor(angle / step).long()
    b = b.clamp(max=params['n_oma']-1)
    chambervision[b] = dist

    # interpolate / extrapolate gaps
    notfilled = torch.ones(chambervision.shape, device=device)
    notfilled[b] = 0
    false_t = torch.Tensor([False]).to(device)
    t1s = torch.nonzero((torch.cat((false_t,notfilled[:-1]))==False) & (notfilled==True))
    t2s = torch.nonzero((notfilled==True) & (torch.cat((notfilled[1:],false_t))==False))
    for c in range(t1s.shape[0]):
        t1 = t1s[c]
        t2 = t2s[c]
        t2_n = (t2+1).clamp(max=chambervision.shape[0]-1)
        t1_p = (t2-1).clamp(min=0)
        nt = t2_n-t1_p
        chunk = (chambervision[t2_n]-chambervision[t1_p])/nt
        chambervision[t1:t2+1] = chambervision[t1_p] + torch.arange(1,nt.item(), device=device)*chunk

    #import pdb; pdb.set_trace()
    flyvision = flyvision / params['PPM']
    chambervision = chambervision / params['PPM']
  
    #global fly_vision
    #global chamber_vision
    #if np.sum(np.isnan(fly_vision)) > 0:
    #    fly_vision[0][np.isnan(fly_vision)[0]] = 0
    #if np.sum(np.isnan(chamber_vision)) > 0:
    #    chamber_vision[0][np.isnan(fly_vision)[0]] =0 
    #fly_vision.append(flyvision)
    #chamber_vision.append(chambervision)

    #global max_fly_vision
    #global max_chamber_vision
    #global min_fly_vision 
    #global min_chamber_vision

    #max_fly_vision     = max([max_fly_vision, np.max(flyvision[flyvision != np.inf])])
    #max_chamber_vision = max([max_chamber_vision, np.max(chambervision[chambervision != np.inf])])
    #min_fly_vision     = min([min_fly_vision, np.min(flyvision[flyvision != -np.inf])])
    #min_chamber_vision = min([min_chamber_vision, np.min(chambervision[chambervision != -np.inf])])

    flyvision_dist = flyvision[:]
    chambervision_dist = chambervision[:]

    flyvision = 1 - torch.clamp(.05 * torch.clamp(flyvision - 1., min=0.)**0.6, max=1.)
    chambervision = torch.clamp(.5**(((chambervision - params['mindist']) * .4) * 1.3), max=1.)

    if distF: return flyvision_dist, chambervision_dist
    return flyvision, chambervision


def run_rnns(models, T, start_positions, start_feat_motion, params):
    num_real_frames = start_positions.values()[0].shape[1]
    hiddens = [m['hidden'] for m in models]
    results = []
    
    for t in range(T):
        if t < num_real_frames:
            # Extract fly positions and motion from real trajectories
            positions = start_positions[:, t, :]
            feat_motion = start_feat_motion[:, :, t, :]
        else:
            # Update fly positions from the last simulated motion prediction
            feat_motion = feat_motion_new
            positions = positions_new
            
        feats = compute_features(positions, params)
        feat_motion_new = feat_motion.clone()
        hiddens_new = []
        for i, (model, hidden) in enumerate(zip(models, hiddens)):
            inds = model['inds']
            feats_m = feats[:, inds, :]
            preds, hidden = one_step_simulate_rnn(model['model'], hidden, feats_m, 
                        num_bin=num_bin, batch_sz=positions.shape[0], teacherForce=0)
            
            preds = preds.reshape([params['n_motions'], params['binedges'].shape[0]-1])
            binscores = preds.flatten()
            motion = binscores2motion(binscores, params)
            feat_motion_new[:, :, :, inds] = motion
            hiddens_new.append(hidden)

        positions_new = update_positions(positions, feat_motion_new, params)
        results.append({'hidden': hiddens_new, 'positions': positions_new,
                        'feat_motion': feat_motion_new})
    return results

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
    
    if t1 is None:
        t1= trx['x'].shape[0] - tsim
        
    simulated_male_flies = torch.arange(len(male_ind), device=device)
    simulated_female_flies = torch.arange(len(male_ind),len(male_ind)+len(female_ind),
                                          device=device)

    predictions_flies, flyvisions = [], []
    vel_errors, pos_errors, theta_errors, wing_ang_errors, wing_len_errors \
                                                = [], [], [], [], []
    acc_rates, loss_rates = [], []
    simtrx_numpys, dataset, dataset_frames = [], [], []

        

    models = [{'name': 'male', 'model': male_model, 'inds': simulated_male_flies},
              {'name': 'female', 'model': female_model, 'inds': simulated_female_flies}
    ]
    for m in models:
        m['hidden'] = m['model'].initHidden(batch_sz * len(v['inds']), use_cuda=use_cuda)
        m['state'] = None
    
    print('Simulation Start %d %d %d...\n' % (t0+t_dim,t1,tsim))
    
    progress = tqdm(enumerate(range(t0+t_dim,t1,tsim*batch_sz)))
    for ii, t in progress:
        print(ii, t)
        
        real_positions, real_feat_motion = \
            get_real_positions(t - tsim, model['inds'], trx, motiondata, tsim, batch_sz)

        # Run the RNN, first using the real fly positions from t-tsim:t
        run_rnns(models,  2 * tsim, real_positions, real_feat_motion)
        
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

