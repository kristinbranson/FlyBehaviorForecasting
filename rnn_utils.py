import torch
import torch.nn.functional as F

from fly_utils import compute_features, update_positions, binprobs2motion
from utils import compute_position_errors

def run_rnns(models, T, num_samples, start_positions, start_feat_motion, params, basesize,
             num_real_frames=None, train=False, motion_method='multinomial',
             num_rand_features=0, debug=0):
    if num_real_frames is None:
        num_real_frames = start_positions.values()[0].shape[1]
    batch_sz = start_positions.values()[0].shape[0]
    num_motion_feat = params['n_motions']
    positions, hiddens, motions, binscores, basesizes = [], [], [], [], []
    results = []
    rand_features = None
    device = start_feat_motion.device

    # Run the RNN model on the initial ground truth observed frames, to accumulate
    # the initial RNN hidden state
    for i, model in enumerate(models):
        inds = model['inds']
        num_flies = len(inds)
        basesizes.append({k: v[inds] for k,v in basesize.items()})
            
        position = {k: v[:, :num_real_frames, inds] for k,v in start_positions.items()}
        motion = start_feat_motion[:, :num_real_frames, inds, :]
        assert(motion.shape[-1] == num_motion_feat)

        if num_rand_features > 0:
            rand_features = torch.zeros(list(motion.shape[:-1]) + [num_rand_features],
                                        device=device)
        feats_m = compute_features(position, motion, params, train=train,
                                   rand_features=rand_features)
        
        # Run num_real_frames steps of the RNN model
        binscore, hidden, motion_new = forward_with_motion(model['model'], feats_m,
                                                           model['hidden'], batch_sz,
                                                           1, params, motion_method)
        
        position_new = update_positions(position, basesizes[i], motion_new, params) 
        debug_check(debug, motion_new, hidden, position_new)
        
        positions.append(position)
        binscores.append(binscore)
        hiddens.append(hidden)
        motions.append(motion)
        results.append([{'hidden': hiddens[i], 'positions': position_new,
                         'binscores': binscores[i]}])

    # Store the last observed real frame
    t = num_real_frames - 1
    for i, model in enumerate(models):
        inds = model['inds']
        num_layers = model['model'].num_layers if hasattr(model['model'], 'num_layers') else 1
        num_flies = len(inds)
        positions[i] = {k: torch.stack([v[:, t:t+1, inds]] * num_samples, 1).\
                        view([-1, 1, num_flies]) for k,v in start_positions.items()}
        motions[i] = torch.stack([start_feat_motion[:, t:t+1, inds, :]] * num_samples, 1).\
                     view([-1, 1, num_flies, start_feat_motion.shape[3]])
        hiddens[i] = [torch.stack([h] * num_samples, 2).view([num_layers, -1, h.shape[2]])\
                      for h in hiddens[i]]

    # Sequentially run the RNN one frame at a time, generating vision features using the
    # simulated position from the previous timestep
    rand_features = [None] * len(models)
    for t in range(num_real_frames, T):
        for i, model in enumerate(models):
            # Compute state features using the current position
            assert(motions[i].shape[-1] == num_motion_feat)
            if num_rand_features > 0 and t == num_real_frames:
                rand_features[i] = torch.rand(list(motions[i].shape[:-1]) + [num_rand_features],
                                            device=device)
            feats_m = compute_features(positions[i], motions[i], params, train=train,
                                       rand_features=rand_features[i])

            # Run one step of the RNN model
            binscores[i], hiddens[i], motions[i] = forward_with_motion(model['model'], feats_m,
                                                                       hiddens[i], batch_sz,
                                                                       num_samples, params,
                                                                       motion_method)
            positions[i] = update_positions(positions[i], basesizes[i], motions[i], params)

            debug_check(debug, motions[i], hiddens[i], positions[i])
            results[i].append({'hidden': hiddens[i], 'positions': positions[i],
                               'binscores': binscores[i]})
            
    return results

def debug_check(debug, motion, hidden, positions_new_m):
    if debug > 0:
        if torch.isnan(motion).sum():
            import pdb; pdb.set_trace()
        for h in hidden:
            if torch.isnan(h).sum():
                import pdb; pdb.set_trace()
        for k, v in positions_new_m.items():
            if torch.isnan(v).sum():
                import pdb; pdb.set_trace()

def forward_with_motion(model, feats_m, hidden, batch_sz, num_samples, params, motion_method):
    T, num_flies = feats_m.shape[0], feats_m.shape[1] // (batch_sz * num_samples)
    num_motion_feat = params['n_motions']
    num_motion_bins = params['binedges'].shape[0] - 1

    output, hidden = model.forward(feats_m, hidden)
    if motion_method == 'direct':
        # Directly output the motion delta from the RNN (train using regression)
        motion = output.view([T, batch_sz * num_samples, num_flies, num_motion_feat])
        motion = motion.transpose(0, 1)
        binscores = None
    else:
        # Bin the possible motion outputs such that the RNN outputs probabilities for each
        # motion bin (usually means training with a bin classification loss)
        binscores = output.contiguous().view([T * batch_sz * num_samples * num_flies,
                                              num_motion_feat, num_motion_bins]) * \
                                              params['binprob_exp']#*.3
        binprobs = F.softmax(binscores, dim=2)
        binprobs = binprobs.reshape([T, batch_sz*num_samples, num_flies, num_motion_feat,
                                     num_motion_bins])

        # Sample a predicted motion from the motion bin probabilities
        motion = binprobs2motion(binprobs, params, method=motion_method).transpose(0, 1)

    return binscores, hidden, motion


# Baseline: update positions by setting positions to the last observed frame
def run_stay_still_baseline(models, T, num_samples, start_positions, start_feat_motion,
                            params, basesize):
    tm = start_positions.values()[0].shape[1] - 1
    return [{'positions': {k: v[:, min(tm,t):min(tm,t)+1, :] \
                           for k, v in start_positions.items()}} for t in range(T)]

# Baseline: update positions based on the velocity of the last observed frame
def run_constant_velocity_baseline(models, T, num_samples, start_positions,
                                   start_feat_motion, params, basesize):
    tm = start_positions.values()[0].shape[1] - 1
    return [{'positions': {k: (v[:, min(tm,t):min(tm,t)+1, :] + \
                               (start_positions['vel' + k][:, min(tm,t):min(tm,t)+1, :] \
                                * (t-tm) if 'vel' + k in start_positions and t > tm else 0.)) \
                           for k, v in start_positions.items()}} for t in range(T)]

