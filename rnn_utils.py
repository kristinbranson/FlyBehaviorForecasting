import torch
import torch.nn.functional as F

from fly_utils import compute_features, update_positions, binscores2motion

def run_rnn_one_step(model, hidden, input_variable, num_motion_feat=8, num_bin=51, 
                          mtype='rnn'):
    model.eval()
    loss = 0
    T, batch_sz, D = input_variable.size()
   
    output, hidden = model.forward(input_variable, hidden)
    output = output.view([batch_sz, num_motion_feat, num_bin])
    prediction = F.softmax(output, dim=2)

    predictions = prediction.reshape([T, batch_sz, num_motion_feat, num_bin])
    return predictions, hidden


def run_stay_still_baseline(models, T, num_samples, start_positions, start_feat_motion, params, basesize):
    tm = start_positions.values()[0].shape[1] - 1
    return [{'positions': {k: v[:, min(tm,t):min(tm,t)+1, :] for k, v in start_positions.items()}} for t in range(T)]

def run_constant_velocity_baseline(models, T, num_samples, start_positions, start_feat_motion, params, basesize):
    tm = start_positions.values()[0].shape[1] - 1
    return [{'positions': {k: (v[:, min(tm,t):min(tm,t)+1, :] + (start_positions['vel' + k][:, min(tm,t):min(tm,t)+1, :] * (t-tm) if 'vel' + k in start_positions and t > tm else 0.)) for k, v in start_positions.items()}} for t in range(T)]

def run_rnns(models, T, num_samples, start_positions, start_feat_motion, params, basesize):
    num_real_frames = start_positions.values()[0].shape[1]
    batch_sz = start_positions.values()[0].shape[0]
    hiddens = [m['hidden'] for m in models]
    results = []
    start_feat_motion = start_feat_motion.permute(0, 2, 3, 1)
    num_motion_feat = params['n_motions']
    num_motion_bins = params['binedges'].shape[0] - 1
    
    for t in range(T):
        if t < num_real_frames:
            # Extract fly positions and motion from real trajectories
            positions = {k: torch.stack([v[:, t:t+1, :]] * num_samples, 1).view([-1, 1, v.shape[2]]) for k,v in start_positions.items()}
            feat_motion = torch.stack([start_feat_motion[:, t:t+1, :, :]] * num_samples, 1).view([-1, 1, start_feat_motion.shape[2], start_feat_motion.shape[3]])
        else:
            # Update fly positions from the last simulated motion prediction
            feat_motion = feat_motion_new
            positions = positions_new
        
        assert(feat_motion.shape[-1] == num_motion_feat)
        feats = compute_features(positions, feat_motion, params)
        num_feats = feats.shape[-1]
        feat_motion_new = feat_motion.clone()
        hiddens_new = []
        for i, (model, hidden) in enumerate(zip(models, hiddens)):
            inds = model['inds']
            num_flies = len(inds)
            feats_m = feats[:, :, inds, :].transpose(0, 1).contiguous().view(1, -1, num_feats)
            preds, hidden = run_rnn_one_step(model['model'], hidden, feats_m,
                                             num_bin=num_motion_bins,
                                             num_motion_feat=num_motion_feat)
            
            preds = preds.reshape([1, batch_sz*num_samples, num_flies, num_motion_feat,
                                   num_motion_bins])
            motion = binscores2motion(preds, params)
            feat_motion_new[:, :, inds, :] = motion.transpose(0, 1)
            hiddens_new.append(hidden)

        positions_new = update_positions(positions, basesize, feat_motion_new, params)
        results.append({'hidden': hiddens_new, 'positions': positions_new,
                        'feat_motion': feat_motion_new})
    return results
