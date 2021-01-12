import torch
import torch.nn.functional as F

from fly_utils import compute_features, update_positions, binprobs2motion
from utils import compute_position_errors

def run_rnns(models, T, num_samples, start_positions, start_feat_motion, params, basesize,
             num_real_frames=None, train=False, motion_method='multinomial',
             num_rand_features=0, debug=0
):
    if num_real_frames is None:
        num_real_frames = start_positions.values()[0].shape[1]
    batch_sz = start_positions.values()[0].shape[0]
    hiddens = [m['hidden'] for m in models]
    results = []
    num_motion_feat = params['n_motions']
    num_motion_bins = params['binedges'].shape[0] - 1

    positions_new, feat_motion_new, binscores_all = {}, {}, {}
    for t in range(T):
        hiddens_new = []
        for i, (model, hidden) in enumerate(zip(models, hiddens)):
            inds = model['inds']
            num_flies = len(inds)
            if t < num_real_frames:
                # Extract fly positions and motion from real trajectories
                positions_m = {k: torch.stack([v[:, t:t+1, inds]] * num_samples, 1) \
                             .view([-1, 1, num_flies]) for k,v in start_positions.items()}
                feat_motion_m = torch.stack([start_feat_motion[:, t:t+1, inds, :]] * num_samples, 1). \
                              view([-1, 1, num_flies, start_feat_motion.shape[3]])
            else:
                # Update fly positions from the last simulated motion prediction
                positions_m = positions_new[i]
                feat_motion_m = feat_motion_new[i]

            # Compute state features using the current position
            assert(feat_motion_m.shape[-1] == num_motion_feat)
            feats_m = compute_features(positions_m, feat_motion_m, params)
            if num_rand_features > 0:
                rand = torch.randn(list(feats_m.shape[:-1]) + [num_rand_features],
                                   device=feats_m.device)
                feats_m = torch.cat([feats_m, rand], 3)
            num_feats = feats_m.shape[-1]        
            feats_m = feats_m.transpose(0, 1)
            feats_m = feats_m.contiguous().view(1, -1, num_feats)
            if train:
                feats_m = torch.autograd.Variable(feats_m, requires_grad=True)

            # Run one step of the RNN model
            binscores, hidden = model['model'].forward(feats_m, hidden)
            binscores = binscores.contiguous().view([batch_sz * num_samples * num_flies,
                                                     num_motion_feat, num_motion_bins]) * \
                                                     params['binprob_exp']
            binscores_all[i] = binscores
            binprobs = F.softmax(binscores, dim=2)
            binprobs = binprobs.reshape([1, batch_sz*num_samples, num_flies, num_motion_feat,
                                         num_motion_bins])

            # Sample a predicted motion from the motion bin probabilities
            motion = binprobs2motion(binprobs, params, method=motion_method).transpose(0, 1)
            feat_motion_new[i] = motion
            
            basesize_m = {k: v[inds] for k,v in basesize.items()}
            positions_new_m = update_positions(positions_m, basesize_m, motion, params)
            positions_new[i] = positions_new_m

            if debug > 0:
                if torch.isnan(motion).sum():
                    import pdb; pdb.set_trace()
                for h in hidden:
                    if torch.isnan(h).sum():
                        import pdb; pdb.set_trace()
                for k, v in positions_new_m.items():
                    if torch.isnan(v).sum():
                        import pdb; pdb.set_trace()
                
            hiddens_new.append(hidden)
                
        results.append({'hidden': hiddens_new, 'positions': positions_new.copy(),
                        'feat_motion': feat_motion_new.copy(), 'binscores': binscores_all.copy()
        })
        
    return results




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

