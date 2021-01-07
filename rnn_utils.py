import torch
import torch.nn.functional as F

from fly_utils import compute_features, update_positions, binprobs2motion
from utils import compute_position_errors

def run_rnns(models, T, num_samples, start_positions, start_feat_motion, params, basesize,
             loss_type=None, num_real_frames=None, train=False, motion_method='multinomial',
             num_rand_features=0, error_types=None
):
    if num_real_frames is None:
        num_real_frames = start_positions.values()[0].shape[1]
    batch_sz = start_positions.values()[0].shape[0]
    hiddens = [m['hidden'] for m in models]
    results = []
    start_feat_motion = start_feat_motion.permute(0, 2, 3, 1)
    num_motion_feat = params['n_motions']
    num_motion_bins = params['binedges'].shape[0] - 1
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    
    for model in models:
        model['loss'] = 0.
        model['errors'] = {}
        if train:
            model['model'].train()
            model['optimizer'].zero_grad()
        else:
            model['model'].eval()

    positions_new, feat_motion_new, errors_all, binscores_all = {}, {}, {}, {}
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
            if torch.isnan(motion).sum():
                import pdb; pdb.set_trace()
            for k, v in positions_new_m.items():
                if torch.isnan(v).sum():
                    import pdb; pdb.set_trace()
                
            hiddens_new.append(hidden)
            
            # If training, add a loss term for the target bins based on the observed trainset
            if loss_type == 'cross_entropy':
                # Multiclass cross entropy loss using the trainset motion bin from t to t+1
                errors = compute_cross_entropy_errors(binscores, start_feat_motion[:, t, inds, :],
                                                      params)
                '''target_bins = motion2bins(start_feat_motion[:, t, inds, :], params)
                target_bins = torch.cat([target_bins.unsqueeze(1)] * num_samples, 1)
                sz = batch_sz * num_samples * num_flies * num_motion_feat
                target_bins = target_bins.contiguous().view([sz])
                binscores_f = binscores.view([sz, num_motion_bins])
                errors = cross_entropy(binscores_f, target_bins)'''
                errors_all[i] = errors
                loss = errors.mean()
                model['loss'] = model['loss'] + loss
            elif loss_type == 'nstep':
                # n-step k-sample error from "Evaluation metrics for behaviour modeling"
                # where n=T-num_real_frames, k=num_samples.  Here our RNN samples k random
                # trajectory samples n-steps into the future, and the loss is the minimum
                # distance among the k samples to the true observed trajectory.  For
                # training, we use a softmin instead of a min
                sim_positions = {k: v.view([batch_sz, num_samples, 1, len(inds)]) \
                                 for k,v in positions_new_m.items()}
                future_positions = {k: v.detach()[:, t+1, inds].view(batch_sz, 1, 1, len(inds)) \
                                    for k,v in start_positions.items()}
                #import pdb; pdb.set_trace()
                errors = compute_position_errors(sim_positions, future_positions, error_types,
                                             num_samples=num_samples,
                                             soft=train)
                errors_all[i] = errors
                for name in error_types:
                    err = errors[name + '_min'].mean() / T
                    model['loss'] = model['loss'] + err
                    model['errors'][name] = model['errors'][name] + err if name in model['errors'] else err/T
            else:
                assert(loss_type is None)
                
        results.append({'hidden': hiddens_new, 'positions': positions_new.copy(),
                        'feat_motion': feat_motion_new.copy(), 'errors': errors_all.copy(),
                        'binscores': binscores_all.copy()
        })

    if train:
        for model in models:
            #print("")
            #register_hooks(model['loss'])
            model['loss'].backward()
            model['optimizer'].step()
        
    return results

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen or fn is None:
            continue
        seen.add(fn)
        if  fn.next_functions is not None:
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            print(fn)
            if not all(t is None or (torch.all(~torch.isnan(t)) and torch.all(~torch.isinf(t))) for t in grad_input):
                import pdb; pdb.set_trace()
            if not all(t is None or (torch.all(~torch.isnan(t)) and torch.all(~torch.isinf(t))) for t in grad_output):
                import pdb; pdb.set_trace()
            assert all(t is None or torch.all(~torch.isnan(t)) for t in grad_input), "{fn} grad_input={grad_input} grad_output={grad_output}"
            assert all(t is None or torch.all(~torch.isnan(t)) for t in grad_output), "{fn} grad_input={grad_input} grad_output={grad_output}"
            
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

def motion2bins(f_motion,params):
    num_bins = params['binedges'].shape[0] - 1
    motion = f_motion.unsqueeze(3)
    edges = params['binedges'].transpose(0, 1)[None,None,:,:]
    binidx = ((motion >= edges).sum(3) - 1).clamp(min=0, max=num_bins - 1)
    return binidx

# TODO
def soft_nstep_loss(binscores, positions, target_positions, params, args, error_types):
    bin_positions # positions + binedges for each k
    # loss = softmin((target_positions - bin_positions) ** 2)


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

def compute_cross_entropy_errors(binscores, true_feat_motion, params):
    batch_sz, num_flies, num_motion_feat = true_feat_motion.shape
    num_samples = binscores.shape[0] // batch_sz
    num_motion_bins = binscores.shape[3]
    
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    target_bins = motion2bins(true_feat_motion, params)
    target_bins = torch.cat([target_bins.unsqueeze(1)] * num_samples, 1)
    sz = batch_sz * num_samples * num_flies * num_motion_feat
    target_bins_f = target_bins.contiguous().view([sz])
    binscores_f = binscores.view([sz, num_motion_bins])
    error = cross_entropy(binscores_f, target_bins_f)
    return error.contiguous().view(target_bins.shape)
