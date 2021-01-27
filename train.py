import torch
import numpy as np
from tqdm import tqdm
import os
import json
import shutil
from torch.utils.tensorboard import SummaryWriter

from flyNetwork_RNN import FlyNetworkGRU, FlyNetworkGRU2, FlyNetworkSKIP6
from utils import get_real_positions_batch, get_real_positions_batch_random, add_velocities, compute_position_errors, makedirs, update_datasetwide_position_errors, plot_errors, replace_in_file, nan_safe, get_modelname, geometrical_steps, register_hooks, get_memory_usage, parse_args
from rnn_utils import run_rnns, run_constant_velocity_baseline, run_stay_still_baseline
from fly_utils import *


def train(video_list, args,
          run_prediction_model=run_rnns,
          error_types=ERROR_TYPES_FLIES, log_errors_by_time=True):
    T_past = args.t_past # number of past frames used to load the RNN hidden state
    batch_sz = args.batch_sz  # batch_sz X num_flies will be processed in each batch
    if args.loss_type == 'cross_entropy':
        num_samples = 1
        T_sim = 0
    else:
        num_samples = args.num_samples  # number of random trajectories per fly
        T_sim = args.t_sim   # number of random frames to simulate a trajectory for
    
    train = [load_video_and_setup(v, args) for v in video_list[TRAIN]]
    val = [load_video_and_setup(v, args) for v in video_list[VALID]]

    if log_errors_by_time:
        log_errors_by_time = geometrical_steps(1.9, T_past - 1) + \
                             geometrical_steps(1.3, T_sim - 1, start=T_past)
    
    models = [{'name': 'male'}, {'name': 'female'}]
    for m in models:
        m['model'] = FlyNetworkSKIP6(args).cuda() if args.rnn_type =='hrnn' \
                     else (FlyNetworkGRU2(args).cuda() if args.rnn_type =='rnnc'\
                           else FlyNetworkGRU(args).cuda())
        m['optimizer'] = torch.optim.Adam(m['model'].parameters(), lr=args.learning_rate)
        m['scheduler'] = lr_scheduler_init(m['optimizer'], args)

    savedir = args.save_dir + "/" + args.dataset_type
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savename = get_modelname(args)

    writer = SummaryWriter()
    progress = tqdm(range(args.num_iters))
    val_str = ''
    for t in progress:
        i = np.random.randint(len(train)) # Random training video
        trx, motiondata, params, basesize = train[i]
        switch_video(train[i], models, batch_sz, num_samples)

        for m in models:
            m['model'].train()
            m['optimizer'].zero_grad()
            m['loss'] = 0
        
        # Batch of random trajectories from video i
        real_positions, real_feat_motion = get_real_positions_batch_random(trx, 
                    T_past + T_sim, batch_sz, basesize=basesize, motiondata=motiondata)

        results = run_rnns(models, T_past + T_sim, num_samples, real_positions,
                           real_feat_motion, params, basesize, train=True,
                           num_real_frames=T_past, motion_method=args.motion_method,
                           num_rand_features=args.num_rand_features,
                           debug=args.debug)
        compute_loss(models, real_positions, real_feat_motion, results, error_types, args,
                     params, loss_type=args.loss_type, train=True)
        
        progress_str = 't=%d' % t
        for mi, m in enumerate(models):
            if args.debug > 1:
                register_hooks(m['loss'])
                
            progress_str += ' train_loss_%s=%f' % (m['name'], m['loss'])
            writer.add_scalar('Loss/train_%s' % m['name'], m['loss'], t)
            if 'errors' in results[mi][0]:
                for n in results[mi][0]['errors']:
                    d = results[mi][0]['errors'][n].ndim
                    dims = [0, 2] if d == 3 else ([0, 1, 3] if d == 4 else [0, 1, 3, 4])
                    E = torch.cat([results[mi][ti]['errors'][n].mean(dims) for ti in range(len(results[mi]))], 0)
                    e = E.mean()
                    progress_str += ' %s=%f' % (n, e)
                    writer.add_scalar('Loss/train_%s_%s' % (m['name'], n), e, t)
                    if log_errors_by_time is not None:
                        for ti in log_errors_by_time:
                            et = E[ti]
                            writer.add_scalar('T_Loss/train_%s_%s_%d' % (m['name'], n, ti), et, t)
                            
            m['loss'].backward()
            #sz, tensors = get_memory_usage()
            #import pdb; pdb.set_trace()
            tensors = None
            m['optimizer'].step()
            m['scheduler'].step()
            del m['loss']
        results = None
            

        if (t + 1) % args.save_freq == 0:
            for m in models:
                save(m['model'], '%s/%s_%s_%d' % (savedir, savename, m['name'], t))
                
        if (t + 1) % args.validation_freq == 0:
            for m in models:
                m['loss'] = 0
            compute_validation_loss(val, models, args, error_types)
            val_str = ''
            for m in models:
                writer.add_scalar('Loss/val_%s' % m['name'], m['loss'], t)
                val_str += ' val_loss_%s=%f' % (m['name'], m['loss'])
                if 'errors' in m:
                    for n, E in m['errors'].items():
                        e = E.mean()
                        val_str += ' %s=%f' % (n, e)
                        writer.add_scalar('Loss/val_%s_%s' % (m['name'], n), e, t)
                        if log_errors_by_time is not None:
                            for ti in log_errors_by_time:
                                et = E[ti]
                                writer.add_scalar('T_Loss_val/%s_%s_%d' % (m['name'], n, ti), et, t)

        progress.set_description((progress_str + val_str))

    for m in models:
        save(m['model'], '%s/%s_%s' % (savedir, savename, m['name']))

    writer.close()


def compute_validation_loss(val, models, args, error_types):
    with torch.no_grad():
        T_past = args.t_past # number of past frames used to load the RNN hidden state
        T_sim = args.t_sim   # number of random frames to simulate a trajectory for
        batch_sz = args.batch_sz  # batch_sz X num_flies will be processed in each batch
        num_samples = args.num_samples  # number of random trajectories per fly
        count = 0
        for m in models:
            m['model'].eval()
            m['loss'] = 0.
            m['errors'] = {}
            m['counts'] = {}
        for v in val:
            trx, motiondata, params, basesize = v
            old_batch_sz_v = None
            for tv in range(1, trx['x'].shape[0] - T_past - T_sim - 2, batch_sz * T_past):
                batch_sz_v = min(batch_sz, (trx['x'].shape[0] - T_sim - 2 - tv) // T_past)
                if batch_sz_v != old_batch_sz_v:
                    switch_video(v, models, batch_sz_v, num_samples)
                    old_batch_sz_v = batch_sz_v
                val_positions, val_feat_motion = get_real_positions_batch(
                    tv, trx, T_past + T_sim, T_past, batch_sz_v, basesize=basesize,
                    motiondata=motiondata)
                results = run_rnns(models, T_past + T_sim, num_samples, val_positions,
                         val_feat_motion, params, basesize, train=False,
                         num_real_frames=T_past, motion_method=args.motion_method,
                         num_rand_features=args.num_rand_features,
                         debug=args.debug
                )
                if args.motion_method != 'direct':
                    compute_loss(models, val_positions, val_feat_motion, results, error_types,
                                 args, params, loss_type='cross_entropy', train=False)
                compute_loss(models, val_positions, val_feat_motion, results, error_types,
                             args, params, loss_type='nstep', train=False)

                for mi, m in enumerate(models):
                    if 'errors' in results[mi][0]:
                        for n in results[mi][0]['errors']:
                            d = results[mi][0]['errors'][n].ndim
                            dims = [0, 2] if d == 3 else ([0, 1, 3] if d == 4 else [0, 1, 3, 4])
                            E = torch.cat([results[mi][ti]['errors'][n].mean(dims) \
                                                         for ti in range(len(results[mi]))], 0)
                            if n not in m['errors']:
                                m['errors'][n] = E
                                m['counts'][n] = torch.ones_like(E)
                            else:
                                m['errors'][n][:E.shape[0]] += E
                                m['counts'][n][:E.shape[0]] += 1
                del results
                count += 1
        for m in models:
            m['loss'] /= count
            for n in m['errors']:
                m['errors'][n] /= m['counts'][n]
            del m['counts']

def compute_loss(models, real_positions, real_feat_motion, results, error_types, args,
                 params, loss_type='nstep', train=True):
    errors_all = {}
    batch_sz = real_positions.values()[0].shape[0]
    for mi, m in enumerate(models):
        inds = m['inds']
        t = 0
        #T = np.sum([r['positions'].values()[0].shape[1]] for r in results[mi]) 
        for result in results[mi]:
            Tc = result['positions'].values()[0].shape[1]
            if loss_type == 'cross_entropy':
                # Multiclass cross entropy loss using the trainset motion bin from t to t+1
                binscores = result['binscores']
                errors = compute_cross_entropy_errors(binscores, real_feat_motion[:, t:t+Tc, inds, :], params)
                loss = errors.mean()
                errors = {'cross_entropy': errors.permute(1, 2, 0, 3, 4) }
                m['loss'] = m['loss'] + loss
            elif loss_type == 'nstep':
                # n-step k-sample error from "Evaluation metrics for behaviour modeling"
                # where n=T-num_real_frames, k=num_samples.  Here our RNN samples k random
                # trajectory samples n-steps into the future, and the loss is the minimum
                # distance among the k samples to the true observed trajectory.  For
                # training, we use a softmin instead of a min
                positions_new_m = result['positions']
                num_samples = positions_new_m.values()[0].shape[0] // batch_sz
                sim_positions = {k: v.view([batch_sz, num_samples, Tc, len(inds)]) \
                                 for k,v in positions_new_m.items()}
                future_positions = {k: v.detach()[:, t:t+Tc, inds].view(batch_sz, 1, Tc, len(inds)) \
                                    for k,v in real_positions.items()}
                errors = compute_position_errors(sim_positions, future_positions, error_types,
                                                 num_samples=num_samples, soft=train)
                for name in error_types:
                    err = errors[name + '_min']
                    m['loss'] = m['loss'] + err.mean() / len(results[mi])
            else:
                assert(args.loss_type is None)
            t += Tc
            if not 'errors' in result:
                result['errors'] = {}
            result['errors'] = errors
    
def compute_cross_entropy_errors(binscores, true_feat_motion, params):
    batch_sz, T, num_flies, num_motion_feat = true_feat_motion.shape
    num_samples = binscores.shape[-3] // (T * batch_sz * num_flies)
    num_motion_bins = binscores.shape[-1]
    
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    target_bins = motion2bins(true_feat_motion, params)
    target_bins = torch.cat([target_bins.unsqueeze(2)] * num_samples, 2)
    sz = batch_sz * T * num_samples * num_flies * num_motion_feat
    target_bins_f = target_bins.contiguous().view([sz])
    binscores_f = binscores.view([sz, num_motion_bins])
    error = cross_entropy(binscores_f, target_bins_f)
    return error.contiguous().view(target_bins.shape)
                    
def motion2bins(f_motion, params):
    num_bins = params['binedges'].shape[0] - 1
    motion = f_motion.unsqueeze(4)
    edges = params['binedges'].transpose(0, 1)[None, None, None, :, :]
    binidx = ((motion >= edges).sum(4) - 1).clamp(min=0, max=num_bins - 1)
    return binidx.permute(1, 0, 2, 3)


def lr_scheduler_init(optimizer, args):

    if args.lr_sched_type == 'lambda':
        # Assuming optimizer has two groups.
        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    elif args.lr_sched_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.gamma)

    elif args.lr_sched_type == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    elif args.lr_sched_type == 'multstep':
         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.num_iters*i // 4 for i in range(1, 5)], gamma=args.gamma)

    elif args.lr_sched_type == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer)
    else:
        return None
    return scheduler

def switch_video(v, models, batch_sz, num_samples):
    trx, motiondata, params, basesize = v
    compute_model_inds(models, basesize)
    for m in models:
        m['hidden'] = m['model'].initHidden(batch_sz * len(m['inds']),
                                            device=trx.values()[0].device)

def train_args(parser):
    parser.add_argument('--dataset_path', type=str, default=FLY_DATASET_PATH,
                        help='Location of real videos of observed trajectories')


if __name__ == '__main__':
    args = parse_args(train_args)
    video_list = video16_path[args.dataset_type]
    train(video_list, args)
