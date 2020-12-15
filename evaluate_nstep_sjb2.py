import torch
import numpy as np
from tqdm import tqdm
import argparse
import os

from utils import get_real_positions_batch, add_velocities, compute_position_errors, makedirs, update_datasetwide_position_errors
from rnn_utils import run_rnns, run_constant_velocity_baseline, run_stay_still_baseline
from fly_utils import load_fly_models, compute_model_inds, load_video, feature_dims, ERROR_TYPES_FLIES, \
    VELOCITY_FIELDS_FLIES, FLY_DATASET_PATH, video16_path, TEST

                                    
def compute_nstep_errors_on_video(models, video, exp_name, video_dir, args,
                                  run_prediction_model=run_rnns,
                                  error_types=None, velocity_fields=None, all_errors=None):
    """
    Compute and save nstep errors on a single video for simulated trajectories of
    length args.t_sim frames using args.t_past frames from the past
    """
    trx,motiondata,params,basesize = video
    t_stride = args.t_stride  # spacing between frames to sample trajectories in each video
    T_past = args.t_past # number of past frames used to load the RNN hidden state
    T_sim = args.t_sim   # number of random frames to simulate a trajectory for
    batch_sz = args.batch_sz  # batch_sz X num_flies will be processed in each batch
    
    t_start = args.t_start if args.t_start is not None else 0
    t_end = args.t_end if args.t_end is not None else trx['x'].shape[0]

    for m in models:
        m['hidden'] = m['model'].initHidden(batch_sz * len(m['inds']), use_cuda=args.use_cuda)
    
    print('Simulation Start %d %d %d...\n' % (t_start, t_end, T_sim))

    all_simulated_positions = {}
    progress = tqdm(enumerate(range(t_start + T_past, t_end - T_sim, t_stride * batch_sz)))
    for ii, t in progress:
        batch_sz = min(args.batch_sz, (trx['x'].shape[0] - t) // t_stride)
        if batch_sz != args.batch_sz:
            for m in models:
                m['hidden'] = m['model'].initHidden(batch_sz * len(m['inds']),
                                                    use_cuda=args.use_cuda)
        
        # Extract real fly positions from the dataset
        past_positions, past_feat_motion = \
            get_real_positions_batch(t - T_past, trx, T_past, t_stride, batch_sz,
                                     basesize=basesize, motiondata=motiondata)
        add_velocities(past_positions, velocity_fields)
        future_positions = \
            get_real_positions_batch(t, trx, T_sim, t_stride, batch_sz, basesize=basesize)
        add_velocities(future_positions, velocity_fields, prev=past_positions)
                                    
        # Run the RNN, first using the real fly positions from t-T_past:t, then using
        # simulated frames from t:t+T_sim
        results = run_prediction_model(models, T_past + T_sim, past_positions, past_feat_motion, params,
                           basesize)

        # Store simulated fly positions
        simulated_positions = {k: torch.cat([r['positions'][k] for r in results[T_past:]], 1) \
                               for k in results[0]['positions']}
        add_velocities(simulated_positions, velocity_fields, prev=results[T_past-1]['positions'])
        all_simulated_positions = {k: ((all_simulated_positions[k] \
                                        if k in all_simulated_positions else []) + \
                                       [v.cpu().numpy()]) for k, v in simulated_positions.items()}

        # compute nstep errors
        errors = compute_position_errors(simulated_positions, future_positions, error_types)

        progress_str = "t=%d, Mean Errors: " % t
        if all_errors is not None:
            progress_str += update_datasetwide_position_errors(all_errors, errors) 
        progress.set_description((progress_str))

    # Save simulated fly positions
    makedirs('%s/simtrx/%s/' % (args.basepath, video_dir), exist_ok=True)   
    makedirs('%s/simtrx/%s/%s' % (args.basepath, video_dir, args.model_type), exist_ok=True)
    for k, v in all_simulated_positions.items():
        np.save('%s/simtrx/%s/%s/%s_%s.mat' % (args.basepath, video_dir, args.model_type,
                                               exp_name, k), np.asarray(v))

    # Save simulated fly nstep errors
    makedirs('%s/metrics/%s/' % (args.basepath, video_dir), exist_ok=True)   
    makedirs('%s/metrics/%s/%s' % (args.basepath, video_dir, args.model_type), exist_ok=True)  
    for k, v in all_errors['all_errors'].items():
        np.save('%s/metrics/%s/%s/%s_%s.mat' % (args.basepath, video_dir, args.model_type,
                                                exp_name, k), np.asarray(v))
        print('Final %s error: %s' % (k, np.nanmean(np.asarray(v), axis=(0, 1, 3))))

    return all_simulated_positions
    
def compute_nstep_errors_on_dataset(video_list, args,
                                    run_prediction_model=run_rnns,
                                    error_types=ERROR_TYPES_FLIES,
                                    velocity_fields=VELOCITY_FIELDS_FLIES):
    """


    Compute and save nstep errors on a dataset of videos for simulated trajectories of
    length args.t_sim frames using args.t_past frames from the past
    """
    models = None
    device = torch.device("cuda" if args.use_cuda else "cpu")
    errors = {'all_errors': {}, 'sum_errors': {}, 'sum_sqr_errors': {}, 'counts': {}}
    
    for testvideo_num, video_dir in enumerate(video_list):
        # Load video dataset file
        matfile = '%s/%s/%s' % (args.dataset_path, video_dir, args.trx_name)
        video = load_video(matfile, device=device)
        trx,motiondata,params,basesize = video

        nans = torch.isnan(motiondata)
        if nans.any():
            print("Found nans in motiondata, replacing with 0s")
            motiondata[nans] = 0.
        if 'perc' in args.bin_type:
            binedges = np.load('./bins/percentile_%dbins.npy' % args.num_motion_bins)
            params['binedges'] = torch.tensor(binedges).to(device)
        else:
            binedges = params['binedges']

        params['mtype'] = args.model_type
        if models is None:
            args.y_dim = (binedges.shape[0] - 1) * params['n_motions']
            args.x_dim = feature_dims(args, params)
            models = []
            exp_name = args.model_type
            if args.model_type == 'baseline_const_vel':
                velocity_fields = [k for k in trx.keys()]
                run_prediction_model = run_constant_velocity_baseline
            elif args.model_type == 'baseline_const_vel_pos':
                velocity_fields = ['x', 'y', 'theta']
                run_prediction_model = run_constant_velocity_baseline
            elif args.model_type == 'baseline_still':
                run_prediction_model = run_stay_still_baseline
            else:
                run_prediction_model = run_rnns
                models = load_fly_models(args)
                compute_model_inds(models, basesize)
                exp_name = args.save_path_male.split('/')[-1]
        
        for i in range(args.num_random):
            print('testvideo %d/%d, random iter %d/%d %s' % \
                   (testvideo_num, len(video_list), i, args.num_random, video_dir))
            compute_nstep_errors_on_video(models, video, exp_name + "_" + str(i), video_dir,
                                          args, run_prediction_model=run_prediction_model,
                                          error_types=error_types,
                                          velocity_fields=velocity_fields,
                                          all_errors=errors
            )
            errors['all_errors'] = {}
            
            
            

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of FlyNetwork collections"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--t_stride', type=int, default=30, help='Compute nstep error over trajectories sampled every t_stride frames for each video')
    parser.add_argument('--t_past', type=int, default=30, help='Use t_past frames from the past to initial the RNN state before simulating future frames')
    parser.add_argument('--t_sim', type=int, default=30, help='Number of frames to simulate for each trajectory')
    parser.add_argument('--num_random', type=int, default=10, help='Number of frames to simulate for each trajectory')
    parser.add_argument('--dataset_path', type=str, default=FLY_DATASET_PATH, help='Location of real videos of observed trajectories')
    parser.add_argument('--basepath', type=str, default='./', help='Location to store results')
    parser.add_argument('--trx_name', type=str, default='eyrun_simulate_data.mat', help='Name of each video file containing tracked trajectory positions')
    parser.add_argument('--save_path_male', type=str, default='./models/gmr/flyNet_gru50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_maleflies_10000')
    parser.add_argument('--save_path_female', type=str, default='./models/gmr/flyNet_gru50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_femaleflies_10000')
    parser.add_argument('--use_cuda', type=int, default=1, help='Whether or not to run on the GPU')
    parser.add_argument('--dataset_type', type=str, default='gmr', help='Name of the dataset')
    parser.add_argument('--bin_type', type=str, default='perc', help='Method used to bin RNN predicted motion outputs')
    parser.add_argument('--model_type', type=str, default='rnn50', help='Model architecture')
    parser.add_argument('--h_dim', type=int, default=100, help='RNN hidden state dimensions')
    parser.add_argument('--num_motion_bins', type=int, default=101, help='number of motion bins for RNN output')
    parser.add_argument('--t_start', type=int, default=20, help='First frame to sample trajectories from')
    parser.add_argument('--t_end', type=int, default=None, help='Last frame to sample trajectories from')
    parser.add_argument('--batch_sz', type=int, default=1024, help='Number of trajectories in each batch')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    video_list = video16_path[args.dataset_type][TEST]
    compute_nstep_errors_on_dataset(video_list, args)

