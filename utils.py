import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
import argparse

def get_real_positions_batch(t, trx, T, t_stride, batch_sz, basesize=None, motiondata=None):
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
        p = [v[s : s + T, :] for s in range(t, t + t_stride * batch_sz, t_stride)]
        positions[k] = torch.stack(p, 0)
    if basesize is not None:
        positions['b'] = basesize['minax'][None, None, :].repeat([batch_sz, T, 1])

    if motiondata is None:
        return positions
    else:
        motion_feats = torch.stack([motiondata[:, s : s + T, :] for s in range(t, t + t_stride * batch_sz, t_stride)]).permute(0, 2, 3, 1)
        return positions, motion_feats

def get_real_positions_batch_random(trx, T, batch_sz, basesize=None, motiondata=None, t_pad=2):
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
    device = trx.values()[0].device
    N = trx.values()[0].shape[0]
    ts = torch.randint(t_pad, N - T - t_pad - 1, [batch_sz, 1], device=device)
    Ts = ts + torch.arange(T, device=device).unsqueeze(0)
    positions = {k: v[Ts, :] for k,v in trx.items()}
    if basesize is not None:
        positions['b'] = basesize['minax'][None, None, :].repeat([batch_sz, T, 1])

    if motiondata is None:
        return positions
    else:
        motion_feats = motiondata[:, Ts, :].permute(1, 2, 3, 0)
        return positions, motion_feats

def add_velocities(positions, fields, prev=None):
    """
    Compute velocities based on fly positions for a subset of fields (e.g. x,y).  Each
    position field is a batch_sz X T X num_flies tensor
    prev is an optional tensor of positions for timestep t=-1
    """
    for k in fields:
        device = positions[k].device
        first = positions[k][:, 0:1, :] - prev[k][:, -1:, :] if prev is not None else \
                torch.zeros([positions[k].shape[0], 1, positions[k].shape[2]], device=device)
        positions['vel' + k] = torch.cat([first, positions[k][:, 1:, :] - positions[k][:, :-1, :]], 1)

def compute_position_errors(simulated_positions, true_positions, error_types, num_samples=1,
                            soft=False, eps=1e-8):
    """
    Helper function to compute different loss functions (L1, L2, and angle difference),
    possibly over multiple fields (e.g. x, y)
    """
    errors = {}
    softmin = torch.nn.Softmin(1)
    for name, error in error_types.items():
        weight = 1.
        if ':' in error:
            # The error is computed over multiple fields separated by '|'
            split = error.split(':')
            err = split[0]
            fields = split[1].split('|')
            weight = float(split[2])
        else:
            # The error is computed over a single field
            err = error
            fields = [name]

        for f in fields:
            sim, true = simulated_positions[f], true_positions[f]
            invalid = torch.isnan(true)
            invalid_s = invalid.repeat([s1//s2 for s1, s2 in zip(sim.shape, true.shape)])
            true[invalid] = 0
            sim[invalid_s] = 0
            if sim.ndim < 4:
                sim = sim.view([-1, num_samples] + list(sim.shape[1:]))
                true = true.view([-1, 1] + list(true.shape[1:]))
            if err == 'ang':
                diff = (torch.remainder(sim, 2 * np.pi) - torch.remainder(true, 2 * np.pi)).abs()
                e = torch.where(diff > np.pi, 2 * np.pi - diff, diff)
            elif err == 'L1':
                e = (sim - true).abs()
            elif err == 'L2':
                e = (sim - true) ** 2
            else:
                assert(False, "Unknown error type %s" % err)
            
            errors[name] = errors[name] + e if name in errors else e

        if err == 'L2':
            errors[name] = torch.sqrt(errors[name] + eps)
        elif err == 'L1' or err == 'ang' and len(fields) > 1:
            errors[name] /= len(fields)

        if weight != 1. and soft:
            errors[name] *= weight

        if soft:
            # TODO (SJB): need scalar/weight before applying softmin
            errors[name + '_min'] = (softmin(errors[name]) * errors[name]).sum(1)
        else:
            errors[name + '_min'] = errors[name].min(1)[0]
            
    return errors

def update_datasetwide_position_errors(errors, new_errors):
    progress_str = ""
    for k, v in new_errors.items():
        e_orig = e = v.cpu().numpy()
        invalid = np.isnan(e)
        valid = ~invalid
        e[invalid] = 0
        ax = (0, 1) if e.ndim == 4 else 0
        if k in errors['all_errors']:
            errors['all_errors'][k].append(e_orig)
            errors['sum_errors'][k] += e.sum(ax)
            errors['sum_sqr_errors'][k] += (e ** 2).sum(ax)
            errors['counts'][k] += valid.astype(np.float64).sum(ax)
        else:
            errors['all_errors'][k] = [e_orig]
            errors['sum_errors'][k] = e.sum(ax)
            errors['sum_sqr_errors'][k] = (e ** 2).sum(ax)
            errors['counts'][k] = valid.astype(np.float64).sum(ax)
        progress_str += " %s=%f" % (k, (errors['sum_errors'][k] / errors['counts'][k]).mean())
    return progress_str

def plot_errors(args, error_types, colors=['blue','red','green', 'magenta', 'purple', 'black','cyan','pink','orange'], lines=['-', '-', '-', '-', '-', '-', '-', '-', '-']):
    sum_errors, sum_sqr_errors, counts = {}, {}, {}
    x = np.arange(1,args.t_sim + 1)
    exp_names = args.exp_names.split(',')
    model_types = args.model_type.split(',')
    assert(len(exp_names) == len(model_types))
    labels = args.labels.split(',') if args.labels is not None else exp_names
    savepath = '%s/figs/nstep%s' % (args.basepath, '/' + args.plot_name if args.plot_name is not None else '')
    makedirs('%s/figs' % (args.basepath), exist_ok=True)
    makedirs('%s/figs/nstep' % (args.basepath), exist_ok=True)  
    makedirs(savepath, exist_ok=True)  
    for exp_name, model_type in zip(exp_names, model_types):
        sum_errors[exp_name], sum_sqr_errors[exp_name], counts[exp_name] = {}, {}, {}
        with open('%s/metrics/%s/%s.npy' % (args.basepath, model_type, exp_name)) as f:
            keys = np.load(f)
            for k in keys:
                sum_errors[exp_name][k] = np.load(f)
                sum_sqr_errors[exp_name][k] = np.load(f)
                counts[exp_name][k] = np.load(f)

    for k in keys:
        plt.figure()
        ax = plt.axes([0,0,1,1])
        for i, exp_name in enumerate(exp_names):
            if k in sum_errors[exp_name]:
                err = sum_errors[exp_name][k].sum(1) / counts[exp_name][k].sum(1)
                sq_err = sum_sqr_errors[exp_name][k].sum(1) / counts[exp_name][k].sum(1)
                plt.errorbar(np.arange(1,err.shape[0] + 1), err, ls=lines[i], color=colors[i], label=labels[i], lw=3, alpha=0.8)  #, yerr=np.sqrt(-err ** 2 + sq_err)

        plt.title(k + " error")
        plt.xlabel('N-steps')
        plt.ylabel('Error rate')
        matplotlib.rc('font', size=22)
        matplotlib.rc('axes', titlesize=22)
        ax.legend(fontsize=20, loc='lower center', bbox_to_anchor=(0.5,1.25), ncol=3)
        plt.savefig('%s/%s_eval.pdf' \
            % (savepath, k), format='pdf', bbox_inches='tight')
        
def makedirs(dir_name, exist_ok=True):
    if not exist_ok or not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
def replace_in_file(filein, fileout, replace):
    f = open(filein,'r')
    filedata = f.read()
    f.close()

    for k, v in replace.items():
        filedata = filedata.replace(k, v)

    f = open(fileout,'w')
    f.write(filedata)
    f.close()

def nan_safe(v):
    v[np.isnan(v)] = 0
    return v

def geometrical_steps(f, max_v, start=0):
    if max_v < 0:
        return []
    ret = [start]
    curr = 1
    while curr <= max_v:
        i = start + int(curr)
        if i != ret[-1]:
            ret.append(i)
        curr *= f
    if ret[-1] != start + max_v:
        ret.append(start + max_v)
    return ret

def get_modelname(args, prefix="flynet"):
    return prefix + "_" + args.rnn_type \
        + str(args.num_iters)+'steps_'\
        + str(args.batch_sz)+'batch_sz_'\
        + str(args.learning_rate)+'lr_'\
        + str(args.t_past)+'tpast_'\
        + str(args.t_sim)+'tsim_'\
        + str(args.num_samples)+'samp_'\
        + str(args.num_motion_bins)+'bins_'\
        + str(args.h_dim)+'hids_'\
        + str(args.gamma)+'gamma_'\
        + str(args.num_rand_features)+'gamma_'\
        + '_btype:'+str(args.bin_type)\
        + '_mtype:'+str(args.model_type)\
        + '_rtype:'+str(args.rnn_type)\
        + '_stype:'+str(args.lr_sched_type)\
        + '_mot:'+str(args.motion_method) if args.exp_name is None else args.exp_name

def iter_graph(root, callback, args):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen or fn is None:
            continue
        seen.add(fn)
        if  fn.next_functions is not None:
            for next_fn, _ in fn.next_functions:
                if next_fn is not None and not hasattr(next_fn, 'variable'):
                    queue.append(next_fn)
        callback(fn, args)

stack = None
def register_hooks(var):
    fn_dict = {}
    global stack
    stack = []
    def hook_cb(fn, args):
        stack = args
        stack += [fn]
        def register_grad(grad_input, grad_output):
            if not all(t is None or (torch.all(~torch.isnan(t)) and torch.all(~torch.isinf(t))) for t in grad_input):
                import pdb; pdb.set_trace()
            if not all(t is None or (torch.all(~torch.isnan(t)) and torch.all(~torch.isinf(t))) for t in grad_output):
                import pdb; pdb.set_trace()
            assert all(t is None or torch.all(~torch.isnan(t)) for t in grad_input), "{fn} grad_input={grad_input} grad_output={grad_output}"
            assert all(t is None or torch.all(~torch.isnan(t)) for t in grad_output), "{fn} grad_input={grad_input} grad_output={grad_output}"
            
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb, stack)
    return stack

def get_memory_usage():
    tensors = []
    total_sz = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                sz = obj.element_size() * obj.nelement()
                tensors.append((sz, obj))
                total_sz += sz / 1000000000.0
        except:
            pass
    return total_sz, [t[1] for t in sorted(tensors, key=lambda x: x[0])]


"""parsing and configuration"""
def parse_args(fn = None):
    desc = "Pytorch implementation of FlyNetwork collections"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--t_past', type=int, default=30,
                        help='Use t_past frames from the past to initial the RNN state before simulating future frames')
    parser.add_argument('--t_sim', type=int, default=30,
                        help='Number of frames to simulate for each trajectory')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of random trajectory samples to simulate for each trajectory')
    parser.add_argument('--basepath', type=str, default='./',
                        help='Location to store results')
    parser.add_argument('--trx_name', type=str, default='eyrun_simulate_data.mat',
                        help='Name of each video file containing tracked trajectory positions')
    parser.add_argument('--use_cuda', type=int, default=1,
                        help='Whether or not to run on the GPU')
    parser.add_argument('--dataset_type', type=str, default='gmr',
                        help='Name of the dataset')
    parser.add_argument('--bin_type', type=str, default='perc',
                        help='Method used to bin RNN predicted motion outputs')
    parser.add_argument('--model_type', type=str, default='rnn50', help='Model architecture')
    parser.add_argument('--h_dim', type=int, default=100, help='RNN hidden state dimensions')
    parser.add_argument('--num_motion_bins', type=int, default=101,
                        help='number of motion bins for RNN output')
    parser.add_argument('--batch_sz', type=int, default=None,
                        help='Number of trajectories in each batch')
    parser.add_argument('--loss_type', type=str, default='nstep',
                        help='Type of training loss function')
    parser.add_argument('--rnn_type', type=str, default='rnn', help='Type of RNN model')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--lr_sched_type', type=str, default='multstep',
                        help='Learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--num_iters', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--r_dim', type=int, default=128)
    parser.add_argument('--init_weights', type=str, default='kaiming')
    
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--validation_freq', type=int, default=1000,
                        help='Frequency of evaluation validation loss in terms of training iterations')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='Frequency of saving model in terms of training iterations')
    parser.add_argument('--save_dir', type=str, default="./models")
    parser.add_argument('--motion_method', type=str, default=None,
                        help='Method used to predict motion from bin probabilities (use multinomial or softmax)')
    parser.add_argument('--num_rand_features', type=int, default=None,
                        help='Add random vector of this dimensionality to state features as an input to the RNN, to represent stochasticity of behaviors')
    parser.add_argument('--save_path_male', type=str, default='./models/gmr/flyNet_gru50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_maleflies_10000')
    parser.add_argument('--save_path_female', type=str, default='./models/gmr/flyNet_gru50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_femaleflies_10000')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--plot_name', type=str, default=None)
    if fn is not None:
        fn(parser)
    
    args = parser.parse_args()

    if args.loss_type == 'nstep':
        # Default to training using nstep loss over 10 samples and 30 timesteps
        args.motion_method = 'softmax' if args.motion_method is None else args.motion_method
        args.num_iters = 30000 if args.num_iters is None else args.num_iters
        args.learning_rate = 0.005 if args.learning_rate is None else args.learning_rate
        args.batch_sz = 128 if args.batch_sz is None else args.batch_sz
    else:
        # Train using behavioral cloning with cross entropy loss
        assert(args.loss_type == 'cross_entropy')
        args.motion_method = 'multinomial' if args.motion_method is None else args.motion_method
        args.num_iters = 10000 if args.num_iters is None else args.num_iters
        args.learning_rate = 0.001 if args.learning_rate is None else args.learning_rate
        args.batch_sz = 256 if args.batch_sz is None else args.batch_sz
        
    if args.num_rand_features is None:
        args.num_rand_features = 0 if args.motion_method == 'multinomial' else 20

    import pdb; pdb.set_trace()
    return args
