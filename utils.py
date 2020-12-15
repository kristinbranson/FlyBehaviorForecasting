import torch
import numpy as np
import os

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
        motion_feats = torch.stack([motiondata[:, s : s + T, :] for s in range(t, t + t_stride * batch_sz, t_stride)])
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

def compute_position_errors(simulated_positions, true_positions, error_types):
    """
    Helper function to compute different loss functions (L1, L2, and angle difference),
    possibly over multiple fields (e.g. x, y)
    """
    errors = {}
    for name, error in error_types.items():
        if ':' in error:
            # The error is computed over multiple fields separated by '|'
            split = error.split(':')
            err = split[0]
            fields = split[1].split('|')
        else:
            # The error is computed over a single field
            err = error
            fields = [name]

        for f in fields:
            if err == 'ang':
                diff = (torch.remainder(simulated_positions[f], 2 * np.pi) -
                        true_positions[f]).abs()
                e = torch.where(diff > np.pi, 2 * np.pi - diff, diff)
            elif err == 'L1':
                e = (simulated_positions[f] - true_positions[f]).abs()
            elif err == 'L2':
                e = (simulated_positions[f] - true_positions[f]) ** 2
            else:
                assert(False, "Unknown error type %s" % err)
            
            errors[name] = errors[name] + e if name in errors else e

        if err == 'L2':
            errors[name] = torch.sqrt(errors[name])
        elif err == 'L1' or err == 'ang' and len(fields) > 1:
            errors[name] /= len(fields)
            
    return errors

def update_datasetwide_position_errors(errors, new_errors):
    progress_str = ""
    for k, v in new_errors.items():
        e = v.cpu().numpy()
        valid = ~np.isnan(e)
        if k in errors['all_errors']:
            errors['all_errors'][k].append(e)
            errors['sum_errors'][k] += (e * valid).sum(0)
            errors['sum_sqr_errors'][k] += ((e * valid) ** 2).sum(0)
            errors['counts'][k] += valid.astype(np.float64).sum(0)
        else:
            errors['all_errors'][k] = [e]
            errors['sum_errors'][k] = (e * valid).sum(0)
            errors['sum_sqr_errors'][k] = ((e * valid) ** 2).sum(0)
            errors['counts'][k] = valid.astype(np.float64).sum(0)
        progress_str += " %s=%f" % (k, (errors['sum_errors'][k] / errors['counts'][k]).mean())
    return progress_str

def makedirs(dir_name, exist_ok=True):
    if not exist_ok or not os.path.exists(dir_name):
        os.makedirs(dir_name)
