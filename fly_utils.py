import torch
import h5py
import numpy as np

ERROR_TYPES_FLIES = {'x': 'L1', 'y': 'L1', 'position': 'L2:x|y', 'velocity': 'L2:velx|vely',
                     'theta': 'ang', 'wing_ang': 'ang:l_wing_ang|r_wing_ang',
                     'wing_len': 'L1:l_wing_len|r_wing_len'}
VELOCITY_FIELDS_FLIES = ['x', 'y']

MALE=0
FEMALE=1
TRAIN=0
VALID=1
TEST=2
FLY_DATASET_PATH = '/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Data/bowl/'

pbd_4u = [[
    'pBDPGAL4U_TrpA_Rig1Plate10BowlA_20110323T114748/',\
    'pBDPGAL4U_TrpA_Rig1Plate10BowlC_20110610T160613/',\
    #'pBDPGAL4U_TrpA_Rig1Plate15BowlA_20121220T161640',\ #21
    'pBDPGAL4U_TrpA_Rig1Plate15BowlB_20120203T150713/',\
    'pBDPGAL4U_TrpA_Rig1Plate15BowlB_20120713T083042/',\
    'pBDPGAL4U_TrpA_Rig1Plate15BowlC_20110831T100911/',\
    'pBDPGAL4U_TrpA_Rig1Plate15BowlD_20111117T092639/'],\
    [\
    'pBDPGAL4U_TrpA_Rig1Plate15BowlD_20120425T130405/',\
    #'pBDPGAL4U_TrpA_Rig2Plate14BowlB_20110504T102739',\ #19
    'pBDPGAL4U_TrpA_Rig2Plate14BowlB_20110720T140728/'],\
    [
    'pBDPGAL4U_TrpA_Rig2Plate17BowlA_20120315T142016/',\
    'pBDPGAL4U_TrpA_Rig2Plate17BowlA_20120823T144547/',\
    #'pBDPGAL4U_TrpA_Rig2Plate17BowlC_20111007T140348',\ #19
    #'pBDPGAL4U_TrpA_Rig2Plate17BowlC_20120601T150533',\ #19
    'pBDPGAL4U_TrpA_Rig2Plate17BowlD_20111209T134749/',\
    'pBDPGAL4U_TrpA_Rig2Plate17BowlD_20120104T103048/']
    ]

gmr_71 = [[\
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlB_20110707T154653/',
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlC_20110707T154934/', #19
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlA_20110916T155922/',
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlD_20110916T155353/',
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlA_20110921T085351/'],\
    [\
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110921T085346/',\
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/',
    ],
    [\
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/', #19
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110916T155917/',\
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110921T084823/']]
    #'GMR_71G01_AE_01_TrpA_Rig1Plate15BowlA_20120316T144027/',
    #'GMR_71G01_AE_01_TrpA_Rig1Plate15BowlB_20120316T144030/', #21
    #'GMR_71G01_AE_01_TrpA_Rig1Plate15BowlC_20120316T144000/', #30
    #'GMR_71G01_AE_01_TrpA_Rig1Plate15BowlD_20120316T144003/', #27

gmr_91 = [[\
    'GMR_91B01_AE_01_TrpA_Rig1Plate15BowlA_20120329T131415/',
    'GMR_91B01_AE_01_TrpA_Rig1Plate15BowlB_20120329T131418/', #19
    'GMR_91B01_AE_01_TrpA_Rig1Plate15BowlC_20120329T131338/',
    'GMR_91B01_AE_01_TrpA_Rig1Plate15BowlD_20120329T131343/',
    'GMR_91B01_AE_01_TrpA_Rig2Plate17BowlA_20120614T085804/'],\
    [\
    'GMR_91B01_AE_01_TrpA_Rig2Plate17BowlB_20120614T085806/',\
    'GMR_91B01_AE_01_TrpA_Rig2Plate17BowlD_20120614T090114/' #22
    ],
    [\
    'GMR_91B01_AE_01_TrpA_Rig2Plate17BowlA_20120614T085804/',\
    'GMR_91B01_AE_01_TrpA_Rig2Plate17BowlC_20120614T090112/']]

gmr_26 = [[\
    'GMR_26E01_AE_01_TrpA_Rig2Plate14BowlA_20110610T141315/',
    'GMR_26E01_AE_01_TrpA_Rig2Plate14BowlB_20110610T141310/', #19
    'GMR_26E01_AE_01_TrpA_Rig2Plate14BowlD_20110610T141503/',
    'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlA_20120531T140054/',
    'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlB_20120223T101810/'],\
    [\
    'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlB_20120531T140057/',\
    'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlC_20120531T140341/',\
    ],
    [\
    'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlC_20120223T101853/',
    #'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlD_20120223T101856/',\
    'GMR_26E01_AE_01_TrpA_Rig2Plate17BowlD_20120531T140344/']]



video16_path = {'gmr':gmr_71, 'pdb':pbd_4u, 'gmr91':gmr_91, 'gmr26':gmr_26}

video16_july = [[\
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlB_20110707T154653/',
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlC_20110707T154934/'],  
    [
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',\
    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/'],\
    [\
    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/' #19
    ]]

video16_original = [['GMR_71G01_AE_01_TrpA_Rig2Plate14BowlB_20110707T154653/'],\
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110921T085346/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/',
                    ],
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/', #19
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110916T155917/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110921T084823/']]

video16_v2 = [['GMR_71G01_AE_01_TrpA_Rig2Plate17BowlA_20110916T155922/'],\
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110921T085346/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/',
                    ],
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/', #19
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110916T155917/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110921T084823/']]

video16_v3 = [['GMR_71G01_AE_01_TrpA_Rig2Plate17BowlA_20110921T085351/'],\
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110921T085346/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/',
                    ],
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/', #19
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110916T155917/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110921T084823/']]


video16_v4 = [['GMR_71G01_AE_01_TrpA_Rig2Plate17BowlD_20110916T155353/'],\
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110921T085346/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/',
                    ],
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/', #19
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110916T155917/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110921T084823/']]


video16_v5 = [['GMR_71G01_AE_01_TrpA_Rig2Plate17BowlA_20110921T085351/'],\
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110921T085346/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110916T155358/',
                    ],
                    [\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/',
                    'GMR_71G01_AE_01_TrpA_Rig2Plate14BowlD_20110707T154929/', #19
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlB_20110916T155917/',\
                    'GMR_71G01_AE_01_TrpA_Rig2Plate17BowlC_20110921T084823/']]


default_params = {}
default_params['n_motions'] = 8
default_params['binprob_exp'] = 1.2
default_params['mindist'] = 3.885505
default_params['n_oma'] = 72
default_params['I'] = None
default_params['J'] = None
default_params['PPM'] = 7.790785
default_params['chamber_outline_matfile'] = \
    '/groups/branson/home/bransonk/behavioranalysis/code/SSRNN/SSRNN/Code/chamber_outline.mat'
default_params['ranges'] = \
    np.array([24.449421,13.040234,9.382661,0.143254,0.464102,0.506068,0.216591,0.220717])
default_params['FPS'] = 30.
default_params['arena_center_x'] = 512.5059
default_params['arena_center_y'] = 516.4722
default_params['arena_radius'] = 476.3236



def feature_dims(args, params):
    return params['n_oma'] * 2 + params['n_motions']

def compute_features(positions, feat_motion, params):
    return torch.cat(list(compute_fly_vision_features(positions, params)) + [feat_motion], 3)


def compute_fly_vision_features(positions, params, distF=0):
    """
    This is a ported version of Kristin/Daniel's compute_vision() function, which
    compute features that simulate a fly's vision based on the distance to other flies
    and the chamber wall (as described in Eyrun's paper).  This is a port to pytorch and
    adds batching support on all dimensions: batch_sz X T X num_flies
    """
    device = positions.values()[0].device

    # Fly center positions (x,y), ellipse angle and minor/major axes (theta, a, b)
    x = positions['x']  # batch_sz X T X num_flies
    y = positions['y']  # batch_sz X T X num_flies
    theta = positions['theta']  # batch_sz X T X num_flies
    a = positions['a'] / 2  # batch_sz X T X num_flies
    b = positions['b'] / 2  # batch_sz X T X num_flies

    batch_sz, T, num_flies = x.shape
    num_bins, num_bins_chamber = params['n_oma'], params['n_oma']
    chamber_xs, chamber_ys = params['J'].flatten(), params['I'].flatten()
    num_chamber_pts = chamber_xs.shape[0]

    if torch.isnan(x).any() or torch.isnan(y).any() or torch.isnan(theta).any() or torch.isnan(a).any() or torch.isnan(b).any():
        raise Exception(ValueError, 'compute_features position has nans')
    
    # vision angle bin size
    step = 2. * np.pi / num_bins
    step_chamber = 2. * np.pi / num_bins_chamber
    
    # flip
    theta_r = -theta + np.pi  # batch_sz X T X num_flies
    theta_r_pi2 = -theta + 3 * np.pi / 2  # batch_sz X T X num_flies
    
    # xs, ys are batch_sz X T X num_flies X 4 tensors containing the 4 points
    # of the ellipse minor/major axes
    cos_theta_r, sin_theta_r = torch.cos(theta_r), torch.sin(theta_r)
    cos_theta_pi2_r, sin_theta_pi2_r = torch.cos(theta_r_pi2), torch.sin(theta_r_pi2)
    xs = torch.stack([x + cos_theta_r * a, x - cos_theta_r * a,
                      x + cos_theta_pi2_r * b, x - cos_theta_pi2_r * b], 3)
    ys = torch.stack([y + sin_theta_r * a, y - sin_theta_r * a,
                      y + sin_theta_pi2_r * b, y - sin_theta_pi2_r * b], 3)

    # other flies positions ends of axes, xs_o and y_o are
    # batch_sz X T X num_flies X num_flies X 4
    xs_o = xs.unsqueeze(3).repeat((1, 1, 1, num_flies, 1))
    ys_o = ys.unsqueeze(3).repeat((1, 1, 1, num_flies, 1))
    
    # convert to this fly's coord system
    dx = -(x[:, :, :, None, None].repeat([1, 1, 1, 1, 4]) - xs.unsqueeze(2))
    dy = -(y[:, :, :, None, None].repeat([1, 1, 1, 1, 4]) - ys.unsqueeze(2))
    dist = torch.sqrt(dx ** 2 + dy ** 2) # batch_sz X T X num_flies X num_flies X 4
    dx = dx / dist # batch_sz X T X num_flies X num_flies X 4
    dy = dy / dist # batch_sz X T X num_flies X num_flies X 4
    cos_theta_r_e = cos_theta_r[:, :, :, None, None]
    sin_theta_r_e = sin_theta_r[:, :, :, None, None]
    angle = torch.atan2(cos_theta_r_e * dy - sin_theta_r_e * dx,
                        cos_theta_r_e * dx + sin_theta_r_e * dy)
    angle = torch.remainder(angle, 2.*np.pi) # batch_sz X T X num_flies X num_flies X 4

    # no -1 because of zero-indexing
    angle_bin = torch.floor(angle / step).int() # batch_sz X T X num_flies X num_flies X 4

    # dists_bin is a batch_sz X T X num_flies X num_flies X num_bins tensor.  Each
    # dist_bins[n,t,i,j, :] is set to the distance from fly i to fly j for all angle
    # bins subtended by the fly ellipse j (or infinity otherwise).
    # dists_bin_min[n,t,i,j] first stores the minimum distance from fly i to the 4 ellipse
    # extrema of fly j
    # min_bin[n,t,i,j], max_bin[n,t,i,j] store the minimum and maximum angle bin among
    # the 4 ellipse points of fly j in fly i's reference system
    # dists_bin[n,t,i,j,:] copies the minimum distances in dists_bin_min for all angle bins
    # between min_bin[n,t,i,j] and max_bin[n,t,i,j]
    # TODO: this tries to replicate logic version of the numpy function compute_vision()
    # for flies that pass through 2pi->0, which I'm not sure is completely correct
    dists_bin_inf = torch.full([batch_sz, T, num_flies, num_flies, num_bins], np.inf, device=device)
    dists_bin_min = dist.min(4)[0].unsqueeze(4).repeat([1, 1, 1, 1, num_bins]) 
    min_bin = angle_bin.min(4)[0].unsqueeze(4).repeat([1, 1, 1, 1, num_bins])
    max_bin = angle_bin.max(4)[0].unsqueeze(4).repeat([1, 1, 1, 1, num_bins])
    R = torch.arange(num_bins, device=device, dtype=min_bin.dtype)
    R = R.repeat([batch_sz, T, num_flies, num_flies, 1])
    dists_bin = torch.where((max_bin - min_bin) * 2 < num_bins - 1,
                            torch.where((R >= min_bin) & (R <= max_bin),
                                        dists_bin_min, dists_bin_inf),
                            torch.where((R <= min_bin) | (R >= max_bin),
                                        dists_bin_min, dists_bin_inf))

    # Ignore distances dist_bins[:,:,i,i,:] from one fly to itself
    E = torch.eye(num_flies, dtype=torch.bool, device=device)
    E = E[None, None, :, :, None].repeat([batch_sz, T, 1, 1, num_bins])
    dists_bin[E] = np.inf

    # vision features are based on the closest fly in each angle bin
    flyvision = dists_bin.min(3)[0]
    
   
    # compute chamber view.  dx, dy, dist, and angle are
    # batch_sz X T X num_flies X num_chamber_pts
    dx = -(x.unsqueeze(3) - chamber_xs.repeat([batch_sz, T, 1, 1]))
    dy = -(y.unsqueeze(3) - chamber_ys.repeat([batch_sz, T, 1, 1]))
    dist = torch.sqrt(dx ** 2 + dy ** 2)
    dx = dx / dist 
    dy = dy / dist
    cos_theta_r_e = cos_theta_r.unsqueeze(3) 
    sin_theta_r_e = sin_theta_r.unsqueeze(3) 
    angle = torch.atan2(cos_theta_r_e * dy - sin_theta_r_e * dx,
                        cos_theta_r_e * dx + sin_theta_r_e * dy)
    angle = torch.remainder(angle, 2. * np.pi)  # batch_sz X T X num_flies
    if torch.isnan(angle).any():
        raise Exception(ValueError, 'angle to arena wall is nan')
    angle_bin = torch.floor(angle / step_chamber).long()
    angle_bin = angle_bin.clamp(max = num_bins_chamber - 1)  # batch_sz X T X num_flies

    # batch_sz X T X num_flies, num_bins_chamber
    # TODO: use reduce='min' or reduce='mean' when it's supported by pytorch.  Currently, when
    # multiple chamber points map to the same angle bin, the numpy and pytorch implementations
    # will choose a particular one which probably won't match
    chambervision = torch.full([batch_sz, T, num_flies, num_bins_chamber], np.inf, device=device)
    chambervision.scatter_(3, angle_bin, dist) # , reduce='min')

    
    # interpolate / extrapolate gaps in the chambervision
    # For each bin of chambervision[n,t,i,:] that is unfilled (np.inf), interpolate between the
    # previous and next filled bin in chambervision[n,t,i,:]
    valid = chambervision != np.inf
    invalid = chambervision == np.inf
    ids = torch.arange(np.prod(chambervision.shape), device=device).reshape(chambervision.shape)
    fly_ids = torch.arange(np.prod(chambervision.shape[:3]), device=device) \
                   .reshape(chambervision.shape[:3]).unsqueeze(3).repeat([1, 1, 1, num_bins_chamber])
    chambervision_f = chambervision.flatten()
    chambervision_f, fly_ids_f, valid_f = chambervision.flatten(), fly_ids.flatten(), valid.flatten()
    filled_ids = ids[valid]
    unfilled_ids = ids[invalid]
    id_to_prev_filled_ind = valid_f.cumsum(0) - 1
    id_to_next_filled_ind = valid_f.sum() - valid_f.int().flip(0).cumsum(0).flip(0)
    prev_id = filled_ids[id_to_prev_filled_ind[unfilled_ids]]
    prev_valid = fly_ids_f[prev_id] == fly_ids_f[unfilled_ids]
    next_id = filled_ids[id_to_next_filled_ind[unfilled_ids]]
    next_valid = fly_ids_f[next_id] == fly_ids_f[unfilled_ids]
    chambervision_f[unfilled_ids] = \
            torch.where(prev_valid,
                torch.where(next_valid,
                            chambervision_f[prev_id] + (unfilled_ids - prev_id) *
                            (chambervision_f[next_id] - chambervision_f[prev_id]) /
                            (next_id - prev_id),
                            chambervision_f[prev_id]
                ),
                torch.where(next_valid,
                            chambervision_f[next_id],
                            torch.full(unfilled_ids.shape, np.inf, device=device))
                )
    '''
    Test cases:
    In [148]: chambervision[0,0,0, 32] = np.inf
    In [149]: chambervision[0,0,3, 28:58] = np.inf
    In [150]: chambervision[0,0,6, 0:10] = np.inf
    In [151]: chambervision[0,0,9, 60:] = np.inf
    '''

    flyvision = flyvision / params['PPM']
    chambervision = chambervision / params['PPM']

    if not distF:
        flyvision = 1 - torch.clamp(.05 * torch.clamp(flyvision - 1., min=0.)**0.6, max=1.)
        chambervision = torch.clamp(.5**(((chambervision - params['mindist']) * .4) * 1.3), max=1.)

    return flyvision, chambervision


def update_positions(positions_prev, basesize, f_motion, params): 
    """
    This is a ported version of Kristin/Daniel's update_position() function, which
    updates each flies position based on predicted motion outputs (f_motion).  
    This is a port to pytorch and adds batching support on all dimensions:
    batch_sz X T X num_flies
    """
    dfwd, dside, dtheta, dmajax, dawing1, dawing2, dlwing1, dlwing2 = \
        motion2pixels_and_frames(f_motion, params)
    xprev, yprev, thetaprev = positions_prev['x'], positions_prev['y'], positions_prev['theta']
    majax, awing1, awing2, lwing1, lwing2 = \
        basesize['majax'], basesize['awing1'], basesize['awing2'], basesize['lwing1'], basesize['lwing2']

    # update pose
    thetanext = torch.remainder(thetaprev + dtheta+np.pi,2.*np.pi)-np.pi
    # I would have used thetaprev, but Eyrun's code uses thetanext
    costh = torch.cos(-thetanext)
    sinth = torch.sin(-thetanext)
    dx = dfwd * costh - dside * sinth
    dy = dfwd * sinth + dside * costh
    xnext = xprev + dx
    ynext = yprev + dy

    # force to stay within the arena
    dx_arena = xnext - params['arena_center_x']
    dy_arena = ynext - params['arena_center_y']
    r_arena = torch.sqrt(dx_arena**2. + dy_arena**2.)
    theta_arena = torch.atan2(dy_arena,dx_arena)

    anext = (dmajax + 1.) * majax
    l_wing_ang_next = -awing1 - dawing1
    r_wing_ang_next = awing2 + dawing2
    l_wing_len_next = (dlwing1 + 1.) * lwing1
    r_wing_len_next = (dlwing2 + 1.) * lwing2

    positions_next = {'x': xnext, 'y': ynext, 'theta': thetanext, 'a': anext, 'b': positions_prev['b'],
                 'l_wing_ang': l_wing_ang_next, 'r_wing_ang': r_wing_ang_next, 'l_wing_len': l_wing_len_next,
                 'r_wing_len': r_wing_len_next}

    return positions_next

def motion2pixels_and_frames(f_motion, params):
    # compute x and y displacement given fwd and sideways displacement
    # and convert to pixels and frames
    mults = torch.tensor([params['PPM'] / params['FPS'], params['PPM'] / params['FPS'],
                          1 / params['FPS'], 1, 1, 1, 1, 1], device=params['ranges'].device)
    moves = f_motion * (params['ranges'] * mults)
    return [moves[...,i] for i in range(moves.shape[-1])]


def binscores2motion(binscores,params,noiseF=True):
    """
    This is a ported version of Kristin/Daniel's binscores2motion() function, which
    randomly samples a motion for each fly based on predicted bin scores from an RNN.
    This is a port to pytorch and adds batching support on all dimensions:
    batch_sz X T X num_flies
    """
    device = params['binedges'].device
    n_bins = params['binedges'].shape[0]-1
    f_motion = torch.zeros(binscores.shape[:-1], device=device)

    prob = binscores ** params['binprob_exp'] 
    prob = prob / prob.sum(binscores.ndim - 1).unsqueeze(binscores.ndim - 1)
    idx = torch.multinomial(prob.view(-1, n_bins), 1).view(binscores.shape[:-1])

    for v in range(params['n_motions']):
        binstart = params['binedges'][idx[..., v], v]
        binend = params['binedges'][idx[..., v] + 1, v]
        binwidth = binend - binstart
        if noiseF:
            f_motion[..., v] = binstart + torch.rand(1, device=device) * (binend - binstart)
        else:
            f_motion[..., v] = binstart

    return f_motion


def load_rnn(args, load_path):
    if 'skip' in args.model_type or 'hrnn' in args.model_type:
        from flyNetwork_RNN import FlyNetworkSKIP6
        model = FlyNetworkSKIP6(args)
    elif 'rnn' in args.model_type:
        from flyNetwork_RNN import FlyNetworkGRU
        model = FlyNetworkGRU(args)

    if args.use_cuda: model = model.cuda()
    print('Model Load %s' % load_path)
    model.load_state_dict(torch.load(load_path + '.pkl', map_location=lambda storage, 
                                loc: storage)[0])

    return model

def load_fly_models(args):
    """
    Load saved male/female fly RNN models
    """
    male_model = load_rnn(args, args.save_path_male)
    female_model = load_rnn(args, args.save_path_female)
    models = [{'name': 'male', 'model': male_model},
              {'name': 'female', 'model': female_model}]
    return models

def gender_classify(basesize):
    male_ind = torch.where(basesize < 19.5)[0]
    female_ind = torch.where(basesize > 19.5)[0]
    return male_ind, female_ind

def compute_model_inds(models, basesize):
    """
    Compute which fly indices are applicable to a male/female model
    """
    male_ind, female_ind = gender_classify(basesize['majax'])
    device = basesize['majax'].device
    simulated_male_flies = torch.arange(len(male_ind), device=device)
    simulated_female_flies = torch.arange(len(male_ind),len(male_ind)+len(female_ind),
                                          device=device)
    models[MALE]['inds'] = simulated_male_flies
    models[FEMALE]['inds'] = simulated_female_flies


def load_video(matfile, device=None):
    trx,motiondata,params,basesize = load_eyrun_data(matfile)
    if device is not None:
        trx = {k: torch.Tensor(v).to(device) for k,v in trx.items()}
        motiondata = torch.Tensor(motiondata).to(device)
        basesize = {k: torch.Tensor(v).to(device) for k,v in basesize.items()}
        params = {k: torch.Tensor(v).to(device) if type(v)==np.ndarray else v for k,v in params.items()}
    return trx,motiondata,params,basesize
    
def load_eyrun_data(matfile):
    f = h5py.File(matfile,'r')
    trx = {}
    trx['x'] = np.array(f['trx_x'])-1.
    trx['y'] = np.array(f['trx_y']) - 1.
    trx['theta'] = np.array(f['trx_theta'])
    trx['a'] = np.array(f['trx_a'])
    trx['b'] = np.array(f['trx_b'])
    trx['l_wing_ang'] = np.array(f['trx_l_wing_ang'])
    trx['l_wing_len'] = np.array(f['trx_l_wing_len'])
    trx['r_wing_ang'] = np.array(f['trx_r_wing_ang'])
    trx['r_wing_len'] = np.array(f['trx_r_wing_len'])

    motiondata = np.array(f['motiondata'])

    params = default_params.copy()
    params['mindist'] = f['mindist'][0]
    params['n_oma'] = int(f['n_oma'][0])
    params['I'] = np.array(f['I'])-1.
    params['J'] = np.array(f['J'])-1.
    params['PPM'] = f['PPM'][0]
    params['FPS'] = f['FPS'][0]
    params['binedges'] = np.array(f['binedges'])
    params['bincenters'] = np.array(f['bincenters'])
    params['ranges'] = np.array(f['ranges']).flatten()
    params['bg'] = np.array(f['bg']).T

    # arena is a circle, find the parameters
    params['arena_center_x'] = np.mean(params['J'])
    params['arena_center_y'] = np.mean(params['I'])
    
    # fly actually can't get all the way out to the arena boundary
    dctr =  np.sqrt( (trx['x']-params['arena_center_x'])**2. + \
                     (trx['y'] - params['arena_center_y'])**2. )
    params['arena_radius'] = np.nanmax(dctr)

    # median
    basesize = {}
    basesize['majax'] = np.nanmedian(trx['a'],axis=0)
    basesize['minax'] = np.nanmedian(trx['b'],axis=0)
    basesize['awing1'] = -np.nanmedian(trx['l_wing_ang'],axis=0)
    basesize['lwing1'] = np.nanmedian(trx['l_wing_len'],axis=0)
    basesize['awing2'] = np.nanmedian(trx['r_wing_ang'],axis=0)
    basesize['lwing2'] = np.nanmedian(trx['r_wing_len'],axis=0)

    return (trx,motiondata,params,basesize)
