import numpy as np
import h5py
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import scipy.io as sio
import torch

from util import copy_hiddens

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




# compute vision features
#
# inputs:
# xs: x-coordinate of centroid for each fly in the current frame. shape = (n_flies).
# ys: y-coordinate of centroid for each fly in the current frame. shape = (n_flies).
# thetas: orientation for each fly in the current frame. shape = (n_flies).
# a_s: major-axis length for each fly in the current frame. shape = (n_flies).
# bs: minor-axis length for each fly in the current frame. shape = (n_flies).
# fly: which fly to compute vision features for. scalar int.
# mindist: optional. minimum distance from a fly to a chamber feature point. constant scalar.
# n_oma: optional. number of bins representing visual scene. constant scalar.
# I: optional. y-coordinates in pixels of points along the chamber outline. If None, then
# this value is taken from either default_I (if this is set) or read from chamber_outline_matfile.
# shape = (n_points).
# J: optional. x-coordinates in pixels of points along the chamber outline. If None, then
# this value is taken from either default_I (if this is set) or read from chamber_outline_matfile.
# shape = (n_points).
# PPM: optional. pixels per millimeter. constant scalar.
# chamber_outline_matfile: optional. mat file containing I and J
# (y- and x-coordinates in pixels of points along the chamber outline). string.
#
# outputs:
# flyvision: appearance of other flies to input fly. shape = (n_oma).
# chambervision: appearance of arena to input fly. shape = (n_oma).
def compute_vision(xs,ys,thetas,a_s,bs,fly,params, distF=0):
    device = xs.device

    other_inds = torch.cat((torch.arange(0,fly, device=device),torch.arange(fly+1,len(xs), device=device)), 0)
    n_flies = len(other_inds)
    
    x = xs[fly].item()
    y = ys[fly].item()
    theta = thetas[fly].item()

    if np.isnan(x) or np.isnan(y) or np.isnan(theta):
        flyvision = torch.zeros((params['n_oma']), device=device)
        chambervision = torch.zeros((params['n_oma']), device=device)
        flyvision_dist = torch.zeros((params['n_oma']), device=device)
        chambervision_dist = torch.zeros((params['n_oma']), device=device)
        return (flyvision, chambervision)
    
    # vision bin size
    step = 2.*np.pi/params['n_oma']
    
    # flip
    theta = -theta + np.pi
    
    # for rotating into fly's coordinate system
    rotvec = np.array([np.cos(theta),np.sin(theta)])

    # compute other flies view

    # initialize everything to be infinitely far away
    flyvision = torch.full([params['n_oma']], np.inf, device=device)

    # other flies positions
    x_ = xs[other_inds]
    y_ = ys[other_inds]
    ax_a = a_s[other_inds]/2.
    ax_b = bs[other_inds]/2.
    th = -thetas[other_inds]+np.pi

    xs = torch.zeros((n_flies,4), device=device)
    ys = torch.zeros((n_flies,4), device=device)
    
    # ends of axes
    xs[:,0] = x_ + torch.cos(th)*ax_a
    xs[:,1] = x_ - torch.cos(th)*ax_a
    ys[:,0] = y_ + torch.sin(th)*ax_a
    ys[:,1] = y_ - torch.sin(th)*ax_a
    xs[:,2] = x_ + torch.cos(th+np.pi/2)*ax_b
    xs[:,3] = x_ - torch.cos(th+np.pi/2)*ax_b
    ys[:,2] = y_ + torch.sin(th+np.pi/2)*ax_b
    ys[:,3] = y_ - torch.sin(th+np.pi/2)*ax_b
   
    # convert to this fly's coord system
    dx = xs-x
    dy = ys-y
    dist = torch.sqrt(dx**2+dy**2)
    dx = dx/dist
    dy = dy/dist
    angle = torch.atan2(rotvec[0]*dy - rotvec[1]*dx, 
                       rotvec[0]*dx + rotvec[1]*dy)
    angle = torch.remainder(angle,2.*np.pi)
    #angle[np.isnan(angle)] = 0. ##THIS LINE ADDED BY DANIEL.

    # no -1 because of zero-indexing
    b_all = torch.floor(angle/step).int()
   
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


# computes motion from frame t-1 to t
def compute_motion(xprev,yprev,thetaprev,xcurr,ycurr,thetacurr,
                   a,l_wing_ang,r_wing_ang,l_wing_len,r_wing_len,
                   basesize,t,fly,params=default_params):

    # forward and sideways displacement
    # I would have done this with thetaprev, but it seems Eyrun did this with thetacurr
    costh=np.cos(-thetacurr)
    sinth=np.sin(-thetacurr)
    dx = xcurr-xprev
    dy = ycurr-yprev
    dfwd = dx*costh + dy*sinth
    dside = -dx*sinth + dy*costh

    # orientation change
    dtheta = np.mod(thetacurr-thetaprev+np.pi,2*np.pi) - np.pi

    # displacement from relaxed pose
    dmajax = a / basesize['majax'][fly]-1.
    dlwing1 = l_wing_len/basesize['lwing1'][fly]-1.
    dlwing2 = r_wing_len/basesize['lwing2'][fly]-1.
    dawing1 = -l_wing_ang - basesize['awing1'][fly]
    dawing2 = r_wing_ang-basesize['awing2'][fly]

    # convert to mm and s
    dfwd = dfwd / params['PPM'] * params['FPS']
    dside=dside/params['PPM']*params['FPS']
    dtheta = dtheta*params['FPS']

    moves = np.array([dfwd,dside,dtheta,dmajax,dawing1,dawing2,dlwing1,dlwing2])

    # normalize
    f_motion = moves / params['ranges']

    return f_motion


def motion2pixels_and_frames(f_motion,params):

    # unnormalize motion
    moves = f_motion * params['ranges']

    # compute x and y displacement given fwd and sideways displacement
    # and convert to pixels and frames
    dfwd = moves[0] * params['PPM'] / params['FPS']
    dside = moves[1] * params['PPM'] / params['FPS']
    dtheta = moves[2] / params['FPS']
    dmajax = moves[3]
    dawing1 = moves[4]
    dawing2 = moves[5]
    dlwing1 = moves[6]
    dlwing2 = moves[7]

    return (dfwd,dside,dtheta,dmajax,dawing1,dawing2,dlwing1,dlwing2)


def update_position(xprev,yprev,thetaprev,
                    majax,awing1,awing2,lwing1,lwing2,
                    f_motion,params):


    (dfwd,dside,dtheta,dmajax,dawing1,dawing2,dlwing1,dlwing2) = motion2pixels_and_frames(f_motion,params)

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

    #margin = 25
    #if r_arena > params['arena_radius'] : #+ margin:
    #    xnext = np.cos(theta_arena)*params['arena_radius'] + params['arena_center_x']
    #    ynext = np.sin(theta_arena)*params['arena_radius'] + params['arena_center_y']

    anext = (dmajax + 1.) * majax
    l_wing_ang_next = -awing1 - dawing1
    r_wing_ang_next = awing2 + dawing2
    l_wing_len_next = (dlwing1 + 1.) * lwing1
    r_wing_len_next = (dlwing2 + 1.) * lwing2

    return xnext,ynext,thetanext,anext,l_wing_ang_next,r_wing_ang_next,l_wing_len_next,r_wing_len_next


def binscores2motion(binscores,params=default_params,noiseF=True):
    device = params['binedges'].device
    n_bins = params['binedges'].shape[0]-1
    f_motion = torch.zeros(params['n_motions'], device=device)
    for v in range(params['n_motions']):

        # select bin according to output probability
        inds = torch.arange(v*n_bins,(v+1)*n_bins, device=device)
        prob = binscores[inds]

        prob = prob**params['binprob_exp']
        prob = prob / torch.sum(prob)
        idx = torch.multinomial(prob, 1)
        #idx = np.argmax(prob, axis=0)

        binstart = params['binedges'][idx,v]
        binwidth = params['binedges'][idx+1,v]-params['binedges'][idx,v]
        if noiseF:
            f_motion[v] = binstart + torch.rand(1, device=device)*binwidth
        else:
            f_motion[v] = binstart

    return f_motion


def motion2binidx(f_motion,params=default_params):

    n_bins = params['binedges'].shape[0]-1
    binidx = np.zeros(params['n_motions'],dtype=int)
    for v in range(params['n_motions']):
        if f_motion[v] < params['binedges'][1,v]:
            binidx[v] = 0
        else:
            binidx[v] = np.nonzero(params['binedges'][1:-1,v]<=f_motion[v])[0][-1]+1

        if binidx[v] >= n_bins:
            raise
    return binidx


def plot_t_step_future(curpos,simtrx,params,ax=None,figsize=(15,15),colors=None,simulated_flies=None):

    if not isinstance(simtrx,list):
        simtrx = [simtrx]

    nsim = len(simtrx)
    n_flies = simtrx[0]['x'].shape[1]

    if simulated_flies is None:
        simulated_flies = np.arange(n_flies)

    if ax is None:
        fig,ax=plt.subplots(figsize=figsize)

    if colors is None:
        colors=get_default_fly_colors(n_flies)

    hbg=plt.imshow(params['bg'],cmap=cm.gray,vmin=0.,vmax=1.)
    
    hbodies,hflies,htexts=draw_flies(curpos['x'],curpos['y'],curpos['a'],curpos['b'],curpos['theta'],
                                     curpos['l_wing_ang'],curpos['r_wing_ang'],curpos['l_wing_len'],
                                     curpos['r_wing_len'],ax=ax,colors=np.append(colors[:,:-1],.5+np.zeros((n_flies,1)),axis=1))

    htrx=[]
    for simi in range(nsim):
        for flyi in range(len(simulated_flies)):
            fly=simulated_flies[flyi]
            if simi == nsim-1:
                htrxcurr,=ax.plot(simtrx[simi]['x'][:,fly],simtrx[simi]['y'][:,fly],\
                            '-',color=np.append(colors[fly,:-1],1),linewidth=3)
            else:
                htrxcurr,=ax.plot(simtrx[simi]['x'][:,fly],simtrx[simi]['y'][:,fly],\
                            '-',color=np.append(colors[fly,:-1],1),linewidth=1)
            htrx.append(htrxcurr)

    plt.axis('image')

    return (ax,hbg,htrx,hbodies,hflies,htexts)


def get_default_fly_colors(n_flies):

    colors=plt.get_cmap('tab20')(np.arange(n_flies))
    colors[:,:-1]*=.7
    return colors

def get_default_fly_colors_rb(n_flies):

    blue_colors=plt.get_cmap('Blues')(np.arange(n_flies//2)*15+50)
    red_colors =plt.get_cmap('Reds')(np.arange(n_flies//2)*15+50)
    colors = np.vstack([blue_colors, red_colors])
    #colors=plt.get_cmap('seismic')(np.arange(n_flies))
    colors[:,:-1]*=.6
    return colors


def get_default_fly_colors_pp(male_n_flies, female_n_flies):

    blue_colors=plt.get_cmap('Purples')(np.arange(male_n_flies)*15+50)
    red_colors =plt.get_cmap('pink')(np.arange(female_n_flies)*15+50)
    colors = np.vstack([blue_colors, red_colors])
    #colors=plt.get_cmap('seismic')(np.arange(n_flies))
    colors[:,:-1]*=.6
    return colors



def get_default_fly_colors_black(n_flies):

    blue_colors=plt.get_cmap('Blues')(np.arange(0,1)*15+100)[::-1]
    black_colors=plt.get_cmap('Purples')(np.arange(1,n_flies//2)*15+50)
    #blue_colors=plt.get_cmap('Blues')(np.arange(n_flies//2-1, n_flies//2)*15+50)
    red_colors =plt.get_cmap('Reds')(np.arange(0,1)*15+100)[::-1]                      
    #red_colors =plt.get_cmap('Reds')(np.arange(n_flies//2-1, n_flies//2)*15+50) 
    colors = np.vstack([blue_colors, black_colors, \
                        red_colors, black_colors])
    #colors=plt.get_cmap('seismic')(np.arange(n_flies))
    colors[:,:-1]*=.8
    return colors

def get_default_fly_colors_single(fly_j, n_flies):

    colors=plt.get_cmap('Blues')(np.arange(0,n_flies)*15+100)[::-1]
    #blue_colors=plt.get_cmap('Blues')(np.arange(fly_j+1,n_flies)*15+100)[::-1]
    #red_colors =plt.get_cmap('Reds')(np.arange(fly_j,fly_j+1)*15+100)[::-1] 

    colors[fly_j] = np.asarray([1,0,0,1]) 
    #colors = np.vstack([blue_colors, red_colors, blue_colors])
    colors[:,:-1]*=.8
    return colors

def get_default_fly_colors_gr(signal, n_flies):

    green_colors=np.tile(np.asarray([0.,1.,0.,1.]), [n_flies,1])
    #green_colors=plt.get_cmap('Greens')(np.arange(0,n_flies)*15+100)[::-1]
    #blue_colors=plt.get_cmap('Blues')(np.arange(fly_j+1,n_flies)*15+100)[::-1]
    colors =plt.get_cmap('Reds')(np.arange(0,n_flies)*15+100)[::-1][::-1]                      
    ind = np.argwhere(signal).flatten()
    colors[ind] = green_colors[ind]
    #colors = np.vstack([blue_colors, red_colors, blue_colors])
    colors[:,:-1]*=.8
    return colors

def get_default_fly_colors_female_quiz(n_flies):

    #blue_colors=plt.get_cmap('Blues')(np.arange(0,1)*15+100)[::-1]
    black_colors=plt.get_cmap('Purples')(np.arange(0,n_flies//2)*15+50)
    #blue_colors=plt.get_cmap('Blues')(np.arange(n_flies//2-1, n_flies//2)*15+50)
    red_colors =plt.get_cmap('Reds')(np.arange(0,1)*15+100)[::-1]                      
    black_colors=plt.get_cmap('Purples')(np.arange(0,n_flies//2)*15+50)
    colors = np.vstack([black_colors, red_colors, black_colors])
    #colors=plt.get_cmap('seismic')(np.arange(n_flies))
    colors[:,:-1]*=.8
    return colors


def get_default_fly_colors_male_quiz(n_flies):

    #blue_colors=plt.get_cmap('Blues')(np.arange(0,1)*15+100)[::-1]
    black_colors=plt.get_cmap('Purples')(np.arange(0,n_flies//2-1)*15+50)
    #blue_colors=plt.get_cmap('Blues')(np.arange(n_flies//2-1, n_flies//2)*15+50)
    blue_colors =plt.get_cmap('Blues')(np.arange(0,1)*15+150)[::-1]                      
    black_colors2=plt.get_cmap('Purples')(np.arange(0,n_flies//2+1)*15+50)
    colors = np.vstack([black_colors, blue_colors, black_colors2])
    #colors=plt.get_cmap('seismic')(np.arange(n_flies))
    colors[:,:-1]*=.8
    return colors





def draw_flies(x,y,a,b,theta,l_wing_ang=None,r_wing_ang=None,l_wing_len=None,r_wing_len=None,
             colors=None,linewidth=4,ax=None,doplotflyid=True, textOff=False, alpha=1):

    n_flies = len(x)
    if colors is None:
        colors = get_default_fly_colors(n_flies)

    hbodies=[]
    hwings = []
    htexts = []
    for fly in range(n_flies):
        if doplotflyid:
            flyid = fly
        else:
            flyid = None

        if (l_wing_ang is not None) and (l_wing_len is not None) and (r_wing_ang is not None) and (r_wing_len is not None):

            (hbody,hwing,htext) = draw_fly(x[fly],y[fly],a[fly],b[fly],theta[fly],
                                                l_wing_ang[fly],r_wing_ang[fly],
                                                l_wing_len[fly],r_wing_len[fly],
                                                color=colors[fly,:],
                                                linewidth=linewidth,
                                                ax=ax,
                                                fly=flyid,\
                                                textOff=textOff,\
                                                alpha=alpha)
        else:
            (hbody,hwing,htext) = draw_fly(x[fly],y[fly],a[fly],b[fly],theta[fly],
                                                color=colors[fly,:],
                                                linewidth=linewidth,
                                                ax=ax,
                                                fly=flyid,\
                                                textOff=textOff,\
                                                alpha=alpha)

        hbodies.append(hbody)
        hwings.append(hwing)
        htexts.append(htext)

    return hbodies,hwings,htexts


def draw_attention(x0, y0, theta, prob, r0=10, scale=1000, \
                    alpha=0.5, nbin=72, ax=None,color='b'):
  
    if ax is None:
        ax = plt

    angle_tick = 360/nbin
    angles = np.arange(nbin) * angle_tick
    
    x = (r0+prob*scale) * np.cos(np.pi*angles/180)
    y = (r0+prob*scale) * np.sin(np.pi*angles/180)
    pts = np.vstack([x,y]).T

    c = np.cos(-theta-np.pi)
    s = np.sin(-theta-np.pi)
    R = np.array([[c,s],[-s,c]])
    pts = pts.dot(R)

    pts[:,0] = pts[:,0] + x0
    pts[:,1] = pts[:,1] + y0
    atten_plt, = ax.plot(pts[:,0], pts[:,1], \
                        linestyle='-', color=color, alpha=alpha,\
                        linewidth=2)
    return atten_plt



def draw_triangle(x,y,a,b,theta, 
             color='b',linewidth=3,ax=None,fly=None, textOff=False):

    c = np.cos(-theta)
    s = np.sin(-theta)

    pts = np.array([[-a/2.,-b/2.],[-a/2.,b/2.],[a/2.,0.]])
    R = np.array([[c,s],[-s,c]])
    pts = pts.dot(R)
    pts[:,0] = pts[:,0] + x
    pts[:,1] = pts[:,1] + y

    if ax is None:
        ax = plt

    hbody, = ax.plot(pts[[0,1,2,0],0],pts[[0,1,2,0],1],color=color,linewidth=linewidth)
    return hbody

def draw_fly(x,y,a,b,theta,l_wing_ang=None,r_wing_ang=None,l_wing_len=None,r_wing_len=None,
             color='b',linewidth=3,ax=None,fly=None, textOff=False, alpha=1):

    c = np.cos(-theta)
    s = np.sin(-theta)

    pts = np.array([[-a/2.,-b/2.],[-a/2.,b/2.],[a/2.,0.]])
    R = np.array([[c,s],[-s,c]])
    pts = pts.dot(R)
    pts[:,0] = pts[:,0] + x
    pts[:,1] = pts[:,1] + y

    if ax is None:
        ax = plt
    
    hbody, = ax.plot(pts[[0,1,2,0],0],pts[[0,1,2,0],1],color=color,linewidth=linewidth, alpha=alpha)
    if fly is not None and not textOff:
        htext = ax.text(x,y,str(fly))
    else:
        htext = None

    if (l_wing_ang is not None) and (l_wing_len is not None) and (r_wing_ang is not None) and (r_wing_len is not None):

        pts = np.array([[np.cos(-l_wing_ang+np.pi)*l_wing_len,np.sin(-l_wing_ang+np.pi)*l_wing_len],[0,0],
                        [np.cos(r_wing_ang+np.pi)*r_wing_len,np.sin(r_wing_ang+np.pi)*r_wing_len]])
        pts = pts.dot(R)
        pts[:,0] = pts[:,0] + x
        pts[:,1] = pts[:,1] + y

        hwing, = ax.plot(pts[:,0],pts[:,1],color=(color*.7+.3), linewidth=linewidth, alpha=alpha)
    else:

        hwing = None

    return (hbody,hwing,htext)


def update_flies(hbodies,hwings,htexts,x,y,a,b,theta,l_wing_ang=None,r_wing_ang=None,l_wing_len=None,r_wing_len=None, colors=None):

    n_flies = len(hbodies)
    for fly in range(n_flies):

        color = colors[fly] if colors is not None else None
        update_fly(hbodies[fly],hwings[fly],htexts[fly],x[fly],y[fly],a[fly],b[fly],theta[fly],
                   l_wing_ang[fly],r_wing_ang[fly],
                   l_wing_len[fly],r_wing_len[fly], color=color)

    return



def update_fly(hbody,hwing,htext,x,y,a,b,theta,l_wing_ang=None,r_wing_ang=None,l_wing_len=None,r_wing_len=None, color=None):

    c=np.cos(-theta)
    s=np.sin(-theta)

    pts=np.array([[-a/2.,-b/2.],[-a/2.,b/2.],[a/2.,0.]])
    R=np.array([[c,s],[-s,c]])
    pts=pts.dot(R)
    pts[:,0]=pts[:,0]+x
    pts[:,1]=pts[:,1]+y

    hbody.set_data(pts[[0,1,2,0],0],pts[[0,1,2,0],1])
    if  color is not None: 
        hbody.set_color(color)
        hwing.set_color(color)

    if htext is not None:
        htext.set_position((x,y))

    if (hwing is not None) and \
            (l_wing_ang is not None) and (l_wing_len is not None) and (r_wing_ang is not None) and (r_wing_len is not None):

        pts=np.array([[np.cos(-l_wing_ang+np.pi)*l_wing_len,np.sin(-l_wing_ang+np.pi)*l_wing_len],[0,0],
                      [np.cos(r_wing_ang+np.pi)*r_wing_len,np.sin(r_wing_ang+np.pi)*r_wing_len]])
        pts=pts.dot(R)
        pts[:,0]=pts[:,0]+x
        pts[:,1]=pts[:,1]+y

        hwing.set_data(pts[:,0],pts[:,1])

    return


def refresh_plot(drawn_artists,ax,fig):

    for drawn_artist in drawn_artists:
        ax.draw_artist(drawn_artist)

    fig.canvas.draw()
    fig.canvas.flush_events()




