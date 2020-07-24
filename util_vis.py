import os, math
import numpy as np
import h5py
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from matplotlib import cm
from matplotlib.lines import Line2D
import scipy.io as sio
PPM = 7.790785


def zeropad_hists(hists):

    lens = [hist.shape[0] for hist in hists]
    maxlen = max(lens) 
    new_hists = [] 
    for hist in hists:

        hist_len = hist.shape[0]
        if hist_len < maxlen:
            diff_len = maxlen - hist_len
            zeros = np.zeros((diff_len,))
            new_hist = np.hstack([hist, zeros])
        else:
            new_hist = hist

        assert(new_hist.shape[0] == maxlen)
        new_hists.append(new_hist)

    return new_hists

def manual_histogram(X, bin_width=4.76, normalize=True, min_val=0):
    
    N = len(X)
    if min_val < 0:
        X = X - min_val

    maxd = np.nanmax(X)
    num_bin = int(np.ceil(maxd/bin_width))
    counts = np.zeros(num_bin, dtype='float32')

    for x in X:
        if not np.isnan(x):
            bin_ind = int(x/bin_width)
            counts[bin_ind]+=1

    if normalize: counts = counts / np.sum(counts)
    return counts



def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_bar(width, bookkeep, ylabel, xlabel, fdir, ftag, \
        labels=['LR7', 'NN REG', 'RNN50 REG', 'HRNN50 REG', 'RNN50 (OLD)', 'RNN50', 'HRNN50', 'TRAIN DATA'],\
        colors=['red', 'green', 'magenta', 'blue', 'orange', 'purple','mediumpurple', 'cyan', 'yellow']):

    rects = []
    #fig, ax = plt.subplots(figsize=(12, 12))
    plt.figure()
    ax = plt.axes([0,0,1,1])
    for ii in range(bookkeep.shape[0]):
        rect = plt.bar(ii*width, bookkeep[ii], width, label=labels[ii], color=colors[ii])
    #rects.append(rect)
    #autolabel(rect)
    #for ii in range(bookkeep.shape[0]):
    #    rect = plt.bar(ii*width, bookkeep[ii], width, label=labels[ii], color=colors[ii])
    #    rects.append(rect)
    #    autolabel(rect)
    #ax.set_ylim((0.5, 1.0))
    #ax.set_xticks(np.arange(bookkeep.shape[0]))
    ax.set_xticklabels(labels)
    #plt.bar(0*width, bookkeep[0,1], width, label='LR7', color='green')
    #plt.bar(1*width, bookkeep[1,1], width, label='NN REG', color='brown')
    #plt.bar(2*width, bookkeep[2,1], width, label='RNN50 REG', color='orange')
    #plt.bar(3*width, bookkeep[3,1], width, label='HRNN50 REG', color='cyan')
    #plt.bar(4*width, bookkeep[4,1], width, label='RNN50', color='magenta')
    #plt.bar(5*width, bookkeep[5,1], width, label='HRNN50', color='blue')
    #plt.bar(6*width, bookkeep[6,1], width, label='TRAIN DATA')
    #plt.title(xlabel)
    ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5,1.25), ncol=3)
    plt.ylabel(ylabel)
    ax.get_xaxis().set_visible(False)

    SMALL_SIZE = 24
    matplotlib.rc('font', size=SMALL_SIZE)
    matplotlib.rc('axes', titlesize=SMALL_SIZE)
    plt.rcParams.update({'font.size': 24})
    plt.savefig('./figs/'+fdir+'/histdist_'+ftag+'.png', format='png', bbox_inches='tight') 
    plt.close()

def plot_histo_ind(bin_loc, hists, keep_perc, min_val, max_val, \
            fdir, ftag, bin_width=10, left_chop=0, right_chop=0, lw=5,\
            labels=['TEST DATA', 'LR7', 'RNN', 'HRNN',  \
                            'RNNREG', 'TRAIN DATA', 'NN7'],\
            colors=['red', 'green', 'skyblue', 'deepskyblue', 'yellow', \
                    'limegreen', 'palegreen','darkblue',\
                    'magenta', 'blue', 'orange', 'mediumpurple', 'cyan', \
                    'darkgreen', \
                    ]):

    if left_chop:
        min_notkeep_val = abs(min_val) 
    else:
        min_notkeep_val = abs(min_val) * (1.0-keep_perc)
    left_cutoff  = int(math.ceil(min_notkeep_val / bin_width))
    
    if right_chop:
        max_notkeep_val = max_val
    else:
        max_notkeep_val = max_val * (1.0-keep_perc)

    right_cutoff = int(math.ceil(max_notkeep_val / bin_width))
    right_cutoff = bin_loc.shape[0] - right_cutoff
    bin_loc_ = bin_loc[left_cutoff:right_cutoff]

    plt.figure() 
    ax = plt.axes([0,0,1,1])
    #plt.plot(bin_loc_, hists[0][left_cutoff:right_cutoff], 'r-', label='TEST DATA', alpha=0.7, lw=3)
    #plt.plot(bin_loc_, hists[2][left_cutoff:right_cutoff], 'm-', label='RNN' , alpha=0.7, lw=3)
    ##plt.plot(bin_loc, hists[3], 'b-', label='HRNN' , alpha=0.8, lw=3)
    #plt.plot(bin_loc_, hists[1][left_cutoff:right_cutoff], 'g-', label='LR7', alpha=0.7, lw=3)
    #plt.plot(bin_loc_, hists[4][left_cutoff:right_cutoff], ls='-', label='RNNREG', color='orange', alpha=0.7, lw=3)
    #plt.plot(bin_loc_, hists[5][left_cutoff:right_cutoff], ls='-', label='TRAIN DATA', color='mediumpurple', alpha=0.7, lw=3)
    ##plt.plot(bin_loc, hists[2], ls='-', label='NN7', alpha=0.6, lw=3, color='brown')
    if 'entre' in ftag or 'ist' in ftag:
        bin_loc_ = bin_loc_ / PPM

    for ii in range(len(hists)):
        plt.plot(bin_loc_, hists[ii][left_cutoff:right_cutoff], ls='-', label=labels[ii], color=colors[ii], alpha=0.7, lw=lw)
    #plt.plot(bin_loc_, hists[0][left_cutoff:right_cutoff], ls='-', label=labels[0], color=colors[0], alpha=0.7, lw=3)
    #plt.plot(bin_loc_, hists[1][left_cutoff:right_cutoff], ls='-', label=labels[1], color=colors[1], alpha=0.7, lw=3)
    #plt.plot(bin_loc_, hists[2][left_cutoff:right_cutoff], ls='-', label=labels[2], color=colors[2], alpha=0.7, lw=3)
    #plt.plot(bin_loc_, hists[3][left_cutoff:right_cutoff], ls='-', label=labels[3], color=colors[3], alpha=0.8, lw=3)
    #plt.plot(bin_loc_, hists[4][left_cutoff:right_cutoff], ls='-', label=labels[4], color=colors[4], alpha=0.7, lw=3)
    #plt.plot(bin_loc_, hists[5][left_cutoff:right_cutoff], ls='-', label=labels[5], color=colors[5], alpha=0.7, lw=3)
    #plt.ylim(0,0.035)
    plt.ylabel('Histogram')
    if 'elocity' in ftag : plt.xlabel('mm/s')
    if 'entre' in ftag or 'istance' in ftag \
            or 'Inter' in ftag: plt.xlabel('mm')
    if 'ngle' in ftag or 'Delta' in ftag: plt.xlabel('Radians')
    #plt.legend()
    #ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5,1.32), ncol=2)
    ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5,1.32), ncol=3)
    ax.get_xaxis().set_visible(True)

    plt.rcParams.update({'font.size': 22})
    plt.savefig('./figs/'+fdir+'/test_hist_'+ftag+str(bin_width)+'bw.pdf', format='pdf', bbox_inches='tight') 
    plt.close()


def plot_histo(hists, fdir, ftag, tt=200, bin_width=10, radianceF=False,\
                    labels=['TEST DATA', 'LR7', 'RNN', 'HRNN',  \
                            'RNNREG', 'TRAIN DATA', 'NN7']):

    plt.figure() 
    ax = plt.axes([0,0,1,1])
    ind = np.arange(hists[0].shape[0]) * bin_width + bin_width / 2
    if radianceF: 
        mid_ind = ind.shape[0] // 2
        mid_point = ind[mid_ind]
        ind = ind - mid_point
        tt0 =  tt//2
        tt1 = -tt//2
    else:
        tt0 = 0
        tt1 = tt

    #plt.plot(ind[tt0:tt1], hists[0][tt0:tt1], 'r-', label='TEST DATA', alpha=0.7, lw=3)
    #plt.plot(ind[tt0:tt1], hists[2][tt0:tt1], 'm-', label='RNN' , alpha=0.7, lw=3)
    ##plt.plot(ind[tt0:tt1], hists[3][tt0:tt1], 'b-', label='HRNN' , alpha=0.8, lw=3)
    #plt.plot(ind[tt0:tt1], hists[1][tt0:tt1], 'g-', label='LR7', alpha=0.7, lw=3)
    #plt.plot(ind[tt0:tt1], hists[4][tt0:tt1], ls='-', label='RNNREG', color='orange', alpha=0.7, lw=3)
    #plt.plot(ind[tt0:tt1], hists[5][tt0:tt1], ls='-', label='TRAIN DATA', color='mediumpurple', alpha=0.7, lw=3)
    ##plt.plot(ind[tt0:tt1], hists[2][tt0:tt1], ls='-', label='NN7', alpha=0.6, lw=3, color='brown')

    plt.plot(ind[tt0:tt1], hists[0][tt0:tt1], 'r-', label=labels[0], alpha=0.7, lw=3)
    plt.plot(ind[tt0:tt1], hists[1][tt0:tt1], 'g-', label=labels[1], alpha=0.7, lw=3)
    plt.plot(ind[tt0:tt1], hists[2][tt0:tt1], 'm-', label=labels[2], alpha=0.7, lw=3)
    plt.plot(ind[tt0:tt1], hists[3][tt0:tt1], 'b-', label=labels[3], alpha=0.8, lw=3)
    plt.plot(ind[tt0:tt1], hists[4][tt0:tt1], ls='-', label=labels[4], color='orange', alpha=0.7, lw=3)
    plt.plot(ind[tt0:tt1], hists[5][tt0:tt1], ls='-', label=labels[5], color='mediumpurple', alpha=0.7, lw=3)
    #plt.plot(ind[tt0:tt1], hists[2][tt0:tt1], ls='-', label='NN7', alpha=0.6, lw=3, color='brown')

    ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5,1.25), ncol=3)
    ax.get_xaxis().set_visible(True)
    plt.ylabel('Histogram')
    if 'elocity' in ftag : plt.xlabel('mm/s')
    if 'entre' in ftag or 'distance' in ftag : plt.xlabel('mm')
    if 'ngle' in ftag : plt.xlabel('radiance')
    plt.savefig('./figs/'+fdir+'/hist_'+ftag+str(bin_width)+'bw.png', format='png', bbox_inches='tight') 
    plt.close()


def plot_histo_inter(hists, fdir, ftag, tt=200, bin_width=10):

    plt.figure() 
    ind = np.arange(hists[0].shape[0]) * bin_width + bin_width / 2

    plt.plot(ind[:tt], hists[0][:tt], 'r-', label='TEST DATA', alpha=0.7, lw=3)
    plt.plot(ind[:tt], hists[5][:tt], 'm-', label='RNN' , alpha=0.7, lw=3)
    #plt.plot(ind[:tt], hists[3][:tt], 'b-', label='HRNN' , alpha=0.8, lw=3)
    plt.plot(ind[:tt], hists[1][:tt], 'g-', label='LR7', alpha=0.7, lw=3)
    plt.plot(ind[:tt], hists[3][:tt], ls='-', label='RNNREG', color='orange', alpha=0.7, lw=3)
    plt.plot(ind[:tt], hists[7][:tt], ls='-', label='TRAIN DATA', color='mediumpurple', alpha=0.7, lw=3)
    #plt.plot(ind[:tt], hists[2][:tt], ls='-', label='NN7', alpha=0.6, lw=3, color='brown')
    plt.ylabel('Histogram')
    if 'elocity' in ftag : plt.xlabel('mm/s')
    if 'entre' in ftag or 'distance' in ftag : plt.xlabel('pixels')
    if 'ngle' in ftag : plt.xlabel('radiance')
    plt.legend()
    plt.savefig('./figs/'+fdir+'/hist_'+ftag+str(bin_width)+'bw.png', format='png', bbox_inches='tight') 


def histogram(x, interpolation='spline36', fname=None, title=None):

    plt.figure()
    bins = np.linspace(0, 1.0, 100)
    #hist, _, _ = plt.hist((x/hmax), bins=bins, density=True)
    hist, _, _ = plt.hist(x, bins=bins, density=True)
    #plt.yticks(range(0,vmax,1000))
    #plt.yticks(range(0,1000))
    plt.title(title)
    plt.tight_layout()
    if fname:
        plt.savefig('./figs/'+fname+'.png', bbox_inches='tight')
    return hist


def density_map(grid, mtype, gender=0, vmax=0.005, interpolation='spline36'):

    methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
               'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
               'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    
    # Fixing random state for reproducibility
   

    plt.figure()   
    plt.imshow(grid, interpolation=interpolation, \
                    cmap='RdBu_r', vmin=0.0, vmax=vmax)

    plt.tight_layout()
    if gender == 0:
        plt.savefig('./figs/'+mtype+'_heatmap_male_'+interpolation+'_2.png', bbox_inches='tight', format='png')
    elif gender == 1:
        plt.savefig('./figs/'+mtype+'_heatmap_female_'+interpolation+'_2.png', bbox_inches='tight', format='png')
    else:
        plt.savefig('./figs/'+mtype+'_heatmap_all_'+interpolation+'_2.png', bbox_inches='tight', format='png')



