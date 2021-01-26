import os, sys, csv
import pandas
import numpy as np
from util_vis import error_bar_plot

def pred2int(pred):

    if 'fake' in pred:
        return -1
    elif 'real' in pred:
        return 1
    return np.nan

def title2label(name):

    if 'data' in name:
        return 1
    else: 
        return -1

def title2model(name):

    if 'data' in name:
        return 0
    elif 'lr' in name:
        return 1
    elif 'conv' in name:
        return 2
    elif 'rnn' in name:
        return 3
    elif 'skip' in name:
        return 4


def barplot(times, model_names, color_list, labels, ftag, N, ylabel='Median Time (sec)'):

    times = np.asarray(times)
    error_bar_plot(times, model_names, color_list, 'human_exp', ftag, \
        N=N, text_labels=labels, vmax=1.,\
        ylabel=ylabel)



def get_data(labeler, model_names, prefix='labeler'):

    path ='/groups/branson/bransonlab/kwaki/forDaniel/'
    path = './huamn_eval_csv/'
    os.makedirs('./figs/human_exp/', exist_ok=True)
    
    models, names, labels, preds, times = [], [], [], [], []
    with open(path+'%s%s.csv' % (prefix, labeler)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:

            if len(row) == 3:
                names.append(row[0])
                models.append(title2model(row[0]))
                labels.append(title2label(row[0]))
                preds.append(pred2int(row[1]))
                times.append(int(row[2]))
    
  
    models = np.asarray(models)
    labels = np.asarray(labels)
    preds  = np.asarray(preds)
    times  = np.asarray(times)

    accs, mt = get_acc(models, preds, labels, times)
    return accs, mt

def get_acc(models, preds, labels, times):

    accs, mt = [], []
    for i in range(5):
        
        mask = (models == i)
        preds_i  = preds[mask]
        labels_i = labels[mask]

        label1 = labels_i == 1
        label0 = labels_i == -1
        pred1 = preds_i[label1]
        pred0 = preds_i[label0]
        #import pdb; pdb.set_trace()
 
        true_positive = sum(pred1>0) / len(label1)
        true_negative = sum(pred0<0) / len(label0)
        accuracy = sum((preds_i * labels_i)>0) / len(labels_i)
        med_time = np.median(times[mask])
        avg_time = np.mean(times[mask])
        std_time = np.std(times[mask])
        num_positive = sum(preds_i>0)
        num_negative = sum(preds_i<0)
        print('%s TP %f TN %f ACC %f #pos %d #neg %d Time %f %f +- %f' \
                % (model_names[i], true_positive, true_negative, accuracy, num_positive, num_negative, med_time, avg_time, std_time))

        accs.append(accuracy)
        mt.append(med_time)

    accuracy = sum((preds * labels)>0) / len(labels)
    print ("Overall Accuracy : %f" % accuracy)

    return accs, mt

def get_long_acc(models, preds, labels, frames, times):

    accs, mt, acc_breakdown_list = [], [], []
    for i in range(5):
        
        mask = (models == i)
        preds_i  = preds[mask]
        labels_i = labels[mask]
        frames_i = frames[mask]

        label1 = labels_i == 1
        label0 = labels_i == -1
        pred1 = preds_i[label1]
        pred0 = preds_i[label0]
        #import pdb; pdb.set_trace()
 
        true_positive = sum(pred1>0) / len(label1)
        true_negative = sum(pred0<0) / len(label0)
        accuracy = sum((preds_i * labels_i)>0) / len(labels_i)
        med_time = np.median(times[mask])
        avg_time = np.mean(times[mask])
        std_time = np.std(times[mask])
        num_positive = sum(preds_i>0)
        num_negative = sum(preds_i<0)
        print('%s TP %f TN %f ACC %f #pos %d #neg %d Time %f %f +- %f' \
                % (model_names[i], true_positive, true_negative, accuracy, num_positive, num_negative, med_time, avg_time, std_time))

        accs.append(accuracy)
        mt.append(med_time)

        acc_breakdowns = []
        for j in range(4):

            mask_j = frames_i == j
            preds_ij = preds_i[mask_j]
            labels_ij = labels_i[mask_j]

            accuracy_j = sum((preds_ij * labels_ij)>0) / len(labels_ij)
            acc_breakdowns.append(accuracy_j)
        acc_breakdown_list.append(acc_breakdowns)

    accuracy = sum((preds * labels)>0) / len(labels)
    print ("Overall Accuracy : %f" % accuracy)

    return accs, mt, acc_breakdown_list


def get_long_data(labeler, model_names, prefix='labeler'):

    data_frame = {'60': [25500, 10500, 29500, 26500], '120': [11000, 20500, 24500, 18500],\
            '240':[27000, 15500, 17500, 28500], '480':[19000,12500,21500,16500]} 

    cnn_frame = {'60': [20000], '120': [29000], '240':[19000], '480':[23000]} 
    lr_frame  = {'60': [15000], '120': [17000], '240':[26000], '480':[24000]} 
    skip_frame= {'60': [25000], '120': [12000], '240':[21000], '480':[16000]} 
    rnn_frame = {'60': [10000, 25000], '120': [22000, 12000], '240':[28000, 21000], '480':[13000, 16000]} 

    mlist = {'rnn50':rnn_frame, 'skip50':skip_frame, 'conv4':cnn_frame, 'lr50':lr_frame, 'data':data_frame}

    path ='/groups/branson/bransonlab/kwaki/forDaniel/'
    path = './huamn_eval_csv/'

    
    models, names, labels, preds, frames, times = [], [], [], [], [], []
    with open(path+'%s%s.csv' % (prefix, labeler)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:

            if len(row) == 3:
                name = row[0]
                names.append(name)
                models.append(title2model(row[0]))
                labels.append(title2label(row[0]))
                preds.append(pred2int(row[1]))
                times.append(int(row[2]))

                index = name.split('_')[-1].replace('.mp4', '')
                model = row[0].split('_')[2].split('/')[-1]
                for i, key in enumerate(mlist[model].keys()):
                    if int(index) in mlist[model][key]:
                        frames.append(i)

 
    models = np.asarray(models)
    labels = np.asarray(labels)
    frames = np.asarray(frames)
    preds  = np.asarray(preds)
    times  = np.asarray(times)

    accs, mt, acc_breakdown_list = get_long_acc(models, preds, labels, frames, times)
    return accs, mt, acc_breakdown_list


def acc_curve(acc_bd):


    colors = ['red', 'blue', 'green', 'purple', 'orange']
    plt.figure()
    for i in range(5):
      
        plt.plot(np.arange(4), acc_bd[i], '-', color=colors[i] )

    plt.ylabel('Accuracy')
    plt.xlabel('Num Frames')

if __name__ == '__main__':

    if 1:
        labels = ['Labeler A', 'Labeler B', "Labeler C"]
        color_list = ['gray', 'red', 'green', 'deepskyblue', 'mediumpurple']
        model_names = ['DATA', 'LINEAR', 'CNN', 'RNN', 'HRNN']
        accs_labeler, mt_labeler = [], []
        for labeler in ['5','7','9']:
            accs, mt = get_data(labeler, model_names)
            accs_labeler.append(accs)
            mt_labeler.append(mt)

        barplot(mt_labeler[1:], model_names, color_list, labels, 'median_time', N=5)
        barplot(accs_labeler[1:], model_names, color_list, labels, 'accuracy', 5, ylabel='Human Accuracy')

    if 0:
        labels = ['Labeler A', 'Labeler B', "Labeler C"]
        color_list = ['gray', 'red', 'green', 'deepskyblue', 'mediumpurple']
        model_names = ['DATA', 'LINEAR', 'CNN', 'RNN', 'HRNN']
        accs_labeler, mt_labeler = [], []
        for labeler in ['a', 'b']:
            #accs, mt = get_data(labeler, model_names, 'long_')
            accs, mt, acc_bd = get_long_data(labeler, model_names, 'long_')
            accs_labeler.append(accs)
            mt_labeler.append(mt)

            barplot(mt_labeler, model_names, color_list, labels, 'long_median_time', N=5)
            barplot(accs_labeler, model_names, color_list, labels, 'long_accuracy', 5, ylabel='Human Accuracy')
            
            print(acc_bd)
            
    #error_bar_plot(acc_bd, model_names, color_list, 'human_exp', ftag, \
    #    N=N, text_labels=labels, vmax=None,\
    #    ylabel=ylabel)
    #import pdb; pdb.set_trace()

  


