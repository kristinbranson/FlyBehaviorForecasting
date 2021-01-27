#!/bin/bash


# Python/system setup
export SINGULARITY_BINDPATH="/nrs/branson,/groups/branson/home,/groups/branson/bransonlab,/scratch,/media"
singularity exec --nv /misc/local/singularity/branson_cuda10_mayank.simg /bin/bash
source /groups/branson/home/bransons/env_forecasting/bin/activate


# Train
python train.py

# Look at training progress at http://bransonk-ws3:6006/, where bransonk-ws3 is the name of the
# machine we are running tensorboard:
tensorboard --bind_all --logdir runs


# Evaluate nstep loss.  In this example, we evaluate twice using a t_past=30 and t_past=8 frames of
# observed context frames per trajectory
python eval_nstep.py --save_path_male=models/gmr/flynet_rnn30000steps_128batch_sz_0.005lr_30tpast_30tsim_10samp_101bins_100hids_0.3gamma_20gamma__btype:perc_mtype:rnn50_rtype:rnn_stype:multstep_mot:softmax_male_29999 --save_path_female=models/gmr/flynet_rnn30000steps_128batch_sz_0.005lr_30tpast_30tsim_10samp_101bins_100hids_0.3gamma_20gamma__btype:perc_mtype:rnn50_rtype:rnn_stype:multstep_mot:softmax_female_29999 --save_vis=1 --exp_name=nstep_tpast30
python eval_nstep.py --save_path_male=models/gmr/flynet_rnn30000steps_128batch_sz_0.005lr_30tpast_30tsim_10samp_101bins_100hids_0.3gamma_20gamma__btype:perc_mtype:rnn50_rtype:rnn_stype:multstep_mot:softmax_male_29999 --save_path_female=models/gmr/flynet_rnn30000steps_128batch_sz_0.005lr_30tpast_30tsim_10samp_101bins_100hids_0.3gamma_20gamma__btype:perc_mtype:rnn50_rtype:rnn_stype:multstep_mot:softmax_female_29999 --t_past=8 --save_vis=1 --exp_name=nstep_tpast8

# To visualize results, look at http://bransonk-ws3:8000/simtrx/GMR_71G01_AE_01_TrpA_Rig2Plate14BowlA_20110707T154658/rnn50/nstep_tpast30/, where bransonk-ws3 is the name of the machine used to run a webserver:
python -m SimpleHTTPServer

# Plotnstep error, where plots are stored in ./figs/nstep/tpast/
python eval_nstep.py --exp_names=nstep_tpast30,nstep_tpast10 --model_type=rnn50,rnn50 --plot_name=tpast
