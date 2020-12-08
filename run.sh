#!/bin/bash
## SINGULARITY IMAGE : branson_cuda10.simg
#

# Generate feratures
python gen_dataset.py --onehotF 0 --num_bin 101 --bin_type perc

# Train RNN models (note: this relies on pre-existing feature files in /groups, because paths are wrong)
CUDA_VISIBLE_DEVICES=0 nohup python -u main_gru_test.py --rnn_type gru --gender 0 --num_bin 101 --bin_type 'perc' &> nohup_gru0.out &; CUDA_VISIBLE_DEVICES=0 nohup python -u main_gru_test.py --rnn_type hrnn --gender 0 --num_bin 101 --bin_type 'perc' &> nohup_hrnn0.out
CUDA_VISIBLE_DEVICES=1 nohup python -u main_gru_test.py --rnn_type gru --gender 1 --num_bin 101 --bin_type 'perc' &> nohup_gru1.out &; CUDA_VISIBLE_DEVICES=1 nohup python -u main_gru_test.py --rnn_type hrnn --gender 1 --num_bin 101 --bin_type 'perc' &> nohup_hrnn1.out


CUDA_VISIBLE_DEVICES=0 nohup python simulate_rnn.py --dtype 'gmr' --mtype 'rnn50' --sim_type 'SMSF' --basepath='.' --model_epoch 10000 --save_path_male='./models/gmr/flyNet_gru50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_maleflies_10000' --save_path_female='./models/gmr/flyNet_gru50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_femaleflies_10000' &> nohup_simulate_gmr_smsf.out &
CUDA_VISIBLE_DEVICES=1 nohup python simulate_rnn.py --dtype 'hrnn' --mtype 'skip50' --sim_type 'SMSF' --basepath='.' --model_epoch 10000 --save_path_male='./models/gmr/flyNet_hrnn50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_maleflies_10000' --save_path_female='./models/gmr/flyNet_hrnn50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_femaleflies_10000' &> nohup_simulate_hrnn_smsf.out &
CUDA_VISIBLE_DEVICES=0 nohup python simulate_rnn.py --dtype 'gmr' --mtype 'rnn50' --sim_type 'LOO' --basepath='.' --model_epoch 10000 --save_path_male='./models/gmr/flyNet_gru50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_maleflies_10000' --save_path_female='./models/gmr/flyNet_gru50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_femaleflies_10000' &> nohup_simulate_gmr_loo.out &
CUDA_VISIBLE_DEVICES=1 nohup python simulate_rnn.py --dtype 'hrnn' --mtype 'skip50' --sim_type 'LOO' --basepath='.' --model_epoch 10000 --save_path_male='./models/gmr/flyNet_hrnn50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_maleflies_10000' --save_path_female='./models/gmr/flyNet_hrnn50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_femaleflies_10000' &> nohup_simulate_hrnn_loo.out &

python evaluate_nstep.py --dtype 'gmr' --mtype 'rnn50' --tsim 30 --basepath './' --save_path_male='./models/gmr/flyNet_gru50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_maleflies_10000' --save_path_female='./models/gmr/flyNet_gru50steps_512batch_sz_10000epochs_0.01lr_101bins_100hids__onehot0_visionF1_vtype:full_dtype:gmr_btype:perc_femaleflies_10000'

############################
##### DATA GENERATION ######
############################

## When you try with new bin sizes, you must run this code to
## generated bins and bin assignment (coverting from real value motions to bins) 
## The below script will generate in "./bins/[video_path]/":
##  - bin_motion_indx_[onehotF]_[bin_type]_[num_bin].npy (Bin Index Values)
##  - motion[num_bin]_[onehotF]_[bin_type].npy (Label - bin values)
##  - percentile_[num_bin].ny - binedges (only if bin_type set to percentile)

## OnehotF=1 : onehot encoding
python gen_dataset.py --onehotF 1 --num_bin 101 --bin_type linear

## OnehotF=0 : Gaussian smelaring over onehot label -- always used this one
python gen_dataset.py --onehotF 0 --num_bin 101 --bin_type linear


## SIDE NOTE: Data are [vision features + motion feature] x num_sequence
## Ex) [144 (72vision+72chamber) + 8 motion] x 50steps -> (x_dim, num_steps) : (152,50)


##########################
##### TRANING MODEL ######
##########################
## Trains GRU or HRNN model and saves trained model to "./models/[gmr/gmr91/pdb]/"

# RUN RNN
# gender = 0 -> male
python -u main_gru.py --rnn_type gru --gender 0 --num_bin 101 --bin_type 'perc' --save_dir './models/gmr/'
# gender = 1 -> female
python -u main_gru.py --rnn_type gru --gender 1 --num_bin 101 --bin_type 'perc'

# RUN HRNN 
python -u main_gru.py --rnn_type hrnn --gender 0 --num_bin 101 --bin_type 'perc'
python -u main_gru.py --rnn_type hrnn --gender 1 --num_bin 101 --bin_type 'perc'

# RUN CONV 
python -u main_conv.py --save_dir ./runs/conv4_cat50/ --epoch 25000 --gender 1 --vtype full --visionOnly 0 --vision 1 --lr 0.01 --h_dim 128 --t_dim 50  --dtype gmr
 


########################
##### SIMULATION  ######
########################
## Requirement: trained model
## Saves simulated trajectories "./trx/[gmr/gmr91/pdb/"
## Saves simulated motion "./motion/[video_path]/"
## Saves simulated velocity "./velocity/[video_path]/"
## Saves simulated centredist "./centredist/[video_path]/"
## Generates simulation in image file then set debug=1 (runs slower)

## You can specify male and female model using savepath_male 
## and savepath_female options. Otherwise, it will use pretrained models

## Simulate all made & female
python simulate_rnn.py --dtype 'gmr' --mtype 'rnn50' --sim_type 'SMSF' --model_epoch 200000
python simulate_autoreg.py --dtype 'gmr' --mtype 'conv4_cat50' --sim_type 'SMSF' --model_epoch 20000
python simulate_autoreg.py --dtype 'gmr' --mtype 'lr50' --sim_type 'SMSF'

## Leave One Out (LOO)  - one female AND one male simulated
python simulate_rnn.py --dtype 'gmr' --mtype 'rnn50' --sim_type 'LOO' --model_epoch 200000
python simulate_auto.py --dtype 'gmr' --mtype 'conv4_cat50' --sim_type 'LOO' --model_epoch 20000
python simulate_rnn.py --dtype 'gmr' --mtype 'rnn50' --sim_type 'Single' --model_epoch 200000 --fly_k 0
#python simulate_rnn.py --dtype 'gmr' --mtype 'rnn50' --sim_type 'Single' --model_epoch 200000 --fly_k 10
## Other options are RMSF and SMRF (Simulated Male Real Female)


########################
##### EVALUATION  ######
########################
##### N step prediction with min distance among 10 samples

## 1. For saving the error rates for male and female flies (Requires PlotF=0)
## Requirement: trained models (lr50, rnn50, skip50, conv4_cat50) 
## Saves error rates for 8 motions in './simtrx/[video_path]/[model_type]/"

python evaluate_nstep.py --dtype 'gmr' --mtype 'rnn50' --tsim 30
python evaluate_nstep.py --dtype 'gmr' --mtype 'skip50' --num_bin 51 
python evaluate_nstep.py --dtype 'gmr' --mtype 'conv4_cat50'

## 2. Figure the nstep prediction results once you gathered the
##    nstep error results for all models
## Requirement: run evaluate_nstep.py --plotF0 and have saved error rates for all model 
python evaluate_nstep.py --plotF 1 --gender 0 --dtype 'gmr' --mtype 'rnn50' --tsim 30   ##Fig8
python evaluate_nstep.py --plotF 1 --gender 1


##### Discriminator Evaluation

##1. Dataset Creation Process: gather simulated trajectories  
## Requirement: trained models (lr50, rnn50, skip50, conv4_cat50) 
## Saves simulated trajectory data at './fakedata/[video_path]/[model_type]/"

python evaluate_gan.py --operation faketrx --mtype rnn50 --dtype gmr 
python evaluate_gan.py --operation faketrx --mtype rnn50 --dtype gmr91
python evaluate_gan.py --operation faketrx --mtype rnn50 --dtype pdb

##2. Classify real vs simulated trajectroies
## Requirement: run evaluate_gan.py --operation faketrx for all model 
python evaluate_gan.py --operation eval_disc --dtype gmr    #Fig9
python evaluate_gan.py --operation eval_disc --dtype gmr91  #Fig9
python evaluate_gan.py --operation eval_disc --dtype pdb    #Fig9

##### Distribution Distance 
python evaluate_model.py --atype diff_models --dtype gmr   ##Fig5
python evaluate_model.py --atype diff_models --dtype gmr91 ##Fig5
python evaluate_model.py --atype diff_models --dtype pdb   ##Fig5

python evaluate_model.py --atype diff_simtype --dtype gmr   ##Fig14
python evaluate_model.py --atype diff_simtype --dtype gmr91 ##Fig14
python evaluate_model.py --atype diff_simtype --dtype pdb   ##Fig14
python evaluate_model.py --atype diff_nstep 


##### Chase Classifier
## Open MATLAB and run ScriptJAABAClassifySimData20200305.m
## You need to choose the model by choosing the correct model path



### Human Evaluation
python evaulate_huamn.py 


