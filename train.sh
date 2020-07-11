#!/bin/bash

#output="-oo $(pwd)/out.txt"
cores="4"
memory="4000"  # MB per core
scratch="10000"  # MB per core
gpus="1"
clock="4:00"  # time limit: 4:00, 24:00, or 120:00
gpu_memory="10240"  # minimum GPU memory
warn="-wt 10 -wa INT"  # interrupt signal 10 min before timeout

cmd="bsub
    -n $cores
    -W $clock $output
    $warn
    -R 'select[gpu_mtotal0>=$gpu_memory] rusage[mem=$memory,scratch=$scratch,ngpus_excl_p=$gpus]'
    
    python lstm_train_test.py --test --test_features '/cluster/scratch/vjose/argoverse/forecasting_features_val.pkl' --model_path '/cluster/scratch/vjose/argoverse/' --use_social --use_delta --normalize --obs_len 20 --pred_len 30 --traj_save_path '/cluster/scratch/vjose/argoverse/traj' --model_path '/cluster/home/vjose/argoverse-forecasting/saved_models/lstm_social/LSTM_rollout30.pth.tar'
    "
    #python lstm_train_test.py --test --train_features '/cluster/scratch/vjose/argoverse/forecasting_features_train.pkl' --val_features '/cluster/scratch/vjose/argoverse/forecasting_features_val.pkl' --model_path '/cluster/scratch/vjose/argoverse/' --use_social --use_delta --normalize --obs_len 20 --pred_len 30 --traj_save_path '/cluster/scratch/vjose/argoverse/'
echo $cmd
eval $cmd

# https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs
