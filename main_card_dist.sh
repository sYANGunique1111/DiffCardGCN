#!/bin/bash

args=(
    #train setting
    "--epochs" "100" 
    "--number_of_frames" "243"
    "--batch_size" "1024"
    "--learning_rate" "0.00006"
    "--lr_decay" "0.993"
    "--dataset" "h36m"
    "--keypoints" "cpn_ft_h36m_dbb"
    "--timestep" "25"
    # "--evaluate" ""
    "--checkpoint" "checkpoint/model_h36m" 
    "--nolog"
    # "--evalu_mpjpe"
    "--checkpoint" "experiments" 
    #data setting
    # "--mode" "gt"
    # distributed
    "--world_size" "1" 
    "--master_port" "7500" 
    "--master_addr" "127.0.0.1" 
    "--reduce_rank" "0"
)

CUDA_VISIBLE_DEVICES="0,1" python main_card_dist.py ${args[@]}
