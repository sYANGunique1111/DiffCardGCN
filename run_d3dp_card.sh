#!/bin/bash

args=(
    #train setting
    "--epochs" "400" 
    "--stride" "243" 
    "--batch-size" "1024"
    "--learning-rate" "0.00006"
    "--lr-decay" "0.993"
    "--dataset" "h36m"
    "--keypoints" "cpn_ft_h36m_dbb"
    # "--evaluate" ""
    "--checkpoint" "checkpoint/model_h36m" 
    "--nolog"
    "-gpu" "0,1"
    #data setting
    # "--mode" "gt"
    # "--mode" "gt"
    # "--dataset" "h36m"
    # "--path" "./dataset/shuoyang67/H36m/annot"
    # "--checkpoint" "./checkpoint" 
    # vit setting
    # "--norm_adj"
    # "--norm_mode" "normal"
    # "--weightsharemode" "pre"
    # distributed
    # "--world_size" "2" 
    # "--master_port" "7500" 
    # "--master_addr" "127.0.0.1" 
    # "--reduce_rank" "0"
)

CUDA_VISIBLE_DEVICES="0,1" python main_card.py ${args[@]}