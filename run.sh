#!/bin/bash

export PYTHONUNBUFFERED=1

# Base output directory for experiment artifacts
experiments_base_dir="./experiments"

# Generate timestamp-based directory to store results (prevents overwriting)
timestamp=$(date +"%Y%m%d%H%M")
#timestamp=202604081158

# Experiment configuration parameters
# num_clients_list="10 20 40 60 80"
# dataset_list=("mnist" "cifar10" "tiny_imagenet")
# model_list=("lenet" "resnet18" "vgg16")
num_clients_list="20"
dataset_list=("tiny_imagenet")
model_list=("vgg16")
# data_distribution_list=("equal_size_equal_class" "unequal_size_equal_class" "equal_size_unequal_class" "unequal_size_unequal_class")
# attack_type_list=("pgd" "mi_fgsm" "ni_fgsm" "si_ni_fgsm" "vmi_fgsm" "emi_fgsm" "margin") 
data_distribution_list=("equal_size_equal_class")
attack_type_list=("pgd") 

echo -e "===== 实验标记 ===== \n ${timestamp}"

length=${#model_list[@]}
batch_size=64


for ((i=0; i<length; i++)); do

    model=${model_list[i]}
    dataset=${dataset_list[i]}

    for num_clients in $num_clients_list; do

        for data_distribution in  "${data_distribution_list[@]}"; do

            experiments_dir="${experiments_base_dir}/${timestamp}/${dataset}/${model}/clients_${num_clients}/${data_distribution}"

            # Set the learning rate
            if [ "$dataset" = "mnist" ]; then
                learning_rate=0.1
            else
                learning_rate=0.01
            fi            
        
            # step1
            start_time=$(date +%s.%N)
            python3 step1_training.py \
                --num_clients $num_clients \
                --experiments_dir $experiments_dir \
                --dataset $dataset \
                --model $model \
                --batch_size $batch_size \
                --learning_rate $learning_rate \
                --data_distribution $data_distribution \
                --num_rounds 20
            end_time=$(date +%s.%N)
            run_time=$(echo "$end_time - $start_time" | bc)
            echo "step1 execution time: $run_time 秒"   

           
            # step2
            for attack_type in "${attack_type_list[@]}"; do
                echo "======================================"
                echo "Running attack: ${attack_type}"
                echo "======================================"
                start_time=$(date +%s.%N)
                python3 step2_gen_watermarks.py \
                    --num_clients $num_clients \
                    --experiments_dir $experiments_dir \
                    --dataset $dataset \
                    --batch_size $batch_size \
                    --model $model \
                    --attack_type $attack_type
                end_time=$(date +%s.%N)
                run_time=$(echo "$end_time - $start_time" | bc)
                echo "step2 execution time: $run_time 秒"  


                # step3
                start_time=$(date +%s.%N)
                python3 step3_verification.py \
                    --num_clients $num_clients \
                    --experiments_dir $experiments_dir \
                    --dataset $dataset \
                    --model $model \
                    --attack_type $attack_type
                end_time=$(date +%s.%N)
                run_time=$(echo "$end_time - $start_time" | bc)
                echo "step3 execution time: $run_time 秒"  
            done
        done
    done
done
